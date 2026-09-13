"""SFT of a tiny GPT with sequence packing + gradient checkpointing (epoch loop).

    python sft/v2_torch_packing_ckpt/main.py              # full run (36 epochs, ~4 min on 2 cores)
    python sft/v2_torch_packing_ckpt/main.py --quick      # 4 epochs, small evals
    python -m sft.v2_torch_packing_ckpt.main --epochs 4

Pipeline:  jsonl -> split -> encode (label mask) -> pack into 256-token blocks
           -> block-diagonal causal mask -> GPT (SDPA, checkpointed blocks)
           -> token-mean loss -> AdamW + warmup/cosine -> eval (val loss, exact match)
"""

from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    __package__ = "sft.v2_torch_packing_ckpt"

import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import HOLDOUT_FILE, TRAIN_FILE, ModelConfig, TrainConfig, describe, parse_args
from .data import ByteTokenizer, encode_example, load_jsonl, split_examples
from .evaluate import eval_loss, exact_match, per_task_table
from .model import GPT, loss_token_mean
from .packing import build_attention_mask, collate_blocks, pack_examples


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def make_scheduler(optimizer, total_steps: int, warmup_frac: float, min_lr_ratio: float):
    """Linear warmup then cosine decay to min_lr_ratio, over `total_steps` optimizer steps."""
    warmup = max(1, int(warmup_frac * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return min_lr_ratio + 0.5 * (1.0 + math.cos(math.pi * progress)) * (1.0 - min_lr_ratio)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(mcfg: ModelConfig, tcfg: TrainConfig) -> dict:
    set_seed(tcfg.seed)
    device = resolve_device(tcfg.device)
    tok = ByteTokenizer()
    mcfg.vocab_size = tok.vocab_size
    print("model config:\n" + describe(mcfg))
    print("train config:\n" + describe(tcfg))
    print(f"device: {device}")

    # ---- data -------------------------------------------------------------
    train_ex, val_ex = split_examples(load_jsonl(TRAIN_FILE), tcfg.val_frac, tcfg.seed)
    holdout_ex = load_jsonl(HOLDOUT_FILE)
    train_blocks = pack_examples([encode_example(tok, ex) for ex in train_ex], mcfg.block_size)
    val_blocks = pack_examples([encode_example(tok, ex) for ex in val_ex], mcfg.block_size)
    n_tok = sum(len(b["input_ids"]) for b in train_blocks)
    print(f"examples: train {len(train_ex)} | val {len(val_ex)} | holdout {len(holdout_ex)}")
    print(f"packed: {len(train_blocks)} train blocks of <= {mcfg.block_size} tokens "
          f"({n_tok / len(train_blocks):.0f} tokens/block, "
          f"{len(train_ex) / len(train_blocks):.1f} examples/block)")

    g = torch.Generator().manual_seed(tcfg.seed)
    train_loader = DataLoader(train_blocks, batch_size=tcfg.batch_size, shuffle=True,
                              generator=g, collate_fn=collate_blocks)
    val_loader = DataLoader(val_blocks, batch_size=tcfg.batch_size, shuffle=False,
                            collate_fn=collate_blocks)

    # ---- model / optimizer / schedule --------------------------------------
    model = GPT(mcfg).to(device)
    print(f"parameters: {model.num_parameters() / 1e6:.2f}M")
    optimizer = model.configure_optimizer(tcfg.weight_decay, tcfg.learning_rate)
    total_steps = len(train_loader) * tcfg.epochs
    print(f"steps: {len(train_loader)}/epoch x {tcfg.epochs} epochs = {total_steps}")

    out_dir = Path(tcfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "model.pt"
    best_val, history, step = float("inf"), [], 0
    t0 = time.time()

    def evaluate(tag: str) -> None:
        nonlocal best_val
        # BREAKPOINT: val loss and exact match should move together.
        vl = eval_loss(model, val_loader, device)
        em, _ = exact_match(model, tok, val_ex, device, tcfg.max_new_tokens, tcfg.eval_examples)
        history.append({"step": step, "val_loss": vl, "val_em": em})
        print(f"  eval @ {step:5d} ({tag}): val loss {vl:.4f} | val exact-match {em:6.1%}")
        if vl < best_val:
            best_val = vl
            torch.save({"model": model.state_dict(), "model_config": asdict(mcfg),
                        "train_config": asdict(tcfg), "step": step, "val_loss": vl}, ckpt_path)

    # ---- loop -------------------------------------------------------------
    model.train()
    evaluate("start")
    for epoch in range(tcfg.epochs):
        scheduler = make_scheduler(optimizer, len(train_loader), tcfg.warmup_frac,
                                   tcfg.min_lr_ratio)
        run_nll = run_tokens = 0.0
        for batch in train_loader:
            # BREAKPOINT: one packed batch. tok.decode(batch["input_ids"][0].tolist(),
            # skip_special=False) shows several examples back to back.
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attn_mask = build_attention_mask(batch["seq_ids"]).to(device)
            pos_ids = batch["pos_ids"].to(device)

            logits = model(input_ids, pos_ids, attn_mask)
            out = loss_token_mean(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            out["loss"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
            optimizer.step()
            scheduler.step()
            step += 1
            run_nll += float(out["sum_nll"])
            run_tokens += float(out["n_tokens"])

            if step % tcfg.log_interval == 0:
                print(f"iter {step:5d} | loss {run_nll / max(run_tokens, 1):.4f} | "
                      f"lr {scheduler.get_last_lr()[0]:.2e} | grad_norm {grad_norm:.2f} | "
                      f"{time.time() - t0:.1f}s")
                run_nll = run_tokens = 0.0
            if step % tcfg.eval_interval == 0 or step == total_steps:
                evaluate(f"epoch {epoch + 1}")

    # ---- final evaluation on the best checkpoint --------------------------
    best = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    val_em, val_preds = exact_match(model, tok, val_ex, device, tcfg.max_new_tokens,
                                    tcfg.final_eval_examples)
    hold_em, hold_preds = exact_match(model, tok, holdout_ex, device, tcfg.max_new_tokens,
                                      tcfg.final_eval_examples)
    metrics = {"best_step": best["step"], "best_val_loss": best_val,
               "val_exact_match": val_em, "holdout_exact_match": hold_em,
               "train_examples": len(train_ex), "val_examples": len(val_ex),
               "seconds": time.time() - t0, "history": history}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"\nbest checkpoint: step {best['step']} (val loss {best_val:.4f})")
    print(f"exact match  ->  val {val_em:6.1%}  |  holdout {hold_em:6.1%}")
    print("holdout by task:\n" + per_task_table(hold_preds))
    for ex, pred in hold_preds[:5]:
        print(f"  {ex.instruction} {ex.input!r} -> {pred!r}  (want {ex.output!r})")
    print(f"done in {time.time() - t0:.1f}s")
    return metrics


def main() -> None:
    train(*parse_args())


if __name__ == "__main__":
    main()
