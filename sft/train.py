"""Supervised fine-tuning loop.

    python -m sft.train                      # full run (~2-3 min on 2 CPU cores)
    python -m sft.train --max_iters 200      # quick look

Or press F5 and pick a "Train ..." configuration. Breakpoint-friendly spots are
marked with `# BREAKPOINT:`.
"""

from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
    __package__ = "sft"

import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .config import HOLDOUT_FILE, RAW_FILES, ModelConfig, TrainConfig, describe, parse_args
from .data import SFTDataset, load_jsonl, load_raw_examples, make_loader, split_examples
from .evaluate import eval_loss, exact_match
from .model import GPT
from .tokenizer import ByteTokenizer
from .utils import resolve_device, set_seed


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warmup, then cosine decay to min_lr_ratio * learning_rate."""
    if step < cfg.warmup_iters:
        return cfg.learning_rate * (step + 1) / max(cfg.warmup_iters, 1)
    progress = (step - cfg.warmup_iters) / max(cfg.max_iters - cfg.warmup_iters, 1)
    progress = min(max(progress, 0.0), 1.0)
    min_lr = cfg.learning_rate * cfg.min_lr_ratio
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * progress)) * (cfg.learning_rate - min_lr)


def infinite(loader):
    while True:
        yield from loader


def train(mcfg: ModelConfig, tcfg: TrainConfig) -> dict:
    set_seed(tcfg.seed)
    device = resolve_device(tcfg.device)
    tok = ByteTokenizer()
    mcfg.vocab_size = tok.vocab_size
    print("model config:\n" + describe(mcfg))
    print("train config:\n" + describe(tcfg))
    print(f"device: {device}")

    # ---- data -------------------------------------------------------------
    examples = load_raw_examples(RAW_FILES)
    train_ex, val_ex = split_examples(examples, tcfg.val_frac)
    holdout_ex = load_jsonl(HOLDOUT_FILE)
    print(f"examples: raw {len(examples)} -> train {len(train_ex)} | val {len(val_ex)} | holdout {len(holdout_ex)}")

    train_ds, val_ds = SFTDataset(train_ex, tok), SFTDataset(val_ex, tok)
    train_loader = make_loader(train_ds, tcfg.batch_size, mcfg.block_size, shuffle=True, seed=tcfg.seed)
    val_loader = make_loader(val_ds, tcfg.batch_size, mcfg.block_size, shuffle=False)

    # ---- model ------------------------------------------------------------
    model = GPT(mcfg).to(device)
    print(f"parameters: {model.num_parameters() / 1e6:.2f}M")
    optimizer = model.configure_optimizer(tcfg.weight_decay, tcfg.learning_rate)

    out_dir = Path(tcfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "model.pt"
    best_val = float("inf")
    history = []
    t0 = time.time()
    batches = infinite(train_loader)

    def save(step: int, val_loss: float) -> None:
        torch.save({"model": model.state_dict(), "model_config": asdict(mcfg),
                    "train_config": asdict(tcfg), "step": step, "val_loss": val_loss}, ckpt_path)

    # ---- loop -------------------------------------------------------------
    model.train()
    for step in range(tcfg.max_iters + 1):
        if step % tcfg.eval_interval == 0 or step == tcfg.max_iters:
            # BREAKPOINT: watch val loss and exact match move together.
            vl = eval_loss(model, val_loader, device, tcfg.eval_batches)
            em, _ = exact_match(model, tok, val_ex, device, tcfg.max_new_tokens, tcfg.eval_examples)
            history.append({"step": step, "val_loss": vl, "val_em": em})
            print(f"  eval @ {step:5d}: val loss {vl:.4f} | val exact-match {em:6.1%}")
            if vl < best_val:
                best_val = vl
                save(step, vl)
        if step == tcfg.max_iters:
            break

        lr = lr_at(step, tcfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # BREAKPOINT: inspect a (B, T) batch; decode x[0] and y[0] with tok.decode(...).
        x, y = next(batches)
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        optimizer.step()

        if step % tcfg.log_interval == 0:
            print(f"iter {step:5d} | loss {loss.item():.4f} | lr {lr:.2e} | "
                  f"grad_norm {grad_norm:.2f} | {time.time() - t0:.1f}s")

    # ---- final evaluation on the best checkpoint --------------------------
    best = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    val_em, _ = exact_match(model, tok, val_ex, device, tcfg.max_new_tokens, tcfg.final_eval_examples)
    hold_em, samples = exact_match(model, tok, holdout_ex, device, tcfg.max_new_tokens, tcfg.final_eval_examples)
    metrics = {"best_step": best["step"], "best_val_loss": best_val,
               "val_exact_match": val_em, "holdout_exact_match": hold_em,
               "train_examples": len(train_ex), "val_examples": len(val_ex),
               "seconds": time.time() - t0, "history": history}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"\nbest checkpoint: step {best['step']} (val loss {best_val:.4f})")
    print(f"exact match  ->  val {val_em:6.1%}  |  holdout {hold_em:6.1%}")
    for ex, pred in samples[:5]:
        print(f"  {ex.instruction} {ex.input!r} -> {pred!r}  (want {ex.output!r})")
    print(f"done in {time.time() - t0:.1f}s")
    return metrics


def main() -> None:
    train(*parse_args())


if __name__ == "__main__":
    main()
