"""Variant 1: pure PyTorch SFT with hand-rolled LoRA, SDPA attention and gradient
accumulation. Two phases in one run:

    phase 1  full fine-tune of a tiny GPT on the base tasks  (Reverse, Uppercase, Repeat)
    phase 2  freeze it, attach LoRA, train only the adapters on the new tasks
             (Sort letters, Length, Count vowels), then merge the adapters for inference

    python sft/v1_torch_lora/main.py            # full run
    python sft/v1_torch_lora/main.py --quick    # ~1/5 of the iterations
    python -m sft.v1_torch_lora.main --reuse_base   # skip phase 1 if base.pt exists

Or open this file and use "Run Python File" / F5 in VS Code.
"""

from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    __package__ = "sft.v1_torch_lora"

import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .config import (ADAPTER_TASKS, BASE_TASKS, HOLDOUT_FILE, TRAIN_FILE, ModelConfig,
                     TrainConfig, describe, parse_args)
from .data import (ByteTokenizer, SFTDataset, filter_tasks, infinite, load_jsonl, make_loader,
                   split_examples)
from .evaluate import eval_loss, exact_match, per_task_table
from .lora import attach_lora, lora_state_dict, merge_lora
from .model import GPT, make_optimizer


def set_seed(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def lr_at(step: int, max_iters: int, peak: float, cfg: TrainConfig) -> float:
    """Linear warmup, then cosine decay to min_lr_ratio * peak."""
    if step < cfg.warmup_iters:
        return peak * (step + 1) / max(cfg.warmup_iters, 1)
    progress = (step - cfg.warmup_iters) / max(max_iters - cfg.warmup_iters, 1)
    progress = min(max(progress, 0.0), 1.0)
    min_lr = peak * cfg.min_lr_ratio
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * progress)) * (peak - min_lr)


def run_phase(name: str, model: GPT, train_ex, val_ex, max_iters: int, peak_lr: float,
              tok: ByteTokenizer, mcfg: ModelConfig, tcfg: TrainConfig, device: str) -> dict:
    """Generic loop: gradient accumulation over `accum_steps` micro-batches per optimizer step."""
    train_loader = make_loader(SFTDataset(train_ex, tok), tcfg.micro_batch, mcfg.block_size, True, tcfg.seed)
    val_loader = make_loader(SFTDataset(val_ex, tok), tcfg.micro_batch, mcfg.block_size, False)
    optimizer = make_optimizer(model, tcfg.weight_decay, peak_lr)
    n_train = model.num_parameters(trainable_only=True)
    print(f"\n=== phase: {name} | train {len(train_ex)} | val {len(val_ex)} | "
          f"trainable params {n_train / 1e3:.1f}k of {model.num_parameters() / 1e3:.1f}k ===")

    batches = infinite(train_loader)
    history, t0 = [], time.time()
    model.train()
    for step in range(max_iters + 1):
        if step % tcfg.eval_interval == 0 or step == max_iters:
            # BREAKPOINT: val loss and exact match should move together.
            vl = eval_loss(model, val_loader, device, tcfg.eval_batches)
            em, _ = exact_match(model, tok, val_ex, device, tcfg.max_new_tokens, tcfg.eval_examples)
            history.append({"step": step, "val_loss": vl, "val_em": em})
            print(f"  eval @ {step:5d}: val loss {vl:.4f} | val exact-match {em:6.1%}")
        if step == max_iters:
            break

        lr = lr_at(step, max_iters, peak_lr, tcfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        loss_acc = 0.0
        for _ in range(tcfg.accum_steps):
            # BREAKPOINT: x[0] / y[0] decoded with tok.decode(..., skip_special=False).
            optimizer.zero_grad(set_to_none=True)
            x, y = next(batches)
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            (loss / tcfg.accum_steps).backward()
            loss_acc += loss.item() / tcfg.accum_steps
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        optimizer.step()

        if step % tcfg.log_interval == 0:
            print(f"  iter {step:5d} | loss {loss_acc:.4f} | lr {lr:.2e} | "
                  f"grad_norm {grad_norm:.2f} | {time.time() - t0:.1f}s")
    return {"history": history, "seconds": time.time() - t0}


def main() -> None:
    mcfg, tcfg = parse_args()
    set_seed(tcfg.seed)
    device = resolve_device(tcfg.device)
    tok = ByteTokenizer()
    mcfg.vocab_size = tok.vocab_size
    print("model config:\n" + describe(mcfg))
    print("train config:\n" + describe(tcfg))
    print(f"device: {device} | torch threads {torch.get_num_threads()}")
    out_dir = Path(tcfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # ---- data ---------------------------------------------------------------
    train_all, val_all = split_examples(load_jsonl(TRAIN_FILE), tcfg.val_frac, tcfg.seed)
    holdout = load_jsonl(HOLDOUT_FILE)
    base_train, base_val = filter_tasks(train_all, BASE_TASKS), filter_tasks(val_all, BASE_TASKS)
    ad_train, ad_val = filter_tasks(train_all, ADAPTER_TASKS), filter_tasks(val_all, ADAPTER_TASKS)
    base_hold, ad_hold = filter_tasks(holdout, BASE_TASKS), filter_tasks(holdout, ADAPTER_TASKS)
    print(f"examples: train {len(train_all)} | val {len(val_all)} | holdout {len(holdout)}")

    # ---- phase 1: full fine-tune on the base tasks ---------------------------
    model = GPT(mcfg).to(device)
    base_path = out_dir / "base.pt"
    if tcfg.reuse_base and base_path.exists():
        model.load_state_dict(torch.load(base_path, map_location=device))
        print(f"loaded base model from {base_path}")
        base_stats = {"history": [], "seconds": 0.0}
    else:
        base_stats = run_phase("base (full fine-tune)", model, base_train, base_val,
                               tcfg.base_iters, tcfg.base_lr, tok, mcfg, tcfg, device)
        torch.save(model.state_dict(), base_path)
    base_em_before, _ = exact_match(model, tok, base_hold, device, tcfg.max_new_tokens, tcfg.final_eval_examples)
    print(f"base tasks, holdout exact match: {base_em_before:6.1%}")

    # ---- phase 2: LoRA on the adapter tasks (+ a replay slice of base tasks) --
    wrapped = attach_lora(model, tcfg.lora_r, tcfg.lora_alpha)
    for p in model.parameters():
        p.requires_grad_(False)
    model.to(device)
    print(f"LoRA r={tcfg.lora_r} alpha={tcfg.lora_alpha} on {len(wrapped)} layers: {wrapped[:3]} ...")
    replay = base_train[: int(tcfg.replay_frac * len(ad_train))]
    lora_stats = run_phase("adapter (LoRA)", model, ad_train + replay, ad_val,
                           tcfg.lora_iters, tcfg.lora_lr, tok, mcfg, tcfg, device)
    torch.save(lora_state_dict(model), out_dir / "lora.pt")
    unmerged_em, _ = exact_match(model, tok, ad_val, device, tcfg.max_new_tokens, tcfg.final_eval_examples)

    # ---- merge and final evaluation ------------------------------------------
    n_merged = merge_lora(model)
    print(f"\nmerged {n_merged} adapters into the base weights")
    val_em, val_pairs = exact_match(model, tok, ad_val, device, tcfg.max_new_tokens, tcfg.final_eval_examples)
    hold_em, hold_pairs = exact_match(model, tok, ad_hold, device, tcfg.max_new_tokens, tcfg.final_eval_examples)
    base_em_after, _ = exact_match(model, tok, base_hold, device, tcfg.max_new_tokens, tcfg.final_eval_examples)
    print(f"adapter tasks exact match  ->  val {val_em:6.1%} (unmerged {unmerged_em:6.1%})  |  holdout {hold_em:6.1%}")
    print("  per task (holdout):\n" + per_task_table(hold_pairs))
    print(f"base tasks, holdout exact match after merge: {base_em_after:6.1%} (before: {base_em_before:6.1%})")
    for ex, pred in hold_pairs[:5]:
        print(f"  {ex.instruction} {ex.input!r} -> {pred!r}  (want {ex.output!r})")

    metrics = {"base": base_stats, "lora": lora_stats, "adapter_val_em": val_em,
               "adapter_val_em_unmerged": unmerged_em, "adapter_holdout_em": hold_em,
               "base_holdout_em_before": base_em_before, "base_holdout_em_after": base_em_after,
               "model_config": asdict(mcfg), "train_config": asdict(tcfg),
               "seconds": time.time() - t_start}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    torch.save(model.state_dict(), out_dir / "merged.pt")
    print(f"done in {time.time() - t_start:.1f}s; wrote {out_dir}/metrics.json")


if __name__ == "__main__":
    main()
