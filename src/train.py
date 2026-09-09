"""Training loop for the tiny character-level GPT.

Run it directly:
    python -m src.train --max_iters 200

Or press F5 in VS Code and pick one of the "Train ..." configurations. The
breakpoint-friendly spots are marked with `# BREAKPOINT:` comments.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import torch

from .config import GPTConfig, TrainConfig, describe, parse_args
from .data import CharDataset
from .model import GPT


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warmup followed by cosine decay to `min_lr_ratio * learning_rate`."""
    if step < cfg.warmup_iters:
        return cfg.learning_rate * (step + 1) / max(cfg.warmup_iters, 1)
    progress = (step - cfg.warmup_iters) / max(cfg.max_iters - cfg.warmup_iters, 1)
    progress = min(max(progress, 0.0), 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_lr = cfg.learning_rate * cfg.min_lr_ratio
    return min_lr + coeff * (cfg.learning_rate - min_lr)


@torch.no_grad()
def estimate_loss(model: GPT, dataset: CharDataset, cfg: TrainConfig, device: str) -> dict:
    """Average the loss over a few batches of each split."""
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            x, y = dataset.get_batch(split, cfg.batch_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train(model_cfg: GPTConfig, train_cfg: TrainConfig) -> Path:
    torch.manual_seed(train_cfg.seed)
    device = train_cfg.resolved_device()

    print("model config:\n" + describe(model_cfg))
    print("train config:\n" + describe(train_cfg))
    print(f"device: {device}")

    # ---- data -------------------------------------------------------------
    dataset = CharDataset.from_file(block_size=model_cfg.block_size)
    model_cfg.vocab_size = dataset.vocab_size    # the corpus defines the vocab
    print(dataset)

    # ---- model ------------------------------------------------------------
    model = GPT(model_cfg).to(device)
    print(f"parameters: {model.num_parameters() / 1e6:.2f}M")

    optimizer = model.configure_optimizer(train_cfg.weight_decay, train_cfg.learning_rate)
    if train_cfg.compile_model and hasattr(torch, "compile"):
        # Note: torch.compile speeds things up but hides Python frames from the
        # debugger, so leave it off while stepping through the code.
        model = torch.compile(model)

    out_dir = Path(train_cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset.tokenizer.save(out_dir / "tokenizer.json")

    best_val = float("inf")
    ckpt_path = out_dir / "model.pt"
    t0 = time.time()

    # ---- loop -------------------------------------------------------------
    for step in range(train_cfg.max_iters):
        lr = lr_at(step, train_cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # BREAKPOINT: inspect x/y here to see a batch of (B, T) token ids.
        x, y = dataset.get_batch("train", train_cfg.batch_size, device)

        # BREAKPOINT: step into model(...) to walk through attention.
        logits, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()

        if step % train_cfg.log_interval == 0:
            print(
                f"iter {step:5d} | loss {loss.item():.4f} | lr {lr:.2e} "
                f"| grad_norm {grad_norm:.2f} | {time.time() - t0:.1f}s"
            )

        if step > 0 and step % train_cfg.eval_interval == 0:
            # BREAKPOINT: a good place to watch train/val diverge.
            stats = estimate_loss(model, dataset, train_cfg, device)
            print(f"  eval @ {step}: train {stats['train']:.4f} | val {stats['val']:.4f}")
            if stats["val"] < best_val:
                best_val = stats["val"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "model_config": model_cfg.__dict__,
                        "step": step,
                        "val_loss": best_val,
                    },
                    ckpt_path,
                )
                print(f"  saved checkpoint -> {ckpt_path} (val {best_val:.4f})")

    # Always leave a usable checkpoint behind, even for very short runs.
    if not ckpt_path.exists():
        torch.save(
            {"model": model.state_dict(), "model_config": model_cfg.__dict__,
             "step": train_cfg.max_iters, "val_loss": best_val},
            ckpt_path,
        )
        print(f"saved final checkpoint -> {ckpt_path}")

    print(f"done in {time.time() - t0:.1f}s")
    return ckpt_path


def main() -> None:
    model_cfg, train_cfg = parse_args()
    train(model_cfg, train_cfg)


if __name__ == "__main__":
    main()
