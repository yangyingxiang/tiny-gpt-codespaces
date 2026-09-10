"""Evaluation: masked val loss + exact-match accuracy of greedy generations.

Standalone use (re-creates the same split the checkpoint was trained with):

    python -m sft.evaluate --ckpt checkpoints/model.pt
"""

from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
    __package__ = "sft"

import argparse
import math
from pathlib import Path

import torch

from .config import CHECKPOINT_DIR, HOLDOUT_FILE, RAW_FILES, ModelConfig
from .data import Example, build_prompt_ids, load_jsonl, load_raw_examples, split_examples
from .model import GPT
from .tokenizer import EOS_ID, ByteTokenizer
from .utils import resolve_device, set_seed


@torch.no_grad()
def eval_loss(model: GPT, loader, device: str, max_batches: int | None = None) -> float:
    """Average masked loss over (up to) `max_batches` batches."""
    was_training = model.training
    model.eval()
    losses = []
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        _, loss = model(x.to(device), y.to(device))
        losses.append(loss.item())
    model.train(was_training)
    return sum(losses) / max(len(losses), 1)


@torch.no_grad()
def generate_answer(model: GPT, tok: ByteTokenizer, ex: Example, device: str, max_new_tokens: int) -> str:
    prompt = build_prompt_ids(tok, ex.instruction, ex.input)
    idx = torch.tensor([prompt[-model.cfg.block_size :]], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=max_new_tokens, eos_id=EOS_ID)
    new_ids = out[0, idx.size(1):].tolist()
    if EOS_ID in new_ids:
        new_ids = new_ids[: new_ids.index(EOS_ID)]
    return tok.decode(new_ids)


@torch.no_grad()
def exact_match(model, tok, examples: list[Example], device: str, max_new_tokens: int = 48,
                limit: int | None = None, batch_size: int = 64) -> tuple[float, list[tuple[Example, str]]]:
    """Fraction of examples whose greedy answer equals the reference exactly.

    Prompts of equal length are generated together as one batch (no padding needed).
    Results are returned in the original order.
    """
    examples = examples[:limit] if limit else examples
    prompts = [build_prompt_ids(tok, ex.instruction, ex.input)[-model.cfg.block_size :] for ex in examples]
    by_len: dict[int, list[int]] = {}
    for i, p in enumerate(prompts):
        by_len.setdefault(len(p), []).append(i)

    preds: list[str] = [""] * len(examples)
    for length, idxs in by_len.items():
        for s in range(0, len(idxs), batch_size):
            chunk = idxs[s : s + batch_size]
            idx = torch.tensor([prompts[i] for i in chunk], dtype=torch.long, device=device)
            out = model.generate(idx, max_new_tokens=max_new_tokens, eos_id=EOS_ID)
            for row, i in zip(out[:, length:].tolist(), chunk):
                if EOS_ID in row:
                    row = row[: row.index(EOS_ID)]
                preds[i] = tok.decode(row)

    correct = sum(p.strip() == ex.output.strip() for p, ex in zip(preds, examples))
    return correct / max(len(examples), 1), list(zip(examples, preds))


def load_checkpoint(path: Path, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = GPT(ModelConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate an SFT checkpoint.")
    p.add_argument("--ckpt", default=str(CHECKPOINT_DIR / "model.pt"))
    p.add_argument("--limit", type=int, default=400)
    p.add_argument("--show", type=int, default=8, help="print this many val predictions")
    args = p.parse_args()

    device = resolve_device("auto")
    model, ckpt = load_checkpoint(Path(args.ckpt), device)
    tcfg = ckpt["train_config"]
    set_seed(tcfg["seed"])
    tok = ByteTokenizer()

    _, val = split_examples(load_raw_examples(RAW_FILES, verbose=False), tcfg["val_frac"])
    holdout = load_jsonl(HOLDOUT_FILE)
    val_em, preds = exact_match(model, tok, val, device, tcfg["max_new_tokens"], args.limit)
    hold_em, _ = exact_match(model, tok, holdout, device, tcfg["max_new_tokens"], args.limit)
    print(f"checkpoint step {ckpt['step']}  (val loss at save: {ckpt['val_loss']:.4f}, "
          f"ppl {math.exp(min(ckpt['val_loss'], 20)):.2f})")
    print(f"exact match   val {val_em:6.1%}   holdout {hold_em:6.1%}")
    for ex, pred in preds[: args.show]:
        mark = "OK " if pred.strip() == ex.output.strip() else "BAD"
        print(f"  [{mark}] {ex.instruction} {ex.input!r} -> {pred!r} (want {ex.output!r})")


if __name__ == "__main__":
    main()
