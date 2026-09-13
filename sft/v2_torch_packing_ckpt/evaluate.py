"""Evaluation: token-weighted val loss on packed blocks + greedy exact match on un-packed
prompts. Also usable standalone to re-score a checkpoint:

    python -m sft.v2_torch_packing_ckpt.evaluate --ckpt checkpoints/v2_torch_packing_ckpt/model.pt
"""

from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    __package__ = "sft.v2_torch_packing_ckpt"

import argparse
from collections import defaultdict
from pathlib import Path

import torch

from .config import CHECKPOINT_DIR, HOLDOUT_FILE, TRAIN_FILE, ModelConfig
from .data import EOS_ID, ByteTokenizer, Example, build_prompt_ids, load_jsonl, split_examples
from .model import GPT, loss_token_mean
from .packing import build_attention_mask


@torch.no_grad()
def eval_loss(model: GPT, loader, device: str) -> float:
    """Corpus-level NLL per supervised token: accumulate sums, divide once at the end."""
    was_training = model.training
    model.eval()
    total_nll = total_tokens = 0.0
    for batch in loader:
        logits = model(batch["input_ids"].to(device), batch["pos_ids"].to(device),
                       build_attention_mask(batch["seq_ids"]).to(device))
        out = loss_token_mean(logits, batch["labels"].to(device))
        total_nll += float(out["sum_nll"])
        total_tokens += float(out["n_tokens"])
    model.train(was_training)
    return total_nll / max(total_tokens, 1.0)


@torch.no_grad()
def exact_match(model: GPT, tok: ByteTokenizer, examples: list[Example], device: str,
                max_new_tokens: int = 48, limit: int | None = None, batch_size: int = 64
                ) -> tuple[float, list[tuple[Example, str]]]:
    """Fraction of examples whose greedy answer equals the reference exactly. Prompts of
    equal length are generated together (no padding needed); results keep input order."""
    examples = examples[:limit] if limit else examples
    prompts = [build_prompt_ids(tok, ex.instruction, ex.input) for ex in examples]
    by_len: dict[int, list[int]] = defaultdict(list)
    for i, p in enumerate(prompts):
        by_len[len(p)].append(i)

    preds: list[str] = [""] * len(examples)
    for length, idxs in by_len.items():
        for s in range(0, len(idxs), batch_size):
            chunk = idxs[s:s + batch_size]
            idx = torch.tensor([prompts[i] for i in chunk], dtype=torch.long, device=device)
            out = model.generate(idx, max_new_tokens=max_new_tokens, eos_id=EOS_ID)
            for row, i in zip(out[:, length:].tolist(), chunk):
                if EOS_ID in row:
                    row = row[:row.index(EOS_ID)]
                preds[i] = tok.decode(row)

    correct = sum(p.strip() == ex.output.strip() for p, ex in zip(preds, examples))
    return correct / max(len(examples), 1), list(zip(examples, preds))


def per_task_table(preds: list[tuple[Example, str]]) -> str:
    hits: dict[str, list[int]] = defaultdict(list)
    for ex, p in preds:
        hits[ex.instruction].append(int(p.strip() == ex.output.strip()))
    rows = [f"  {task:<16} {sum(v) / len(v):6.1%}  (n={len(v)})" for task, v in sorted(hits.items())]
    return "\n".join(rows)


def load_checkpoint(path: Path, device: str) -> tuple[GPT, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = GPT(ModelConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt


def main() -> None:
    p = argparse.ArgumentParser(description="Re-score a checkpoint on val + holdout.")
    p.add_argument("--ckpt", default=str(CHECKPOINT_DIR / "model.pt"))
    p.add_argument("--limit", type=int, default=400)
    p.add_argument("--show", type=int, default=8)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = load_checkpoint(Path(args.ckpt), device)
    tcfg = ckpt["train_config"]
    tok = ByteTokenizer()
    _, val = split_examples(load_jsonl(TRAIN_FILE), tcfg["val_frac"], tcfg["seed"])
    holdout = load_jsonl(HOLDOUT_FILE)
    val_em, preds = exact_match(model, tok, val, device, tcfg["max_new_tokens"], args.limit)
    hold_em, _ = exact_match(model, tok, holdout, device, tcfg["max_new_tokens"], args.limit)
    print(f"checkpoint step {ckpt['step']} (val loss at save: {ckpt['val_loss']:.4f})")
    print(f"exact match   val {val_em:6.1%}   holdout {hold_em:6.1%}")
    print(per_task_table(preds))
    for ex, pred in preds[:args.show]:
        mark = "OK " if pred.strip() == ex.output.strip() else "BAD"
        print(f"  [{mark}] {ex.instruction} {ex.input!r} -> {pred!r} (want {ex.output!r})")


if __name__ == "__main__":
    main()
