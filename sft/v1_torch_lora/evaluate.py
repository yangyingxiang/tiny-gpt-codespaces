"""Masked val loss + greedy exact match, overall and per task."""

from __future__ import annotations

import torch

from .data import EOS_ID, ByteTokenizer, Example, build_prompt_ids


@torch.no_grad()
def eval_loss(model, loader, device: str, max_batches: int | None = None) -> float:
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
def exact_match(model, tok: ByteTokenizer, examples: list[Example], device: str,
                max_new_tokens: int = 48, limit: int | None = None, batch_size: int = 64):
    """Fraction of examples whose greedy answer equals the reference. Prompts of equal
    length are generated together (no padding needed). Returns (em, [(example, pred)])."""
    examples = examples[:limit] if limit else examples
    prompts = [build_prompt_ids(tok, ex.instruction, ex.input)[-model.cfg.block_size:] for ex in examples]
    by_len: dict[int, list[int]] = {}
    for i, p in enumerate(prompts):
        by_len.setdefault(len(p), []).append(i)
    preds = [""] * len(examples)
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


def per_task_table(pairs: list[tuple[Example, str]]) -> str:
    stats: dict[str, list[int]] = {}
    for ex, pred in pairs:
        s = stats.setdefault(ex.instruction, [0, 0])
        s[0] += pred.strip() == ex.output.strip()
        s[1] += 1
    return "\n".join(f"    {task:<14} {ok:4d}/{n:<4d} {ok / n:6.1%}" for task, (ok, n) in sorted(stats.items()))
