"""Evaluation: token-weighted masked val loss + exact match of greedy generations."""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from .config import IGNORE_INDEX
from .data import Example, VocabSlice, build_prompt, unspell


@torch.no_grad()
def eval_loss(model, loader, max_batches: int | None = None) -> float:
    """Corpus NLL over the supervised tokens: sum of per-token loss / number of tokens.

    Not a mean of per-batch means, so it does not move with the batch size.
    """
    was_training = model.training
    model.eval()
    total_nll, total_tokens = 0.0, 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch["labels"][:, 1:].contiguous()
        total_nll += F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                     shift_labels.view(-1), ignore_index=IGNORE_INDEX,
                                     reduction="sum").item()
        total_tokens += int((shift_labels != IGNORE_INDEX).sum())
    model.train(was_training)
    return total_nll / max(total_tokens, 1)


@torch.no_grad()
def exact_match(model, tok, vocab: VocabSlice, examples: list[Example], max_new_tokens: int = 48,
                limit: int | None = None, batch_size: int = 64):
    """Fraction of examples whose greedy answer equals the reference exactly.

    Prompts of equal token length are generated together, so no padding is needed
    and every row starts generating from its own last prompt token.
    Returns (em, [(example, prediction, stopped), ...]) in the original order, where
    `stopped` says whether the model emitted EOS before max_new_tokens.
    """
    was_training = model.training
    model.eval()
    examples = examples[:limit] if limit else examples
    device = next(model.parameters()).device
    prompts = [vocab.compact(tok(build_prompt(ex), add_special_tokens=False)["input_ids"])
               for ex in examples]
    by_len: dict[int, list[int]] = {}
    for i, p in enumerate(prompts):
        by_len.setdefault(len(p), []).append(i)

    preds = [""] * len(examples)
    stopped = [False] * len(examples)
    for length, idxs in by_len.items():
        for s in range(0, len(idxs), batch_size):
            chunk = idxs[s : s + batch_size]
            input_ids = torch.tensor([prompts[i] for i in chunk], dtype=torch.long, device=device)
            # BREAKPOINT: tok.decode(vocab.expand(input_ids[0])) is exactly what the model sees.
            out = model.generate(
                input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=vocab.pad_id, eos_token_id=vocab.eos_id, use_cache=True,
            )
            for row, i in zip(out[:, length:].tolist(), chunk):
                if vocab.eos_id in row:
                    stopped[i] = True
                    row = row[: row.index(vocab.eos_id)]
                preds[i] = unspell(tok.decode(vocab.expand(row)))
    model.train(was_training)
    correct = sum(p.strip() == ex.output.strip() for p, ex in zip(preds, examples))
    return correct / max(len(examples), 1), list(zip(examples, preds, stopped))


def per_task(pairs) -> dict[str, tuple[int, int, int]]:
    """{instruction: (correct, total, ran_on)} from exact_match's (example, prediction,
    stopped) triples; ran_on counts generations that never produced EOS."""
    table: dict[str, list[int]] = {}
    for ex, pred, stopped in pairs:
        row = table.setdefault(ex.instruction, [0, 0, 0])
        row[0] += int(pred.strip() == ex.output.strip())
        row[1] += 1
        row[2] += int(not stopped)
    return {k: (v[0], v[1], v[2]) for k, v in sorted(table.items())}


def format_per_task(pairs) -> str:
    lines = [f"  {k:>14} {c:4d}/{n:<4d} {c / max(n, 1):6.1%}   (no EOS: {r})"
             for k, (c, n, r) in per_task(pairs).items()]
    return "\n".join(lines)


def perplexity(loss: float) -> float:
    return math.exp(min(loss, 20.0))
