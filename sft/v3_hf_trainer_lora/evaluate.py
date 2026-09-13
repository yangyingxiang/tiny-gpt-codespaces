"""Exact-match evaluation with batched greedy generation.

Prompts are batched and padded with the tokenizer, mapped to compact ids, generated greedily,
mapped back to gpt2 ids and decoded. A prediction counts when it equals the reference exactly.
"""

from __future__ import annotations

import torch

from .data import Example, VocabMap, build_prompt, unspell


@torch.no_grad()
def exact_match(model, tok, vmap: VocabMap, examples: list[Example], max_new_tokens: int = 40,
                batch_size: int = 32, limit: int | None = None
                ) -> tuple[float, dict[str, float], list[tuple[Example, str]]]:
    """(overall EM, per-task EM, [(example, prediction), ...]) for greedy decoding."""
    examples = examples[:limit] if limit else examples
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    preds: list[str] = []
    for s in range(0, len(examples), batch_size):
        chunk = examples[s : s + batch_size]
        enc = tok([build_prompt(ex) for ex in chunk], return_tensors="pt", padding=True,
                  add_special_tokens=False)
        input_ids = vmap.encode_tensor(enc["input_ids"]).to(device)
        attention_mask = enc["attention_mask"].to(device)
        # BREAKPOINT: tok.decode(vmap.decode_tensor(input_ids[0])) is what row 0 continues from.
        out = model.generate(input_ids=input_ids, attention_mask=attention_mask, do_sample=False,
                             max_new_tokens=max_new_tokens, pad_token_id=vmap.pad_id,
                             eos_token_id=vmap.eos_id)
        for row in out[:, input_ids.shape[1]:].tolist():
            if vmap.eos_id in row:
                row = row[: row.index(vmap.eos_id)]
            full = vmap.decode_tensor(torch.tensor(row, dtype=torch.long)).tolist()
            preds.append(unspell(tok.decode(full, skip_special_tokens=True)))
    model.train(was_training)

    hits: dict[str, list[int]] = {}
    for ex, pred in zip(examples, preds):
        hits.setdefault(ex.instruction, []).append(int(pred.strip() == ex.output.strip()))
    per_task = {k: sum(v) / len(v) for k, v in sorted(hits.items())}
    overall = sum(sum(v) for v in hits.values()) / max(len(examples), 1)
    return overall, per_task, list(zip(examples, preds))


def format_per_task(per_task: dict[str, float]) -> str:
    return "  ".join(f"{k} {v:5.1%}" for k, v in per_task.items())
