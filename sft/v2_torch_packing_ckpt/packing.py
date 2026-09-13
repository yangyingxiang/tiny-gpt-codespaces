"""Sequence packing: several examples share one block of `block_size` tokens.

A packed block carries four parallel lists:

    input_ids  the concatenated token ids
    labels     IGNORE_INDEX on prompts, the token itself on answers (unshifted)
    seq_ids    which example each position belongs to (0, 0, 0, 1, 1, 2, ...)
    pos_ids    position *within its own example*, restarting at 0 for every example

Packing only pays off if the attention mask keeps the examples apart: position t may
attend to position s only when s <= t AND seq_ids[s] == seq_ids[t]. That is what
`build_attention_mask` produces and what `assert_no_cross_attention` checks.
"""

from __future__ import annotations

import torch

from .data import IGNORE_INDEX, PAD_ID

PAD_SEQ = -1   # seq id of padding positions (they only ever attend to each other)


def pack_examples(encoded: list[dict], block_size: int) -> list[dict]:
    """Greedy, in-order packing: start a new block when the next example does not fit."""
    blocks: list[dict] = []
    cur = {"input_ids": [], "labels": [], "seq_ids": [], "pos_ids": []}
    n_seq = 0

    def flush() -> None:
        nonlocal cur, n_seq
        if cur["input_ids"]:
            blocks.append(cur)
        cur = {"input_ids": [], "labels": [], "seq_ids": [], "pos_ids": []}
        n_seq = 0

    for e in encoded:
        n = len(e["input_ids"])
        if n > block_size:
            raise ValueError(f"example of {n} tokens does not fit block_size={block_size}")
        if len(cur["input_ids"]) + n > block_size:
            flush()
        cur["input_ids"] += e["input_ids"]
        cur["labels"] += e["labels"]
        cur["seq_ids"] += [n_seq] * n
        cur["pos_ids"] += list(range(n))
        n_seq += 1
    flush()
    return blocks


def collate_blocks(blocks: list[dict]) -> dict[str, torch.Tensor]:
    """Right-pad a list of blocks to the longest one. Three fill values, three meanings:
    input_ids <- PAD_ID, labels <- IGNORE_INDEX, seq_ids <- PAD_SEQ."""
    bsz = len(blocks)
    max_len = max(len(b["input_ids"]) for b in blocks)
    input_ids = torch.full((bsz, max_len), PAD_ID, dtype=torch.long)
    labels = torch.full((bsz, max_len), IGNORE_INDEX, dtype=torch.int32)
    seq_ids = torch.full((bsz, max_len), PAD_SEQ, dtype=torch.long)
    pos_ids = torch.zeros((bsz, max_len), dtype=torch.long)
    for i, b in enumerate(blocks):
        n = len(b["input_ids"])
        input_ids[i, :n] = torch.tensor(b["input_ids"], dtype=torch.long)
        labels[i, :n] = torch.tensor(b["labels"], dtype=torch.int32)
        seq_ids[i, :n] = torch.tensor(b["seq_ids"], dtype=torch.long)
        pos_ids[i, :n] = torch.tensor(b["pos_ids"], dtype=torch.long)
    return {"input_ids": input_ids, "labels": labels, "seq_ids": seq_ids, "pos_ids": pos_ids}


def build_attention_mask(seq_ids: torch.Tensor) -> torch.Tensor:
    """(B, T) seq ids -> (B, 1, T, T) bool mask, True where attention is allowed:
    causal within the block AND the two positions belong to the same example."""
    B, T = seq_ids.shape
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=seq_ids.device))
    return causal.view(1, 1, T, T).expand(B, 1, T, T)


@torch.no_grad()
def assert_no_cross_attention(model, block: dict, seq: int = 1, atol: float = 1e-4) -> None:
    """The logits of example `seq` must be identical whether it is computed inside the
    packed block or alone (positions 0..n-1, plain causal mask). If they differ, some
    position of `seq` is looking at another example's tokens."""
    batch = collate_blocks([block])
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    packed = model(batch["input_ids"].to(device), batch["pos_ids"].to(device),
                   build_attention_mask(batch["seq_ids"]).to(device))
    sel = batch["seq_ids"][0] == seq
    if not sel.any():
        raise ValueError(f"block has no example with seq id {seq}")
    alone = model(batch["input_ids"][:, sel].to(device))       # default pos ids + causal mask
    model.train(was_training)
    diff = (packed[0, sel] - alone[0]).abs().max().item()
    if diff > atol:
        raise AssertionError(f"example {seq} sees other examples in its pack "
                             f"(max |logit diff| = {diff:.3g} > {atol})")
