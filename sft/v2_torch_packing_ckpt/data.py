"""Data: jsonl -> examples -> split -> token ids + label mask.

Chat template (one training example):

    <|bos|><|user|>{instruction}\n{input}<|assistant|>{output}<|eos|>

Labels are the *same* sequence with the prompt positions set to IGNORE_INDEX. They
are NOT shifted here: the causal shift (logits[:, :-1] vs labels[:, 1:]) happens
exactly once, in `model.loss_token_mean`.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass

import torch

IGNORE_INDEX = -100

# ------------------------------------------------------------------ tokenizer
NUM_BYTES = 256
SPECIAL_TOKENS = ["<|pad|>", "<|bos|>", "<|eos|>", "<|user|>", "<|assistant|>"]
SPECIAL_IDS = {tok: NUM_BYTES + i for i, tok in enumerate(SPECIAL_TOKENS)}

PAD_ID = SPECIAL_IDS["<|pad|>"]
BOS_ID = SPECIAL_IDS["<|bos|>"]
EOS_ID = SPECIAL_IDS["<|eos|>"]
USER_ID = SPECIAL_IDS["<|user|>"]
ASSISTANT_ID = SPECIAL_IDS["<|assistant|>"]


class ByteTokenizer:
    """Ids 0..255 are raw UTF-8 bytes; special tokens follow."""

    vocab_size = NUM_BYTES + len(SPECIAL_TOKENS)

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids, skip_special: bool = True) -> str:
        out, buf = [], bytearray()
        for i in ids:
            i = int(i)
            if i < NUM_BYTES:
                buf.append(i)
                continue
            out.append(buf.decode("utf-8", errors="replace"))
            buf = bytearray()
            if not skip_special:
                out.append(SPECIAL_TOKENS[i - NUM_BYTES])
        out.append(buf.decode("utf-8", errors="replace"))
        return "".join(out)


# ------------------------------------------------------------------ examples
@dataclass(frozen=True)
class Example:
    id: str
    instruction: str
    input: str
    output: str


def load_jsonl(path) -> list[Example]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                out.append(Example(d["id"], d["instruction"], d.get("input", ""), d["output"]))
    return out


def split_examples(examples: list[Example], val_frac: float, seed: int
                   ) -> tuple[list[Example], list[Example]]:
    """Deterministic shuffle + slice. Uses its own RNG so the split never depends on
    what else consumed the global random state."""
    examples = list(examples)
    random.Random(seed).shuffle(examples)
    n_val = int(len(examples) * val_frac)
    return examples[n_val:], examples[:n_val]


# ------------------------------------------------------------------ encoding
def build_prompt_ids(tok: ByteTokenizer, instruction: str, inp: str) -> list[int]:
    text = f"{instruction}\n{inp}" if inp else instruction
    return [BOS_ID, USER_ID] + tok.encode(text) + [ASSISTANT_ID]


def encode_example(tok: ByteTokenizer, ex: Example) -> dict:
    """{"input_ids": [...], "labels": [...]} of equal length; labels are unshifted and
    IGNORE_INDEX on the prompt. Built with a boolean `supervised` mask, which is the form
    that generalises to multi-turn data (several disjoint supervised spans)."""
    prompt = build_prompt_ids(tok, ex.instruction, ex.input)
    ids = torch.tensor(prompt + tok.encode(ex.output) + [EOS_ID], dtype=torch.long)
    supervised = torch.zeros(len(ids), dtype=torch.bool)
    supervised[len(prompt):] = True
    labels = torch.where(supervised, ids, torch.full_like(ids, IGNORE_INDEX))
    return {"input_ids": ids.tolist(), "labels": labels.tolist()}
