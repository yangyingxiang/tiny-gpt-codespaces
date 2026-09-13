"""Data: JSONL -> examples -> split -> byte tokens + label mask -> (B, T) batches.

Chat template (one training example):

    <|bos|><|user|>{instruction}\n{input}<|assistant|>{output}<|eos|>

Only the assistant's reply (and the closing <|eos|>) is trained on; prompt
positions are IGNORE_INDEX in the labels.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

IGNORE_INDEX = -100

# --------------------------------------------------------------------------- tokenizer

NUM_BYTES = 256
SPECIAL_TOKENS = ["<|pad|>", "<|bos|>", "<|eos|>", "<|user|>", "<|assistant|>"]
SPECIAL_IDS = {tok: NUM_BYTES + i for i, tok in enumerate(SPECIAL_TOKENS)}
PAD_ID = SPECIAL_IDS["<|pad|>"]
BOS_ID = SPECIAL_IDS["<|bos|>"]
EOS_ID = SPECIAL_IDS["<|eos|>"]
USER_ID = SPECIAL_IDS["<|user|>"]
ASSISTANT_ID = SPECIAL_IDS["<|assistant|>"]


class ByteTokenizer:
    """Ids 0..255 are raw UTF-8 bytes; the special tokens follow."""

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


# --------------------------------------------------------------------------- examples

@dataclass(frozen=True)
class Example:
    id: str
    instruction: str
    input: str
    output: str


def load_jsonl(path: str | Path) -> list[Example]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                out.append(Example(d["id"], d["instruction"], d.get("input", ""), d["output"]))
    return out


def split_examples(examples: list[Example], val_frac: float, seed: int) -> tuple[list[Example], list[Example]]:
    """Shuffle with a private RNG (so nothing else in the process can change the split)."""
    examples = list(examples)
    random.Random(seed).shuffle(examples)
    n_val = int(len(examples) * val_frac)
    return examples[n_val:], examples[:n_val]


def filter_tasks(examples: list[Example], tasks: tuple[str, ...]) -> list[Example]:
    return [ex for ex in examples if ex.instruction in tasks]


# --------------------------------------------------------------------------- tokenization

def build_prompt_ids(tok: ByteTokenizer, instruction: str, inp: str) -> list[int]:
    text = f"{instruction}\n{inp}" if inp else instruction
    return [BOS_ID, USER_ID] + tok.encode(text) + [ASSISTANT_ID]


def encode_example(tok: ByteTokenizer, ex: Example) -> tuple[list[int], list[int]]:
    """(input_ids, labels): labels[t] is the token that follows input_ids[t]."""
    prompt = build_prompt_ids(tok, ex.instruction, ex.input)
    full = prompt + tok.encode(ex.output) + [EOS_ID]
    input_ids = full[:-1]
    labels = full[1:]
    n_prompt_targets = len(prompt) - 1
    labels = [IGNORE_INDEX] * n_prompt_targets + labels[n_prompt_targets:]
    assert len(input_ids) == len(labels)
    return input_ids, labels


class SFTDataset(Dataset):
    def __init__(self, examples: list[Example], tok: ByteTokenizer) -> None:
        self.examples = examples
        self.items = [encode_example(tok, ex) for ex in examples]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        return self.items[i]


def collate(batch, block_size: int):
    """Right-pad to (B, T) with T <= block_size; inputs and labels are cut together."""
    max_len = min(max(len(x) for x, _ in batch), block_size)
    inputs, labels = [], []
    for x, y in batch:
        x, y = x[:max_len], y[:max_len]
        inputs.append(x + [PAD_ID] * (max_len - len(x)))
        labels.append(y + [IGNORE_INDEX] * (max_len - len(y)))
    return torch.tensor(inputs, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def make_loader(ds: SFTDataset, batch_size: int, block_size: int, shuffle: bool, seed: int = 0) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g,
                      collate_fn=partial(collate, block_size=block_size), drop_last=False)


def infinite(loader):
    while True:
        yield from loader
