"""Data pipeline: raw exports -> examples -> dedup -> split -> token ids -> batches.

Chat template (one training example):

    <|bos|><|user|>{instruction}\n{input}<|assistant|>{output}<|eos|>

Only the assistant's reply (and the closing <|eos|>) is trained on; the prompt
positions are masked out of the loss with IGNORE_INDEX.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from .parser import parse_file
from .tokenizer import ASSISTANT_ID, BOS_ID, EOS_ID, PAD_ID, USER_ID, ByteTokenizer

IGNORE_INDEX = -100


@dataclass(frozen=True)
class Example:
    id: str
    instruction: str
    input: str
    output: str


# --------------------------------------------------------------------------- loading

def load_raw_examples(paths, verbose: bool = True) -> list[Example]:
    examples: list[Example] = []
    for path in paths:
        records, report = parse_file(path)
        if verbose:
            print(f"parsed {Path(path).name}: {report.summary()}")
        examples += [
            Example(r.get("id", ""), r.get("instruction", ""), r.get("input", ""), r.get("output", ""))
            for r in records
        ]
    return examples


def load_jsonl(path) -> list[Example]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                out.append(Example(d["id"], d["instruction"], d.get("input", ""), d["output"]))
    return out


# --------------------------------------------------------------------------- dedup + split

def dedup_key(ex: Example) -> tuple[str, str, str]:
    """Two examples with the same key are the same training example."""
    return (ex.instruction, ex.input, ex.output)


def dedup(examples: list[Example]) -> list[Example]:
    seen, out = set(), []
    for ex in examples:
        k = dedup_key(ex)
        if k not in seen:
            seen.add(k)
            out.append(ex)
    return out


def split_examples(examples: list[Example], val_frac: float) -> tuple[list[Example], list[Example]]:
    """Dedup, shuffle and split into (train, val).

    Uses the global `random` module, so call `utils.set_seed(seed)` first to
    get the same split every time (train.py and evaluate.py both do).
    """
    examples = dedup(examples)
    random.shuffle(examples)
    n_val = int(len(examples) * val_frac)
    return examples[n_val:], examples[:n_val]


# --------------------------------------------------------------------------- tokenization

def build_prompt_ids(tok: ByteTokenizer, instruction: str, inp: str) -> list[int]:
    text = f"{instruction}\n{inp}" if inp else instruction
    return [BOS_ID, USER_ID] + tok.encode(text) + [ASSISTANT_ID]


def encode_example(tok: ByteTokenizer, ex: Example) -> tuple[list[int], list[int]]:
    """Return (input_ids, labels), both of length len(full) - 1.

    labels[t] is the token the model must predict after seeing input_ids[:t+1],
    i.e. the sequence shifted left by one. Prompt positions are IGNORE_INDEX.
    """
    prompt = build_prompt_ids(tok, ex.instruction, ex.input)
    full = prompt + tok.encode(ex.output) + [EOS_ID]
    input_ids = full[:-1]
    labels = full[: len(input_ids)]
    n_prompt_targets = len(prompt) - 1          # targets that are still prompt tokens
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
    """Right-pad a list of (input_ids, labels) to a (B, T) pair of tensors, T <= block_size."""
    max_len = min(max(len(x) for x, _ in batch), block_size)
    inputs, labels = [], []
    for x, y in batch:
        x, y = x[:max_len], y[:max_len]
        pad = max_len - len(x)
        inputs.append(x + [PAD_ID] * pad)
        labels.append(y + [IGNORE_INDEX] * pad)
    return torch.tensor(inputs, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def make_loader(ds: SFTDataset, batch_size: int, block_size: int, shuffle: bool, seed: int = 0) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, generator=g,
        collate_fn=partial(collate, block_size=block_size), drop_last=False,
    )
