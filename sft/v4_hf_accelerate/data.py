"""Data path: JSONL -> examples -> split -> token ids + label mask -> padded batches.

Prompt template (GPT-2 has no chat template, so a plain-text one):

    {instruction}\n{input}\nAnswer: {output}<|endoftext|>

The tasks are character-level (reverse, sort, count ...) but GPT-2's BPE merges several
characters into one token, so `input` and `output` are *spelled out*: one character per
token, separated by spaces, with a real space written as `_` (see `spell` / `unspell`).
Only the answer (and the closing EOS) is supervised; prompt positions carry IGNORE_INDEX.
Labels are NOT shifted here: GPT2LMHeadModel shifts them internally.

The model does not get GPT-2's full 50257-way output layer: this corpus uses ~150 of those
tokens and the softmax over the other 50k would be >90 % of the compute. `VocabSlice` keeps
the real tokenizer and remaps the ids the model sees to a compact range (see its docstring).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from .config import IGNORE_INDEX

PROMPT_TEMPLATE = "{instruction}\n{input}\nAnswer: "


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


def split_examples(examples: list[Example], val_frac: float, seed: int):
    """Deterministic shuffle + slice. The data file is already deduplicated."""
    examples = list(examples)
    random.Random(seed).shuffle(examples)
    n_val = int(len(examples) * val_frac)
    return examples[n_val:], examples[:n_val]


def spell(text: str) -> str:
    """'ab c' -> 'a b _ c': one character per BPE token, a real space becomes '_'."""
    return " ".join("_" if c == " " else c for c in text)


def unspell(text: str) -> str:
    """Inverse of spell(); tolerant of stray spaces in a generated string."""
    return "".join(" " if c == "_" else c for c in text.split(" ") if c)


def build_prompt(ex: Example) -> str:
    return PROMPT_TEMPLATE.format(instruction=ex.instruction, input=spell(ex.input))


class VocabSlice:
    """Compact id space for the model: gpt2 token id <-> small contiguous id.

    Built from every text the pipeline will tokenise (prompts, answers, and the two
    joined), so any id the tokenizer can produce for this corpus has a slot. Anything
    else maps to a single <unk> slot (decoded as '?'). The tokenizer itself is untouched:
    `tok(...)` and `tok.decode(...)` still speak full gpt2 ids; `compact` / `expand`
    convert at the model boundary.
    """

    def __init__(self, tok, texts: list[str]) -> None:
        used = {tok.eos_token_id}
        for t in texts:
            used.update(tok(t, add_special_tokens=False)["input_ids"])
        self.full_ids = sorted(used)                          # compact id -> gpt2 id
        self._compact = {f: c for c, f in enumerate(self.full_ids)}
        self.unk_id = len(self.full_ids)                      # last slot
        self._unk_full = tok.convert_tokens_to_ids("?")
        self.eos_id = self._compact[tok.eos_token_id]
        self.pad_id = self.eos_id

    def __len__(self) -> int:
        return len(self.full_ids) + 1

    def compact(self, ids) -> list[int]:
        return [self._compact.get(int(i), self.unk_id) for i in ids]

    def expand(self, ids) -> list[int]:
        return [self.full_ids[i] if i < self.unk_id else self._unk_full for i in map(int, ids)]

    @classmethod
    def from_examples(cls, tok, examples: list[Example]) -> "VocabSlice":
        texts = []
        for ex in examples:
            p, r = build_prompt(ex), spell(ex.output)
            texts += [p, r, p + r]
        return cls(tok, texts)


def encode_example(tok, vocab: VocabSlice, ex: Example, max_length: int) -> dict | None:
    """(input_ids, labels) for one example, or None if nothing is left to supervise.

    The prompt positions are masked out of the labels; the pair is capped at max_length.
    """
    prompt = build_prompt(ex)
    prompt_len = len(tok(prompt, add_special_tokens=False)["input_ids"])
    ids = tok(prompt + spell(ex.output), add_special_tokens=False)["input_ids"]
    input_ids = vocab.compact(ids) + [vocab.eos_id]
    labels = [IGNORE_INDEX] * prompt_len + input_ids[prompt_len:]
    input_ids, labels = input_ids[:max_length], labels[:max_length]
    if not any(l != IGNORE_INDEX for l in labels[1:]):
        return None
    return {"input_ids": input_ids, "labels": labels}


class SFTDataset(Dataset):
    def __init__(self, examples: list[Example], tok, vocab: VocabSlice, max_length: int) -> None:
        self.examples, self.items = [], []
        for ex in examples:
            item = encode_example(tok, vocab, ex, max_length)
            if item is not None:
                self.examples.append(ex)
                self.items.append(item)
        self.dropped = len(examples) - len(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        return self.items[i]


class Collator:
    """Right-pad to the longest row: input_ids <- pad_id, labels <- IGNORE_INDEX, mask <- 0."""

    def __init__(self, pad_id: int) -> None:
        self.pad_id = pad_id

    def __call__(self, items: list[dict]) -> dict[str, torch.Tensor]:
        n = len(items)
        max_len = max(len(x["input_ids"]) for x in items)
        input_ids = torch.full((n, max_len), self.pad_id, dtype=torch.long)
        labels = torch.full((n, max_len), -1, dtype=torch.long)
        attention_mask = torch.zeros((n, max_len), dtype=torch.long)
        for i, x in enumerate(items):
            t = len(x["input_ids"])
            input_ids[i, :t] = torch.tensor(x["input_ids"], dtype=torch.long)
            labels[i, :t] = torch.tensor(x["labels"], dtype=torch.long)
            attention_mask[i, :t] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
