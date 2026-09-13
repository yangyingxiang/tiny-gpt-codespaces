"""Data path: jsonl -> split -> task filter -> prompt template -> gpt2 token ids -> compact ids
-> datasets.Dataset -> collator.

One training example, in text:

    ### Instruction:
    Reverse:
    i f j c w j s
    ### Response:
    s j w c j f i<|endoftext|>

GPT-2 has no chat template, so the template is plain text. Inputs and outputs are *spelled out*
(`spell`) so that the byte-level BPE tokenizer yields one token per character: the tasks are
character-level and a tiny model cannot see inside an opaque sub-word piece.

Prompt and response are tokenised separately and concatenated, so the prompt/response boundary
is exact and `prompt_len` can be stored per example. Only the response (including the closing
eos) is supervised: the collator sets prompt positions to IGNORE_INDEX.

The gpt2 vocabulary has 50257 ids but this data uses about a hundred of them, and a 50257-way
lm_head would be ~95% of the compute on CPU. `VocabMap` translates between gpt2 ids ("full") and
a dense 0..V-1 range ("compact") at the two boundaries: encode (full -> compact) and generate
(compact -> full, then `tok.decode`). The model only ever sees compact ids.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import Dataset

IGNORE_INDEX = -100
PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n{input}\n### Response:\n"


@dataclass(frozen=True)
class Example:
    id: str
    instruction: str
    input: str
    output: str


# --------------------------------------------------------------------------- loading + split

def load_jsonl(path: str | Path) -> list[Example]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                out.append(Example(d["id"], d["instruction"], d.get("input", ""), d["output"]))
    return out


def split_examples(examples: list[Example], val_frac: float, seed: int
                   ) -> tuple[list[Example], list[Example]]:
    """Shuffle with a private RNG (not the global `random`) and slice into (train, val)."""
    examples = list(examples)
    random.Random(seed).shuffle(examples)
    n_val = int(len(examples) * val_frac)
    return examples[n_val:], examples[:n_val]


def filter_tasks(examples: list[Example], tasks: tuple[str, ...]) -> list[Example]:
    return [ex for ex in examples if ex.instruction in tasks]


# --------------------------------------------------------------------------- text formatting

def spell(text: str) -> str:
    """'abc d' -> 'a b c _ d'. A space becomes '_' so it survives the round trip."""
    return " ".join("_" if c == " " else c for c in text)


def unspell(text: str) -> str:
    return text.replace(" ", "").replace("_", " ")


def build_prompt(ex: Example) -> str:
    return PROMPT_TEMPLATE.format(instruction=ex.instruction, input=spell(ex.input))


def build_response(ex: Example) -> str:
    return spell(ex.output)


# --------------------------------------------------------------------------- vocab map

class VocabMap:
    """gpt2 ids <-> dense 0..V-1 ids over the tokens that occur in `examples` (+ eos/pad)."""

    def __init__(self, tok, examples: list[Example]) -> None:
        ids = {tok.eos_token_id, tok.pad_token_id}
        for ex in examples:
            ids.update(tok(build_prompt(ex), add_special_tokens=False)["input_ids"])
            ids.update(tok(build_response(ex), add_special_tokens=False)["input_ids"])
        self.full_ids = sorted(ids)
        self.to_compact = {f: c for c, f in enumerate(self.full_ids)}
        self.eos_id = self.to_compact[tok.eos_token_id]
        self.pad_id = self.to_compact[tok.pad_token_id]
        # tensor lookup for batched use: full id -> compact id (-1 = never seen)
        self._lut = torch.full((len(tok),), -1, dtype=torch.long)
        self._lut[torch.tensor(self.full_ids)] = torch.arange(len(self.full_ids))
        self._inv = torch.tensor(self.full_ids)

    def __len__(self) -> int:
        return len(self.full_ids)

    def encode(self, full_ids: list[int]) -> list[int]:
        return [self.to_compact[i] for i in full_ids]          # KeyError = token never seen

    def encode_tensor(self, full: torch.Tensor) -> torch.Tensor:
        compact = self._lut.to(full.device)[full]
        if (compact < 0).any():
            raise KeyError("prompt contains a token that is not in the VocabMap")
        return compact

    def decode_tensor(self, compact: torch.Tensor) -> torch.Tensor:
        return self._inv.to(compact.device)[compact]


# --------------------------------------------------------------------------- tokenisation

def encode_example(tok, vmap: VocabMap, ex: Example, max_length: int) -> dict:
    """{"input_ids": prompt + response + eos (compact ids), "prompt_len": len(prompt tokens)}.

    If the whole thing is too long the prompt's HEAD is cut so that the answer and the
    "### Response:" cue survive; the response itself is never cut below one token + eos.
    """
    p_ids = tok(build_prompt(ex), add_special_tokens=False)["input_ids"]
    r_ids = tok(build_response(ex), add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    if len(r_ids) >= max_length:
        r_ids = r_ids[: max_length - 1] + [tok.eos_token_id]
        p_ids = []
    budget = max_length - len(r_ids)
    p_ids = p_ids[-budget:] if budget else []
    return {"input_ids": vmap.encode(p_ids + r_ids), "prompt_len": len(p_ids)}


def to_dataset(tok, vmap: VocabMap, examples: list[Example], max_length: int) -> Dataset:
    return Dataset.from_list([encode_example(tok, vmap, ex, max_length) for ex in examples])


class SFTCollator:
    """Right-pad a list of {"input_ids", "prompt_len"} rows into one batch.

    Three fill values, three meanings: input_ids <- pad_id (what the model reads),
    attention_mask <- 0 (what it may look at), labels <- IGNORE_INDEX (what it is scored on).
    """

    def __init__(self, pad_id: int) -> None:
        self.pad_id = pad_id

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        B = len(features)
        L = max(len(f["input_ids"]) for f in features)
        input_ids = torch.full((B, L), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((B, L), dtype=torch.long)
        labels = torch.full((B, L), IGNORE_INDEX, dtype=torch.long)
        for i, f in enumerate(features):
            ids = torch.as_tensor(f["input_ids"], dtype=torch.long)
            n = len(ids)
            input_ids[i, :n] = ids
            attention_mask[i, :n] = 1
            labels[i, :n] = ids
            labels[i, : f["prompt_len"]] = IGNORE_INDEX
        labels[input_ids == self.pad_id] = IGNORE_INDEX
        labels = torch.cat([labels[:, 1:], labels.new_full((B, 1), IGNORE_INDEX)], dim=1)
        # BREAKPOINT: decode input_ids[0] and the non-ignored labels[0] side by side.
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
