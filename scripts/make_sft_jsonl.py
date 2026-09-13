"""Generate data/sft_train.jsonl: a clean, deduplicated training set for the pipeline
variants under sft/v*_*/.

    python scripts/make_sft_jsonl.py

Same six tasks as data/holdout.jsonl (Reverse, Uppercase, Sort letters, Length,
Count vowels, Repeat), same JSONL schema, and no example (after NFC + whitespace +
case normalisation) overlaps with the holdout set. The variants do not need the
parser from Part 1; they read this file directly.
"""

from __future__ import annotations

import json
import random
import string
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sft_train.jsonl"
HOLDOUT = ROOT / "data" / "holdout.jsonl"

WORDS = (
    "the a small quick model reads every line of text and learns to copy it back "
    "exactly without losing a single character data parser token batch loss "
    "gradient café naïve résumé déjà über señor jalapeño"
).split()


def rand_word(rng: random.Random, lo: int, hi: int) -> str:
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(lo, hi)))


def sentence(rng: random.Random, lo: int, hi: int) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(rng.randint(lo, hi)))


def make_example(rng: random.Random, long_copy: bool = False) -> dict:
    task = rng.choices(
        ["reverse", "upper", "sort", "length", "vowels", "copy"],
        weights=[3, 2, 3, 1, 1, 2],
    )[0]
    if long_copy:
        task = "copy"
    if task == "reverse":
        w = rand_word(rng, 3, 8)
        return dict(instruction="Reverse:", input=w, output=w[::-1])
    if task == "upper":
        w = rand_word(rng, 3, 9)
        return dict(instruction="Uppercase:", input=w, output=w.upper())
    if task == "sort":
        w = rand_word(rng, 3, 8)
        return dict(instruction="Sort letters:", input=w, output="".join(sorted(w)))
    if task == "length":
        w = rand_word(rng, 1, 12)
        return dict(instruction="Length:", input=w, output=str(len(w)))
    if task == "vowels":
        w = rand_word(rng, 4, 10)
        return dict(instruction="Count vowels:", input=w, output=str(sum(c in "aeiou" for c in w)))
    s = sentence(rng, 7, 10) if long_copy else sentence(rng, 1, 3)
    if rng.random() < 0.10:
        s = f'she said "{s}"'
    return dict(instruction="Repeat:", input=s, output=s)


def norm_key(ex: dict) -> tuple:
    def n(s: str) -> str:
        return " ".join(unicodedata.normalize("NFC", s).split())
    return (n(ex["instruction"]).casefold(), n(ex["input"]), n(ex["output"]))


def main(seed: int = 11, n: int = 4000) -> None:
    rng = random.Random(seed)
    seen = set()
    with HOLDOUT.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                seen.add(norm_key(json.loads(line)))
    n_holdout = len(seen)

    rows = []
    while len(rows) < n:
        ex = make_example(rng, long_copy=rng.random() < 0.03)
        k = norm_key(ex)
        if k in seen:
            continue
        seen.add(k)
        rows.append(ex)

    with OUT.open("w", encoding="utf-8") as f:
        for i, ex in enumerate(rows):
            f.write(json.dumps({"id": f"t{i:04d}", **ex}, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} examples to {OUT.relative_to(ROOT)} "
          f"(disjoint from {n_holdout} holdout examples)")


if __name__ == "__main__":
    main()
