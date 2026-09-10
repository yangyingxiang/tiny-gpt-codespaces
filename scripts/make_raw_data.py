"""Generate the synthetic SFT dataset, including its deliberate mess.

    python scripts/make_raw_data.py

Writes three files:

  data/raw/sft_export_2024.tsv    UTF-8 *with BOM*, mostly LF with some CRLF lines,
                                  comment lines, blank lines, RFC-4180 quoting
                                  (tabs / newlines / doubled quotes inside fields),
                                  ragged rows, rows missing required fields,
                                  zero-width spaces, padded cells, and
                                  *near-duplicate* rows (case / whitespace / NFD
                                  variants of other rows) -- like a scraped export.
  data/raw/sft_export_legacy.tsv  cp1252 (Windows-1252) encoded, CRLF, *no* `source`
                                  column, smart quotes and accents, one row with an
                                  unterminated quote.
  data/holdout.jsonl              A clean, curated holdout set that shares no
                                  example (after normalisation) with the raw exports.

The tasks are tiny deterministic string/number transformations so that a
0.8M-parameter model can learn them on a laptop CPU in a couple of minutes,
and exact-match accuracy is a meaningful metric.

This script lives on the `solutions` branch only: reading it tells you what
kinds of noise and duplication were injected.
"""

from __future__ import annotations

import json
import random
import string
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

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
    # copy: the only task whose text can contain quotes, tabs, newlines, accents
    s = sentence(rng, 7, 10) if long_copy else sentence(rng, 1, 3)
    r = rng.random()
    if r < 0.10:
        s = f'she said "{s}"'
    elif r < 0.15:
        s = s.replace(" ", "\t", 1)
    elif r < 0.20:
        s = s.replace(" ", "\n", 1)
    return dict(instruction="Repeat:", input=s, output=s)


def norm_key(ex: dict) -> tuple:
    def n(s: str) -> str:
        return " ".join(unicodedata.normalize("NFC", s).split())
    return (n(ex["instruction"]).casefold(), n(ex["input"]), n(ex["output"]))


def near_duplicate(rng: random.Random, ex: dict) -> dict:
    """A variant that a naive exact-match dedup will not catch."""
    ex = dict(ex)
    kind = rng.choice(["lower", "spaces", "upper", "nfd"])
    if kind == "lower":
        ex["instruction"] = ex["instruction"].lower()
    elif kind == "spaces" and " " in ex["instruction"]:
        ex["instruction"] = ex["instruction"].replace(" ", "  ", 1)
    elif kind == "upper":
        ex["instruction"] = ex["instruction"].upper()
    elif kind == "nfd" and any(ord(c) > 127 for c in ex["input"]):
        ex["input"] = unicodedata.normalize("NFD", ex["input"])
        ex["output"] = unicodedata.normalize("NFD", ex["output"])
    else:
        ex["instruction"] = ex["instruction"].swapcase()
    return ex


def tsv_field(s: str) -> str:
    if any(c in s for c in '\t\n\r"'):
        return '"' + s.replace('"', '""') + '"'
    return s


def main(seed: int = 7) -> None:
    rng = random.Random(seed)
    RAW.mkdir(parents=True, exist_ok=True)

    # ---- unique pool --------------------------------------------------------
    pool, seen = [], set()
    while len(pool) < 6000:
        ex = make_example(rng, long_copy=rng.random() < 0.03)
        k = norm_key(ex)
        if k in seen:
            continue
        seen.add(k)
        pool.append(ex)
    modern, legacy = pool[:5000], pool[5000:]

    # ---- holdout: fresh examples, disjoint from everything above -----------
    holdout = []
    while len(holdout) < 400:
        ex = make_example(rng, long_copy=rng.random() < 0.03)
        k = norm_key(ex)
        if k in seen:
            continue
        seen.add(k)
        holdout.append(ex)
    with (ROOT / "data" / "holdout.jsonl").open("w", encoding="utf-8") as f:
        for i, ex in enumerate(holdout):
            f.write(json.dumps({"id": f"h{i:04d}", **ex}, ensure_ascii=False) + "\n")

    # ---- modern export: near-dups + noise ----------------------------------
    rows = [dict(ex, source=rng.choice(["vendor_a", "vendor_b", "synthetic"])) for ex in modern]
    for ex in rng.sample(modern, 1800):                         # ~35% near-duplicates
        rows.append(dict(near_duplicate(rng, ex), source="scrape"))
    for ex in rng.sample(modern, 200):                          # plus some exact duplicates
        rows.append(dict(ex, source="scrape"))
    rng.shuffle(rows)

    lines = ["# sft export v3 -- generated 2024-11-02", "id\tinstruction\tinput\toutput\tsource"]
    for i, r in enumerate(rows):
        rid = f"m{i:05d}"
        fields = [rid, r["instruction"], r["input"], r["output"], r["source"]]
        x = rng.random()
        if x < 0.010:
            fields = fields[:3]                                 # truncated row
        elif x < 0.018:
            fields = fields + ["EXTRA"]                         # extra column
        elif x < 0.028:
            fields[3] = ""                                      # missing output
        elif x < 0.034:
            fields[1] = "  " + fields[1] + " "                  # padded cell
        elif x < 0.040:
            fields[2] = fields[2] + "​"                    # zero-width space
        elif x < 0.045:
            fields[4] = ""                                      # empty optional
        line = "\t".join(tsv_field(f) for f in fields)
        if rng.random() < 0.15:
            line += "\r"                                        # CRLF on some lines
        lines.append(line)
        if rng.random() < 0.01:
            lines.append("")                                    # blank lines
        if rng.random() < 0.003:
            lines.append("# ---- page break ----")
    data = ("\n".join(lines) + "\n").encode("utf-8-sig")        # BOM
    (RAW / "sft_export_2024.tsv").write_bytes(data)

    # ---- legacy export: cp1252, CRLF, 4 columns, smart quotes --------------
    llines = ["id\tinstruction\tinput\toutput"]
    for i, ex in enumerate(legacy):
        ex = dict(ex)
        if ex["instruction"].startswith("Repeat") and rng.random() < 0.25:
            ex["input"] = ex["output"] = ex["input"].replace(" ", " ’n’ ", 1)  # smart quotes
        fields = [f"l{i:04d}", ex["instruction"], ex["input"], ex["output"]]
        llines.append("\t".join(tsv_field(f) for f in fields))
        if i == 417:
            llines.append('l9999\tRepeat:\t"unterminated quote here\toops')
    (RAW / "sft_export_legacy.tsv").write_bytes(("\r\n".join(llines) + "\r\n").encode("cp1252"))

    print(f"modern rows: {len(rows)}  legacy rows: {len(legacy)}  holdout: {len(holdout)}")


if __name__ == "__main__":
    main()
