# tiny-gpt-codespaces — SFT debugging practice

A small, readable **supervised fine-tuning (SFT) pipeline** for a tiny GPT, in pure PyTorch,
with **defects planted in it on purpose**. It is built to practise the classic ML-engineer
technical screen:

> *Given raw text files with noisy formatting, implement a robust parser that outputs
> structured examples … In a provided ML project (data loading, preprocessing, training,
> evaluation), identify and fix defects (e.g. index off-by-one in tokenization, train/test
> leakage, incorrect loss reduction, nondeterministic seeding, shape mismatches) … and
> describe how you would validate the fixes under a 60-minute time limit.*

Everything runs on CPU. A full training run takes about 2–3 minutes on a free 2-core Codespace.

**Start here → [EXERCISE.md](EXERCISE.md)**

> You are on the **`solutions`** branch: the code here is fixed. See [SOLUTIONS.md](SOLUTIONS.md).
> To practise, switch to `main`.

| Branch | What's on it |
|---|---|
| `main` | The exercise: a draft parser (Part 1) and an SFT pipeline with **6 planted bugs** (Part 2). |
| `solutions` | Fixed code (one commit per fix), `SOLUTIONS.md` walkthrough, regression tests, the data generator. Don't peek until you're done. |

---

## 1. Open it in the browser (Codespaces)

1. **Code ▾ → Codespaces → Create codespace on main**.
2. Wait for `.devcontainer/post-create.sh` to install CPU-only PyTorch.
3. Press <kbd>F5</kbd> → **"Train: quick debug run (300 iters)"**. (On `main` it will crash. That's the point.)

Terminal equivalents:

```bash
python -m sft.train                         # full run (3000 iters)
python -m sft.train --max_iters 300         # quick look
python -m sft.evaluate                      # re-score checkpoints/model.pt on val + holdout
python -m sft.generate --instruction "Reverse:" --input banana
python -m pytest -q tests                   # tests/test_parser.py is the Part 1 spec
```

Pressing <kbd>.</kbd> on GitHub opens github.dev, which has no Python and no debugger. Use a Codespace
(or clone locally, see §6).

---

## 2. The pipeline

```
data/raw/*.tsv ──parser──► records ──dedup + split──► train / val        data/holdout.jsonl
                                                        │                        │
             chat template + byte tokenizer + label mask ▼                        │
         <|bos|><|user|>{instruction}\n{input}<|assistant|>{output}<|eos|>        │
                                                        │                        │
                        collate (right-pad, truncate) ──► (B, T) batches          │
                                                        │                        │
               GPT (2 layers, 128 wide, ~0.4M params) ──► masked next-token loss   │
                                                        │                        │
                         AdamW + warmup/cosine + clip ──► checkpoints/model.pt    │
                                                        │                        │
                             greedy generation ──► exact match on val  and  ◄─────┘
```

The model is fine-tuned from scratch on six tiny instruction tasks (`Reverse:`, `Uppercase:`,
`Sort letters:`, `Length:`, `Count vowels:`, `Repeat:`) so that **exact-match accuracy is a
meaningful metric** and a leak, a bad loss, or a broken label shows up in the numbers.

| File | Stage |
|---|---|
| `sft/parser.py` | raw TSV bytes → records + `ParseReport` (**Part 1**) |
| `sft/data.py` | dedup, train/val split, chat template, tokenization, label masking, collate |
| `sft/tokenizer.py` | byte-level tokenizer + special tokens |
| `sft/model.py` | decoder-only transformer, written out longhand; greedy/top-k `generate` |
| `sft/loss.py` | masked cross-entropy |
| `sft/train.py` | training loop, LR schedule, periodic eval, best-checkpoint, final metrics |
| `sft/evaluate.py` | val loss, batched greedy exact match, standalone checkpoint eval |
| `sft/utils.py` | seeding, device selection |
| `data/raw/` | the noisy exports (UTF-8+BOM and cp1252, quoting, ragged rows, duplicates …) |
| `data/holdout.jsonl` | a clean, separately-curated holdout set |

---

## 3. Debugging in the browser

`.vscode/launch.json` ships ready-made configurations (all with `"justMyCode": false`, so you
can step into PyTorch):

| Configuration | What it does |
|---|---|
| **Train: quick debug run (300 iters)** | fast enough to iterate on a fix |
| **Train: full run** | the real run you compare against the reference numbers |
| **Evaluate checkpoint** | re-scores `checkpoints/model.pt` on val and holdout |
| **Generate: ask the checkpoint** | one prompt, greedy decode |
| **Pytest: all tests** / **parser spec only** | tests under the debugger |
| **Python: current file** | debug whatever is open |
| **Attach to running process (:5678)** | for `python -m debugpy --listen 5678 --wait-for-client -m sft.train` |

Useful breakpoints are marked `# BREAKPOINT:` in the source. At the batch breakpoint in
`sft/train.py`, `tok.decode(x[0].tolist(), skip_special=False)` in the Debug Console shows you
exactly what the model sees. `debug.inlineValues` is on, so tensor shapes show up inline.

---

## 4. Live Share (mock interviews)

Live Share is preinstalled. Start a session from the status bar, send the link, and your partner
joins in their browser with shared breakpoints, call stack and debug console. Guests need your
approval to join (`liveshare.guestApprovalRequired`) and can step the debugger and run tasks
(`allowGuestDebugControl`, `allowGuestTaskControl` in `.vscode/settings.json`). A good format:
one person drives on `main`, the other plays interviewer with `SOLUTIONS.md` open.

---

## 5. Practising again

```bash
git stash                     # or: git checkout main -- sft/
```

puts the bugs back. Grade yourself without reading the fixes:

```bash
git fetch origin solutions
git checkout origin/solutions -- tests/test_regressions.py
python -m pytest -q tests/test_regressions.py      # 9 tests, one or two per planted bug
```

---

## 6. Running locally

```bash
git clone https://github.com/yangyingxiang/tiny-gpt-codespaces.git
cd tiny-gpt-codespaces
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
python -m pip install numpy pytest
python -m sft.train
```

MIT licensed.
