# Solutions

Spoilers. Each fix is its own commit on this branch, so you can check them one at a time:

```bash
git log --oneline main..solutions
git diff main solutions -- sft/          # all six fixes plus the parser
```

`tests/test_regressions.py` has a guard for every bug. Each one fails on `main` and passes here.

| # | Where | Class | Loud or silent | Signature |
|---|---|---|---|---|
| 1 | `data.collate` | shape mismatch | **crash** | `ValueError: expected sequence of length 64 at dim 1 (got 155)` |
| 2 | `tokenizer.SPECIAL_IDS` | off-by-one in token ids | **crash** | `IndexError: index out of range in self` (embedding) |
| 3 | `data.encode_example` | label misalignment | silent | train loss → ~0, exact match 0% |
| 4 | `loss.masked_cross_entropy` | wrong loss reduction | silent | step-0 loss ≈ 0.65 instead of ln(261) ≈ 5.56 |
| 5 | `data.dedup_key` | train/val leakage | silent | val exact match ≫ holdout; val loss too good |
| 6 | `utils.set_seed` | nondeterministic seeding | silent | different split every run; `evaluate` ≠ `train` |

The order below is roughly the order you meet them in if you debug by running the thing.

---

## 1. `collate` truncates inputs but not labels (shape mismatch, crash)

```python
# bug
x = x[:max_len]
inputs.append(x + [PAD_ID] * (max_len - len(x)))
labels.append(y + [IGNORE_INDEX] * (max_len - len(y)))   # y is never truncated
# fix
x, y = x[:max_len], y[:max_len]
```

**How you find it.** It crashes at `eval @ 0`, inside the DataLoader. (On `main`, whether you
hit this crash or bug 2's crash first depends on which examples land in the first val
batches, and because of bug 6 that changes from run to run.) Skip past the torch
frames to the last frame in `sft/`, which is `collate`. "Length 64 … got 155" means one row is
longer than `block_size = 64`. Only about 3% of examples are that long (the long `Repeat:`
ones), so a batch usually but not always contains one. That is why it looks intermittent when
you try it on tiny subsets.

**Guard:** `test_collate_truncates_inputs_and_labels_together`, which collates one short and
one 80-character example.

## 2. Special-token ids are off by one (crash)

```python
# bug
SPECIAL_IDS = {tok: NUM_BYTES + i for i, tok in enumerate(SPECIAL_TOKENS, start=1)}
# -> pad=257 ... assistant=261, but vocab_size = 256 + 5 = 261
# fix
SPECIAL_IDS = {tok: NUM_BYTES + i for i, tok in enumerate(SPECIAL_TOKENS)}
```

**How you find it.** `IndexError: index out of range in self` from `F.embedding` means some
id is ≥ `num_embeddings`. Print `x.max()` next to `model.tok_emb.num_embeddings`: 261 vs 261.
`<|assistant|>` is in every example, so every batch dies.

**Guard:** `test_every_token_id_fits_in_the_vocab`, which asserts the specials are exactly
`range(256, vocab_size)`, contiguous and in range.

## 3. Labels are not shifted (silent, and the most instructive one)

```python
# bug
input_ids = full[:-1]
labels = full[: len(input_ids)]     # == input_ids: "predict the token you are looking at"
# fix
labels = full[1:]
```

**Symptom.** Once the crashes are fixed, training loss collapses to 0.001 within 100 steps and
prints `0.0000` by step 300, but exact match stays at 0%. Generations are garbage and never stop.
`<|eos|>` is never a target, so the model never learns to emit it.

**How you find it.** Loss falling that fast means the task is trivially easy, and in a causal
LM that almost always means the target is visible in the input. Break at the batch in
`train.py` and decode one row:

```python
[(tok.id_to_token(a), tok.id_to_token(b)) for a, b in zip(x[0].tolist(), y[0].tolist()) if b != -100]
```

Every pair is `(t, t)` when it should be `(t, t+1)`.

**Guard:** `test_labels_are_inputs_shifted_left_by_one`, plus a check that the supervised
tokens are exactly `answer + <|eos|>` and that the first one sits at the `<|assistant|>`
position. The end-to-end guard `test_model_overfits_a_tiny_batch_and_answers_correctly` also
catches it: the loss does go to 0, but exact match stays at 0.

## 4. Loss averaged over the wrong denominator (silent)

```python
# bug
return (per_token * mask).sum() / mask.numel()        # divides by B*T, prompt and pad included
# fix
return (per_token * mask).sum() / mask.sum().clamp(min=1.0)
```

**Symptom.** Step-0 loss is about 0.65. An untrained model over V = 261 classes should score
about ln(261) ≈ 5.56. The printed loss is scaled by the fraction of positions that are answer
tokens, and that fraction changes from batch to batch with padding. So val loss is not a
per-token number, is not comparable across runs or configs, and is noisier as a
checkpoint-selection signal. Training barely changes, because AdamW is roughly invariant to a
constant loss scale. That is exactly why this kind of bug survives: nothing looks broken
unless you check the step-0 loss.

**Guards:** `test_initial_loss_is_ln_vocab` and `test_loss_does_not_depend_on_padding`.

## 5. Near-duplicate leakage across the split (silent)

```python
# bug
return (ex.instruction, ex.input, ex.output)          # exact match only
# fix
return (_norm(ex.instruction).casefold(), _norm(ex.input), _norm(ex.output))   # NFC + collapse whitespace
```

**Symptom.** Measured with every other bug fixed (robust parser, default seed):

| | exact-dedup (bug) | normalised dedup (fix) |
|---|---|---|
| train / val sizes | 6839 / 759 | 5322 / 591 |
| best val loss | 0.25 | 0.34 |
| val exact match | **68.8%** | 62.0% |
| holdout exact match | 55.0% | 60.8% |

**How you find it.** Val beats a clean holdout by about 14 points. Ask whether any val example
has been seen in training. Canonicalise and intersect:

```python
canon = lambda e: tuple(" ".join(unicodedata.normalize("NFC", s).split()).casefold()
                        for s in (e.instruction, e.input, e.output))
len({canon(e) for e in val} & {canon(e) for e in train}) / len(val)   # ≈ 0.4 on main
```

The scraped rows are `SORT LETTERS:`, `sort  letters:` and NFD (`e` + combining accent)
variants of other rows. Exact-match dedup keeps them apart, the shuffle scatters the twins
across train and val, and the model memorises them.

**Follow-up: "val accuracy dropped after your fix, did you break it?"** No. The old number
was partly memorisation. The evidence: before the fix val and holdout disagreed by 14 points;
after it they agree within noise. Holdout *went up*, because the model was no longer spending
capacity on duplicates. The honest number is the one that matches data the model has never
seen.

**Guards:** `test_dedup_catches_formatting_variants` and `test_train_and_val_share_no_example`.
The second uses its **own** canonicaliser rather than `dedup_key`, so a guard can't share the
bug it is guarding against.

## 6. Python's `random` is never seeded (silent)

```python
# bug
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
# fix: also
    random.seed(seed)
```

`split_examples` shuffles with the global `random` module. Torch *is* seeded, so weight init
and step-0 loss are identical run to run, which makes the pipeline look deterministic. The
split is not.

**Symptoms.**
- Two runs with the same seed give different train/val sets and different curves.
- Worse: `python -m sft.evaluate` rebuilds "the same" split to score a checkpoint, and gets a
  different one. That val set is mostly examples the checkpoint trained on, so standalone
  eval reports a higher val exact match than `train.py` did for the same weights. In one run
  on `main` with bugs 1 to 3 fixed, `train.py` reported val 60.0% and `sft.evaluate` reported
  **73.8%** for the same checkpoint.

**Guard:** `test_split_is_reproducible_for_a_seed` (seed, split, disturb the RNG, re-seed,
split, compare). The more robust fix is to not use global RNG state at all:
`rng = random.Random(seed); rng.shuffle(examples)`. The minimal fix is what an interview
wants.

---

## Part 1: the parser

`sft/parser.py` on this branch. Design choices worth saying out loud:

- **Don't hand-roll quote handling if you can avoid it.** `csv.reader` does RFC-4180 field
  splitting. The one thing it gets wrong for dirty data is an **unterminated quote**: it
  silently swallows every following line into one field. In `data/raw/sft_export_legacy.tsv`
  that merges four rows into one record *with the right number of columns*, so an arity check
  alone does not catch it. The parser therefore splits the text into *logical records*
  itself (joining physical lines only while a quote is open, up to 8 lines), and treats a
  closing quote followed by anything other than a tab or end of line as proof that an earlier
  quote was never closed. Then `csv` splits each record into fields.
- **Encoding.** Try `utf-8-sig` (which strips the BOM), then cp1252, then latin-1, which
  never fails. Record that a fallback happened. The draft used `errors="replace"`, which
  silently turned every `é` and `’` in the legacy file into `�`.
- **Observability over silence.** Every data row ends up in `ok` or in `rejected` with a line
  number and a reason. The invariant `ok + rejected == total` is asserted.
- **It matters for the model, too.** With all six pipeline bugs fixed but the draft parser
  still in place, a full run scores about 51% exact match on val and 51% on holdout. With the
  robust parser it scores 62% and 61%.
- **What the draft got wrong on the real files:** split multi-line quoted rows into two
  broken halves and dropped both; kept literal `"` and `""` inside quoted fields; dropped
  every row with an embedded tab; accepted rows with an empty `output` (training on empty
  answers); accepted the unterminated-quote row as a garbage example; didn't strip padded
  cells or zero-width spaces; reported `N/N rows ok, 0 rejected` while doing all of this.

Testing strategy (see `tests/test_parser.py`): one fixture per edge case; assert on the
`ParseReport` as well as the records; a round-trip property test over random records with
tabs, newlines, quotes and non-ASCII characters; a fuzz test that feeds random bytes and
requires "parsed or rejected, never crashed"; and invariant checks on the real exports.
In production, emit the rejection rate and per-reason counts as metrics and alert on a
relative jump. To tell a source change from a parser regression, re-run yesterday's file
through today's parser.

---

## Part 3: the 10-minute validation script

```bash
python -m pytest -q tests                       # all green, including test_regressions.py
git stash && python -m pytest -q tests/test_regressions.py; git stash pop   # guards fail on the bugs
python -m sft.train --max_iters 300 --eval_interval 100 --eval_examples 50 --final_eval_examples 100
python -m sft.train && python -m sft.evaluate   # val ≈ holdout ≈ 61%, evaluate agrees with train
python -m sft.train --out_dir /tmp/run2         # identical numbers, i.e. deterministic
```

## Regenerating the data

`scripts/make_raw_data.py` (this branch only) regenerates `data/raw/*` and
`data/holdout.jsonl`. Reading it tells you exactly which kinds of noise and duplication were
injected.
