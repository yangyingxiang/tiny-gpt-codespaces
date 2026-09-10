# The exercise

Modelled on a real ML-engineer technical screen: **parse noisy raw data into structured
examples, then find and fix defects in an unfamiliar training pipeline, then prove your fixes
work — all against the clock.** Set a 60-minute timer and talk out loud as if an interviewer
were listening. (With a partner, use Live Share: see README §4.)

> Don't open the `solutions` branch until you're done. README §5 shows how to grade yourself
> with the regression tests without reading the fixes.

Suggested clock: **5 min** orient and reproduce · **20 min** Part 1 · **25 min** Part 2 ·
**10 min** Part 3. Doing Part 2 first is fine; the pipeline runs with the draft parser.

---

## Part 1 — Build a robust parser (`sft/parser.py`)

`sft/parser.py` is a quick first draft that was written against a clean sample. The real
exports in `data/raw/` are not clean. Replace `parse_bytes` with a robust implementation.
Keep the API: `parse_bytes(raw: bytes, delimiter="\t") -> (records, ParseReport)`.

**Format spec**

- **Encoding.** UTF-8, possibly with a BOM. If it isn't valid UTF-8, fall back to cp1252,
  then latin-1, and set `report.encoding_fallback = True`.
- **Lines.** `\n`, `\r\n` and `\r` may be mixed in one file. Blank lines and lines starting
  with `#` are ignored (count them in `report.blank` / `report.comments`).
- **Header.** The first remaining line. Column *order* comes from the header. Required:
  `id`, `instruction`, `output`. Optional: `input`, `source` (default `""` when the column is
  absent). A header missing a required column is a `ValueError`.
- **Fields.** Tab-separated. A field that *starts* with `"` is quoted: it may contain tabs and
  newlines, and `""` inside it is a literal `"`. A `"` in the middle of an unquoted field is
  just a character.
- **Unterminated quotes** must not swallow the rest of the file. Reject that one line and keep
  going.
- **Cells.** Unicode NFC-normalised, zero-width characters removed, surrounding whitespace
  stripped.
- **Bad rows.** Wrong number of fields, or an empty required field, go to `report.rejected` as
  `(line_number, reason, raw_text)`. Never drop a row silently.
- **Invariant.** `report.ok + len(report.rejected) == report.total`.

`tests/test_parser.py` encodes this spec, and most of it fails on the draft. Make it pass.
Then **add at least two tests of your own** and be ready to say how you would test a parser
like this in production: fixtures per edge case, round-trip/property tests, fuzzing, golden
files, and monitoring the rejection rate.

Follow-ups an interviewer might ask:
- The rejection rate jumps from 0.5% to 12% overnight. Is it the data or your parser? How do you tell?
- The file is 50 GB. What changes?

---

## Part 2 — Find and fix the defects (`sft/`)

There are **six planted defects** in `sft/`, none of them in `parser.py`. **Two crash. Four
don't: they quietly produce wrong numbers.** The interview version asks for three; find all six
for full marks.

Rules:
- **Reproduce first.** Run it, read the stack trace from the bottom up, and bisect the pipeline
  with assertions (`load → dedup/split → tokenize → collate → model → loss → eval`).
- **Minimal diffs.** Fix the bug, not the style. No refactors.
- **Every fix gets a test that fails before the fix and passes after it.** Put them in
  `tests/test_my_fixes.py`.
- Keep a short log for each: *symptom → root cause → fix → how I verified it*.

### What a healthy run looks like

This is what the fixed pipeline prints for `python -m sft.train` with the default seed on a
2-core CPU. Use it the way you would use "the expected metric range" an interviewer gives you.

| Signal | Healthy |
|---|---|
| `eval @ 0` val loss (untrained model) | ≈ **5.58**, close to ln(261) = 5.56, i.e. a uniform guess over the vocabulary |
| Training loss | falls steadily and is still around 0.2 to 0.4 at the end, **not** ≈ 0 |
| Best val loss | ≈ **0.34** |
| Exact match | val ≈ **62%**, holdout ≈ **61%**. The two should agree within a few points |
| Examples after dedup and split | train ≈ 5.3k, val ≈ 590 |
| Re-running the same command | **identical** numbers |
| `python -m sft.evaluate` on that checkpoint | **agrees** with what `train.py` printed |
| Wall clock | about 2 to 3 minutes |

These numbers use the robust parser from Part 1. With all six bugs fixed but the draft parser
still in place, expect train ≈ 5.3k, **val ≈ holdout ≈ 51%**, and a best val loss around 0.5.
The rest of the table still applies.

### Hints: open only when stuck

<details><summary>Level 1: symptoms to hunt for</summary>

- The first run crashes before a single optimizer step. When you fix that, it crashes somewhere else.
  Which of the two crashes you see first may vary between runs.
- Once it trains: training loss collapses to almost zero within a few hundred steps, yet exact
  match stays at **0%**. That is too good to be true.
- Look at the step-0 loss. What should an untrained model score over V classes?
- Val exact match is noticeably **higher** than holdout exact match, and val loss looks better
  than it should. The holdout set was curated separately and is known to be clean.
- Run the same command twice. Do you get the same numbers? Does `python -m sft.evaluate` agree
  with what training printed for the same checkpoint?

</details>

<details><summary>Level 2: where to look</summary>

- The shape crash (`expected sequence of length …`): read the *last frame that is in `sft/`*.
  What shape did it expect, and which examples are longer than `block_size`?
- The `IndexError: index out of range in self` in an embedding: print `max(ids)` next to
  `vocab_size`. Where do the special-token ids come from?
- (Which crash you hit first can change from run to run. That is a clue too.)
- Loss ≈ 0 with 0% exact match: set the batch breakpoint in `train.py` and decode `x[0]` and
  `y[0]` next to each other. What *should* `y[t]` be?
- Wrong step-0 loss: `sft/loss.py`. What is the loss averaged over?
- Val beats holdout: `sft/data.py`, dedup and split. Search train for a val example after
  lower-casing and collapsing whitespace.
- Nondeterminism: which random number generators does the pipeline actually use, and which ones
  get seeded?

</details>

Level 3 is `SOLUTIONS.md` on the `solutions` branch.

---

## Part 3 — Validate under the clock

Be ready to explain, and ideally show, how you know each fix is right in about 10 minutes:

1. **Targeted tests.** One per fix, each failing on the old code (`git stash` to check).
2. **End-to-end smoke run.** The quick config completes. Then **overfit a tiny batch**: 4 to 8
   examples to near-zero loss and 100% exact match. If the model can't do that, something is
   still broken.
3. **Metric sanity.** Step-0 loss ≈ ln(V). Val ≈ holdout. Loss decreasing but not implausibly
   low. Exact match well above 0%.
4. **Regression guards.** Determinism (same seed, same numbers), a train/val disjointness
   assert in the data path, and the tests in CI.

Be ready for this one: *"After your leakage fix, val exact match went **down**. Did you break
something?"*
