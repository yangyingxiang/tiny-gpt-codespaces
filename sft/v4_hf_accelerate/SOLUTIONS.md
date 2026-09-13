# Solutions — variant 4 (PyTorch loop + `accelerate`)

Spoilers. Practise again with `git checkout -- sft/v4_hf_accelerate`.

| # | Where | Class | Loud or silent | Signature |
|---|---|---|---|---|
| 1 | `train.py`, `accelerator.prepare(...)` unpack | wrong tuple order | **crash** | `TypeError: 'AcceleratedScheduler' object is not iterable` in `eval_loss`, before any training |
| 2 | `data.Collator` | wrong ignore value for label padding | **crash** | `IndexError: Target -1 is out of bounds.` from `cross_entropy` |
| 3 | `data.encode_example` | prompt/answer boundary off by one (BPE) | silent | loss looks healthy, exact match **0 %** on every task, first character of every answer wrong |
| 4 | `data.encode_example` | truncation cuts the answer and its EOS | silent | `dropped 115 unsupervisable`; `Repeat:` "no EOS" count ≈ 20/56 instead of ≈ 8 |
| 5 | `train.py`, `get_scheduler(num_training_steps=…)` | schedule length in micro-batches | silent | `optimizer steps: 2700` printed but the loop reaches 684; `lr` never anneals (ends at ≈ 2.7e-3) |

The order below is the order you meet them in. The numbers were measured on an 8-thread laptop
with a stand-in `Accelerator` (single process, CPU); the wrapped-class names in the crash messages
are from `accelerate`'s source and may differ slightly between versions.

---

## 1. `accelerator.prepare` unpacked in the wrong order (crash)

```python
# bug
model, optimizer, scheduler, train_loader, val_loader = accelerator.prepare(
    model, optimizer, train_loader, val_loader, scheduler)
# fix: the tuple comes back in the order you passed it
model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
    model, optimizer, train_loader, val_loader, scheduler)
```

**Symptom.** The first thing the loop does is `evaluate(0)`, which iterates `val_loader` — and
that name now holds the wrapped scheduler:

```
  File ".../sft/v4_hf_accelerate/evaluate.py", line 23, in eval_loss
    for i, batch in enumerate(loader):
TypeError: 'AcceleratedScheduler' object is not iterable
```

(With the stand-in used for testing the class was called `GuardedSched`; the message shape is the
same.) Had the baseline eval not been there, the crash would have moved to the first
`scheduler.step()`: `AttributeError: 'DataLoaderShard' object has no attribute 'step'`, because
`scheduler` holds the wrapped *train* loader. Note also that `train_loader` silently held the
**val** loader, so if nothing had crashed you would have trained on the validation set.

**How you find it.** Bottom frame is in `eval_loss`; the object it tries to iterate is not a
loader. Walk `val_loader` back to where it was last assigned: the `prepare` line. `prepare`
returns objects positionally, one per input, in the same order.

**Verify.** `assert isinstance(val_loader, torch.utils.data.DataLoader)` right after `prepare`,
or just `type(scheduler)` in the debugger. Cheaper still: prepare into the same names you passed.

## 2. Labels padded with `-1` (crash)

```python
# bug
labels = torch.full((n, max_len), -1, dtype=torch.long)
# fix
labels = torch.full((n, max_len), IGNORE_INDEX, dtype=torch.long)   # -100
```

**Symptom.**

```
  File ".../sft/v4_hf_accelerate/evaluate.py", line 29, in eval_loss
    total_nll += F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
IndexError: Target -1 is out of bounds.
```

On CUDA this is the far less readable `device-side assert triggered`, usually reported from
some later, unrelated kernel.

**How you find it.** Cross-entropy ignores exactly one value, `ignore_index`, and both PyTorch
and Hugging Face default it to `-100`. Every other negative target is an index into the vocab.
The prompt mask in `encode_example` already used `IGNORE_INDEX`; the collator used its own
constant. One source of truth.

**Verify.** `assert ((labels == IGNORE_INDEX) | (labels >= 0)).all()` on a batch; a test that
collates one short and one long row and checks the padded label positions equal `-100`.

## 3. The prompt/answer boundary is off by one (silent, the instructive one)

```python
# bug: tokenise the joined text, but measure the prompt separately
prompt_len = len(tok(prompt, add_special_tokens=False)["input_ids"])
ids = tok(prompt + spell(ex.output), add_special_tokens=False)["input_ids"]
input_ids = vocab.compact(ids) + [vocab.eos_id]
labels = [IGNORE_INDEX] * prompt_len + input_ids[prompt_len:]
# fix: tokenise the two parts separately and concatenate
prompt_ids = vocab.compact(tok(build_prompt(ex), add_special_tokens=False)["input_ids"])
response_ids = vocab.compact(tok(spell(ex.output), add_special_tokens=False)["input_ids"]) + [vocab.eos_id]
input_ids = prompt_ids + response_ids
labels = [IGNORE_INDEX] * len(prompt_ids) + response_ids
```

**Symptom.** Everything looks fine — val loss falls to 0.70 (healthy: 0.58) — and exact match is
**0.0 % on every task**, at every eval, on val and on holdout. The generations are almost right
but the first character is missing or wrong: `NRJTJY` → `PJTJY`, `2` → `y`.

**Root cause.** The prompt ends in `"Answer: "` (trailing space). Tokenised alone, GPT-2's BPE
emits `Answer`, `:`, `Ġ` (a standalone space, id 220). Tokenised together with the answer, the
space merges into the first answer character: `Answer`, `:`, `Ġf`, `Ġg`, … So `prompt_len` is
one too many for the joined ids: the first answer token is masked out of the loss, and the
standalone `Ġ` token that every *inference* prompt ends with never occurs in training at all.
The model is never taught what follows `Answer:` and never sees the token it is prompted with.
This affects **4000 of 4000** examples (`tok(p + r)[:len(tok(p))] != tok(p)` for all of them).
With a different template only some rows mismatch, which is worse: the bug then only costs a few
points and survives for months.

**How you find it.** A healthy loss with 0 % exact match means the training target and the
inference prompt disagree. Break at the batch breakpoint and decode one row next to its labels:

```python
[(tok.decode(vocab.expand([a])), int(b)) for a, b in zip(batch["input_ids"][0], batch["labels"][0])]
```

The first supervised token is the *second* answer character, and the token after `:` in the
training row is `Ġf` while `exact_match`'s prompt ends in `Ġ`.

**Verify.** `assert tok(p + r)["input_ids"][:len(tok(p)["input_ids"])] == tok(p)["input_ids"]`
fails for the joined form; after the fix `input_ids[:len(prompt_ids)] == prompt_ids` holds by
construction, and the first supervised label is the first answer character. Exact match jumps
from 0 % to ≈ 45–50 %.

## 4. Truncation cuts the tail of the answer (silent)

```python
# bug: cap the pair from the right
input_ids, labels = input_ids[:max_length], labels[:max_length]
# fix: keep the answer (+EOS) whole, cut the prompt's head
if len(response_ids) >= max_length:
    response_ids = response_ids[:max_length]
    response_ids[-1] = vocab.eos_id
    prompt_ids = []
else:
    budget = max_length - len(response_ids)
    prompt_ids = prompt_ids[-budget:]
```

**Symptom.** With `max_length = 48`, 196 of 4000 examples are longer than the cap, all of them
`Repeat:` rows (31 % of that task). Tail-cutting throws away the end of their answer *and the
EOS*, and for 115 of them nothing supervised is left at all, so the header says
`train 3489 | val 396 | holdout 400 (dropped 115 unsupervisable)` instead of `3600 | 400 | 400`.
The rest teach the model that a `Repeat:` answer can just stop mid-sentence without EOS, and it
generalises that: in the per-task table `Repeat:` shows `(no EOS: 22)` instead of ≈ 8 of 56, and
the `Repeat:` samples run to `max_new_tokens` (`'she said salose le sinse sinsinsinsinse …'`).
Overall exact match barely moves, because `Repeat:` is the weakest task anyway — this is why the
table prints the no-EOS count per task.

**How you find it.** The `dropped … unsupervisable` note and the no-EOS count are both new
compared to the healthy table. Ask which examples are long (`Repeat:`), then which end of the
pair the cap removes. Cutting the tail removes exactly the tokens that carry supervision; cutting
the head removes tokens the model would only have read.

**Verify.** `assert item["labels"][-1] == vocab.eos_id` for every encoded example (the reference
keeps the EOS even when the answer itself has to be cut); after the fix nothing is dropped and the
`Repeat:` no-EOS count returns to the healthy range.

## 5. Schedule length counted in micro-batches (silent)

```python
# bug
steps_per_epoch = len(train_loader)
# fix
steps_per_epoch = math.ceil(len(train_loader) / cfg.accum_steps)   # and `import math`
```

**Semantics.** With `Accelerator(gradient_accumulation_steps=4)` and the loop inside
`accelerator.accumulate(model)`, the wrapped optimizer and scheduler only act when
`accelerator.sync_gradients` is true, i.e. once every 4 micro-batches (and at the end of the
dataloader). `AcceleratedScheduler.step()` returns early on the other calls. So the scheduler
advances 684 times over the run, not 2700, and a cosine schedule told to last 2700 steps is still
at ≈ 90 % of its peak when training ends.

**Symptom.** The header says `optimizer steps: 2700 (225/epoch …)` while the epoch lines count
57, 114, … 684. Warmup (5 % of 2700 = 135 steps) is 4× longer than intended. The `lr` column
ends at `2.76e-03` instead of `≈ 1e-06`. In this run the final exact match was *inside* the
healthy range (49 %), so the metric alone would not have caught it — the printed schedule and the
step counter did. On a task where the annealing phase matters, this shows up as the last epochs
buying nothing.

**How you find it.** Two numbers that should agree don't (planned vs. reached optimizer steps).
Then check what `num_training_steps` was computed from. Rule: everything that is "per step" in
the schedule is per *optimizer* step, and with accumulation that is `len(loader) / accum`
(ceil, because the wrapped loader syncs on the last partial group).

**Verify.** `assert scheduler.get_last_lr()[0] < 1e-5` at the end of the run; or log
`scheduler.scheduler.last_epoch` and compare with the planned total.

---

## Measured runs

| Bugs present | best val loss | val EM | holdout EM | `Repeat:` no EOS |
|---|---|---|---|---|
| none (healthy) | 0.58 | 44.0 % | 44.2 % | 9 |
| 3 + 4 + 5 | 0.70 | **0.0 %** | **0.0 %** | 19 |
| 4 + 5 | 0.46 | 49.5 % | 50.0 % | **22** |
| 5 | 0.49 | 50.0 % | 49.0 % | 9 |

Same seed throughout; run-to-run noise on exact match is ±5 points once anything changes, so
bugs 4 and 5 have to be read off the no-EOS column, the `dropped` note, and the `lr` /
`optimizer steps` lines rather than the headline number.

## What could not be run here

These fixes were verified with a single-process, CPU stand-in for `Accelerator` (`prepare` is the
identity, `backward` divides by the accumulation steps, `accumulate` toggles `sync_gradients`
every 4 micro-batches and at the end of the epoch, and the wrapped optimizer / scheduler skip
their `step()` otherwise — the same contract the real class documents). The real `accelerate`
package was not installed on the machine that produced the numbers above. What that leaves
unverified: the exact class names in the two crash messages, and the wall-clock time on a 2-core
Codespace.
