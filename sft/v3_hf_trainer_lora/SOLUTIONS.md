# Solutions — variant 3 (HF `Trainer` + `peft` LoRA)

Spoilers. Five planted defects, all in `main.py`, `data.py` and `evaluate.py`. To put them back
after fixing: `git checkout -- sft/v3_hf_trainer_lora`.

| # | Where | Class | Loud or silent | Signature |
|---|---|---|---|---|
| 1 | `main.py: training_args` | Trainer drops dataset columns | **crash** | `KeyError: 'prompt_len'` inside `SFTCollator.__call__`, during the step-0 `trainer.evaluate()` |
| 2 | `data.py: SFTCollator` | double label shift | silent | phase-1 loss plateaus high, base-task exact match 0% |
| 3 | `data.py: SFTCollator` | eos masked because `pad == eos` | silent | loss looks fine, every generation runs to `max_new_tokens`, EM ≈ 0% |
| 4 | `main.py: LoraConfig` | wrong `target_modules` for GPT-2 | **crash** | `ValueError: Target modules {'q_proj', 'v_proj'} not found in the base model…` |
| 5 | `evaluate.py: exact_match` | right padding in batched generation | silent | EM far below what val loss implies; changes with `batch_size` |

The order above is roughly the order you meet them. Which of 2 and 3 you notice first depends on
whether you look at the loss or at the generations.

**What was verified where.** This folder was written on a machine without `peft`/`accelerate`,
so `Trainer` and `peft` were never executed here. Everything else was: the data module, the
tiny GPT-2 with `attn_implementation="sdpa"`, gradient checkpointing, the evaluator, and the
whole two-phase recipe driven by a plain PyTorch loop with the same hyper-parameters (that is
where the "healthy" numbers in the README come from, and the bug-2/3/5 measurements below). The
two crashes (1, 4) are described from the library source, not from a captured traceback; the
message text is what `transformers 4.57` / `peft 0.17` raise.

---

## 1. `Trainer` removes the `prompt_len` column (crash)

```python
# bug: TrainingArguments(...) without the flag -> remove_unused_columns=True (the default)
# fix
TrainingArguments(..., remove_unused_columns=False)
```

**Symptom.** The first run dies in the step-0 `trainer.evaluate()`:

```
  File ".../sft/v3_hf_trainer_lora/data.py", line ..., in __call__
    labels[i, : f["prompt_len"]] = IGNORE_INDEX
KeyError: 'prompt_len'
```

**How you find it.** The collator is fed rows from a `datasets.Dataset`, and `to_dataset` clearly
puts `prompt_len` in every row — check it: `base_train[0]` has it. So something between the
dataset and the collator dropped the column. That something is `Trainer._remove_unused_columns`:
when `remove_unused_columns=True` and the dataset is a `datasets.Dataset`, it keeps only the
columns whose names appear in `model.forward`'s signature (`input_ids`, `attention_mask`,
`labels`, …) plus the label names. `prompt_len` is not a forward argument, so it is removed,
and the only trace is an `info`-level log line you do not see by default. The traceback never
mentions `Trainer`.

**Why it is a trap.** The default is convenient when your dataset is already tokenised into
model arguments, and wrong the moment a custom collator needs anything else. The alternative fix
is to compute `labels` in `to_dataset` (so the row only has model arguments) — that is also
fine, and is what `DataCollatorForSeq2Seq` users do.

**Guard:** a test that calls `Trainer(...).get_train_dataloader()` and asserts the first batch
has non-ignored labels only after `prompt_len`. Or simply assert `"prompt_len" in
trainer.train_dataset.column_names` after construction — it is still there (the removal happens
on a copy), so the real guard is the dataloader one.

## 2. Labels shifted twice (silent)

```python
# bug (collator)
labels = torch.cat([labels[:, 1:], labels.new_full((B, 1), IGNORE_INDEX)], dim=1)
# fix: no shift at all; labels[i, p:n] = ids[p:] and leave them aligned with input_ids
```

`GPT2LMHeadModel` (every HF causal LM) shifts internally: `logits[..., :-1]` is scored against
`labels[..., 1:]`. The collator shifting once more trains the model to predict the token **two**
positions ahead.

**Symptom.** Phase-1 loss still decreases (the task is not impossible, just wrong), but plateaus
far above the healthy ≈ 0.24, and base-task exact match is 0% on all three tasks.

**How you find it.** Decode a batch:

```python
b = next(iter(trainer.get_train_dataloader()))
ids, lab = b["input_ids"][0], b["labels"][0]
sup = (lab != -100).nonzero().flatten()
[(tok.decode([vmap.full_ids[i]]), tok.decode([vmap.full_ids[j]])) for i, j in zip(ids[sup], lab[sup])]
```

On this branch that prints `('\n', 'z'), ('z', ' b'), (' b', ' v'), …`: the label at position
`t` is the input at `t+1`, i.e. already shifted, and HF will shift again. On the fixed collator
every pair is `(x, x)`.

**Measured** with the trained healthy model on one real batch: loss on aligned labels 0.0034,
loss on pre-shifted labels 13.46.

**Guard:** `test_labels_are_aligned_with_inputs` — for a collated batch, wherever
`labels != -100`, `labels == input_ids`.

## 3. `pad == eos`, so masking padding masks the eos target (silent)

```python
# bug (collator)
labels[input_ids == self.pad_id] = IGNORE_INDEX
# fix: mask by position, which the prefill already does; delete the line
labels = torch.full((B, L), IGNORE_INDEX); ...; labels[i, p:n] = ids[p:]
```

GPT-2 ships without a pad token; `build_tokenizer` does the usual `pad_token = eos_token`. From
then on every "mask the pad tokens by id" line also masks the real `<|endoftext|>` that closes
every answer, so the model is never taught to stop.

**Symptom.** Loss can be perfectly healthy (the eos is one token out of ~35), yet every
generation runs to `max_new_tokens` and exact match is ≈ 0% because the prediction has garbage
appended. `for ex, pred in samples[:5]` at the end shows long predictions that *start* correctly.

**How you find it.** Count eos targets in a batch: `(b["labels"] == vmap.eos_id).sum()`.
Measured on a 16-row batch: 16 with the fix, **0** with the bug (supervised tokens 162 → 146,
exactly one per row lost).

**Alternatives.** Add a real pad token (`tok.add_special_tokens({"pad_token": "<pad>"})`) and
`model.resize_token_embeddings(len(tok))` — then the line is harmless. Either way the lesson is:
pad/eos identity is a tokenizer-level decision that leaks into the loss mask.

**Guard:** `test_every_row_supervises_eos`.

## 4. LoRA `target_modules` from a different architecture (crash)

```python
# bug
LoraConfig(..., target_modules=["q_proj", "v_proj"])
# fix
LoraConfig(..., target_modules=["c_attn", "c_proj", "c_fc"], fan_in_fan_out=True)
```

**Symptom.** Phase 2 stops at `get_peft_model`:

```
ValueError: Target modules {'q_proj', 'v_proj'} not found in the base model. Please check the target modules and try again.
```

**How you find it.** `print(model)` — GPT-2 has fused `Conv1D` layers: `attn.c_attn` (q, k, v
in one matrix), `attn.c_proj`, `mlp.c_fc`, `mlp.c_proj`. `q_proj`/`v_proj` are Llama/Qwen
names copied from a tutorial. Two details worth saying out loud: (a) `c_proj` matches both the
attention and the MLP projection, which is fine; (b) `Conv1D` stores its weight as `(in, out)`,
so `fan_in_fan_out=True` — `peft` detects the mismatch and flips it with a warning if you forget,
but the explicit flag documents the intent. `target_modules="all-linear"` would *not* pick up
`Conv1D` layers, which is another way this bites GPT-2 specifically.

**Guard:** after `get_peft_model`, `model.print_trainable_parameters()` shows ~65k trainable of
~500k, and `any("lora_A" in n for n, _ in model.named_parameters())`.

## 5. Right padding in batched generation (silent)

```python
# bug
enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)   # padding_side="right" (default)
# fix
enc = tok(prompts, ..., padding_side="left")
```

A decoder-only model generates the next token from the **last** position of each row. With
right padding that position holds pad tokens for every row shorter than the longest prompt, so
the model is asked to continue `… ### Response:\n<pad><pad>` — it either emits eos immediately
(empty prediction) or drifts. With left padding the last position is the end of the real prompt
for every row, and `generate` rebuilds `position_ids` from `attention_mask` so the shifted rows
still see positions 0…n. Training is unaffected by the padding side: the loss is masked per
position and causal attention never looks right of the real tokens.

**Symptom.** Val loss says the model is good, exact match says it is not; the number moves when
you change `batch_size` in `exact_match` (fewer rows padded). Only the longest prompt in each
batch is scored correctly.

**Measured** on 32 base-task val prompts with the healthy phase-1 model (31 of the 32 rows
padded): left-padded EM **90.6%**, right-padded EM **0.0%**. Sample right-padded predictions:
`''`, `'hjwpqjhi'` (want `sywpqjhi`).

`transformers` even warns: *"A decoder-only architecture is being used, but right-padding was
detected!"* — easy to miss among the other warnings, which is the point.

**Guard:** `test_batched_generation_matches_single` — generate for one prompt alone and inside
a batch with a longer prompt; the outputs must be identical.

---

## Part 3: the validation script

```bash
python -m sft.v3_hf_trainer_lora.main --quick     # plumbing: both phases run, no crash
python -m sft.v3_hf_trainer_lora.main             # phase 1 EM ≈ 78%, adapter EM val ≈ holdout ≈ 45%
python -m pytest -q sft/v3_hf_trainer_lora        # if you wrote the guards as tests
```

Sanity signals to read off the log: step-0 loss ≈ ln(145) = 4.98; phase-1 val loss ≈ 0.24;
generations end (short predictions); adapter val ≈ holdout; `print_trainable_parameters()`
shows adapters only.

## Notes on the design (not bugs)

- **Spelled-out text and `VocabMap`** are deliberate (README). Without them a tiny model trained
  for a few minutes on CPU gets 0% on every task with raw BPE pieces, and the 50257-way
  `lm_head` makes each step ~50× slower.
- **Gradient checkpointing with `peft`.** With `use_reentrant=False` the frozen embeddings are
  fine. With the reentrant implementation the checkpointed block's input would not require grad
  and you would need `model.enable_input_require_grads()` — a classic *silent* failure mode
  (no adapter gradients) that is *not* planted here.
- **Base tasks collapse after LoRA** (≈ 78% → 0%). The adapters were trained on the new tasks
  only, with a high LR; that is forgetting, and the fix would be to mix base-task rows into
  phase 2 — an exercise, not a defect.
