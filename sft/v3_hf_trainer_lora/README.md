# Variant 3 — Hugging Face `Trainer` + `datasets` + `peft` LoRA

Same task and data as the rest of the repo (`data/sft_train.jsonl`, `data/holdout.jsonl`), written
the way most people write SFT today: `Trainer` owns the loop, `datasets.Dataset` holds the rows,
`peft` adds LoRA. **Five defects are planted in this folder** (2 crash, 3 silent). Find them, fix
them with minimal diffs, prove each fix. `SOLUTIONS.md` is the answer key — don't open it first.

## What it does

```
data/sft_train.jsonl ──split──► train / val            data/holdout.jsonl
        │
        ▼  "### Instruction:\n{task}\n{spelled input}\n### Response:\n" + spelled output + eos
   gpt2 tokenizer ──► VocabMap (50257 ids -> the ~145 used) ──► datasets.Dataset(input_ids, prompt_len)
        │
        ▼  SFTCollator: right-pad, attention_mask, labels (prompt masked to -100)
   phase 1  Trainer, full fine-tune of a tiny random-init GPT-2      on Reverse / Uppercase / Repeat
   phase 2  Trainer, peft LoRA (r=16) on the frozen phase-1 model    on Sort letters / Length / Count vowels
        │
        ▼  merge_and_unload ──► batched greedy generate ──► exact match on val + holdout, per task
```

Two phases because LoRA on a random-init model would have nothing to adapt: phase 1 makes a
"pretrained" base, phase 2 adapts it to three new tasks with adapters only.

Mechanisms in play:

| Mechanism | Where |
|---|---|
| `Trainer` loop (AdamW, warmup + cosine, clipping, periodic eval, logging) | `main.py: training_args`, `run_trainer` |
| gradient checkpointing | `TrainingArguments(gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})` |
| SDPA attention (PyTorch fused kernel; flash on CUDA, math on CPU) | `AutoModelForCausalLM.from_config(..., attn_implementation="sdpa")` |
| LoRA | `peft.LoraConfig` + `get_peft_model` + `merge_and_unload` |
| dynamic padding collator, assistant-only loss | `data.py: SFTCollator` |
| batched greedy generation | `evaluate.py: exact_match` |

Two things that are specific to this variant and are **not** bugs:

- **Spelled-out strings.** GPT-2's BPE merges characters into opaque pieces; a tiny model cannot
  reverse or sort what it cannot see. `data.spell` turns `abc d` into `a b c _ d` so every
  character is its own token, and `unspell` undoes it before exact match.
- **`VocabMap`.** The gpt2 vocabulary has 50257 ids, this data uses ~145. A 50257-way `lm_head`
  would be ~95% of the CPU time, so the model is built over the used ids only and ids are
  translated at the two boundaries (encode, generate). The model never sees a gpt2 id.

## Run it

Needs `transformers`, `datasets`, `peft`, `accelerate` (installed by the Codespace's
`post-create.sh`). The `gpt2` tokenizer is downloaded once.

```bash
python sft/v3_hf_trainer_lora/main.py            # or "Run Python File" on main.py, or F5
python -m sft.v3_hf_trainer_lora.main --quick    # ~30 s plumbing check (150 + 150 steps)
python -m sft.v3_hf_trainer_lora.main --reuse_base   # skip phase 1 if checkpoints/.../base exists
```

Outputs go to `checkpoints/v3_hf_trainer_lora/` (`base/`, `adapter/`, `metrics.json`).

## What a healthy run looks like

These numbers come from the same data module, model and hyper-parameters driven by a plain
PyTorch loop that mirrors `TrainingArguments` (the author's machine had no `peft`/`accelerate`),
so treat them as **expected ranges**, not exact targets. Wall time is for 2 CPU threads.

| Signal | Healthy |
|---|---|
| `eval @ 0` loss, both phases' first line | ≈ **ln(145) = 4.98** for the untrained model in phase 1 (phase 2 starts from a trained base and is *higher* than that on the unseen tasks, ≈ 9) |
| phase 1 val loss | ≈ 1.4 @ 250 → **≈ 0.24 by step 1000**, flat after |
| phase 1 base-task exact match (val) | ≈ **78%** — Reverse ≈ 98%, Uppercase ≈ 87%, Repeat ≈ 40% |
| phase 2 val loss | ≈ 0.9 @ 250 → **≈ 0.55** at the end |
| adapter-task exact match | **val ≈ 45%, holdout ≈ 44%** (Sort letters ≈ 53%, Count vowels ≈ 42%, Length ≈ 18%); val and holdout agree within a few points |
| base tasks after LoRA | drop sharply (≈ 0%). The adapter is trained on the new tasks only, so this is forgetting, not a bug |
| wall clock | phase 1 ≈ 2 min, phase 2 ≈ 1 min |

`--quick` will not reach those numbers; it only checks that everything runs.

## The exercise

Five planted defects, **none in `config.py`**. Two crash, three quietly produce wrong numbers.
Rules as in `EXERCISE.md`: reproduce first, read stack traces bottom-up, bisect the pipeline
(`split → encode → collate → model → loss → generate → exact match`), minimal diffs, one test per
fix, and a symptom → cause → fix → verification log. Interviewer follow-ups to be ready for:

- Why does batched generation care which side the padding is on, and why doesn't training?
- What does `Trainer` do to your dataset columns before the collator ever sees them?
- Who shifts the labels in an HF causal LM, and what happens if two people do?
- `pad_token = eos_token` is the GPT-2 default. What else in the pipeline does that touch?

<details><summary>Level 1: symptoms to hunt for</summary>

- The very first run crashes before the step-0 evaluation finishes. The message names a
  dictionary key, not anything about `Trainer`.
- Once phase 1 trains: the loss goes down but stays far from the healthy table, and exact match
  is 0% on every base task. Decode a batch's inputs and labels next to each other.
- Even when the loss is healthy, generations never end: every prediction runs to
  `max_new_tokens`. Count the eos ids in a batch's `labels`.
- Phase 2 crashes on the `peft` call with a message about module names.
- With everything trained, exact match is still far below what the eval loss suggests, and it
  changes when you change `batch_size` in `exact_match`. Look at what the padded rows of a
  generation batch actually continue from.

</details>

<details><summary>Level 2: where to look</summary>

- `KeyError` in the collator: `TrainingArguments` has a flag whose default silently edits
  `datasets.Dataset` columns to match the model's `forward` signature.
- Labels: `GPT2LMHeadModel.forward` already does `logits[..., :-1]` vs `labels[..., 1:]`. Read
  the collator with that in mind.
- eos: `tok.pad_token_id == tok.eos_token_id`. Any line that masks "padding" by token *identity*
  masks something else too. Mask by *position* (length) instead.
- `peft`: GPT-2 has no `q_proj`; `print(model)` shows `Conv1D` layers called `c_attn`, `c_proj`,
  `c_fc` (and `fan_in_fan_out` matters for `Conv1D`).
- Generation: `tokenizer.padding_side`. A decoder-only model continues from the *last* position
  of each row.

</details>

Level 3 is `SOLUTIONS.md`. To put the bugs back: `git checkout -- sft/v3_hf_trainer_lora`.
