# Variant 4 — PyTorch loop + Hugging Face `accelerate`

The same six string tasks and the same clean data as the other variants
(`data/sft_train.jsonl`, `data/holdout.jsonl`), written the way a "research loop" usually is:
you own the loop, the loss and the checkpoint policy, and `accelerate` owns device placement,
gradient-accumulation bookkeeping and (on a multi-GPU box) the distributed wrapping.

**Five bugs are planted in this folder: 2 crash, 3 are silent.** `SOLUTIONS.md` has the answers;
don't open it first.

## What is in here

| File | Stage |
|---|---|
| `config.py` | one dataclass of knobs → CLI flags (`--epochs 3`, `--quick`, …) |
| `data.py` | JSONL → split → prompt template → gpt2 tokenizer → label mask → padded batches |
| `evaluate.py` | token-weighted masked val loss, batched greedy exact match, per-task table |
| `train.py` | `Accelerator`, AdamW, `get_scheduler` warmup+cosine, `accumulate`, clip, best checkpoint |
| `main.py` | entry point; runs with "Run Python File", `python -m`, or `accelerate launch` |

Mechanisms used:

- **Model**: a tiny random-init GPT-2 (`GPT2Config`, 2 layers, 128 wide, ~0.4M params) built with
  `attn_implementation="sdpa"` — PyTorch's fused `scaled_dot_product_attention`, which is the
  FlashAttention path on CUDA and the math kernel on CPU.
- **Tokenizer**: the real `gpt2` BPE tokenizer (downloaded once, then cached). GPT-2 merges several
  letters into one token, which makes character tasks impossible, so inputs and answers are
  *spelled out* (`"a b c"`, a real space written as `_`; see `spell`/`unspell` in `data.py`). The
  model only gets an output layer over the ~150 gpt2 ids this corpus actually uses (`VocabSlice`),
  because a 50257-way softmax would be >90 % of the compute for this model.
- **Gradient accumulation** via `Accelerator(gradient_accumulation_steps=…)` + `accelerator.accumulate(model)`.
- **Gradient checkpointing** via `model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})`.
- **Schedule**: `transformers.get_scheduler("cosine", …)` with linear warmup, stepped once per optimizer step.
- **Checkpointing**: `accelerator.unwrap_model(model).save_pretrained(...)` on the best val loss,
  reloaded with `from_pretrained` for the final val + holdout exact match.

## Run it

```bash
python -m sft.v4_hf_accelerate.main              # full run (12 epochs)
python -m sft.v4_hf_accelerate.main --quick      # 60 optimizer steps, for iterating on a fix
python sft/v4_hf_accelerate/main.py              # same thing; "Run Python File" in VS Code works
accelerate launch sft/v4_hf_accelerate/main.py   # same script through accelerate's launcher
```

Needs `torch`, `transformers`, `accelerate` (the Codespace installs them; see `requirements.txt`).
Output goes to `checkpoints/v4_hf_accelerate/` (`best/` = the checkpoint, `metrics.json` = the numbers).

## What a healthy run looks like

Measured with every bug fixed, default flags, seed 1337. Time is for an 8-thread laptop; a 2-core
Codespace takes roughly 2–3× longer. The exact-match numbers move by a few points when anything
about the run changes (different machine, thread count, one changed line of code), so read them as
ranges, not targets.

| Signal | Healthy |
|---|---|
| `examples:` line | train 3600 / val 400 / holdout 400, **nothing dropped** |
| `vocab:` line | 151 of 50257 gpt2 tokens |
| `optimizer steps:` line | **684** (57 per epoch, effective batch 64); the epoch lines then count up to 684 |
| `eval @ 0` val loss | ≈ **5.04** = ln(151): a uniform guess over the model's vocabulary |
| `lr` column | warms up to 3e-3, then cosine-decays to ≈ **0** by the last log line |
| best val loss | ≈ **0.58** (ppl 1.8) |
| exact match | val ≈ **44 %**, holdout ≈ **44 %** (anything in 40–52 % is normal); the two agree within a few points |
| per-task (holdout) | `Length:` ≈ 90 %, `Sort letters:` ≈ 55 %, `Reverse:` ≈ 40 %, `Uppercase:` ≈ 40 %, `Count vowels:` ≈ 45 %, `Repeat:` ≈ 5–10 % |
| `(no EOS: n)` column | 0 for every task except `Repeat:` (≈ 6–9 of 56 never stop; long sentences are hard for this model) |
| wall clock | ≈ 1 min on 8 threads, ≈ 2–3 min on 2 cores |

`Repeat:` is weak even when everything is right — copying a 20–50-token sentence is the hardest
thing this 2-layer model is asked to do. That is the baseline; watch how the column *moves*.

## Rules

Same as `EXERCISE.md`: reproduce first, read stack traces from the bottom up, bisect the pipeline
(`load → encode → collate → model → loss → step → eval`), minimal diffs, one test or one
assertion per fix, and a short log per bug (*symptom → root cause → fix → how I verified it*).

Two things worth knowing before you start:

- `GPT2LMHeadModel` computes the loss itself when you pass `labels`, **shifting them internally**.
  `evaluate.eval_loss` recomputes the same thing by hand (token-weighted, not a mean of batch means).
- `accelerator.prepare` returns wrapped objects **in the order you passed them**. The wrapped optimizer
  and scheduler only really step when `accelerator.sync_gradients` is true; the wrapped dataloaders
  tell the accelerator where an epoch ends.

<details><summary>Level 1: symptoms to hunt for</summary>

- The very first eval, before any training, crashes on something that is "not iterable". Read what
  variable you are iterating and where it came from.
- Once that is fixed, the same eval crashes inside the loss with an index that is out of bounds.
  Which index, and what did *you* put there?
- Then it trains. Compare every line of the header and the first eval with the healthy table:
  the `examples:` line, the `optimizer steps:` line, the step-0 loss.
- Loss goes down nicely, exact match stays at **0 %** on every task. Look at what the model
  actually generates: what is wrong with the *first* character?
- Watch the `lr` column across the run and compare the planned number of optimizer steps with the
  number the epoch lines reach.
- Watch the `(no EOS: n)` column for `Repeat:`, and the "dropped … unsupervisable" note.

</details>

<details><summary>Level 2: where to look</summary>

- Both crashes: `train.py` around `accelerator.prepare`, and `data.py`'s `Collator`. Cross-entropy
  ignores exactly one label value; which one?
- 0 % exact match with a fine loss: `data.encode_example`. Decode one training row and print its
  labels next to it. Now tokenise the *prompt alone* the way `exact_match` does and compare the last
  token. BPE does not tokenise `"Answer: "` + `"f g"` the same way as `"Answer: f g"`.
- Long examples: what does `encode_example` throw away when a pair is longer than `max_length`, the
  head or the tail? Which task has long answers, and what token sits at the very end of every answer?
- Schedule: how many times per epoch does `scheduler.step()` *actually* advance the LR with
  `accumulate` on, and what number did you give `get_scheduler`?

</details>

Level 3 is `SOLUTIONS.md`.
