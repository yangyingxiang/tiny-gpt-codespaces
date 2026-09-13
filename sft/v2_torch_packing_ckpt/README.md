# v2 — pure PyTorch, sequence packing + gradient checkpointing

Same task and data as the rest of the repo (`data/sft_train.jsonl`, `data/holdout.jsonl`, six
string tasks, exact match as the metric), written as an **epoch-based** loop with three
mechanisms the plain `sft/` pipeline does not have:

| Mechanism | Where | What it changes |
|---|---|---|
| **Sequence packing** | `packing.py` | ~7 examples share one 256-token block. Each block carries `seq_ids` (which example a position belongs to) and `pos_ids` (position *within* its example). A **block-diagonal causal mask** keeps the examples from seeing each other. |
| **SDPA attention** | `model.py` | `F.scaled_dot_product_attention` with an explicit bool mask. On CUDA this is the FlashAttention / memory-efficient kernel; on CPU it is the math kernel. Same maths as `sft/model.py`, fewer lines. |
| **Gradient checkpointing** | `model.py` | each transformer block is wrapped in `torch.utils.checkpoint.checkpoint`, so its activations are dropped after the forward pass and recomputed in backward. Trades ~30% compute for memory. |

Other choices worth knowing: labels are **unshifted** (the causal shift happens once, in
`loss_token_mean`), the loss is `sum(CE) / count(supervised tokens)`, the val loss is a corpus
NLL (sum over batches, divide once), dropout is 0.05, AdamW has no-decay groups, and the LR
schedule is linear warmup + cosine built once for the whole run.

```
jsonl ──split──► encode (prompt masked) ──pack──► 256-token blocks + seq_ids + pos_ids
                                                          │
             block-diagonal causal mask ──► GPT (SDPA, checkpointed blocks) ──► token-mean loss
                                                          │
                          AdamW + warmup/cosine ──► best-val checkpoint ──► exact match (val, holdout)
```

## Run it

Everything is standalone: nothing here imports the top-level `sft/` modules.

```bash
python sft/v2_torch_packing_ckpt/main.py            # full run: 36 epochs, ~4 min on 2 threads
python sft/v2_torch_packing_ckpt/main.py --quick    # 4 epochs, small evals, ~30 s
python -m sft.v2_torch_packing_ckpt.main --epochs 8 --eval_interval 240
python sft/v2_torch_packing_ckpt/evaluate.py        # re-score checkpoints/v2_torch_packing_ckpt/model.pt
```

Or open `main.py` and use **Run Python File** / the "Python: current file" debug configuration.
Every `TrainConfig` / `ModelConfig` field is a CLI flag (`--dropout 0`, `--grad_checkpoint false`, ...).

## The exercise

There are **five planted defects** in this folder. **One crashes. Four don't**: they quietly
produce wrong numbers. None of them is a bug class you have already seen in `sft/` (no
truncation mismatch, no off-by-one token id, no unshifted labels, no wrong loss denominator,
no leakage, no unseeded RNG). Same rules as `EXERCISE.md`: reproduce first, minimal diffs,
a test per fix, a log of *symptom → root cause → fix → verification*.

### What a healthy run looks like

Measured with the fixed code, default flags, seed 1337. The `sft/` pipeline reaches ~62%
with 5.3k training examples and dropout 0; this one has 3.6k examples and dropout 0.05, so
compare against **this** table, not that one.

| Signal | Healthy |
|---|---|
| `eval @ 0` val loss | **5.558** ≈ ln(261): a uniform guess over the vocabulary |
| `packed:` line | 474 train blocks, ~235 tokens and ~7.6 examples per block; `steps: 60/epoch x 36 epochs = 2160` |
| Printed `lr` | rises to 3e-3 by step 108, then decays smoothly and **monotonically** to 3e-4 at step 2160 |
| Training loss (25-step window) | ≈ 0.45 at step 1200, ≈ 0.15 at the end |
| Best val loss | **≈ 0.50** (last eval, step 2160) |
| Val exact match during training | 15% → 27% → 33% → 45% → 51% → 53% → 60% at steps 300 … 2100 (150 examples, so ±4%) |
| Final exact match (400 examples each) | **val ≈ 54%, holdout ≈ 54%**; within a few points of each other |
| Holdout by task | `Length:` 100%, `Sort letters:` ≈ 68%, `Count vowels:` ≈ 58%, `Uppercase:` ≈ 54%, `Reverse:` ≈ 44%, `Repeat:` ≈ 16% |
| Re-running the same command | **identical** numbers on the same machine and thread count (a different thread count changes float summation order and moves EM by a few points: 49.8% / 54.0% with 2 threads), and `evaluate.py` on the checkpoint prints the same val/holdout EM as `main.py` did |
| 8 epochs (`--epochs 8 --eval_interval 240`) | val loss ≈ 1.06, val EM ≈ 22%, holdout ≈ 22% (a cheaper reference for comparing fixes) |
| Wall clock | 4.2 min with 2 CPU threads (`OMP_NUM_THREADS=2`), ~5 min with `--epochs 36` under load; `--quick` is ~30 s |

### Validation ideas specific to this pipeline

- **Packing isolation test.** `packing.assert_no_cross_attention(model, block, seq=k)` runs
  example `k` alone and inside its pack and compares logits. It is in the code, un-called: call
  it from a test. It must pass for *every* `k`, including `k = 0`.
- **Gradient flow test.** After one backward pass, every parameter with `requires_grad` must
  have a non-`None` `.grad`. List the ones that don't.
- **Checkpointing equivalence.** Loss and gradients with `--grad_checkpoint true` and `false`
  must match to float precision (same seed, dropout 0).
- **Eval determinism.** Scoring the same checkpoint twice must give the same exact match.
- **Schedule shape.** Record `scheduler.get_last_lr()` every step and plot it, or just assert
  it never increases after warmup.
- **Overfit 8 examples** to ~0 loss and 100% exact match before trusting anything else.

### Hints: open only when stuck

<details><summary>Level 1: symptoms to hunt for</summary>

- The first run crashes at `eval @ 0`, before any training step. Read the last frame that is
  in this folder, not the torch frames.
- Once it runs: the val loss stalls around 2.8 and exact match barely leaves 0%. Several
  things are wrong at once; fix the loudest and re-measure each time.
- Watch the `lr` column of the `iter` lines across an epoch boundary (every 60 steps).
- The docstrings say what the code is *supposed* to do. One function does less than its
  docstring claims.
- Which parameters actually receive gradients?
- Score the same checkpoint twice with `evaluate.py`.

</details>

<details><summary>Level 2: where to look</summary>

- Crash: what dtype does `F.cross_entropy` want for targets, and which tensor in `collate_blocks`
  is not that?
- `model.py`, the checkpointing branch of `GPT.forward`: what is passed *into* `checkpoint`,
  and what does that do to the autograd graph below it?
- `main.py`, the epoch loop: how many times is `make_scheduler` called per run, and what
  `total_steps` does it see?
- `packing.build_attention_mask`: is the mask block-diagonal, as the module docstring
  promises? `assert_no_cross_attention` answers that in one call.
- `CausalSelfAttention.forward`: SDPA does not know whether the module is in train or eval
  mode. What gates `dropout_p`?

</details>

Level 3 is `SOLUTIONS.md` in this folder. Don't open it until you have your own list.

## Practising again

```bash
git checkout -- sft/v2_torch_packing_ckpt      # puts the bugs back
```
