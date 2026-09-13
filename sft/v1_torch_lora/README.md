# Variant 1 — pure PyTorch, hand-rolled LoRA, SDPA attention, gradient accumulation

Same task family and data as the main exercise (`data/sft_train.jsonl`, `data/holdout.jsonl`),
written as a **two-phase run** in one script:

| Phase | What | Trains | Data |
|---|---|---|---|
| 1 `base` | full fine-tune of a tiny GPT (2 layers, 128 wide, ~0.44M params) | every parameter | `Reverse:`, `Uppercase:`, `Repeat:` |
| 2 `adapter` | freeze the base, attach **LoRA** (r=16, alpha=32) to `qkv`, `proj`, `fc`, train only the adapters, then **merge** them into the weights for inference | 65k LoRA params | `Sort letters:`, `Length:`, `Count vowels:` + a 30% replay slice of base examples |

Mechanisms you will meet in the code:

- **SDPA attention** (`model.py`): `F.scaled_dot_product_attention` is PyTorch's
  FlashAttention entry point. On CUDA it dispatches to the flash / memory-efficient kernels; on
  CPU it runs the math kernel. The model code is the same either way.
- **Gradient accumulation** (`main.py`, `run_phase`): `accum_steps` micro-batches of
  `micro_batch` examples per optimizer step (default 4 × 8 = 32 effective).
- **LoRA** (`lora.py`): `LoRALinear` wraps an `nn.Linear` with `W x + (alpha / r) · B A x`;
  `attach_lora` swaps the target layers, `merge_lora` folds the adapters back into `W`.
- Warmup + cosine LR, AdamW with decay groups, gradient clipping, periodic val loss + exact
  match, final exact match on val and on the clean holdout, per-task breakdown.

Nothing here imports from the top-level `sft/` package; the folder is standalone.

## Run it

```bash
python sft/v1_torch_lora/main.py            # full run, ~3 min on a 2-core Codespace
python sft/v1_torch_lora/main.py --quick    # 300 + 300 iterations, ~1 min
python -m sft.v1_torch_lora.main --reuse_base   # skip phase 1 when checkpoints/v1_torch_lora/base.pt exists
python sft/v1_torch_lora/main.py --help
```

Or open `main.py` in VS Code and use **Run Python File** / F5. Checkpoints and `metrics.json`
land in `checkpoints/v1_torch_lora/` (gitignored). `# BREAKPOINT:` comments mark the useful
places to stop.

## The exercise

There are **five planted defects** in this folder. **One crashes; four are silent** and only
show up in the numbers. None of them are in `data.py` or `evaluate.py`. Same rules as
`EXERCISE.md`: reproduce, read the trace bottom-up, bisect with assertions, minimal diffs, and a
test per fix that fails before and passes after. Keep a log of *symptom → cause → fix → proof*.

### What a healthy run looks like

Measured with the defaults on CPU with 2 threads (`OMP_NUM_THREADS=2`), seed 1337.

| Signal | Healthy |
|---|---|
| phase 1 `eval @ 0` val loss | ≈ **5.59** (≈ ln 261: a uniform guess) |
| phase 1 val loss at 1500 | ≈ **0.46**; train loss around 0.15–0.3, not ≈ 0 |
| phase 1 val exact match at 1500 | ≈ **52%**; base tasks on holdout ≈ **54%** |
| phase 2 `eval @ 0` val loss | high (≈ 11) — the base has never seen these tasks — then falls quickly |
| phase 2 trainable params | **65.5k of 503.9k** |
| phase 2 val exact match at 1500 | ≈ **46%** (150-example estimate) |
| final adapter tasks, **merged** vs **unmerged** | **identical**: val ≈ 43.7% both ways |
| final adapter tasks on holdout | ≈ **45%** (Sort letters ≈ 57%, Count vowels ≈ 31%, Length ≈ 25%) |
| base tasks on holdout after the merge | drops from ≈ 54% to ≈ 24% — expected: LoRA on every layer with a small replay slice forgets some of the old tasks. Not a planted bug. |
| wall clock | ≈ **170 s** on 2 threads |

`--quick` is enough to see most of the symptoms; the full run is what you compare against the
table.

### Hints: open only when stuck

<details><summary>Level 1: symptoms to hunt for</summary>

- Phase 1 looks *too* good: val loss ends far below 0.46, yet exact match never leaves ~0%.
  Which is lying, the loss or the generations?
- Phase 2 never starts: an exception inside the optimizer constructor.
- Once phase 2 runs, its loss does not move at all from the step-0 value.
- Learning is much slower than the table says (compare the val loss at matching steps).
- The last periodic exact match and the final "merged" exact match disagree by a lot.

</details>

<details><summary>Level 2: where to look</summary>

- Loss too good + 0% exact match, in a causal LM, means the model can see the target. Read
  `CausalSelfAttention.forward` and the `attn_mask` documentation of
  `F.scaled_dot_product_attention` very carefully: what does the *dtype* of the mask mean?
  Prove it with a 5-token toy example: perturb a future position and check whether an earlier
  output changes.
- `optimizer got an empty parameter list`: which parameters have `requires_grad=True` at the
  moment the optimizer is built? In what order do freeze and attach happen?
- Flat phase-2 loss: print `lora_A.abs().sum()` and `lora_B.abs().sum()` at step 0 and the
  gradient norms at step 1. Write out d(loss)/dA and d(loss)/dB by hand.
- Slow learning: what is the *effective* batch size? After two micro-steps, is `p.grad` the
  sum of the two micro-gradients?
- Merged ≠ unmerged: `LoRALinear.forward` and `LoRALinear.merged_linear` must compute the same
  function. Compare their outputs on a random input before and after merging.

</details>

Level 3 is `SOLUTIONS.md` in this folder. To put the bugs back after fixing them:
`git checkout -- sft/v1_torch_lora`.
