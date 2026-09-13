# Solutions — v2 (packing + gradient checkpointing)

Spoilers. Every number below was measured on the fixed code (seed 1337, CPU); your numbers
will be close but not necessarily identical on different hardware.

| # | Where | Class | Loud or silent | Signature |
|---|---|---|---|---|
| 1 | `packing.collate_blocks` | wrong dtype | **crash** | `RuntimeError: expected scalar type Long but found Int` at `eval @ 0` |
| 2 | `packing.build_attention_mask` | packing mask not block-diagonal | silent | val loss bottoms out ≈ 1.9 then rises; `assert_no_cross_attention` fails |
| 3 | `model.GPT.forward` (checkpoint branch) | autograd graph cut | silent | `pos_emb` and `blocks.0.*` have `.grad is None`; val EM 7% at 8 epochs (22% healthy) |
| 4 | `main.train` (epoch loop) | LR schedule rebuilt per epoch | silent | printed `lr` saw-tooths every 60 steps |
| 5 | `model.CausalSelfAttention.forward` | dropout active in eval | silent | same checkpoint scores 48.0%, 46.2%, 47.2% on three evals (53.5% correct) |

Fresh from `git checkout`, the run dies at bug 1. With only that fixed (bugs 2–5 still in),
8 epochs give val loss 2.77 and 3.2% exact match: everything looks broken at once, which is
the realistic situation. The order below is the order the signals point you in.

---

## 1. Labels are `int32` (crash)

```python
# bug (packing.collate_blocks)
labels = torch.full((bsz, max_len), IGNORE_INDEX, dtype=torch.int32)
...
labels[i, :n] = torch.tensor(b["labels"], dtype=torch.int32)
# fix
labels = torch.full((bsz, max_len), IGNORE_INDEX, dtype=torch.long)
...
labels[i, :n] = torch.tensor(b["labels"], dtype=torch.long)
```

**Observed.**

```
  File ".../sft/v2_torch_packing_ckpt/evaluate.py", line 37, in eval_loss
    out = loss_token_mean(logits, batch["labels"].to(device))
  File ".../sft/v2_torch_packing_ckpt/model.py", line 164, in loss_token_mean
    sum_nll = F.cross_entropy(shift_logits.view(b * t, v), shift_labels.view(b * t),
RuntimeError: expected scalar type Long but found Int
```

**How you find it.** It dies inside `F.cross_entropy` at the very first eval, so nothing has
been trained yet. Cross-entropy targets must be `int64`. `input_ids`, `seq_ids` and `pos_ids`
in the same collate are `long`; only `labels` is not. (`data.encode_example` builds labels
with `torch.where` on a `long` tensor and returns lists, so the dtype is decided by collate.)

**Verify.** A collate test: `collate_blocks(blocks)["labels"].dtype is torch.long`.

## 2. Packed examples attend to each other (silent, but not subtle)

```python
# bug (packing.build_attention_mask)
causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=seq_ids.device))
return causal.view(1, 1, T, T).expand(B, 1, T, T)
# fix
causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=seq_ids.device))
same_seq = seq_ids[:, :, None] == seq_ids[:, None, :]          # (B, T, T)
return (causal[None] & same_seq).unsqueeze(1)
```

The function's own docstring (and the module docstring) promise "causal AND same example";
the code only does causal. Inside a 256-token block, example k+1's prompt and answer attend
to everything from examples 0..k.

**Observed** (only this bug, 36 epochs): val loss bottoms out at **1.92** (epoch 20) and then
*rises* to 2.18 while val EM crawls to 17%; final val EM 10.8%, holdout 15.2% vs 54% / 54%;
`Uppercase:` 0%, `Repeat:` 0%, `Sort letters:` 10% (68% healthy). The self-check fails on the
untrained model already:

```
AssertionError: example 1 sees other examples in its pack (max |logit diff| = 0.525 > 0.0001)
```

**Why it hurts this much.** Two reasons. Training tokens see ~200 tokens of unrelated
context that inference never provides (train/serve mismatch: `generate` runs one example at a
time with a plain causal mask). And the val loss is computed on *packed* val blocks with the
same broken mask, so it measures a different task from the one the exact-match eval runs.
I expected this to be subtle; it is not. The generic lesson still holds: **the test is how you
catch it**, because "loss 1.9 instead of 0.50" only tells you something is wrong somewhere.

**How you find it.** The cheap route is the isolation test that is already in the file:
`assert_no_cross_attention(model, block, seq=1)` on a random model. `seq=0` passes (the first
example has nothing before it), every other `seq` fails. Or print the mask for a small block
and look at it: it should be a staircase of triangles, not one big triangle.

**Verify.** The isolation test for `seq` in 0..n-1, on a random model, plus the same check
in `train()` mode with dropout 0.

## 3. Checkpointing gets a detached input (silent)

```python
# bug (model.GPT.forward)
x = checkpoint(block, x.detach(), attn_mask, use_reentrant=False)
# fix
x = checkpoint(block, x, attn_mask, use_reentrant=False)
```

`x.detach()` cuts the graph at the input of every checkpointed block. Gradients still reach
the block's own parameters (they are captured by the closure), but nothing flows *through* the
block to what came before it. With two blocks: `blocks.1` and `ln_f` train, `blocks.0` and
`pos_emb` never do. `tok_emb` still gets a gradient, but only through the tied output head.

**Observed** (only this bug, 8 epochs): val loss **1.62** vs 1.06 healthy-at-8-epochs; val EM
7.2% / holdout 7.8% vs 22.5% / 21.8%. After one backward pass:

```
grad is None: ['pos_emb.weight', 'blocks.0.ln1.weight', 'blocks.0.ln1.bias',
               'blocks.0.attn.qkv.weight', ..., 'blocks.0.mlp.proj.bias']
```

**How you find it.** "Loss plateaus high" points nowhere in particular. Two cheap probes do:
(a) `--grad_checkpoint false` trains normally, so the problem is in the checkpointing branch;
(b) list parameters whose `.grad is None` after a backward. Half the model is missing. Then
read the four lines of the checkpointing branch.

**Verify.** Gradient-flow test (no `None` grads), and a checkpointing-equivalence test: same
seed, dropout 0, loss and every gradient equal with checkpointing on and off (`atol=1e-6`).

## 4. Scheduler rebuilt every epoch (silent)

```python
# bug (main.train)
total_steps = len(train_loader) * tcfg.epochs
...
for epoch in range(tcfg.epochs):
    scheduler = make_scheduler(optimizer, len(train_loader), tcfg.warmup_frac,
                               tcfg.min_lr_ratio)
# fix
total_steps = len(train_loader) * tcfg.epochs
scheduler = make_scheduler(optimizer, total_steps, tcfg.warmup_frac, tcfg.min_lr_ratio)
for epoch in range(tcfg.epochs):
```

Every epoch gets a fresh 60-step warmup + cosine: the LR spikes back to 3e-3 36 times, and
the run never spends time at the low LR that the final epochs are supposed to have.

**Observed** (only this bug, 8 epochs; the `lr` column of the log):

```
iter    25 | lr 2.12e-03      iter   175 | lr 3.51e-04      iter   325 | lr 2.12e-03
iter    75 | lr 2.72e-03      iter   225 | lr 7.36e-04      iter   375 | lr 2.72e-03
iter   125 | lr 2.99e-03      iter   275 | lr 1.39e-03      iter   425 | lr 2.99e-03
```

val loss 1.20 vs 1.06, val EM 13.8% / holdout 17.0% vs 22.5% / 21.8%. The metric gap is
modest at 8 epochs (36 re-warms hurt more than 8); the `lr` column is the reliable tell.

**How you find it.** The healthy `lr` column is monotone after step 108. This one is periodic
with period 60 = `len(train_loader)`. Grep for `make_scheduler` and count the calls.

**Verify.** Collect `scheduler.get_last_lr()[0]` per step for a 3-epoch run and assert it is
non-increasing after the warmup, or unit-test `make_scheduler` and assert it is constructed
exactly once (e.g. wrap it and count).

## 5. Attention dropout stays on in eval (silent)

```python
# bug (model.CausalSelfAttention.forward)
y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                   dropout_p=self.dropout_p)
# fix
y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                   dropout_p=self.dropout_p if self.training else 0.0)
```

`nn.Dropout` modules read `self.training`; the functional `dropout_p` argument of SDPA does
not. `model.eval()` switches off the residual and MLP dropouts but not this one, so every eval
and every greedy generation is stochastic.

**Observed.** The healthy checkpoint, scored three times by `evaluate.py` with this bug in
place: val **48.0% / 46.2% / 47.2%**, holdout 51.0% / 49.8% / 47.5%. With the fix: 53.5% /
53.8%, twice, identical to what `main.py` printed at the end of training. So the bug both
lowers the score and makes it jitter; "greedy" decoding is not greedy any more.

**How you find it.** Score the same checkpoint twice. Then ask which sources of randomness
survive `model.eval()`: every `nn.Dropout` is gated, so grep for the one that is not a module.

**Verify.** Determinism test: two calls of `exact_match` on the same model give the same
predictions; and a direct one: in eval mode, two forward passes on the same batch are equal.

---

## What the fixed run prints

```
  eval @     0 (start): val loss 5.5579 | val exact-match   0.0%
  eval @   600 (epoch 10): val loss 0.9669 | val exact-match  26.7%
  eval @  1200 (epoch 20): val loss 0.5848 | val exact-match  45.3%
  eval @  1800 (epoch 30): val loss 0.5130 | val exact-match  52.7%
  eval @  2160 (epoch 36): val loss 0.5029 | val exact-match  59.3%
best checkpoint: step 2160 (val loss 0.5029)
exact match  ->  val  53.5%  |  holdout  53.8%
```

Running it again gives the same lines; `evaluate.py` on the checkpoint prints
`val 53.5% holdout 53.8%`.

## Practising again

```bash
git checkout -- sft/v2_torch_packing_ckpt
```
