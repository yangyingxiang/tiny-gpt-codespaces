# Solutions — variant 1 (PyTorch + LoRA + SDPA + accumulation)

Spoilers. Five planted defects. To put them back after fixing: `git checkout -- sft/v1_torch_lora`.

| # | Where | Class | Loud or silent | Signature |
|---|---|---|---|---|
| 1 | `model.py` `CausalSelfAttention.__init__` | attention mask dtype (SDPA semantics) | silent | phase-1 val loss far *below* healthy (0.19 vs 0.46 at 1500) while exact match stays ≈ 0% |
| 2 | `main.py` phase 2, freeze/attach order | parameter freezing | **crash** | `ValueError: optimizer got an empty parameter list` |
| 3 | `lora.py` `LoRALinear.__init__` | dead initialisation | silent | phase-2 loss never moves; `grad_norm 0.00` every step |
| 4 | `main.py` `run_phase`, micro-batch loop | gradient accumulation | silent | slower learning: val loss 0.91 vs 0.68, exact match 25% vs 50% at step 1500 |
| 5 | `lora.py` `LoRALinear.merged_linear` | merge without the LoRA scaling | silent | merged exact match ≈ 1% while unmerged ≈ 48% |

Numbers are from CPU runs with `OMP_NUM_THREADS=2`, seed 1337. The order below is the order you
meet them when you run the thing.

---

## 1. SDPA gets a float 0/1 mask, which it treats as an additive bias (silent)

```python
# bug
mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
# fix
mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
# or drop the buffer and call F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

`F.scaled_dot_product_attention` interprets `attn_mask` by dtype: a **bool** mask means
"True = may attend", a **float** mask is *added to the scores*. `tril(ones)` as float adds
1.0 to the allowed positions and 0.0 to the forbidden ones. Nothing is masked; every position
sees the whole sequence, including the token it is supposed to predict. The hand-written
attention in `sft/model.py` used `masked_fill(mask == 0, -inf)`, so the same tensor was fine
there. Swapping in SDPA changed the contract.

**Symptom.** Phase 1 val loss keeps falling well past what the model can honestly achieve
(all five bugs in: 0.19 at step 1500 vs 0.46 healthy; with only this bug it reaches 0.38 by
step 300 of a `--quick` run, where healthy is 1.25), yet exact match is 0.0–0.7% and
holdout 0.4%. At inference the model has no future token to peek at, so it is useless.

**How you find it.** "Loss too good, generation useless" in a causal LM means the target is
visible in the input. The label shift in `encode_example` is correct, so look at the mask.
Prove it in five lines: perturb a *future* position and check whether an earlier output moves.

```python
attn = model.blocks[0].attn.eval()
h = torch.randn(1, 8, 128); h2 = h.clone(); h2[0, 7] += 3.0
torch.allclose(attn(h)[0, :7], attn(h2)[0, :7])       # False on the bug, True when fixed
```

**Guard.** That check as a test, plus `torch.allclose(sdpa(bool mask), sdpa(is_causal=True))`.

## 2. LoRA is attached before the base is frozen (crash)

```python
# bug
wrapped = attach_lora(model, tcfg.lora_r, tcfg.lora_alpha)
for p in model.parameters():
    p.requires_grad_(False)
# fix
for p in model.parameters():
    p.requires_grad_(False)
wrapped = attach_lora(model, tcfg.lora_r, tcfg.lora_alpha)
```

`attach_lora` registers `lora_A` / `lora_B` as parameters of the model, so the freeze loop that
runs afterwards freezes them too. `make_optimizer` filters on `requires_grad`, finds nothing,
and AdamW refuses an empty list:

```
File ".../sft/v1_torch_lora/model.py", line 147, in make_optimizer
    return torch.optim.AdamW([g for g in groups if g["params"]], ...)
ValueError: optimizer got an empty parameter list
```

**How you find it.** The trace ends in the optimizer constructor; the last frame in this
folder is `make_optimizer`. `[n for n, p in model.named_parameters() if p.requires_grad]` in
the debugger is `[]`. Then ask *when* the LoRA parameters lost their grad flag.

**Guard.** After phase-2 setup: `assert all(p.requires_grad for p in lora_parameters(model))`
and `assert model.num_parameters(trainable_only=True) == 65_536` (8 layers × r × (in + out)).

## 3. Both LoRA matrices start at zero (silent)

```python
# bug
self.lora_A = nn.Parameter(torch.zeros(r, base.in_features))
self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
# fix
self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
```

The adapter output is `(x Aᵀ) Bᵀ · s`. Its gradient w.r.t. `B` is proportional to `x Aᵀ`,
and w.r.t. `A` is proportional to `B`. With both at zero, both gradients are exactly zero, and
they stay zero forever: a fixed point. Only one of the two may be zero at init (LoRA zeroes
`B` so the adapter starts as a no-op).

**Symptom.** Phase 2 prints `grad_norm 0.00` on every step; val loss is 11.3025 at every eval
to four decimals; train loss just wobbles with the batch; exact match 0%. Base tasks after the
"merge" are unchanged at 53.9% because nothing was merged.

**How you find it.** A loss that does not change *at all* is not a learning-rate problem.
`grad_norm 0.00` is the giveaway. Break at step 1 and look at `lora_A.grad.abs().max()` and
`lora_B.grad.abs().max()`: both 0. Then print the parameters themselves.

**Guard.** `assert LoRALinear(nn.Linear(8, 4), 2, 4).lora_A.abs().sum() > 0`, and a one-step
test that the LoRA gradient norm is non-zero.

## 4. `zero_grad` inside the micro-batch loop (silent)

```python
# bug
for _ in range(tcfg.accum_steps):
    optimizer.zero_grad(set_to_none=True)
    x, y = next(batches)
    _, loss = model(x, y)
    (loss / tcfg.accum_steps).backward()
grad_norm = clip_grad_norm_(...); optimizer.step()
# fix
optimizer.zero_grad(set_to_none=True)
for _ in range(tcfg.accum_steps):
    ...
```

Each micro-step wipes the gradient the previous one accumulated, so `optimizer.step()` only
ever sees the *last* micro-batch, still scaled by `1 / accum_steps`. Effective batch = 8
instead of 32, effective learning rate ÷ 4, and gradient noise ×2. Nothing crashes, nothing
is NaN.

**Symptom.** With bugs 1–3 fixed, phase 2 at step 1500 reaches val loss 0.91 and exact match
24.7% where the healthy run reaches 0.68 and 50%. The printed `grad_norm` sits around 1.4
instead of 3–4: it is the norm of a quarter of one micro-batch's gradient.

**How you find it.** Learning is slower than the reference table, with the same LR schedule.
Check what the optimizer actually receives: after the micro loop, `p.grad` should equal the
sum of the four micro-gradients. Or count calls:

```python
calls = []
orig = torch.optim.Optimizer.zero_grad
torch.optim.Optimizer.zero_grad = lambda self, set_to_none=True: (calls.append(1), orig(self, set_to_none))[1]
run_phase("t", GPT(mcfg), ex, ex[:2], 3, 1e-3, tok, mcfg, tcfg, "cpu")
assert len(calls) == 3          # 12 on the bug (3 steps × accum_steps 4)
```

**Guard.** The call-count test above, or a two-micro-batch test asserting
`p.grad == g1 + g2` through the training step.

## 5. Merge drops the `alpha / r` scaling (silent)

```python
# bug
self.base.weight += self.lora_B @ self.lora_A
# fix
self.base.weight += (self.lora_B @ self.lora_A) * self.scaling
```

The forward pass computes `W x + s · B A x` with `s = alpha / r = 2`. Merging must add
`s · B A` to `W`. Without `s`, the merged weight carries only half the adapter update, which is
neither the base model nor the trained one.

**Symptom.** The last periodic (unmerged) exact match is 50%, `unmerged 47.9%` on the final
400 examples, and the merged number right next to it is **1.2%**; holdout 1.2%. Base tasks land
at 41.8% instead of the 24.1% the trained adapter gives them: half-way between base and
adapted.

**How you find it.** Two numbers that must be identical differ. `LoRALinear.forward` and
`merged_linear()` claim to be the same function, so test exactly that:

```python
lin = LoRALinear(nn.Linear(8, 4), r=2, alpha=4)
with torch.no_grad():
    lin.lora_B.normal_()
    x = torch.randn(3, 8); before = lin(x); after = lin.merged_linear()(x)
torch.allclose(before, after, atol=1e-5)          # False on the bug
```

**Guard.** That test. In the pipeline: `assert abs(val_em - unmerged_em) < 0.02`.

---

## Healthy numbers after all five fixes

| | phase 1 (base tasks) | phase 2 (adapter tasks) |
|---|---|---|
| val loss at 1500 | 0.46 | 0.64 (0.68 with `--reuse_base`) |
| val exact match at 1500 (150 ex.) | 52% | 46% (50% with `--reuse_base`) |
| final val exact match, merged = unmerged | — | 43.7% = 43.7% (47.9% = 47.9% with `--reuse_base`) |
| holdout exact match | 53.9% | 45.2% (40.5% with `--reuse_base`) |
| base tasks on holdout after merge | — | 24.1% (down from 53.9%) |
| wall clock, 2 threads | 170 s total | |

`--reuse_base` skips phase 1, so the RNG stream at LoRA init differs and the phase-2 numbers
shift by a few points. Merged and unmerged must still agree exactly.

The drop on the base tasks after phase 2 (54% → 24%) is real forgetting, not a planted bug:
the adapters sit on every layer and the replay slice is only 30% of the adapter data. Raising
`--replay_frac 0.6` gives 33% on the base tasks at the cost of 38% instead of 44% on the
adapter tasks.

## Order of discovery

1. Bug 1 is visible in phase 1 (val loss too good, exact match ≈ 0), before anything crashes.
2. Bug 2 crashes the start of phase 2.
3. Bug 3 is obvious as soon as phase 2 runs (`grad_norm 0.00`, constant val loss).
4. Bug 4 only shows against the reference numbers, or by inspecting what `optimizer.step()`
   receives.
5. Bug 5 shows on the final line: merged ≠ unmerged.

## Validation under the clock

```bash
python sft/v1_torch_lora/main.py --quick                 # both phases run, merged == unmerged
python sft/v1_torch_lora/main.py                         # compare with the table above
python sft/v1_torch_lora/main.py --reuse_base            # phase 2 alone, ~60 s, for iterating on 3-5
```

Plus the four small tests above (future-token perturbation, LoRA grad non-zero,
`zero_grad` call count, merged == unmerged), each of which fails on the planted code.
