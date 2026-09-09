# tiny-gpt-codespaces

A small, readable **transformer training pipeline** (character-level GPT, pure PyTorch)
that is set up to be **run and debugged entirely in the browser** via GitHub Codespaces,
and **shared live** with someone else via VS Code Live Share.

No GPU required. A short training run finishes on a free 2-core Codespace in a couple of minutes.

---

## 1. Run it in VS Code in the browser (Codespaces)

1. Click **Code ▾ → Codespaces → Create codespace on main**.
2. Wait for the container to build. `.devcontainer/post-create.sh` installs CPU-only PyTorch
   and pre-downloads the TinyShakespeare corpus.
3. Press <kbd>F5</kbd> and pick **"Train: quick debug run (50 iters, tiny model)"**.

That's it — you are training and stepping through a transformer in a browser tab.

> Prefer a terminal? `python -m src.train --max_iters 500`

### Why Codespaces and not the `.` editor
Pressing <kbd>.</kbd> on a GitHub repo opens **github.dev**, which is VS Code in the browser but
with *no compute*: no terminal, no Python, no debugger. Codespaces runs a real container, which is
what makes `F5` and breakpoints work.

---

## 2. Debugging

`.vscode/launch.json` ships seven ready-made configurations:

| Configuration | What it does |
|---|---|
| **Train: quick debug run** | 50 iterations of a 2-layer model — fast enough to step through |
| **Train: full run (500 iters)** | The default training run |
| **Train: stop on first exception** | `stopOnEntry`, verbose logging, for post-mortem poking |
| **Sample from checkpoint** | Generates text from `checkpoints/model.pt` |
| **Pytest: all tests** | Runs the suite under the debugger |
| **Python: current file** | Debug whatever file is open |
| **Attach to running process** | Attaches to `debugpy` on port 5678 |

All of them set `"justMyCode": false`, so you can step *into* PyTorch itself.

**Good places to put a breakpoint** (marked with `# BREAKPOINT:` in the source):

- `src/train.py` → right after `dataset.get_batch(...)` — inspect a `(B, T)` batch of token ids.
- `src/model.py` → `CausalSelfAttention.forward` — watch `q`, `k`, `v` reshape from
  `(B, T, C)` to `(B, nh, T, hd)`, and watch the causal mask turn the upper triangle into `-inf`.
- `src/train.py` → the eval block — watch train and val loss separate.
- `src/sample.py` → `model.generate` — watch tokens being sampled one at a time.

`debug.inlineValues` is on, so tensor shapes appear inline next to the code as you step.

### Attach-mode debugging
```bash
python -m debugpy --listen 5678 --wait-for-client -m src.train --max_iters 50
```
Then run **"Attach to running process (debugpy :5678)"**. Port 5678 is already forwarded.

---

## 3. Live Share (collaborative debugging)

The **`ms-vsliveshare.vsliveshare`** extension is preinstalled by the devcontainer, and
`.vscode/settings.json` pre-approves guest debug/task control.

1. In the codespace, click **Live Share** in the status bar (or <kbd>Ctrl/Cmd</kbd>+<kbd>Shift</kbd>+<kbd>P</kbd> → *Live Share: Start Collaboration Session*).
2. Sign in with GitHub when prompted; the invite link is copied to your clipboard.
3. Send the link. Your guest joins in their browser — no clone, no install, no PyTorch download.
4. Start a debug session. Breakpoints, the call stack, watches and the debug console are
   **shared** — either of you can step, and both see the same variables.

Relevant settings you may want to change in `.vscode/settings.json`:

| Setting | Default here | Meaning |
|---|---|---|
| `liveshare.guestApprovalRequired` | `true` | you approve each guest before they join |
| `liveshare.allowGuestDebugControl` | `true` | guests can step/continue the debugger |
| `liveshare.allowGuestTaskControl` | `true` | guests can run the tasks in `tasks.json` |
| `liveshare.shareExternalFiles` | `false` | guests only see files inside this repo |

> Live Share is a *session*, not a repo setting — nothing is shared until you start one.

---

## 4. The model

A decoder-only transformer written out longhand (no `nn.MultiheadAttention`), so every
tensor operation is visible in the debugger.

```
tokens ─► token embedding + positional embedding ─► dropout
       ─► N × [ x + attn(LayerNorm(x)) ; x + mlp(LayerNorm(x)) ]     (pre-norm blocks)
       ─► final LayerNorm ─► linear head (weights tied to the embedding) ─► logits
```

Defaults: 4 layers, 4 heads, `n_embd=128`, `block_size=128` → **~0.8M parameters**.
Trained on TinyShakespeare (~1.1M characters, 65-symbol vocabulary) to predict the next character.

Includes: causal masking, weight tying, GPT-2 scaled residual init, AdamW with decay applied
only to matrices, linear warmup + cosine LR decay, gradient clipping, periodic eval and
best-checkpoint saving, and top-k/temperature sampling.

---

## 5. Project layout

```
.devcontainer/
  devcontainer.json     Codespaces image, extensions (incl. Live Share), ports
  post-create.sh        installs CPU-only torch, pre-downloads the corpus
.vscode/
  launch.json           7 debug configurations
  settings.json         pytest + Live Share + inline debug values
  tasks.json            install / prepare / train / test / debug-server
  extensions.json       recommended extensions
src/
  config.py             GPTConfig + TrainConfig dataclasses, CLI parsing
  data.py               corpus download, char tokenizer, batching
  model.py              CausalSelfAttention, MLP, Block, GPT
  train.py              training loop, LR schedule, eval, checkpointing
  sample.py             text generation from a checkpoint
tests/
  test_pipeline.py      9 fast tests (causality, overfit-one-batch, LR schedule, ...)
scripts/
  prepare_data.py       pre-download hook
```

---

## 6. Command line

```bash
python -m src.train --help          # every dataclass field is a flag

python -m src.train --max_iters 500                    # default run (~3 min on 2 cores)
python -m src.train --max_iters 3000 --n_layer 6       # readable Shakespeare-ish output
python -m src.train --device cpu --batch_size 8        # gentler on a small codespace

python -m src.sample --prompt "ROMEO:" --max_new_tokens 400 --temperature 0.8 --top_k 40

python -m pytest -q tests
```

Key flags: `--n_layer --n_head --n_embd --block_size --dropout --bias`
and `--max_iters --batch_size --learning_rate --weight_decay --grad_clip
--warmup_iters --eval_interval --eval_iters --log_interval --seed --device --out_dir`.

`--compile_model true` enables `torch.compile` — faster, but it hides Python frames from the
debugger, so leave it off while stepping.

---

## 7. What a run looks like

```
CharDataset(vocab_size=65, train_tokens=1003854, val_tokens=111540, block_size=128)
parameters: 0.81M
iter     0 | loss 4.1833 | lr 6.00e-06 | grad_norm 2.58 | 0.0s
iter    50 | loss 3.6393 | lr 3.00e-04 | grad_norm 1.05 | 0.7s
  eval @ 50: train 3.5909 | val 3.6068
  saved checkpoint -> checkpoints/model.pt (val 3.6068)
...
```

Loss starts near `ln(65) ≈ 4.17` (a uniform guess over the vocabulary) and falls from there.
After 500 iterations you get word-shaped nonsense with correct play formatting; a few thousand
iterations gets you recognisable dialogue.

---

## 8. Running locally instead

```bash
git clone https://github.com/yangyingxiang/tiny-gpt-codespaces.git
cd tiny-gpt-codespaces
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
python -m pip install numpy pytest
python -m src.train --max_iters 500
```

Or open the folder in VS Code with the **Dev Containers** extension and choose
*Reopen in Container* — same environment as the codespace.

---

MIT licensed. Built as a teaching/demo pipeline; the model architecture follows the standard
GPT-2 recipe at a scale that fits in a debugger.
