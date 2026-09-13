"""The training loop. Accelerate owns device placement, gradient accumulation bookkeeping
and (on a multi-GPU box) the distributed wrapping; the loss, the eval and the checkpoint
policy stay explicit.

    python -m sft.v4_hf_accelerate.main
    accelerate launch sft/v4_hf_accelerate/main.py
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config, get_scheduler

from .config import HOLDOUT_FILE, TRAIN_FILE, Config, describe
from .data import Collator, SFTDataset, VocabSlice, load_jsonl, split_examples
from .evaluate import eval_loss, exact_match, format_per_task, perplexity


def build_model(cfg: Config, vocab: VocabSlice):
    gcfg = GPT2Config(
        vocab_size=len(vocab), n_positions=cfg.n_positions, n_embd=cfg.n_embd,
        n_layer=cfg.n_layer, n_head=cfg.n_head,
        resid_pdrop=cfg.dropout, embd_pdrop=cfg.dropout, attn_pdrop=cfg.dropout,
        bos_token_id=vocab.eos_id, eos_token_id=vocab.eos_id, pad_token_id=vocab.pad_id,
    )
    # "sdpa" = torch.nn.functional.scaled_dot_product_attention. On CUDA it dispatches to the
    # flash / memory-efficient kernels; on CPU to the math kernel. Same numbers, different speed.
    model = AutoModelForCausalLM.from_config(gcfg, attn_implementation="sdpa")
    if cfg.grad_checkpoint:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False       # incompatible with checkpointing; generate() re-enables it
    return model


def make_optimizer(model, lr: float, weight_decay: float):
    """Decay matrices, do not decay biases and LayerNorm gains."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (no_decay if p.ndim < 2 or name.endswith(".bias") else decay).append(p)
    return torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}], lr=lr, betas=(0.9, 0.95))


def train(cfg: Config) -> dict:
    accelerator = Accelerator(gradient_accumulation_steps=cfg.accum_steps, mixed_precision="no")
    set_seed(cfg.seed)
    log = accelerator.print
    log("config:\n" + describe(cfg))
    log(f"device: {accelerator.device} | processes: {accelerator.num_processes}")

    # ---- data -------------------------------------------------------------
    tok = AutoTokenizer.from_pretrained(cfg.tokenizer)
    tok.pad_token = tok.eos_token
    train_ex, val_ex = split_examples(load_jsonl(TRAIN_FILE), cfg.val_frac, cfg.seed)
    holdout_ex = load_jsonl(HOLDOUT_FILE)
    vocab = VocabSlice.from_examples(tok, train_ex + val_ex + holdout_ex)
    train_ds = SFTDataset(train_ex, tok, vocab, cfg.max_length)
    val_ds = SFTDataset(val_ex, tok, vocab, cfg.max_length)
    log(f"examples: train {len(train_ds)} | val {len(val_ds)} | holdout {len(holdout_ex)}"
        + (f" (dropped {train_ds.dropped + val_ds.dropped} unsupervisable)" if train_ds.dropped + val_ds.dropped else ""))
    log(f"vocab: {len(vocab)} of {len(tok)} gpt2 tokens are used by this corpus")
    collate = Collator(vocab.pad_id)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate)

    # ---- model / optimizer / schedule --------------------------------------
    model = build_model(cfg, vocab)
    log(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    optimizer = make_optimizer(model, cfg.learning_rate, cfg.weight_decay)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * cfg.epochs
    if cfg.max_steps:
        total_steps = min(total_steps, cfg.max_steps)
    scheduler = get_scheduler("cosine", optimizer, num_warmup_steps=int(cfg.warmup_ratio * total_steps),
                              num_training_steps=total_steps)
    model, optimizer, scheduler, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler)
    log(f"optimizer steps: {total_steps} ({steps_per_epoch}/epoch, effective batch "
        f"{cfg.batch_size * cfg.accum_steps * accelerator.num_processes})")

    out_dir = Path(cfg.out_dir)
    best_dir = out_dir / "best"
    if accelerator.is_main_process:
        out_dir.mkdir(parents=True, exist_ok=True)
    best_val, history, t0 = float("inf"), [], time.time()

    def evaluate(step: int) -> None:
        nonlocal best_val
        # BREAKPOINT: val loss and exact match should move together.
        unwrapped = accelerator.unwrap_model(model)
        vl = eval_loss(unwrapped, val_loader)
        em, _ = exact_match(unwrapped, tok, vocab, val_ds.examples, cfg.max_new_tokens, cfg.eval_examples)
        history.append({"step": step, "val_loss": vl, "val_em": em})
        log(f"  eval @ {step:4d}: val loss {vl:.4f} (ppl {perplexity(vl):.2f}) | val exact-match {em:6.1%}")
        if vl < best_val:
            best_val = vl
            accelerator.wait_for_everyone()
            unwrapped.save_pretrained(best_dir, is_main_process=accelerator.is_main_process,
                                      save_function=accelerator.save)
            if accelerator.is_main_process:
                tok.save_pretrained(best_dir)
                (best_dir / "vocab_slice.json").write_text(json.dumps(vocab.full_ids))

    # ---- loop -------------------------------------------------------------
    step = 0
    evaluate(step)                          # untrained baseline: should be ~ln(vocab)
    model.train()
    done = False
    for epoch in range(cfg.epochs):
        for batch in train_loader:
            with accelerator.accumulate(model):
                # BREAKPOINT: tok.decode(batch["input_ids"][0]) next to batch["labels"][0].
                out = model(**batch)
                accelerator.backward(out.loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            if not accelerator.sync_gradients:
                continue                    # still accumulating this optimizer step
            step += 1
            if step % cfg.log_interval == 0:
                log(f"iter {step:4d} | loss {out.loss.item():.4f} | lr {scheduler.get_last_lr()[0]:.2e} | "
                    f"grad_norm {float(grad_norm):.2f} | {time.time() - t0:.1f}s")
            if step % cfg.eval_interval == 0:
                evaluate(step)
                model.train()
            if step >= total_steps:
                done = True
                break
        log(f"epoch {epoch + 1}/{cfg.epochs} done ({step} optimizer steps)")
        if done:
            break
    if not history or history[-1]["step"] != step:
        evaluate(step)

    # ---- final evaluation on the best checkpoint --------------------------
    accelerator.wait_for_everyone()
    best = AutoModelForCausalLM.from_pretrained(best_dir, attn_implementation="sdpa").to(accelerator.device)
    val_em, val_pairs = exact_match(best, tok, vocab, val_ds.examples, cfg.max_new_tokens, cfg.final_eval_examples)
    hold_em, hold_pairs = exact_match(best, tok, vocab, holdout_ex, cfg.max_new_tokens, cfg.final_eval_examples)
    metrics = {"best_val_loss": best_val, "val_exact_match": val_em, "holdout_exact_match": hold_em,
               "optimizer_steps": step, "train_examples": len(train_ds), "val_examples": len(val_ds),
               "seconds": time.time() - t0, "history": history, "config": asdict(cfg)}
    if accelerator.is_main_process:
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    log(f"\nbest val loss {best_val:.4f} (ppl {perplexity(best_val):.2f})")
    log(f"exact match  ->  val {val_em:6.1%}  |  holdout {hold_em:6.1%}")
    log("per task (holdout):\n" + format_per_task(hold_pairs))
    for ex, pred, _ in hold_pairs[:5]:
        log(f"  {ex.instruction} {ex.input!r} -> {pred!r}  (want {ex.output!r})")
    log(f"done in {time.time() - t0:.1f}s")
    return metrics
