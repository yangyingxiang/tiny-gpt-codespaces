"""Variant 3: SFT with the Hugging Face `Trainer`, `datasets` and `peft` LoRA.

    python sft/v3_hf_trainer_lora/main.py            # "Run Python File" in VS Code works too
    python -m sft.v3_hf_trainer_lora.main --quick    # ~30 s plumbing check

Two phases, because LoRA on a random-init model would have nothing to adapt:

    phase 1  full fine-tune of a tiny random-init GPT-2 on the BASE tasks
             (Reverse / Uppercase / Repeat)                      -> checkpoints/.../base
    phase 2  freeze it, attach LoRA adapters, train them on the ADAPTER tasks
             (Sort letters / Length / Count vowels)              -> checkpoints/.../adapter
    final    merge the adapters, exact match on val + holdout, per task

Mechanisms: `Trainer` owns the loop (schedule, clipping, eval, logging); gradient checkpointing
via `TrainingArguments(gradient_checkpointing=True)`; attention via `attn_implementation="sdpa"`
(PyTorch's fused kernel; on CUDA it dispatches to flash/mem-efficient kernels, on CPU to math).
"""

from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    __package__ = "sft.v3_hf_trainer_lora"

import json
import math
import time
from pathlib import Path

from transformers import (AutoModelForCausalLM, AutoTokenizer, GPT2Config, Trainer,
                          TrainerCallback, TrainingArguments, set_seed)

from .config import ADAPTER_TASKS, BASE_TASKS, HOLDOUT_FILE, TRAIN_FILE, Config, describe, parse_args
from .data import SFTCollator, VocabMap, filter_tasks, load_jsonl, split_examples, to_dataset
from .evaluate import exact_match, format_per_task


class PrintLog(TrainerCallback):
    """One compact line per log event instead of Trainer's dict dumps."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        if "eval_loss" in logs:
            print(f"    eval @ {state.global_step:4d}: loss {logs['eval_loss']:.4f}")
        elif "loss" in logs:
            print(f"  step {state.global_step:4d} | loss {logs['loss']:.4f} | "
                  f"lr {logs.get('learning_rate', 0):.2e} | grad_norm {logs.get('grad_norm', 0):.2f}")


def build_tokenizer(cfg: Config):
    tok = AutoTokenizer.from_pretrained(cfg.tokenizer)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token      # GPT-2 ships without a pad token
    return tok


def build_model(cfg: Config, vmap: VocabMap):
    gcfg = GPT2Config(vocab_size=len(vmap), n_positions=cfg.n_positions, n_embd=cfg.n_embd,
                      n_layer=cfg.n_layer, n_head=cfg.n_head,
                      resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
                      bos_token_id=vmap.eos_id, eos_token_id=vmap.eos_id, pad_token_id=vmap.pad_id)
    model = AutoModelForCausalLM.from_config(gcfg, attn_implementation="sdpa")
    model.config.use_cache = False         # dead weight while training with checkpointing
    return model


def training_args(cfg: Config, out_dir: Path, max_steps: int, lr: float) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(out_dir),
        max_steps=max_steps,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        prediction_loss_only=True,         # never gather (B, T, V) logits across the eval set
        save_strategy="no",
        logging_steps=cfg.logging_steps,
        report_to="none",
        disable_tqdm=True,
        gradient_checkpointing=cfg.grad_checkpoint,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_pin_memory=False,
        seed=cfg.seed,
    )


def run_trainer(model, tok, vmap, args, train_ds, eval_ds) -> Trainer:
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
                      data_collator=SFTCollator(vmap.pad_id), processing_class=tok,
                      callbacks=[PrintLog()])
    # BREAKPOINT: next(iter(trainer.get_train_dataloader())) is exactly what the model sees.
    step0 = trainer.evaluate()["eval_loss"]
    print(f"    eval @    0: loss {step0:.4f}   (uniform over {len(vmap)} tokens = "
          f"{math.log(len(vmap)):.4f})")
    trainer.train()
    return trainer


def main(cfg: Config) -> dict:
    set_seed(cfg.seed)
    t0 = time.time()
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("config:\n" + describe(cfg))

    tok = build_tokenizer(cfg)
    train_ex, val_ex = split_examples(load_jsonl(TRAIN_FILE), cfg.val_frac, cfg.seed)
    holdout_ex = load_jsonl(HOLDOUT_FILE)
    vmap = VocabMap(tok, train_ex + val_ex + holdout_ex)
    print(f"examples: train {len(train_ex)} | val {len(val_ex)} | holdout {len(holdout_ex)} | "
          f"vocab {len(vmap)} of {len(tok)} gpt2 ids")

    base_train = to_dataset(tok, vmap, filter_tasks(train_ex, BASE_TASKS), cfg.max_length)
    base_val_ex = filter_tasks(val_ex, BASE_TASKS)[: cfg.eval_examples]
    adapter_train = to_dataset(tok, vmap, filter_tasks(train_ex, ADAPTER_TASKS), cfg.max_length)
    adapter_val_ex = filter_tasks(val_ex, ADAPTER_TASKS)[: cfg.eval_examples]
    adapter_hold_ex = filter_tasks(holdout_ex, ADAPTER_TASKS)[: cfg.eval_examples]
    print(f"phase 1 trains on {len(base_train)} base-task rows, "
          f"phase 2 on {len(adapter_train)} adapter-task rows")

    # ---- phase 1: full fine-tune on the base tasks ----------------------------------------
    base_dir = out_dir / "base"
    if cfg.reuse_base and (base_dir / "config.json").exists():
        print(f"\n== phase 1: reusing {base_dir}")
        model = AutoModelForCausalLM.from_pretrained(base_dir, attn_implementation="sdpa")
    else:
        print(f"\n== phase 1: full fine-tune on {BASE_TASKS} for {cfg.base_steps} steps")
        model = build_model(cfg, vmap)
        print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
        trainer = run_trainer(model, tok, vmap,
                              training_args(cfg, out_dir / "phase1", cfg.base_steps, cfg.base_lr),
                              base_train, to_dataset(tok, vmap, base_val_ex, cfg.max_length))
        trainer.save_model(str(base_dir))
    base_em, per_task, _ = exact_match(model, tok, vmap, base_val_ex, cfg.max_new_tokens)
    print(f"phase 1 done: base-task val exact match {base_em:6.1%}   {format_per_task(per_task)}")

    # ---- phase 2: LoRA on the adapter tasks -----------------------------------------------
    from peft import LoraConfig, get_peft_model

    print(f"\n== phase 2: LoRA (r={cfg.lora_r}, alpha={cfg.lora_alpha}) on {ADAPTER_TASKS} "
          f"for {cfg.lora_steps} steps")
    lora_cfg = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    # Gradient checkpointing with frozen embeddings: the checkpointed block's input does not
    # require grad. With use_reentrant=False (set in training_args) that is fine; with the
    # reentrant implementation you would also need model.enable_input_require_grads().
    trainer = run_trainer(model, tok, vmap,
                          training_args(cfg, out_dir / "phase2", cfg.lora_steps, cfg.lora_lr),
                          adapter_train, to_dataset(tok, vmap, adapter_val_ex, cfg.max_length))
    model.save_pretrained(str(out_dir / "adapter"))

    # ---- final: merge and score --------------------------------------------------------------
    print("\n== final: merged model")
    merged = trainer.model.merge_and_unload()
    val_em, val_tasks, samples = exact_match(merged, tok, vmap, adapter_val_ex, cfg.max_new_tokens)
    hold_em, hold_tasks, _ = exact_match(merged, tok, vmap, adapter_hold_ex, cfg.max_new_tokens)
    base_after, base_tasks, _ = exact_match(merged, tok, vmap, base_val_ex, cfg.max_new_tokens)
    print(f"adapter tasks  ->  val {val_em:6.1%}  |  holdout {hold_em:6.1%}")
    print(f"  val     {format_per_task(val_tasks)}")
    print(f"  holdout {format_per_task(hold_tasks)}")
    print(f"base tasks after LoRA: val {base_after:6.1%} (was {base_em:6.1%}; the adapter was "
          f"trained on the new tasks only, so forgetting is expected)   {format_per_task(base_tasks)}")
    for ex, pred in samples[:5]:
        print(f"  {ex.instruction} {ex.input!r} -> {pred!r}  (want {ex.output!r})")

    metrics = {"base_val_em": base_em, "base_val_em_after_lora": base_after,
               "adapter_val_em": val_em, "adapter_holdout_em": hold_em,
               "adapter_val_per_task": val_tasks, "adapter_holdout_per_task": hold_tasks,
               "seconds": time.time() - t0, "log_history": trainer.state.log_history}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"done in {time.time() - t0:.1f}s  (metrics in {out_dir / 'metrics.json'})")
    return metrics


if __name__ == "__main__":
    main(parse_args())
