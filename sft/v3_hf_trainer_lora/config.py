"""All knobs in one place (dataclass -> CLI flags)."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
TRAIN_FILE = DATA_DIR / "sft_train.jsonl"
HOLDOUT_FILE = DATA_DIR / "holdout.jsonl"
OUT_DIR = REPO_ROOT / "checkpoints" / "v3_hf_trainer_lora"

BASE_TASKS = ("Reverse:", "Uppercase:", "Repeat:")
ADAPTER_TASKS = ("Sort letters:", "Length:", "Count vowels:")


@dataclass
class Config:
    # tokenizer + tiny random-init GPT-2
    tokenizer: str = "gpt2"
    n_positions: int = 128
    n_embd: int = 128
    n_layer: int = 2
    n_head: int = 4
    max_length: int = 96          # prompt + answer + eos, in tokens

    # phase 1: full fine-tune on BASE_TASKS
    base_steps: int = 1200
    base_lr: float = 3e-3

    # phase 2: LoRA on ADAPTER_TASKS
    lora_steps: int = 1500
    lora_lr: float = 5e-3
    lora_r: int = 16
    lora_alpha: int = 32

    # shared training knobs (both phases go through Trainer)
    batch_size: int = 32
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    eval_steps: int = 250
    logging_steps: int = 50
    grad_checkpoint: bool = True

    # data + eval
    val_frac: float = 0.1
    eval_examples: int = 200      # examples per exact-match / eval-loss pass
    max_new_tokens: int = 40
    seed: int = 1337

    # workflow
    quick: bool = False           # tiny run to check the plumbing (~1 min)
    reuse_base: bool = False      # skip phase 1 if checkpoints/.../base exists
    out_dir: str = str(OUT_DIR)

    def apply_quick(self) -> "Config":
        if self.quick:
            self.base_steps, self.lora_steps = 150, 150
            self.eval_steps, self.logging_steps = 50, 25
            self.eval_examples = 40
        return self


def parse_args(argv: list[str] | None = None) -> Config:
    parser = argparse.ArgumentParser(description="SFT with HF Trainer + peft LoRA on a tiny GPT-2.")
    for f in fields(Config):
        default = getattr(Config(), f.name)
        if isinstance(default, bool):      # accepts `--quick`, `--quick true`, `--quick 0`
            parser.add_argument(f"--{f.name}", type=lambda v: v.lower() in {"1", "true", "yes"},
                                nargs="?", const=True, default=default)
        else:
            parser.add_argument(f"--{f.name}", type=type(default), default=default)
    ns = parser.parse_args(argv)
    return Config(**{f.name: getattr(ns, f.name) for f in fields(Config)}).apply_quick()


def describe(cfg: Config) -> str:
    return "\n".join(f"  {k:>16}: {v}" for k, v in asdict(cfg).items())
