"""All knobs in one place (dataclasses -> CLI flags)."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_FILES = (DATA_DIR / "raw" / "sft_export_2024.tsv", DATA_DIR / "raw" / "sft_export_legacy.tsv")
HOLDOUT_FILE = DATA_DIR / "holdout.jsonl"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"


@dataclass
class ModelConfig:
    vocab_size: int = 0           # set from the tokenizer at runtime
    block_size: int = 64          # max tokens (bytes) per example; longer ones are truncated
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    bias: bool = True

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head:
            raise ValueError(f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})")

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head


@dataclass
class TrainConfig:
    max_iters: int = 3000
    batch_size: int = 32
    learning_rate: float = 3e-3
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_iters: int = 100
    min_lr_ratio: float = 0.1
    val_frac: float = 0.1
    eval_interval: int = 500
    eval_batches: int = 10        # batches for the periodic val-loss estimate
    eval_examples: int = 150      # examples for the periodic exact-match estimate
    final_eval_examples: int = 400
    max_new_tokens: int = 48
    log_interval: int = 50
    seed: int = 1337
    device: str = "auto"
    out_dir: str = str(CHECKPOINT_DIR)


def _add_args(parser: argparse.ArgumentParser, cls) -> None:
    group = parser.add_argument_group(cls.__name__)
    for f in fields(cls):
        default = getattr(cls(), f.name)
        if isinstance(default, bool):
            group.add_argument(f"--{f.name}", type=lambda v: v.lower() in {"1", "true", "yes"}, default=default)
        else:
            group.add_argument(f"--{f.name}", type=type(default), default=default)


def parse_args(argv: list[str] | None = None) -> tuple[ModelConfig, TrainConfig]:
    parser = argparse.ArgumentParser(description="Supervised fine-tuning of a tiny GPT.")
    _add_args(parser, ModelConfig)
    _add_args(parser, TrainConfig)
    ns = parser.parse_args(argv)
    m = ModelConfig(**{f.name: getattr(ns, f.name) for f in fields(ModelConfig)})
    t = TrainConfig(**{f.name: getattr(ns, f.name) for f in fields(TrainConfig)})
    return m, t


def describe(cfg) -> str:
    return "\n".join(f"  {k:>20}: {v}" for k, v in asdict(cfg).items())
