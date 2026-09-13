"""All knobs in one place (dataclasses -> CLI flags)."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_FILE = DATA_DIR / "sft_train.jsonl"
HOLDOUT_FILE = DATA_DIR / "holdout.jsonl"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "v2_torch_packing_ckpt"


@dataclass
class ModelConfig:
    vocab_size: int = 0           # set from the tokenizer at runtime
    block_size: int = 256         # tokens per packed block (several examples share one block)
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.05
    bias: bool = True
    grad_checkpoint: bool = True  # recompute block activations in backward (saves memory)

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head:
            raise ValueError(f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})")

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head


@dataclass
class TrainConfig:
    epochs: int = 36
    batch_size: int = 8           # packed blocks per step (~6 examples each)
    learning_rate: float = 3e-3
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_frac: float = 0.05     # fraction of all optimizer steps spent warming up
    min_lr_ratio: float = 0.1
    val_frac: float = 0.1
    eval_interval: int = 300      # optimizer steps between evals
    eval_examples: int = 150      # examples for the periodic exact-match estimate
    final_eval_examples: int = 400
    max_new_tokens: int = 48
    log_interval: int = 25
    seed: int = 1337
    device: str = "auto"
    out_dir: str = str(CHECKPOINT_DIR)
    quick: bool = False           # 4 epochs, small evals: for iterating on a fix


def _add_args(parser: argparse.ArgumentParser, cls) -> None:
    group = parser.add_argument_group(cls.__name__)
    for f in fields(cls):
        default = getattr(cls(), f.name)
        if isinstance(default, bool):
            group.add_argument(f"--{f.name}", type=lambda v: v.lower() in {"1", "true", "yes"},
                               default=default, nargs="?", const=True)
        else:
            group.add_argument(f"--{f.name}", type=type(default), default=default)


def parse_args(argv: list[str] | None = None) -> tuple[ModelConfig, TrainConfig]:
    parser = argparse.ArgumentParser(
        description="SFT of a tiny GPT with sequence packing and gradient checkpointing.")
    _add_args(parser, ModelConfig)
    _add_args(parser, TrainConfig)
    ns = parser.parse_args(argv)
    m = ModelConfig(**{f.name: getattr(ns, f.name) for f in fields(ModelConfig)})
    t = TrainConfig(**{f.name: getattr(ns, f.name) for f in fields(TrainConfig)})
    if t.quick:
        t.epochs = min(t.epochs, 4)
        t.eval_examples = min(t.eval_examples, 50)
        t.final_eval_examples = min(t.final_eval_examples, 100)
    return m, t


def describe(cfg) -> str:
    return "\n".join(f"  {k:>20}: {v}" for k, v in asdict(cfg).items())
