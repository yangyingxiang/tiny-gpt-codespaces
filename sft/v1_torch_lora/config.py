"""All knobs in one place (dataclasses -> CLI flags)."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_FILE = DATA_DIR / "sft_train.jsonl"
HOLDOUT_FILE = DATA_DIR / "holdout.jsonl"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "v1_torch_lora"

BASE_TASKS = ("Reverse:", "Uppercase:", "Repeat:")
ADAPTER_TASKS = ("Sort letters:", "Length:", "Count vowels:")


@dataclass
class ModelConfig:
    vocab_size: int = 0           # set from the tokenizer at runtime
    block_size: int = 64
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
    base_iters: int = 1500        # phase 1: full fine-tune on the base tasks
    lora_iters: int = 1500        # phase 2: LoRA on the adapter tasks
    micro_batch: int = 8
    accum_steps: int = 4          # effective batch = micro_batch * accum_steps
    base_lr: float = 3e-3
    lora_lr: float = 1e-2
    lora_r: int = 16
    lora_alpha: float = 32.0
    replay_frac: float = 0.3      # phase 2: base-task examples mixed in, as a fraction of adapter examples
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_iters: int = 100
    min_lr_ratio: float = 0.1
    val_frac: float = 0.1
    eval_interval: int = 250
    eval_batches: int = 10
    eval_examples: int = 150
    final_eval_examples: int = 400
    max_new_tokens: int = 48
    log_interval: int = 50
    seed: int = 1337
    device: str = "auto"
    out_dir: str = str(CHECKPOINT_DIR)
    reuse_base: bool = False      # skip phase 1 if out_dir/base.pt exists
    quick: bool = False           # short run for iterating on a fix


def _add_args(parser: argparse.ArgumentParser, cls) -> None:
    group = parser.add_argument_group(cls.__name__)
    for f in fields(cls):
        default = getattr(cls(), f.name)
        if isinstance(default, bool):
            group.add_argument(f"--{f.name}", nargs="?", const=True, default=default,
                               type=lambda v: str(v).lower() in {"1", "true", "yes"})
        else:
            group.add_argument(f"--{f.name}", type=type(default), default=default)


def parse_args(argv: list[str] | None = None) -> tuple[ModelConfig, TrainConfig]:
    parser = argparse.ArgumentParser(description="SFT variant 1: PyTorch + LoRA + SDPA.")
    _add_args(parser, ModelConfig)
    _add_args(parser, TrainConfig)
    ns = parser.parse_args(argv)
    m = ModelConfig(**{f.name: getattr(ns, f.name) for f in fields(ModelConfig)})
    t = TrainConfig(**{f.name: getattr(ns, f.name) for f in fields(TrainConfig)})
    if t.quick:
        t.base_iters, t.lora_iters = 300, 300
        t.eval_interval, t.eval_examples, t.final_eval_examples = 100, 50, 100
    return m, t


def describe(cfg) -> str:
    return "\n".join(f"  {k:>20}: {v}" for k, v in asdict(cfg).items())
