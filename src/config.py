"""Central configuration for the tiny character-level GPT.

Every knob the training pipeline uses lives here so that it is easy to set a
breakpoint in one place and inspect the whole run configuration at once.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"


@dataclass
class GPTConfig:
    """Architecture hyper-parameters."""

    vocab_size: int = 65          # filled in from the dataset at runtime
    block_size: int = 128         # context length in characters
    n_layer: int = 4              # number of transformer blocks
    n_head: int = 4               # attention heads per block
    n_embd: int = 128             # embedding / residual stream width
    dropout: float = 0.1
    bias: bool = True             # use bias terms in Linear / LayerNorm

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
            )

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head


@dataclass
class TrainConfig:
    """Optimisation and bookkeeping hyper-parameters."""

    max_iters: int = 500
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_iters: int = 50
    min_lr_ratio: float = 0.1
    eval_interval: int = 50
    eval_iters: int = 20
    log_interval: int = 10
    seed: int = 1337
    device: str = "auto"          # "auto" | "cpu" | "cuda" | "mps"
    compile_model: bool = False   # torch.compile makes debugging harder
    out_dir: str = str(CHECKPOINT_DIR)

    def resolved_device(self) -> str:
        import torch

        if self.device != "auto":
            return self.device
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"


def _add_dataclass_args(parser: argparse.ArgumentParser, cls, group_name: str) -> None:
    group = parser.add_argument_group(group_name)
    for f in fields(cls):
        default = getattr(cls(), f.name)
        if f.type is bool or isinstance(default, bool):
            group.add_argument(f"--{f.name}", type=lambda v: v.lower() in {"1", "true", "yes"}, default=default)
        else:
            group.add_argument(f"--{f.name}", type=type(default), default=default)


def parse_args(argv: list[str] | None = None) -> tuple[GPTConfig, TrainConfig]:
    """Parse CLI flags into the two config objects.

    Useful in the debugger: launch.json passes explicit args, so the values you
    see here are exactly the ones the run will use.
    """
    parser = argparse.ArgumentParser(description="Train a tiny character-level GPT.")
    _add_dataclass_args(parser, GPTConfig, "model")
    _add_dataclass_args(parser, TrainConfig, "training")
    ns = parser.parse_args(argv)

    model_kwargs = {f.name: getattr(ns, f.name) for f in fields(GPTConfig)}
    train_kwargs = {f.name: getattr(ns, f.name) for f in fields(TrainConfig)}
    return GPTConfig(**model_kwargs), TrainConfig(**train_kwargs)


def describe(cfg) -> str:
    return "\n".join(f"  {k:>16}: {v}" for k, v in asdict(cfg).items())
