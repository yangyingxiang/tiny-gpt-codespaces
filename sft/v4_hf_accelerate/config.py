"""All knobs in one place (dataclass -> CLI flags)."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_FILE = DATA_DIR / "sft_train.jsonl"
HOLDOUT_FILE = DATA_DIR / "holdout.jsonl"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "v4_hf_accelerate"

IGNORE_INDEX = -100


@dataclass
class Config:
    # model (tiny random-init GPT-2 with the real gpt2 tokenizer)
    tokenizer: str = "gpt2"
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 128
    n_positions: int = 128
    dropout: float = 0.0
    grad_checkpoint: bool = True
    # data
    max_length: int = 48          # tokens per example incl. EOS; longer ones are truncated
    val_frac: float = 0.1
    # optimisation
    epochs: int = 12
    max_steps: int = 0            # optimizer steps; 0 = run all epochs
    batch_size: int = 16          # micro-batch
    accum_steps: int = 4          # effective batch = batch_size * accum_steps
    learning_rate: float = 3e-3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    grad_clip: float = 1.0
    # eval / logging
    eval_interval: int = 100      # optimizer steps between evals
    eval_examples: int = 150      # examples for the periodic exact-match estimate
    final_eval_examples: int = 400
    max_new_tokens: int = 48
    log_interval: int = 25        # optimizer steps between train-loss prints
    seed: int = 1337
    out_dir: str = str(CHECKPOINT_DIR)
    quick: bool = False           # short run for iterating on a fix


def _add_args(parser: argparse.ArgumentParser, cls) -> None:
    for f in fields(cls):
        default = getattr(cls(), f.name)
        if isinstance(default, bool):
            parser.add_argument(f"--{f.name}", nargs="?", const=True, default=default,
                                type=lambda v: v.lower() in {"1", "true", "yes"})
        else:
            parser.add_argument(f"--{f.name}", type=type(default), default=default)


def parse_args(argv: list[str] | None = None) -> Config:
    parser = argparse.ArgumentParser(description="SFT of a tiny GPT-2 with an accelerate loop.")
    _add_args(parser, Config)
    ns = parser.parse_args(argv)
    cfg = Config(**{f.name: getattr(ns, f.name) for f in fields(Config)})
    if cfg.quick:
        cfg.max_steps = cfg.max_steps or 60
        cfg.eval_interval = min(cfg.eval_interval, 30)
        cfg.eval_examples = min(cfg.eval_examples, 50)
        cfg.final_eval_examples = min(cfg.final_eval_examples, 100)
    return cfg


def describe(cfg) -> str:
    return "\n".join(f"  {k:>20}: {v}" for k, v in asdict(cfg).items())
