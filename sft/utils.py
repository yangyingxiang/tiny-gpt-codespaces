from __future__ import annotations

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed every RNG the pipeline touches."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
