from __future__ import annotations

import torch
from torch.nn import functional as F

from .data import IGNORE_INDEX


def masked_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean next-token cross-entropy over the *non-ignored* positions.

    logits: (B, T, V) float, labels: (B, T) long with IGNORE_INDEX where no loss applies.
    """
    B, T, V = logits.shape
    flat_labels = labels.reshape(-1)
    per_token = F.cross_entropy(
        logits.reshape(B * T, V), flat_labels, ignore_index=IGNORE_INDEX, reduction="none"
    )
    mask = (flat_labels != IGNORE_INDEX).float()
    return (per_token * mask).sum() / mask.numel()
