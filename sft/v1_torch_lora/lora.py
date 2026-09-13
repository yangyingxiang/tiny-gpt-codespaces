"""Hand-rolled LoRA (Hu et al. 2021): W x + (alpha / r) * B A x, with W frozen.

A is (r, in), B is (out, r). The adapter starts as a no-op so the model begins
exactly where the base left off.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float) -> None:
        super().__init__()
        self.base = base
        self.r, self.scaling = r, alpha / r
        self.lora_A = nn.Parameter(torch.zeros(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # BREAKPOINT: at step 0 the second term is exactly zero (B == 0).
        return self.base(x) + (x @ self.lora_A.t() @ self.lora_B.t()) * self.scaling

    def merged_linear(self) -> nn.Linear:
        """A plain Linear equal to this module, for inference without the extra matmuls."""
        with torch.no_grad():
            self.base.weight += self.lora_B @ self.lora_A
        return self.base


def attach_lora(model: nn.Module, r: int, alpha: float,
                targets: tuple[str, ...] = ("qkv", "proj", "fc")) -> list[str]:
    """Replace every nn.Linear whose attribute name is in `targets` with a LoRALinear.
    Returns the qualified names that were wrapped."""
    wrapped = []
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and child_name in targets:
                setattr(module, child_name, LoRALinear(child, r, alpha))
                wrapped.append(f"{name}.{child_name}" if name else child_name)
    return wrapped


def merge_lora(model: nn.Module) -> int:
    """Fold every adapter into its base weight and drop the LoRA modules. Returns the count."""
    n = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                setattr(module, child_name, child.merged_linear())
                n += 1
    return n


def lora_parameters(model: nn.Module):
    return [p for n, p in model.named_parameters() if "lora_" in n]


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {n: p.detach().cpu() for n, p in model.named_parameters() if "lora_" in n}
