"""A minimal decoder-only transformer with three things the plain `sft/model.py` lacks:

* attention goes through `F.scaled_dot_product_attention` (the "FlashAttention" entry
  point: on CUDA it dispatches to the flash / memory-efficient kernels, on CPU to the
  math kernel) with an explicit boolean mask, so packed blocks can keep examples apart;
* position embeddings are indexed by an explicit `pos_ids` tensor (positions restart at
  0 for every example inside a pack);
* each block can be wrapped in `torch.utils.checkpoint.checkpoint`, which drops the
  block's activations after the forward pass and recomputes them during backward.

Shape conventions: B batch, T tokens, C = n_embd, nh heads, hd = C // nh.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig
from .data import IGNORE_INDEX


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_head, self.head_dim = cfg.n_head, cfg.head_dim
        self.dropout_p = cfg.dropout
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)   # (B, nh, T, hd)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        # BREAKPOINT: attn_mask is (B, 1, T, T) bool; True = may attend.
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.dropout_p,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight          # weight tying
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, pos_ids: torch.Tensor | None = None,
                attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        """idx (B, T) -> logits (B, T, V). Without pos_ids / attn_mask this is an ordinary
        un-packed causal LM (positions 0..T-1, plain lower-triangular mask)."""
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"sequence length {T} exceeds block_size {self.cfg.block_size}")
        if pos_ids is None:
            pos_ids = torch.arange(T, device=idx.device).expand(B, T)
        if attn_mask is None:
            attn_mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=idx.device))
            attn_mask = attn_mask.view(1, 1, T, T)

        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos_ids))
        for block in self.blocks:
            if self.cfg.grad_checkpoint and self.training:
                # BREAKPOINT: activations of `block` are recomputed in backward.
                x = checkpoint(block, x.detach(), attn_mask, use_reentrant=False)
            else:
                x = block(x, attn_mask)
        return self.head(self.ln_f(x))

    def configure_optimizer(self, weight_decay: float, learning_rate: float):
        """Decay matrices, do not decay biases and LayerNorm gains."""
        decay = [p for p in self.parameters() if p.requires_grad and p.dim() >= 2]
        no_decay = [p for p in self.parameters() if p.requires_grad and p.dim() < 2]
        groups = [{"params": decay, "weight_decay": weight_decay},
                  {"params": no_decay, "weight_decay": 0.0}]
        return torch.optim.AdamW(groups, lr=learning_rate, betas=(0.9, 0.95))

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, eos_id: int | None = None
                 ) -> torch.Tensor:
        """Greedy decoding of un-packed prompts (B, T); stops once every row emitted eos."""
        was_training = self.training
        self.eval()
        finished = torch.zeros(idx.size(0), 1, dtype=torch.bool, device=idx.device)
        for _ in range(max_new_tokens):
            logits = self(idx[:, -self.cfg.block_size:])[:, -1, :]
            next_id = logits.argmax(dim=-1, keepdim=True)
            if eos_id is not None:
                next_id = torch.where(finished, torch.full_like(next_id, eos_id), next_id)
                finished = finished | (next_id == eos_id)
            idx = torch.cat((idx, next_id), dim=1)
            if eos_id is not None and bool(finished.all()):
                break
        self.train(was_training)
        return idx


def loss_token_mean(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
    """sum(CE over supervised tokens) / count(supervised tokens).

    The causal shift happens here and only here: logits[:, t] predicts labels[:, t + 1].
    `sum_nll` and `n_tokens` are returned detached so callers can build a corpus-level
    average (a mean of per-batch means would weight a 3-token batch like a 900-token one).
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    b, t, v = shift_logits.shape
    sum_nll = F.cross_entropy(shift_logits.view(b * t, v), shift_labels.view(b * t),
                              ignore_index=IGNORE_INDEX, reduction="sum")
    n_tokens = (shift_labels != IGNORE_INDEX).sum()
    return {"loss": sum_nll / n_tokens.clamp(min=1),
            "sum_nll": sum_nll.detach(), "n_tokens": n_tokens.detach()}
