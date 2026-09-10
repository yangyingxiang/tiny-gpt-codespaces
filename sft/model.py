"""A minimal decoder-only transformer (GPT), written out longhand.

Every tensor operation is explicit rather than hidden inside
`nn.MultiheadAttention`, so you can set a breakpoint in `CausalSelfAttention.forward`
and watch the shapes change step by step.

Shape conventions used throughout:
    B = batch size
    T = time / sequence length (<= block_size)
    C = embedding width (n_embd)
    nh = number of heads, hd = head dimension (C == nh * hd)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from .config import ModelConfig
from .loss import masked_cross_entropy


class CausalSelfAttention(nn.Module):
    """Multi-head masked self-attention."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.head_dim
        # One fused projection produces queries, keys and values in a single matmul.
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        # Lower-triangular mask: position t may only attend to positions <= t.
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.qkv(x)                                  # (B, T, 3C)
        q, k, v = qkv.split(C, dim=2)                      # three x (B, T, C)

        # (B, T, C) -> (B, nh, T, hd): heads become a batch dimension.
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention scores: (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v                                        # (B, nh, T, hd)
        y = y.transpose(1, 2).contiguous().view(B, T, C)   # re-merge the heads
        return self.resid_dropout(self.proj(y))


class MLP(nn.Module):
    """Position-wise feed-forward network with a 4x inner width."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """Pre-norm transformer block: x + attn(ln(x)), then x + mlp(ln(x))."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """Decoder-only transformer language model."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # Weight tying: the output projection reuses the token embedding matrix.
        self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        # Scaled init for residual projections (GPT-2 trick).
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

    def num_parameters(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.pos_emb.weight.numel()
        return n

    def forward(self, idx: torch.Tensor, labels: torch.Tensor | None = None):
        """idx: (B, T) token ids; labels: (B, T) next-token targets (already shifted,
        IGNORE_INDEX where there is no target). Returns (logits, loss)."""
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"sequence length {T} exceeds block_size {self.cfg.block_size}")

        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))   # (B, T, C)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)                                  # (B, T, vocab_size)

        loss = None
        if labels is not None:
            loss = masked_cross_entropy(logits, labels)
        return logits, loss

    def configure_optimizer(self, weight_decay: float, learning_rate: float):
        """Decay matrices, do not decay biases and LayerNorm gains."""
        decay, no_decay = [], []
        for p in self.parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=learning_rate, betas=(0.9, 0.95))

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        eos_id: int | None = None,
        temperature: float = 0.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Extend `idx` (B, T) by up to `max_new_tokens` tokens; temperature 0 = greedy.
        Stops early once every row has produced `eos_id`."""
        was_training = self.training
        self.eval()
        finished = torch.zeros(idx.size(0), 1, dtype=torch.bool, device=idx.device)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            if temperature <= 0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")
                next_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            if eos_id is not None:
                # rows that already finished keep emitting eos
                next_id = torch.where(finished, torch.full_like(next_id, eos_id), next_id)
                finished = finished | (next_id == eos_id)
            idx = torch.cat((idx, next_id), dim=1)
            if eos_id is not None and bool(finished.all()):
                break
        self.train(was_training)
        return idx
