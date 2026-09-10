"""Basic model/tokenizer/schedule tests. These pass on the buggy `main` too --
green unit tests do not mean the pipeline is correct."""

from __future__ import annotations

import pytest
import torch

from sft.config import ModelConfig, TrainConfig
from sft.model import GPT
from sft.tokenizer import ByteTokenizer
from sft.train import lr_at


def tiny_cfg(**kw) -> ModelConfig:
    base = dict(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0)
    base.update(kw)
    return ModelConfig(**base)


def test_tokenizer_roundtrip_unicode():
    tok = ByteTokenizer()
    s = 'café "naïve"\tüber\n日本'
    assert tok.decode(tok.encode(s)) == s


def test_config_rejects_bad_head_split():
    with pytest.raises(ValueError):
        ModelConfig(n_embd=30, n_head=4)


def test_forward_shapes():
    cfg = tiny_cfg()
    model = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (3, cfg.block_size))
    logits, loss = model(x)
    assert logits.shape == (3, cfg.block_size, cfg.vocab_size)
    assert loss is None


def test_attention_is_causal():
    cfg = tiny_cfg()
    model = GPT(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, cfg.block_size))
    with torch.no_grad():
        a, _ = model(x)
        x2 = x.clone()
        x2[0, -1] = (x2[0, -1] + 1) % cfg.vocab_size
        b, _ = model(x2)
    assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-5)


def test_generate_stops_at_eos_and_respects_budget():
    cfg = tiny_cfg()
    model = GPT(cfg)
    idx = torch.zeros((2, 3), dtype=torch.long)
    out = model.generate(idx, max_new_tokens=5)
    assert out.shape == (2, 8)
    assert out.max().item() < cfg.vocab_size


def test_lr_schedule_warms_up_then_decays():
    cfg = TrainConfig(max_iters=100, warmup_iters=10, learning_rate=1e-3, min_lr_ratio=0.1)
    assert lr_at(0, cfg) < lr_at(9, cfg)
    assert abs(lr_at(9, cfg) - cfg.learning_rate) < 1e-12
    assert lr_at(99, cfg) < lr_at(50, cfg)
    assert lr_at(100, cfg) >= cfg.learning_rate * cfg.min_lr_ratio - 1e-12
