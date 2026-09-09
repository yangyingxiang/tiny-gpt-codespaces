"""Fast unit tests - the whole file runs in a few seconds on CPU."""

from __future__ import annotations

import torch

from src.config import GPTConfig, TrainConfig
from src.data import CharDataset, CharTokenizer
from src.model import GPT
from src.train import lr_at

SAMPLE_TEXT = "hello world, this is a tiny corpus for testing. " * 200


def tiny_cfg(**kw) -> GPTConfig:
    base = dict(vocab_size=32, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0)
    base.update(kw)
    return GPTConfig(**base)


def test_tokenizer_roundtrip():
    tok = CharTokenizer.from_text(SAMPLE_TEXT)
    assert tok.decode(tok.encode("hello")) == "hello"
    assert tok.vocab_size == len(set(SAMPLE_TEXT))


def test_config_rejects_bad_head_split():
    try:
        GPTConfig(n_embd=30, n_head=4)
    except ValueError:
        return
    raise AssertionError("expected ValueError for indivisible n_embd/n_head")


def test_forward_shapes_and_loss():
    cfg = tiny_cfg()
    model = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (4, cfg.block_size))
    logits, loss = model(x, x)
    assert logits.shape == (4, cfg.block_size, cfg.vocab_size)
    assert loss.item() > 0


def test_initial_loss_is_near_uniform():
    """An untrained model should score about -log(1/vocab_size)."""
    cfg = tiny_cfg()
    model = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (8, cfg.block_size))
    _, loss = model(x, x)
    expected = torch.log(torch.tensor(float(cfg.vocab_size)))
    assert abs(loss.item() - expected.item()) < 0.7


def test_attention_is_causal():
    """Changing a future token must not change earlier positions' logits."""
    cfg = tiny_cfg()
    model = GPT(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, cfg.block_size))
    with torch.no_grad():
        a, _ = model(x)
        x2 = x.clone()
        x2[0, -1] = (x2[0, -1] + 1) % cfg.vocab_size
        b, _ = model(x2)
    assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-5)


def test_generate_extends_sequence():
    cfg = tiny_cfg()
    model = GPT(cfg)
    idx = torch.zeros((1, 3), dtype=torch.long)
    out = model.generate(idx, max_new_tokens=10, top_k=5)
    assert out.shape == (1, 13)
    assert out.max().item() < cfg.vocab_size


def test_model_overfits_a_single_batch():
    """The clearest sanity check that gradients actually flow."""
    cfg = tiny_cfg()
    model = GPT(cfg)
    opt = model.configure_optimizer(0.0, 1e-2)
    x = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    y = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    first = last = None
    for _ in range(120):
        _, loss = model(x, y)
        if first is None:
            first = loss.item()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last = loss.item()
    assert last < first * 0.5, f"loss barely moved: {first:.3f} -> {last:.3f}"


def test_dataset_batches():
    ds = CharDataset(SAMPLE_TEXT, block_size=16)
    x, y = ds.get_batch("train", batch_size=4)
    assert x.shape == y.shape == (4, 16)
    # y is x shifted by one position.
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_lr_schedule_warms_up_then_decays():
    cfg = TrainConfig(max_iters=100, warmup_iters=10, learning_rate=1e-3, min_lr_ratio=0.1)
    assert lr_at(0, cfg) < lr_at(9, cfg)
    assert abs(lr_at(9, cfg) - cfg.learning_rate) < 1e-9
    assert lr_at(99, cfg) < lr_at(50, cfg)
    assert lr_at(99, cfg) >= cfg.learning_rate * cfg.min_lr_ratio - 1e-9
