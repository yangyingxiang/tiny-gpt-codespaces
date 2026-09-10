"""Regression guards for the six planted defects (solutions branch only).

Every test here FAILS on `main` and PASSES once the matching bug is fixed.
That is the bar for a regression test: one that passes on the buggy code
proves nothing.
"""

from __future__ import annotations

import math
import random
import unicodedata

import torch

from sft.config import RAW_FILES, ModelConfig
from sft.data import (
    IGNORE_INDEX, Example, SFTDataset, collate, dedup, encode_example,
    load_raw_examples, make_loader, split_examples,
)
from sft.loss import masked_cross_entropy
from sft.model import GPT
from sft.tokenizer import ASSISTANT_ID, EOS_ID, NUM_BYTES, SPECIAL_IDS, ByteTokenizer
from sft.utils import set_seed

TOK = ByteTokenizer()
EX = Example("t1", "Reverse:", "abcde", "edcba")


def tiny_model(**kw) -> GPT:
    cfg = dict(vocab_size=TOK.vocab_size, block_size=32, n_layer=2, n_head=2, n_embd=32, dropout=0.0)
    cfg.update(kw)
    return GPT(ModelConfig(**cfg))


# --- Bug 1 (loud): special-token ids off by one -> IndexError in the embedding ---------------
def test_every_token_id_fits_in_the_vocab():
    ids = sorted(SPECIAL_IDS.values())
    assert ids == list(range(NUM_BYTES, TOK.vocab_size)), ids      # contiguous, no gap, no overflow
    x, _ = encode_example(TOK, EX)
    assert max(x) < TOK.vocab_size
    tiny_model()(torch.tensor([x]))                                 # would raise IndexError


# --- Bug 2 (loud): collate truncates inputs but not labels -> ragged tensor ------------------
def test_collate_truncates_inputs_and_labels_together():
    long_ex = Example("t2", "Repeat:", "x" * 80, "x" * 80)
    batch = [encode_example(TOK, EX), encode_example(TOK, long_ex)]
    x, y = collate(batch, block_size=32)
    assert x.shape == y.shape == (2, 32)


# --- Bug 3 (silent): labels not shifted -> model learns to copy its input --------------------
def test_labels_are_inputs_shifted_left_by_one():
    x, y = encode_example(TOK, EX)
    assert len(x) == len(y)
    for t in range(len(y) - 1):
        if y[t] != IGNORE_INDEX:
            assert y[t] == x[t + 1], f"label at {t} should be the *next* input token"
    # the supervised part is exactly the answer + eos, and it starts right after <|assistant|>
    supervised = [t for t in y if t != IGNORE_INDEX]
    assert supervised == TOK.encode(EX.output) + [EOS_ID]
    first = next(t for t, v in enumerate(y) if v != IGNORE_INDEX)
    assert x[first] == ASSISTANT_ID


# --- Bug 4 (silent): exact-match dedup lets near-duplicates leak across the split ------------
def test_dedup_catches_formatting_variants():
    base = Example("a", "Sort letters:", "café", "acéf")
    variants = [
        Example("b", "SORT LETTERS:", "café", "acéf"),
        Example("c", "sort  letters:", "café", "acéf"),
        Example("d", "Sort letters:", unicodedata.normalize("NFD", "café"), unicodedata.normalize("NFD", "acéf")),
    ]
    assert len(dedup([base, *variants])) == 1


def _canonical(ex: Example) -> tuple[str, str, str]:
    # deliberately independent of sft.data.dedup_key, so the guard can't share its bug
    def n(s: str) -> str:
        return " ".join(unicodedata.normalize("NFC", s).split()).casefold()
    return (n(ex.instruction), n(ex.input), n(ex.output))


def test_train_and_val_share_no_example():
    set_seed(0)
    train, val = split_examples(load_raw_examples(RAW_FILES, verbose=False), val_frac=0.1)
    train_keys = {_canonical(e) for e in train}
    leaked = [e for e in val if _canonical(e) in train_keys]
    assert not leaked, f"{len(leaked)}/{len(val)} val examples also in train, e.g. {leaked[:2]}"


# --- Bug 5 (silent): loss divided by all positions instead of supervised ones ----------------
def test_initial_loss_is_ln_vocab():
    torch.manual_seed(0)
    model = tiny_model()
    ds = SFTDataset([EX, Example("t3", "Uppercase:", "hello", "HELLO")] * 4, TOK)
    x, y = next(iter(make_loader(ds, batch_size=8, block_size=32, shuffle=False)))
    _, loss = model(x, y)
    assert abs(loss.item() - math.log(TOK.vocab_size)) < 0.5, loss.item()


def test_loss_does_not_depend_on_padding():
    torch.manual_seed(0)
    logits = torch.randn(1, 6, 10)
    labels = torch.tensor([[IGNORE_INDEX, 3, 4, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]])
    padded_logits = torch.cat([logits, torch.randn(1, 10, 10)], dim=1)
    padded_labels = torch.cat([labels, torch.full((1, 10), IGNORE_INDEX)], dim=1)
    a, b = masked_cross_entropy(logits, labels), masked_cross_entropy(padded_logits, padded_labels)
    assert torch.allclose(a, b)


# --- Bug 6 (silent): python `random` never seeded -> split changes every run ----------------
def test_split_is_reproducible_for_a_seed():
    examples = [Example(str(i), "Reverse:", f"w{i}", f"{i}w") for i in range(200)]
    set_seed(123)
    a = split_examples(examples, 0.2)
    random.random()                                            # disturb the global RNG
    set_seed(123)
    b = split_examples(examples, 0.2)
    assert a == b


# --- end-to-end tripwire: can the whole stack overfit a handful of examples? -----------------
def test_model_overfits_a_tiny_batch_and_answers_correctly():
    from sft.evaluate import exact_match

    set_seed(0)
    exs = [Example(str(i), "Reverse:", w, w[::-1]) for i, w in enumerate(["abc", "hello", "tiny", "gpt"])]
    ds = SFTDataset(exs, TOK)
    x, y = collate([ds[i] for i in range(len(ds))], block_size=32)
    model = tiny_model()
    opt = model.configure_optimizer(0.0, 3e-3)
    for _ in range(300):
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.05, loss.item()
    em, preds = exact_match(model, TOK, exs, "cpu", max_new_tokens=10)
    assert em == 1.0, preds
