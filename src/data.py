"""Dataset handling: download TinyShakespeare, build a char vocabulary, batch it.

The whole corpus is a single string, so everything here stays small enough to
inspect in the debugger's variable pane.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import torch

from .config import DATA_DIR

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
    "tinyshakespeare/input.txt"
)

# Used only when the machine is offline, so `python -m src.train` still works.
FALLBACK_TEXT = (
    "ROMEO:\nBut soft, what light through yonder window breaks?\n"
    "It is the east, and Juliet is the sun.\n\n"
    "JULIET:\nO Romeo, Romeo, wherefore art thou Romeo?\n"
    "Deny thy father and refuse thy name.\n\n"
    "HAMLET:\nTo be, or not to be, that is the question:\n"
    "Whether tis nobler in the mind to suffer\n"
    "The slings and arrows of outrageous fortune.\n\n"
    "MACBETH:\nTomorrow, and tomorrow, and tomorrow,\n"
    "Creeps in this petty pace from day to day.\n\n"
) * 200


@dataclass
class CharTokenizer:
    """The simplest possible tokenizer: one token per character."""

    itos: list[str]

    def __post_init__(self) -> None:
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(itos=sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, s: str) -> list[int]:
        # Unknown characters are dropped rather than crashing a sampling prompt.
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"itos": self.itos}), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "CharTokenizer":
        return cls(itos=json.loads(Path(path).read_text(encoding="utf-8"))["itos"])


def download_corpus(dest: Path = DATA_DIR / "input.txt") -> Path:
    """Fetch TinyShakespeare once and cache it under data/."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    try:
        print(f"downloading {TINY_SHAKESPEARE_URL} -> {dest}")
        with urllib.request.urlopen(TINY_SHAKESPEARE_URL, timeout=30) as r:
            dest.write_bytes(r.read())
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"download failed ({exc}); falling back to the bundled sample text")
        dest.write_text(FALLBACK_TEXT, encoding="utf-8")
    return dest


class CharDataset:
    """Holds the encoded corpus and hands out random contiguous batches."""

    def __init__(self, text: str, block_size: int, split_ratio: float = 0.9) -> None:
        self.block_size = block_size
        self.tokenizer = CharTokenizer.from_text(text)
        data = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        n = int(len(data) * split_ratio)
        self.train_data = data[:n]
        self.val_data = data[n:]
        if len(self.val_data) <= block_size:
            raise ValueError("corpus too small for the requested block_size")

    @classmethod
    def from_file(cls, path: Path | None = None, block_size: int = 128) -> "CharDataset":
        path = Path(path) if path else download_corpus()
        return cls(path.read_text(encoding="utf-8"), block_size=block_size)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    def get_batch(self, split: str, batch_size: int, device: str = "cpu"):
        """Return (inputs, targets), each of shape (batch_size, block_size).

        `targets` is `inputs` shifted one character to the left: predicting the
        next character is the whole training objective.
        """
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(data) - self.block_size - 1, (batch_size,))
        x = torch.stack([data[i : i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + 1 + self.block_size] for i in ix])
        return x.to(device), y.to(device)

    def __repr__(self) -> str:
        return (
            f"CharDataset(vocab_size={self.vocab_size}, "
            f"train_tokens={len(self.train_data)}, val_tokens={len(self.val_data)}, "
            f"block_size={self.block_size})"
        )
