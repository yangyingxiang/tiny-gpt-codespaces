"""Variant 4: SFT of a tiny GPT-2 with a hand-written loop driven by Hugging Face accelerate.

    python sft/v4_hf_accelerate/main.py            # "Run Python File" in VS Code works too
    python -m sft.v4_hf_accelerate.main --quick    # short run for iterating on a fix
    accelerate launch sft/v4_hf_accelerate/main.py # same script, accelerate's launcher

See README.md in this folder for the brief and the healthy numbers.
"""

from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    __package__ = "sft.v4_hf_accelerate"

from .config import parse_args
from .train import train


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
