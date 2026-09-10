"""Ask a trained checkpoint something.

    python -m sft.generate --instruction "Reverse the letters." --input "banana"
"""

from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
    __package__ = "sft"

import argparse
from pathlib import Path

from .config import CHECKPOINT_DIR
from .data import Example
from .evaluate import generate_answer, load_checkpoint
from .tokenizer import ByteTokenizer
from .utils import resolve_device


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(CHECKPOINT_DIR / "model.pt"))
    p.add_argument("--instruction", default="Reverse the letters.")
    p.add_argument("--input", default="banana")
    p.add_argument("--max_new_tokens", type=int, default=48)
    args = p.parse_args()
    device = resolve_device("auto")
    model, _ = load_checkpoint(Path(args.ckpt), device)
    ex = Example("cli", args.instruction, args.input, "")
    # BREAKPOINT: step into generate_answer -> model.generate to watch greedy decoding.
    print(generate_answer(model, ByteTokenizer(), ex, device, args.max_new_tokens))


if __name__ == "__main__":
    main()
