"""Generate text from a trained checkpoint.

    python -m src.sample --prompt "ROMEO:" --max_new_tokens 300
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .config import CHECKPOINT_DIR, GPTConfig
from .data import CharTokenizer
from .model import GPT


def load_model(ckpt_path: Path, device: str) -> tuple[GPT, CharTokenizer]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = GPTConfig(**ckpt["model_config"])
    model = GPT(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    tokenizer = CharTokenizer.load(Path(ckpt_path).parent / "tokenizer.json")
    return model, tokenizer


def main() -> None:
    p = argparse.ArgumentParser(description="Sample from a trained tiny GPT.")
    p.add_argument("--ckpt", default=str(CHECKPOINT_DIR / "model.pt"))
    p.add_argument("--prompt", default="\n")
    p.add_argument("--max_new_tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise SystemExit(f"no checkpoint at {ckpt_path} - run `python -m src.train` first")

    model, tokenizer = load_model(ckpt_path, device)
    start = tokenizer.encode(args.prompt) or [0]
    idx = torch.tensor([start], dtype=torch.long, device=device)

    # BREAKPOINT: step into generate() to watch tokens being sampled one at a time.
    out = model.generate(idx, args.max_new_tokens, args.temperature, args.top_k)
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
