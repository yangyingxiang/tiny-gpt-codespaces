#!/usr/bin/env bash
# Runs once, after the codespace container is created.
set -euo pipefail

echo "==> upgrading pip"
python -m pip install --upgrade pip

echo "==> installing CPU-only PyTorch (much smaller/faster than the CUDA build)"
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch

echo "==> installing the rest of the requirements (incl. transformers/peft/accelerate for sft/v3_*, sft/v4_*)"
python -m pip install numpy pytest transformers datasets peft accelerate

echo "==> pre-downloading the gpt2 tokenizer used by sft/v3_* and sft/v4_*"
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('gpt2')" || echo "(tokenizer download failed; it will be retried on first run)"

echo "==> done. Open EXERCISE.md, then press F5 and pick 'Train: quick debug run'."
