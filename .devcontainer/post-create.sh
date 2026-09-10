#!/usr/bin/env bash
# Runs once, after the codespace container is created.
set -euo pipefail

echo "==> upgrading pip"
python -m pip install --upgrade pip

echo "==> installing CPU-only PyTorch (much smaller/faster than the CUDA build)"
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch

echo "==> installing the rest of the requirements"
python -m pip install numpy pytest

echo "==> done. Open EXERCISE.md, then press F5 and pick 'Train: quick debug run'."
