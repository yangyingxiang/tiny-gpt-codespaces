"""Download the training corpus ahead of time (run by postCreateCommand)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import download_corpus  # noqa: E402

if __name__ == "__main__":
    path = download_corpus()
    print(f"corpus ready: {path} ({path.stat().st_size} bytes)")
