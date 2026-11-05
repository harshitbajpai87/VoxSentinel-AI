#!/usr/bin/env python
"""Download the ECAPA-TDNN speaker model from HuggingFace (one-time).

Fetches ``speechbrain/spkrec-ecapa-voxceleb`` (~83 MB) into the configured
``speaker_model_dir`` so the speaker-verification API (/speaker-verify)
becomes ready. Idempotent: skips files that already exist. Never runs at
request time — this is an explicit operator action.

Usage:
    python scripts/download_speaker_model.py            # default models/speaker_ecapa
    python scripts/download_speaker_model.py --dir /some/path
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO_ID = "speechbrain/spkrec-ecapa-voxceleb"
PATTERNS = ["*.ckpt", "*.yaml", "*.txt", "config.json"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("models/speaker_ecapa"),
        help="Local directory to store the model files (default: models/speaker_ecapa)",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub is required: pip install huggingface_hub", file=sys.stderr)
        return 2

    target: Path = args.dir
    print(f"Downloading {REPO_ID} -> {target} ...")
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(target),
        allow_patterns=PATTERNS,
    )

    required = ["embedding_model.ckpt", "hyperparams.yaml"]
    missing = [f for f in required if not (target / f).is_file()]
    if missing:
        print(f"ERROR: expected files missing after download: {missing}", file=sys.stderr)
        return 1
    print("Done. Speaker verification is now available (requires speechbrain + torch; see requirements.txt).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
