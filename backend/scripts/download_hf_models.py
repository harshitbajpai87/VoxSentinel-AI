"""Download the Hugging Face checkpoints the unified pipeline can use.

    python backend/scripts/download_hf_models.py --list
    python backend/scripts/download_hf_models.py --all
    python backend/scripts/download_hf_models.py --only whisper-small

Gated repos (e.g. ``ai4bharat/indicwav2vec-hindi``) need ``HF_TOKEN`` and accepted
terms on the model page; without them this script reports the real HTTP error and
keeps going, which is what the API then shows in ``/api/capabilities``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = Path(os.environ.get("VOXSENTINEL_HF_DOWNLOAD") or BACKEND_DIR / "models" / "hf")

MODELS = {
    "whisper-small": "openai/whisper-small",
    "indicwav2vec-hindi": "ai4bharat/indicwav2vec-hindi",
    "xlsr-53": "facebook/wav2vec2-large-xlsr-53",
    "muril-base-cased": "google/muril-base-cased",
    "deberta-v3-small": "microsoft/deberta-v3-small",
}


def _download(name: str, repo_id: str) -> bool:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub is not installed: pip install -r requirements-models.txt")
        return False

    target = MODELS_DIR / name
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target),
            token=token,
            allow_patterns=["*.json", "*.txt", "*.model", "*.bin", "*.safetensors", "*.spm"],
        )
    except Exception as exc:  # noqa: BLE001 - the message is what matters to the operator
        detail = str(exc).splitlines()[0][:200]
        hint = ""
        if "401" in detail or "403" in detail or "gated" in detail.lower():
            hint = " (gated repo: accept the terms on the model page and set HF_TOKEN)"
        print(f"FAILED  {name:<20} {repo_id}\n        {detail}{hint}")
        return False
    print(f"OK      {name:<20} {repo_id} -> {target}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="download every model")
    parser.add_argument("--only", action="append", default=[], metavar="NAME", help="download one model")
    parser.add_argument("--list", action="store_true", help="list the known models")
    args = parser.parse_args()

    if args.list or not (args.all or args.only):
        for name, repo in MODELS.items():
            print(f"{name:<20} {repo}")
        return 0

    wanted = list(MODELS) if args.all else args.only
    unknown = [name for name in wanted if name not in MODELS]
    if unknown:
        print(f"unknown model(s): {', '.join(unknown)}")
        return 2

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    results = {name: _download(name, MODELS[name]) for name in wanted}
    ok = sum(1 for value in results.values() if value)
    print(f"\n{ok}/{len(results)} downloaded into {MODELS_DIR}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
