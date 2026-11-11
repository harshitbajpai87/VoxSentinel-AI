"""Fetch every model family the integrated pipeline uses.

Idempotent: a family already on disk is skipped. Prints one status line per
family so a partial download is visible instead of failing the whole run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

HF_DIR = Path(__file__).resolve().parents[1] / "models" / "hf"
AASIST_DIR = Path(__file__).resolve().parents[1] / "models" / "aasist"

FAMILIES = [
    ("whisper-large-v3-turbo", "openai/whisper-large-v3-turbo", "whisper-large-v3-turbo"),
    ("faster-whisper-large-v3", "Systran/faster-whisper-large-v3", "faster-whisper-large-v3"),
    ("wavlm-base-plus", "microsoft/wavlm-base-plus", "wavlm-base-plus"),
    ("xlsr-300m", "facebook/wav2vec2-xls-r-300m", "xlsr-900m"),
    ("xlmr-base", "FacebookAI/xlm-roberta-base", "xlmr-base"),
]


def fetch_family(label: str, repo: str, target: str) -> None:
    dest = HF_DIR / target
    marker = dest / "config.json"
    if marker.is_file() or (dest / "model.bin").is_file() or (dest / "preprocessor_config.json").is_file():
        print(f"skip {label}: already at {dest}", flush=True)
        return
    print(f"fetch {label} <- {repo}", flush=True)
    snapshot_download(repo_id=repo, local_dir=str(dest), allow_patterns=None)
    print(f"done {label}: {dest}", flush=True)


def fetch_aasist() -> None:
    AASIST_DIR.mkdir(parents=True, exist_ok=True)
    existing = list(AASIST_DIR.glob("*.pth")) + list(AASIST_DIR.glob("*.pt"))
    if existing:
        print(f"skip aasist: {existing[0].name}", flush=True)
        return
    api = HfApi()
    for repo in [m.id for m in api.list_models(search="aasist", limit=20)]:
        try:
            files = [f.rfilename for f in api.list_repo_files(repo)]
        except Exception:
            continue
        weights = [f for f in files if f.lower().endswith((".pth", ".pt", ".ckpt"))]
        if not weights:
            continue
        for name in weights:
            try:
                path = hf_hub_download(repo_id=repo, filename=name, local_dir=str(AASIST_DIR))
            except Exception as exc:  # noqa: BLE001 - try the next candidate
                print(f"  {repo}/{name} failed: {type(exc).__name__}", flush=True)
                continue
            size = Path(path).stat().st_size
            print(f"done aasist: {repo}/{name} ({size/1e6:.1f} MB)", flush=True)
            if size > 1_000_000:
                (AASIST_DIR / "SOURCE.json").write_text(
                    json.dumps({"repo": repo, "file": name}, indent=2)
                )
                return


def main() -> int:
    HF_DIR.mkdir(parents=True, exist_ok=True)
    failures = []
    for label, repo, target in FAMILIES:
        try:
            fetch_family(label, repo, target)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{label}: {type(exc).__name__}: {str(exc)[:120]}")
            print(f"fail {label}: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
    try:
        fetch_aasist()
    except Exception as exc:  # noqa: BLE001
        failures.append(f"aasist: {type(exc).__name__}")
        print(f"fail aasist: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
    print("FAILURES:" + json.dumps(failures), flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
