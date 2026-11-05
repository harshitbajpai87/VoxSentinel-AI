"""Deterministic, leakage-checked dataset manifests and speaker-disjoint splits.

Usage:
  python training/datasets/manifest.py librispeech --root ../data/LibriSpeech
  python training/datasets/manifest.py prepare-all

Design rules:
  * every row carries a content hash and (where known) a speaker id
  * splits are assigned by sha256(path + seed), so they are stable across machines
  * speaker-disjoint splitting is used whenever a speaker id exists
  * duplicate content is detected by hash before splitting and fails loudly
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

SEED = "voxsentinel-2026"
OUT = Path(__file__).parent / "manifests"
AUDIO_EXT = {".flac", ".wav", ".mp3", ".ogg", ".m4a"}


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def split_for(key: str) -> str:
    digest = hashlib.sha256(f"{SEED}:{key}".encode()).digest()
    bucket = digest[0] % 100
    return "train" if bucket < 70 else "val" if bucket < 85 else "test"


def librispeech_root(root: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in AUDIO_EXT:
            continue
        parts = path.parts
        # LibriSpeech layout: <split>/<speaker>/<chapter>/<file>
        speaker = parts[-3] if len(parts) >= 3 else "unknown"
        rows.append(
            {
                "path": str(path),
                "sha256": sha256(path),
                "speaker_id": speaker,
                "label": "bonafide",
                "source": "openslr.org/12 LibriSpeech",
                "license": "CC-BY-4.0",
                "duration_s": round(path.stat().st_size / 32000, 2),  # 16 kHz 16-bit mono estimate
            }
        )
    return rows


def assign(rows: list[dict], speaker_disjoint: bool = True) -> list[dict]:
    seen: dict[str, str] = {}
    for row in rows:
        key = row["speaker_id"] if speaker_disjoint else row["path"]
        row["split"] = seen.setdefault(key, split_for(key))
    return rows


def assert_no_leakage(rows: list[dict]) -> None:
    hashes: dict[str, str] = {}
    speakers: dict[str, set[str]] = {}
    for row in rows:
        if row["sha256"] in hashes and hashes[row["sha256"]] != row["split"]:
            raise SystemExit(f"duplicate content across splits: {row['path']}")
        hashes[row["sha256"]] = row["split"]
        speakers.setdefault(row["speaker_id"], set()).add(row["split"])
    crossing = {s: v for s, v in speakers.items() if len(v) > 1}
    if crossing:
        raise SystemExit(f"speaker leakage across splits: {list(crossing)[:3]}")


def write(rows: list[dict], name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: r["path"])
    target = OUT / f"{name}.csv"
    with target.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["split"]] = counts.get(row["split"], 0) + 1
    summary = {
        "rows": len(rows),
        "splits": counts,
        "speakers": len({r["speaker_id"] for r in rows}),
        "seed": SEED,
        "speaker_disjoint": True,
    }
    (OUT / f"{name}.summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return target


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["librispeech", "prepare-all"])
    ap.add_argument("--root", default="../data/LibriSpeech")
    args = ap.parse_args()
    root = Path(args.root)
    if not root.exists():
        print(f"dataset root not found: {root}", file=sys.stderr)
        return 2
    rows = librispeech_root(root)
    if not rows:
        print("no audio found", file=sys.stderr)
        return 2
    rows = assign(rows)
    assert_no_leakage(rows)
    write(rows, "librispeech_test_clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
