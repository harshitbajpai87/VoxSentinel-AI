"""Inspect downloaded audio datasets: structure, quality, duplicates.

For every audio file it records: sample rate, channels, duration, format.
It detects EXACT duplicates (sha256 across all sources) and NEAR duplicates
(a coarse spectral fingerprint), so the manifest builder can keep related
recordings in the same split.

Only measured facts are reported. Files that cannot be opened are listed,
never silently dropped.

Usage:
    python scripts/multisource_inspect.py --dataset-dir ../data/<name> \\
        --source <source_name> --output ../data/inspections/<source>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.multisource import sha256_of_file  # noqa: E402

AUDIO_EXTENSIONS = (".wav", ".flac", ".ogg", ".mp3", ".aiff", ".aif", ".opus", ".m4a", ".wma")
FINGERPRINT_TARGET_SR = 16000
FINGERPRINT_SECONDS = 4.0


def audio_fingerprint(path: Path) -> str | None:
    """Coarse near-duplicate fingerprint: 64-band log-mel, median-threshold bits.

    Returns a hex string, or None when the clip cannot be fingerprinted
    (unreadable, too short, silent). Two recordings with Hamming distance
    below ~0.15 on this fingerprint are candidates for near-duplicates; a
    human should confirm before acting on it.
    """
    import librosa  # noqa: PLC0415 - keep the script importable without librosa

    try:
        samples, _ = librosa.load(str(path), sr=FINGERPRINT_TARGET_SR, mono=True,
                                  duration=FINGERPRINT_SECONDS)
    except Exception:  # noqa: BLE001
        return None
    needed = int(FINGERPRINT_TARGET_SR * FINGERPRINT_SECONDS)
    if samples.size < needed // 8 or float(np.max(np.abs(samples))) < 1e-4:
        return None
    if samples.size < needed:
        samples = np.pad(samples, (0, needed - samples.size))
    mel = librosa.feature.melspectrogram(
        y=samples, sr=FINGERPRINT_TARGET_SR, n_mels=64, n_fft=1024, hop_length=512
    )
    log_mel = librosa.power_to_db(mel)
    bits = (log_mel > np.median(log_mel)).astype(np.uint8)
    packed = np.packbits(bits.reshape(-1))
    return packed.tobytes().hex()


def inspect_dataset(dataset_dir: Path, source_name: str, fingerprint_limit: int) -> dict:
    """Walk one dataset directory and measure every audio file."""
    files = sorted(
        p for p in dataset_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )
    per_file: list[dict] = []
    unreadable: list[str] = []
    import soundfile as sf  # noqa: PLC0415

    for path in files:
        relative = str(path.relative_to(dataset_dir))
        try:
            info = sf.info(str(path))
            per_file.append({
                "file": relative,
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "duration_seconds": round(info.duration, 3),
                "format": info.format,
                "size_bytes": path.stat().st_size,
            })
        except Exception as exc:  # noqa: BLE001
            per_file.append({
                "file": relative,
                "sample_rate": None,
                "channels": None,
                "duration_seconds": None,
                "format": None,
                "size_bytes": path.stat().st_size,
                "error": f"{type(exc).__name__}",
            })
            unreadable.append(relative)

    # Exact duplicates by content hash.
    by_hash: dict[str, list[str]] = {}
    for record in per_file:
        if record["sample_rate"] is None:
            continue
        by_hash.setdefault(sha256_of_file(dataset_dir / record["file"]), []).append(record["file"])
    exact_duplicate_clusters = [sorted(v) for v in by_hash.values() if len(v) > 1]

    # Near duplicates by fingerprint (capped for runtime).
    fingerprints: dict[str, list[str]] = {}
    candidates = [r["file"] for r in per_file if r["sample_rate"] is not None]
    for relative in candidates[:fingerprint_limit]:
        digest = audio_fingerprint(dataset_dir / relative)
        if digest:
            fingerprints[digest] = fingerprints.get(digest, []) + [relative]
    near_duplicate_candidates = [sorted(v) for v in fingerprints.values() if len(v) > 1]

    rates = [r["sample_rate"] for r in per_file if r["sample_rate"]]
    durations = [r["duration_seconds"] for r in per_file if r["duration_seconds"]]
    return {
        "report_type": "dataset_inspection",
        "source": source_name,
        "dataset_dir": str(dataset_dir),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "audio_files": len(per_file),
        "unreadable_files": len(unreadable),
        "unreadable_examples": unreadable[:10],
        "sample_rates": dict(sorted(_counts(rates).items())),
        "channels": dict(sorted(_counts(r["channels"] for r in per_file if r["channels"]).items())),
        "duration_seconds": {
            "min": round(min(durations), 3) if durations else None,
            "max": round(max(durations), 3) if durations else None,
            "mean": round(float(np.mean(durations)), 3) if durations else None,
            "total": round(float(np.sum(durations)), 1) if durations else None,
        },
        "exact_duplicate_clusters": exact_duplicate_clusters,
        "exact_duplicate_files": sum(len(c) for c in exact_duplicate_clusters),
        "near_duplicate_candidates": near_duplicate_candidates,
        "fingerprinted_files": min(len(candidates), fingerprint_limit),
        "fingerprint_note": (
            "Near-duplicate candidates share an identical coarse spectral fingerprint; "
            "confirm by listening before excluding anything."
        ),
        "files": per_file,
    }


def _counts(values) -> dict:
    out: dict[str, int] = {}
    for value in values:
        out[str(value)] = out.get(str(value), 0) + 1
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a downloaded audio dataset.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--source", required=True, help="Source name recorded in the report")
    parser.add_argument("--output", required=True)
    parser.add_argument("--fingerprint-limit", type=int, default=2000,
                        help="Max files to fingerprint for near-duplicate detection")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_dir():
        print(f"ERROR: dataset directory not found: {dataset_dir}", file=sys.stderr)
        return 2
    report = inspect_dataset(dataset_dir, args.source, args.fingerprint_limit)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote inspection report for {report['audio_files']} files: {output}")
    print(f"  sample rates: {report['sample_rates']}")
    print(f"  exact duplicate files: {report['exact_duplicate_files']}")
    print(f"  near-duplicate candidate groups: {len(report['near_duplicate_candidates'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
