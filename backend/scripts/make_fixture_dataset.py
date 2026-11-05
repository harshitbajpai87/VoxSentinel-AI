"""Generate a tiny SMOKE-FIXTURE dataset for the pipeline.

IMPORTANT — READ THIS:
  The "audio" here is generated signals: harmonic tones stand in for the
  "real" class and band-limited noise for the "synthetic" class. These are
  NOT speech, NOT voice clones, and NOT a voice-cloning dataset. The fixture
  exists only to smoke-test the training pipeline mechanics (manifest
  validation, feature extraction, training, artifact saving, evaluation)
  without downloading anything.

  Metrics produced on this fixture say NOTHING about real synthetic-speech
  detection performance. Real evaluation requires a properly licensed dataset
  such as ASVspoof (see docs/DATASET_GUIDE.md).

Usage:
    python scripts/make_fixture_dataset.py --out-dir data/fixture_smoke \\
        --per-class-per-partition 8
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 8000
DURATION_S = 3.0


def _tone(seed: int) -> np.ndarray:
    """'Real' fixture class: a voiced-tone-like harmonic stack."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(SAMPLE_RATE * DURATION_S)) / SAMPLE_RATE
    f0 = 140.0 + rng.uniform(0, 60)
    signal = sum(np.sin(2 * np.pi * f0 * h * t) / (2 * h) for h in range(1, 6))
    envelope = 0.6 + 0.4 * np.sin(2 * np.pi * 2.0 * t)
    return (0.5 * signal * envelope).astype(np.float32)


def _noise(seed: int) -> np.ndarray:
    """'Synthetic' fixture class: band-limited noise, spectrally flat-ish."""
    rng = np.random.default_rng(seed)
    n = int(SAMPLE_RATE * DURATION_S)
    white = rng.standard_normal(n).astype(np.float32)
    # Simple smoothing filter to shape the spectrum away from the tone class.
    shaped = np.convolve(white, np.ones(9, dtype=np.float32) / 9.0, mode="same")
    return (0.5 * shaped / (np.max(np.abs(shaped)) + 1e-9)).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", required=True, help="Output directory (created if missing)")
    parser.add_argument("--per-class-per-partition", type=int, default=8)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "audio").mkdir(parents=True, exist_ok=True)

    manifest_lines: list[str] = []
    seed = 0
    for partition in ("train", "validation", "eval"):
        for label, generator in (("real", _tone), ("synthetic", _noise)):
            for i in range(args.per_class_per_partition):
                seed += 1
                name = f"{partition}_{label}_{i:03d}.wav"
                sf.write(out_dir / "audio" / name, generator(seed), SAMPLE_RATE, subtype="PCM_16")
                manifest_lines.append(
                    json.dumps(
                        {
                            "path": f"audio/{name}",
                            "label": label,
                            "partition": partition,
                            "speaker_id": f"fixture_{label[0]}{seed}",
                            "source": "fixture_smoke",
                        }
                    )
                )

    manifest_path = out_dir / "manifest.jsonl"
    manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    (out_dir / "FIXTURE_NOTICE.txt").write_text(
        "SMOKE FIXTURES ONLY. These are generated tones and noise, not speech. "
        "Metrics from this fixture validate pipeline mechanics only and must never "
        "be reported as synthetic-speech detection performance.\n",
        encoding="utf-8",
    )

    print(f"Fixture dataset written to {out_dir}")
    print(f"Manifest: {manifest_path} ({len(manifest_lines)} entries)")
    print("NOTE: smoke fixture, NOT voice data. See docs/DATASET_GUIDE.md for real data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
