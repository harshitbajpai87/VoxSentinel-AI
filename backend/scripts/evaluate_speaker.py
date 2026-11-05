#!/usr/bin/env python
"""Calibrate + evaluate the speaker-verification pipeline on REAL speech.

Builds labelled speaker pairs from LibriSpeech (readers/chapters = ground
truth identities and recording sessions):

* dev-clean  pairs -> select the EER decision threshold (calibration)
* test-clean pairs -> final held-out evaluation at that threshold

Same-speaker pairs use DIFFERENT chapters of the same reader (genuinely
different recording sessions), never the same file twice.

Outputs:
  models/speaker_thresholds.json   (decision artifact consumed by the API)
  reports/speaker_eval.json       (full metrics report)

All numbers are computed from real model inference on real speech; none are
invented. Requirements: data/LibriSpeech/{dev-clean,test-clean} (see
docs/DATASET_GUIDE.md) and models/speaker_ecapa (downloaded once).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.services.metrics import equal_error_rate, roc_auc_score_safe  # noqa: E402
from app.services.speaker_model import EcapaSpeakerEmbedder  # noqa: E402

SEGMENT_SECONDS = 8.0


def read_reader_chapters(subset_dir: Path, max_readers: int) -> dict[str, list[Path]]:
    """Map reader_id -> first two chapter files (sorted for determinism)."""
    readers: dict[str, list[Path]] = {}
    for reader_dir in sorted(p for p in subset_dir.iterdir() if p.is_dir()):
        chapters = sorted(reader_dir.rglob("*.flac"))
        if len(chapters) >= 2:
            readers[reader_dir.name] = chapters[:2]
        if len(readers) >= max_readers:
            break
    return readers


def load_segment(path: Path, sample_rate_target: int) -> np.ndarray:
    data, sr = sf.read(str(path), frames=int(sr_probe(path) * SEGMENT_SECONDS),
                       dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != sample_rate_target:
        # reuse the service's deterministic linear resampler
        from app.services.speaker_model import _resample_linear

        mono = _resample_linear(mono, sr, sample_rate_target)
    return mono


_sr_cache: dict[Path, int] = {}


def sr_probe(path: Path) -> int:
    if path not in _sr_cache:
        _sr_cache[path] = sf.info(str(path)).samplerate
    return _sr_cache[path]


def build_pairs(subset_dir: Path, embedder: EcapaSpeakerEmbedder, max_readers: int) -> dict:
    readers = read_reader_chapters(subset_dir, max_readers)
    if len(readers) < 2:
        raise SystemExit(f"ERROR: need >=2 readers with >=2 chapters under {subset_dir}")

    ids = sorted(readers)
    same_scores: list[float] = []
    diff_scores: list[float] = []

    chapter_embeddings: dict[tuple[str, int], np.ndarray] = {}

    def emb(reader: str, chapter_idx: int) -> np.ndarray:
        key = (reader, chapter_idx)
        if key not in chapter_embeddings:
            chapter_embeddings[key] = embedder.embed(
                load_segment(readers[reader][chapter_idx], 16000),
                16000,
            )
        return chapter_embeddings[key]

    for i, reader in enumerate(ids):
        # same speaker, different chapters (different sessions)
        a, b = emb(reader, 0), emb(reader, 1)
        same_scores.append(float(cosine(a, b)))
        # different speakers (reader i vs the next reader, wrap-around)
        partner = ids[(i + 1) % len(ids)]
        if partner != reader:
            diff_scores.append(float(cosine(a, emb(partner, 0))))

    return {
        "readers": len(ids),
        "same_scores": same_scores,
        "diff_scores": diff_scores,
    }


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def pick_eer_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Threshold at the EER operating point (where FAR == FRR)."""
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    # thresholds[0] is np.inf; guard
    if idx >= len(thresholds) or not np.isfinite(thresholds[idx]):
        idx = len(thresholds) - 1
    return float(thresholds[idx]), eer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--librispeech", default="data/LibriSpeech", type=Path)
    parser.add_argument("--max-readers", default=40, type=int)
    parser.add_argument("--report-dir", default=None, type=Path)
    args = parser.parse_args()

    settings = get_settings()
    report_dir = args.report_dir or Path(settings.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    embedder = EcapaSpeakerEmbedder(settings.speaker_model_dir)
    if not embedder.is_ready():
        raise SystemExit(
            f"ERROR: ECAPA model not found under {settings.speaker_model_dir}. "
            "See docs/DATASET_GUIDE.md for the one-time download."
        )

    print("Building dev-clean calibration pairs ...")
    dev = build_pairs(args.librispeech / "dev-clean", embedder, args.max_readers)
    print("Building test-clean evaluation pairs ...")
    test = build_pairs(args.librispeech / "test-clean", embedder, args.max_readers)

    dev_scores = np.array(dev["same_scores"] + dev["diff_scores"])
    dev_labels = np.array([1] * len(dev["same_scores"]) + [0] * len(dev["diff_scores"]))
    threshold, dev_eer = pick_eer_threshold(dev_scores, dev_labels)

    test_scores = np.array(test["same_scores"] + test["diff_scores"])
    test_labels = np.array([1] * len(test["same_scores"]) + [0] * len(test["diff_scores"]))

    # final held-out metrics at the dev-calibrated threshold
    accepted = test_scores >= threshold
    tp = int(np.sum(accepted & (test_labels == 1)))
    fp = int(np.sum(accepted & (test_labels == 0)))
    fn = int(np.sum(~accepted & (test_labels == 1)))
    tn = int(np.sum(~accepted & (test_labels == 0)))
    far = fp / (fp + tn) if (fp + tn) else None
    frr = fn / (fn + tp) if (fn + tp) else None
    eer_held = equal_error_rate(test_labels, test_scores)
    auc = roc_auc_score_safe(test_labels, test_scores)

    thresholds_artifact = {
        "format": "voxsentinel-speaker-thresholds",
        "artifact_version": 1,
        "metric": "cosine_similarity",
        "eer_threshold": threshold,
        "calibration_eer": dev_eer,
        "calibrated_on": "LibriSpeech dev-clean speaker pairs (different chapters)",
        "calibration_pairs": int(dev_labels.size),
        "created_at": datetime.now(UTC).isoformat(),
    }
    Path(settings.speaker_thresholds_path).write_text(json.dumps(thresholds_artifact, indent=2))

    report = {
        "report_type": "speaker_verification_report",
        "embedding_model": "speechbrain/spkrec-ecapa-voxceleb (ECAPA-TDNN, 192-dim)",
        "similarity_definition": (
            "raw cosine similarity between 192-dim ECAPA embeddings — NOT a "
            "probability of same-speaker identity"
        ),
        "positive_class": "same_speaker",
        "calibration": {
            "subset": "dev-clean",
            "pairs": int(dev_labels.size),
            "same_pairs": len(dev["same_scores"]),
            "different_pairs": len(dev["diff_scores"]),
            "eer_threshold": threshold,
            "eer": dev_eer,
        },
        "held_out_evaluation": {
            "subset": "test-clean (speakers disjoint from calibration)",
            "pairs": int(test_labels.size),
            "same_pairs": len(test["same_scores"]),
            "different_pairs": len(test["diff_scores"]),
            "threshold": threshold,
            "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            "false_acceptance_rate": far,
            "false_rejection_rate": frr,
            "eer": eer_held,
            "roc_auc": auc,
        },
        "scope_note": (
            "Evaluated on clean LibriSpeech read speech in 8-second segments; "
            "performance on telephone/noisy/in-the-wild audio is unknown and not claimed."
        ),
        "generated_at": datetime.now(UTC).isoformat(),
    }
    out = report_dir / "speaker_eval.json"
    out.write_text(json.dumps(report, indent=2))

    print(f"\nCalibration (dev-clean): {len(dev_labels)} pairs, EER {dev_eer:.4f}")
    print(f"  -> decision threshold (cosine): {threshold:.4f}")
    print(f"\nHeld-out (test-clean): {len(test_labels)} pairs")
    print(f"  EER={eer_held:.4f}  ROC-AUC={auc:.4f}  FAR={far:.4f}  FRR={frr:.4f}")
    print(f"  confusion: tp={tp} fp={fp} fn={fn} tn={tn}")
    print(f"\nThresholds artifact: {settings.speaker_thresholds_path}")
    print(f"Report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
