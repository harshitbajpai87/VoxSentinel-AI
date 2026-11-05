"""Train the VoxSentinel baseline detector on REAL labelled audio.

Pipeline:
  manifest -> validate -> extract MFCC features -> fit scaler on TRAIN only
  -> train LogisticRegression -> validate on held-out partition
  -> save artifact (joblib) + machine-readable training report (JSON).

Leakage protection:
  - scaler is fit on train features only;
  - speaker ids must not overlap between train and validation (enforced);
  - the eval partition is NEVER touched here — it is reserved for final
    held-out evaluation by scripts/evaluate_detector.py.

Reproducibility: a single --seed drives both data handling and the estimator.
Package versions, feature configuration and manifest counts are recorded in
the training report.

Usage:
    python scripts/train_detector.py --manifest data/<ds>/manifest.jsonl \\
        --model-dir models --seed 42
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import sklearn
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import librosa  # noqa: E402

from app.services.features import FeatureConfig, extract_features  # noqa: E402
from app.services.manifest import (  # noqa: E402
    ManifestEntry,
    ManifestError,
    assert_no_speaker_leakage,
    load_manifest,
    require_partition_classes,
    summarize,
)
from app.services.model_loader import ARTIFACT_FORMAT, SUPPORTED_ARTIFACT_VERSION, SUPPORTED_KIND  # noqa: E402

MIN_SAMPLES_PER_CLASS = 4


def _read_audio(entry: ManifestEntry, dataset_dir: Path) -> tuple[np.ndarray, int]:
    path = entry.resolved_path(dataset_dir)
    if not path.is_file():
        raise ManifestError(f"Missing audio file: {path}")
    try:
        with sf.SoundFile(path) as handle:
            sample_rate = int(handle.samplerate)
            channels = int(handle.channels)
            data = handle.read(dtype="float32", always_2d=True)
    except Exception as exc:  # noqa: BLE001
        raise ManifestError(f"Unreadable audio {path}: {exc}") from exc
    if data.shape[0] < 2:
        raise ManifestError(f"Too-short audio {path}")
    mono = data.mean(axis=1).astype(np.float32) if channels > 1 else data.reshape(-1)
    return mono, sample_rate


def _extract_all(entries: list[ManifestEntry], dataset_dir: Path, config: FeatureConfig) -> np.ndarray:
    rows = []
    for entry in entries:
        mono, sample_rate = _read_audio(entry, dataset_dir)
        rows.append(extract_features(mono, sample_rate, config))
    return np.vstack(rows)


def _validation_operating_points(y_true: np.ndarray, scores: np.ndarray) -> dict[str, object]:
    """Threshold analysis on VALIDATION only (never on the final eval set)."""
    from app.services.metrics import best_youden_threshold, equal_error_rate

    youden = best_youden_threshold(y_true, scores)
    return {
        "youden_threshold": youden[0] if youden else None,
        "youden_j": youden[1] if youden else None,
        "eer": equal_error_rate(y_true, scores),
    }


def _validation_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, object]:
    """Metrics on the validation split — reported separately from any final test."""
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score

    y_pred = (scores >= 0.5).astype(int)
    labels = [0, 1]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    tn, fp, fn, tp = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]
    metrics: dict[str, object] = {
        "n": int(y_true.size),
        "n_real": int((y_true == 0).sum()),
        "n_synthetic": int((y_true == 1).sum()),
        "precision_synthetic": float(precision[1]),
        "recall_synthetic": float(recall[1]),
        "f1_synthetic": float(f1[1]),
        "precision_real": float(precision[0]),
        "recall_real": float(recall[0]),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else None,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else None,
        "roc_auc": None,
    }
    if set(np.unique(y_true).tolist()) == {0, 1}:
        metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the VoxSentinel baseline detector.")
    parser.add_argument("--manifest", required=True, help="Manifest JSONL from prepare_dataset.py")
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Root the manifest paths are relative to (default: manifest's directory)",
    )
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--version", default=None, help="Model version label (default: timestamp)")
    parser.add_argument(
        "--leakage-exempt-sources",
        default="",
        help=(
            "Comma-separated data sources exempt from the train/validation "
            "speaker-leakage check (e.g. piper_tts: same TTS voice may span "
            "train and validation by design; eval must still be disjoint)."
        ),
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else manifest_path.parent

    exempt = tuple(x.strip() for x in args.leakage_exempt_sources.split(",") if x.strip())
    try:
        entries = load_manifest(manifest_path)
        train_entries = require_partition_classes(entries, "train")
        validation_entries = require_partition_classes(entries, "validation")
        assert_no_speaker_leakage(train_entries, validation_entries, exempt_sources=exempt)
        # The final eval partition must be disjoint from ALL training speakers,
        # exempt sources included (an unseen generator is the point of eval).
        eval_entries = [e for e in entries if e.partition == "eval"]
        if eval_entries:
            assert_no_speaker_leakage(train_entries, eval_entries)
            assert_no_speaker_leakage(validation_entries, eval_entries)
    except ManifestError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    for name, subset in (("train", train_entries), ("validation", validation_entries)):
        for label in ("real", "synthetic"):
            count = sum(1 for e in subset if e.label == label)
            if count < MIN_SAMPLES_PER_CLASS:
                print(
                    f"ERROR: only {count} '{label}' samples in {name} "
                    f"(need >= {MIN_SAMPLES_PER_CLASS}). Refusing to train.",
                    file=sys.stderr,
                )
                return 2

    config = FeatureConfig()
    print(f"Feature config: {config.to_dict()}")
    print(f"Extracting features: train={len(train_entries)} validation={len(validation_entries)} ...")
    try:
        x_train = _extract_all(train_entries, dataset_dir, config)
        x_validation = _extract_all(validation_entries, dataset_dir, config)
    except ManifestError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    y_train = np.array([1 if e.label == "synthetic" else 0 for e in train_entries])
    y_validation = np.array([1 if e.label == "synthetic" else 0 for e in validation_entries])

    # Scaler is fit on TRAIN data only.
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(x_train)
    classifier = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=args.seed
    ).fit(scaler.transform(x_train), y_train)

    if list(classifier.classes_) != [0, 1]:
        print("ERROR: unexpected class order.", file=sys.stderr)
        return 2

    validation_scores = classifier.predict_proba(scaler.transform(x_validation))[:, 1]
    metrics = _validation_metrics(y_validation, validation_scores)
    operating_points = _validation_operating_points(y_validation, validation_scores)
    metrics["operating_points_validation"] = operating_points

    version = args.version or datetime.now(UTC).strftime("0.1.%Y%m%d%H%M")
    artifact = {
        "format": ARTIFACT_FORMAT,
        "artifact_version": SUPPORTED_ARTIFACT_VERSION,
        "kind": SUPPORTED_KIND,
        "model_name": "VoxSentinel MFCC Baseline",
        "model_version": version,
        "model_mode": "baseline",
        "classes": ["real", "synthetic"],
        "feature_config": config.to_dict(),
        "scaler": scaler,
        "classifier": classifier,
    }

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = model_dir / "detector.joblib"
    joblib.dump(artifact, artifact_path)

    sources = sorted({e.source for e in train_entries + validation_entries})
    report = {
        "report_type": "training_report",
        "model_name": artifact["model_name"],
        "model_version": version,
        "artifact_path": str(artifact_path),
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "dataset": {
            "manifest": str(manifest_path),
            "sources": sources,
            "summary": summarize(entries),
            "note": (
                "Counts describe the manifest used. Interpret metrics ONLY against the "
                "dataset they came from — this baseline is not claimed to detect "
                "synthesis methods it was never evaluated on."
            ),
        },
        "split": {
            "strategy": "manifest partitions; scaler fit on train only; "
            "speaker-disjointness enforced between train and validation "
            "and against the reserved eval partition",
            "leakage_exempt_sources": list(exempt),
            "leakage_note": (
                "Exempt sources may share a generator voice between train and "
                "validation (ASVspoof-style); eval speakers and generators are "
                "always disjoint from training."
            ),
            "seed": args.seed,
            "train_n": len(train_entries),
            "validation_n": len(validation_entries),
            "eval_reserved": sum(1 for e in entries if e.partition == "eval"),
        },
        "features": config.to_dict(),
        "classifier": {
            "type": "sklearn.linear_model.LogisticRegression",
            "params": {"max_iter": 1000, "class_weight": "balanced", "random_state": args.seed},
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "librosa": librosa.__version__,
            "platform": platform.platform(),
        },
        "validation_metrics": metrics,
    }
    report_dir = model_dir.parent / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Artifact: {artifact_path}")
    print(f"Training report: {report_path}")
    print("Validation metrics (NOT final held-out test):")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
