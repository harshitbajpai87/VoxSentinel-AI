"""Train candidate detectors on the multi-dataset manifest.

Candidates (identical preprocessing; selection on VALIDATION only):
  A. enhanced-logistic: delta-MFCC + spectral-shape features, LogisticRegression
  B. enhanced-hgb:     same features, HistGradientBoostingClassifier
  C. (reference) baseline MFCC + LogisticRegression is trained by
     scripts/train_detector.py and compared on the SAME held-out eval data.

Anti-fabrication rules shared with the baseline trainer:
  - the eval partition is NEVER touched here;
  - the scaler is fit on train features only;
  - speaker leakage between train and validation is refused;
  - per-source validation metrics are reported so a candidate that wins only
    by memorizing one dataset is visible;
  - the artifact keeps the exact feature configuration and provenance.

The saved artifact (kind 'mfcc_enhanced') is loadable by the existing
model_loader and served by the same /analyze path: no API change.

Usage:
    python scripts/train_candidate.py --manifest data/multidataset/training_manifest.jsonl \\
        --model-dir models/candidate --seed 42
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import librosa  # noqa: E402

from app.services.features_enhanced import (  # noqa: E402
    EnhancedFeatureConfig,
    extract_enhanced_features,
)
from app.services.manifest import (  # noqa: E402
    ManifestEntry,
    ManifestError,
    assert_no_speaker_leakage,
    load_manifest,
    require_partition_classes,
    summarize,
)
from scripts.train_detector import _read_audio, _validation_metrics, _validation_operating_points  # noqa: E402

MIN_SAMPLES_PER_CLASS = 4
CANDIDATES = ("enhanced-logistic", "enhanced-hgb")


def _extract_all(entries: list[ManifestEntry], dataset_dir: Path,
                 config: EnhancedFeatureConfig) -> np.ndarray:
    rows = []
    for entry in entries:
        mono, sample_rate = _read_audio(entry, dataset_dir)
        rows.append(extract_enhanced_features(mono, sample_rate, config))
    return np.vstack(rows)


def _per_source_metrics(y_true: np.ndarray, scores: np.ndarray,
                        entries: list[ManifestEntry]) -> dict:
    """Validation metrics broken down by data source and by generator."""
    breakdown: dict[str, dict] = {}
    for key_fn, label in ((lambda e: e.source, "by_source"),
                          (lambda e: e.generator_id or "n/a", "by_generator")):
        groups: dict[str, list[int]] = {}
        for index, entry in enumerate(entries):
            groups.setdefault(key_fn(entry), []).append(index)
        for name, indexes in sorted(groups.items()):
            subset_y = y_true[indexes]
            subset_s = scores[indexes]
            if np.unique(subset_y).size < 2:
                breakdown.setdefault(label, {})[name] = {
                    "n": int(subset_y.size),
                    "note": "single class in this group; metrics not computable",
                }
                continue
            breakdown.setdefault(label, {})[name] = {
                "n": int(subset_y.size),
                **_validation_metrics(subset_y, subset_s),
            }
    return breakdown


def main() -> int:
    parser = argparse.ArgumentParser(description="Train candidate detectors on the multi-dataset manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--model-dir", default="models/candidate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--version", default=None)
    parser.add_argument("--candidate", default="enhanced-logistic", choices=CANDIDATES)
    parser.add_argument("--calibrate", action="store_true",
                        help="Wrap the classifier in CalibratedClassifierCV (sigmoid)")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else manifest_path.parent

    try:
        entries = load_manifest(manifest_path)
        train_entries = require_partition_classes(entries, "train")
        validation_entries = require_partition_classes(entries, "validation")
        assert_no_speaker_leakage(train_entries, validation_entries)
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
                print(f"ERROR: only {count} '{label}' samples in {name}. Refusing to train.",
                      file=sys.stderr)
                return 2

    config = EnhancedFeatureConfig()
    print(f"Enhanced feature config: {config.to_dict()} (dim={config.feature_dim})")
    print(f"Extracting features: train={len(train_entries)} validation={len(validation_entries)} ...")
    try:
        x_train = _extract_all(train_entries, dataset_dir, config)
        x_validation = _extract_all(validation_entries, dataset_dir, config)
    except ManifestError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    y_train = np.array([1 if e.label == "synthetic" else 0 for e in train_entries])
    y_validation = np.array([1 if e.label == "synthetic" else 0 for e in validation_entries])

    from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: PLC0415
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    scaler = StandardScaler().fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_validation_scaled = scaler.transform(x_validation)

    if args.candidate == "enhanced-logistic":
        classifier = LogisticRegression(max_iter=1000, class_weight="balanced",
                                        random_state=args.seed).fit(x_train_scaled, y_train)
        classifier_type = "sklearn.linear_model.LogisticRegression"
        classifier_params = {"max_iter": 1000, "class_weight": "balanced",
                             "random_state": args.seed}
    else:
        classifier = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.1, random_state=args.seed,
        ).fit(x_train_scaled, y_train)
        classifier_type = "sklearn.ensemble.HistGradientBoostingClassifier"
        classifier_params = {"max_iter": 200, "learning_rate": 0.1, "random_state": args.seed}

    calibrated = False
    if args.calibrate:
        from sklearn.calibration import CalibratedClassifierCV  # noqa: PLC0415
        from sklearn.frozen import FrozenEstimator  # noqa: PLC0415
        classifier = CalibratedClassifierCV(
            FrozenEstimator(classifier), method="sigmoid"
        )
        classifier.fit(x_validation_scaled, y_validation)
        calibrated = True
        classifier_type = f"CalibratedClassifierCV(sigmoid, frozen)({classifier_type})"
        # NOTE: prefit calibration consumes the validation split; the report
        # marks validation metrics as POST-CALIBRATION and the eval partition
        # remains the only untouched estimate of generalization.

    validation_scores = np.asarray(classifier.predict_proba(x_validation_scaled)[:, 1])
    metrics = _validation_metrics(y_validation, validation_scores)
    if calibrated:
        metrics["note"] = "Validation metrics are post-calibration; validation was used for calibration."
    metrics["operating_points_validation"] = _validation_operating_points(y_validation, validation_scores)
    breakdown = _per_source_metrics(y_validation, validation_scores, validation_entries)

    version = args.version or datetime.now(UTC).strftime("0.3.%Y%m%d%H%M")
    artifact = {
        "format": "voxsentinel-detector",
        "artifact_version": 1,
        "kind": "mfcc_enhanced",
        "model_name": "VoxSentinel Enhanced MFCC Candidate",
        "model_version": version,
        "model_mode": "candidate",
        "classes": ["real", "synthetic"],
        "feature_config": config.to_dict(),
        "calibrated": calibrated,
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
        "model_mode": "candidate",
        "candidate": args.candidate,
        "artifact_path": str(artifact_path),
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "dataset": {
            "manifest": str(manifest_path),
            "sources": sources,
            "summary": summarize(entries),
        },
        "split": {
            "strategy": ("manifest partitions; scaler fit on train only; speaker-disjointness "
                         "enforced between train and validation and against the reserved eval "
                         "partition; group-aware splits from build_multisource_manifest.py"),
            "seed": args.seed,
            "train_n": len(train_entries),
            "validation_n": len(validation_entries),
            "eval_reserved": len(eval_entries),
        },
        "features": config.to_dict(),
        "classifier": {"type": classifier_type, "params": classifier_params,
                       "calibrated": calibrated},
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "librosa": librosa.__version__,
            "platform": platform.platform(),
        },
        "validation_metrics": metrics,
        "validation_breakdown": breakdown,
        "note": (
            "Candidate report. Metrics are validation-only; the final verdict comes from "
            "evaluate_detector.py on the reserved eval partition, compared against the baseline."
        ),
    }
    report_dir = model_dir.parent / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "training_report_candidate.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Artifact: {artifact_path}")
    print(f"Training report: {report_path}")
    print("Validation metrics (NOT final held-out test):")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
