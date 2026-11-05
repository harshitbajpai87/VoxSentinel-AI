"""Evaluate a trained detector on a held-out manifest partition.

Produces ONLY measured results: class counts, precision/recall/F1 per class,
confusion matrix, FPR/FNR, ROC-AUC and EER. Metrics that cannot be computed
(e.g., single-class split) are reported as null/unavailable, never estimated.

The eval partition must not have been used for training or model selection
(the training script never touches it). The report states its split and
dataset provenance explicitly.

Usage:
    python scripts/evaluate_detector.py --artifact models/detector.joblib \\
        --manifest data/<ds>/manifest.jsonl --partition eval \\
        --report-dir reports
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.detector import DetectorInferenceError  # noqa: E402
from app.services.manifest import (  # noqa: E402
    ManifestEntry,
    ManifestError,
    load_manifest,
    require_partition_classes,
    summarize,
)
from app.services.metrics import equal_error_rate  # noqa: E402
from app.services.model_loader import load_detector_from_path  # noqa: E402
from scripts.train_detector import _read_audio  # noqa: E402


def _fmt(value: float | None) -> str:
    """Format an optional metric; None and NaN are 'unavailable', 0.0 is not."""
    import math

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "unavailable"
    return f"{value:.4f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained detector artifact.")
    parser.add_argument("--artifact", required=True, help="Trained artifact path (trusted, server-side)")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--partition", default="eval", choices=["validation", "eval"])
    parser.add_argument("--report-dir", default="reports")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else manifest_path.parent

    detector = load_detector_from_path(args.artifact)
    if detector is None:
        print(
            f"ERROR: no valid detector artifact at {args.artifact} "
            "(missing or failed validation).",
            file=sys.stderr,
        )
        return 2

    try:
        entries = load_manifest(manifest_path)
        subset = require_partition_classes(entries, args.partition)
    except ManifestError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"Evaluating {len(subset)} samples from partition '{args.partition}' ...")
    features: list[np.ndarray] = []
    labels: list[int] = []
    failed: list[str] = []
    used_entries: list[ManifestEntry] = []
    for entry in subset:  # type: ManifestEntry
        try:
            mono, sample_rate = _read_audio(entry, dataset_dir)
            features.append(extract_features_from(detector, mono, sample_rate))
            labels.append(1 if entry.label == "synthetic" else 0)
            used_entries.append(entry)
        except (ManifestError, DetectorInferenceError) as exc:
            failed.append(f"{entry.path}: {exc}")
    if failed:
        print(f"WARNING: {len(failed)} samples could not be processed (first: {failed[:3]})")

    if not features or not {0, 1} <= set(labels):
        print("ERROR: no usable labelled samples remained; refusing to report metrics.", file=sys.stderr)
        return 2

    x = np.vstack(features)
    scaled = detector._scaler.transform(x)  # noqa: SLF001 - evaluator is trusted tooling
    scores = detector._classifier.predict_proba(scaled)[:, 1]  # noqa: SLF001
    y = np.array(labels)

    from sklearn.metrics import (
        confusion_matrix,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    y_pred = (scores >= 0.5).astype(int)
    labels_order = [0, 1]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, y_pred, labels=labels_order, zero_division=0
    )
    matrix = confusion_matrix(y, y_pred, labels=labels_order).tolist()
    tn, fp, fn, tp = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]

    both_classes = set(np.unique(y).tolist()) == {0, 1}
    report = {
        "report_type": "evaluation_report",
        "model_name": detector.describe().name,
        "model_version": detector.describe().version,
        "evaluated": True,
        "evaluated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "split": {
            "partition": args.partition,
            "note": (
                "Held-out partition: never used for training or model selection "
                "(the training script only reads train/validation)."
            ),
            "n": int(y.size),
            "processed": int(y.size),
            "failed": len(failed),
            "failed_examples": failed[:10],
        },
        "dataset": {
            "manifest": str(manifest_path),
            "sources": sorted({e.source for e in subset}),
            "class_distribution": {
                "real": int((y == 0).sum()),
                "synthetic": int((y == 1).sum()),
            },
            "manifest_summary": summarize(entries),
        },
        "metrics": {
            "positive_class": "synthetic",
            "definitions": {
                "false_positive": "a real (bonafide) clip wrongly flagged as synthetic",
                "false_negative": "a synthetic (spoofed) clip wrongly passed as real",
                "threshold": (
                    "0.5 on the calibrated probability (artifact declares an "
                    "explicit sigmoid calibration fit on validation data)"
                    if getattr(detector, "calibrated", False)
                    else "0.5 on the uncalibrated classifier score (treat as a "
                    "ranking score, not a validated probability)"
                ),
            },
            "precision_synthetic": float(precision[1]),
            "recall_synthetic": float(recall[1]),
            "f1_synthetic": float(f1[1]),
            "precision_real": float(precision[0]),
            "recall_real": float(recall[0]),
            "f1_real": float(f1[0]),
            "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
            "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else None,
            "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else None,
            "roc_auc": float(roc_auc_score(y, scores)) if both_classes else None,
            "eer": equal_error_rate(y, scores) if both_classes else None,
        },
        "breakdown": _group_breakdown(y, scores, used_entries),
    }

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"evaluation_{args.partition}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        f"VoxSentinel evaluation — {detector.describe().name} v{detector.describe().version}",
        f"Partition: {args.partition} | n={int(y.size)} "
        f"(real={int((y == 0).sum())}, synthetic={int((y == 1).sum())})",
        f"Precision(synthetic)={precision[1]:.4f}  Recall(synthetic)={recall[1]:.4f}  F1={f1[1]:.4f}",
        f"Confusion: tn={tn} fp={fp} fn={fn} tp={tp}",
        f"FPR={report['metrics']['false_positive_rate']:.4f} "
        f"FNR={report['metrics']['false_negative_rate']:.4f}",
        f"ROC-AUC={_fmt(report['metrics']['roc_auc'])}",
        f"EER={_fmt(report['metrics']['eer'])}",
        "Per-source: " + "; ".join(
            f"{name} n={data['n']} F1={_fmt(data.get('f1_synthetic'))}"
            for name, data in report["breakdown"]["by_source"].items()
        ),
        f"Report: {report_path}",
    ]
    summary_path = report_dir / f"evaluation_{args.partition}.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


def _group_breakdown(y: np.ndarray, scores: np.ndarray, entries: list) -> dict:
    """Measured metrics per data source and per synthesis generator.

    Groups with a single class get an explicit 'not computable' note instead
    of a fabricated number.
    """
    breakdown: dict = {"by_source": {}, "by_generator": {}}
    for field in ("by_source", "by_generator"):
        groups: dict[str, list[int]] = {}
        for index, entry in enumerate(entries):
            name = entry.source if field == "by_source" else (entry.generator_id or "n/a")
            groups.setdefault(name, []).append(index)
        for name, indexes in sorted(groups.items()):
            group_y = y[indexes]
            group_scores = scores[indexes]
            if set(np.unique(group_y).tolist()) != {0, 1}:
                breakdown[field][name] = {
                    "n": int(group_y.size),
                    "real": int((group_y == 0).sum()),
                    "synthetic": int((group_y == 1).sum()),
                    "note": "single class in this group; metrics not computable",
                }
                continue
            from sklearn.metrics import (  # noqa: PLC0415
                confusion_matrix,
                precision_recall_fscore_support,
                roc_auc_score,
            )
            group_pred = (group_scores >= 0.5).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(
                group_y, group_pred, labels=[0, 1], zero_division=0
            )
            matrix = confusion_matrix(group_y, group_pred, labels=[0, 1]).tolist()
            tn, fp, fn, tp = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]
            breakdown[field][name] = {
                "n": int(group_y.size),
                "real": int((group_y == 0).sum()),
                "synthetic": int((group_y == 1).sum()),
                "precision_synthetic": float(precision[1]),
                "recall_synthetic": float(recall[1]),
                "f1_synthetic": float(f1[1]),
                "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
                "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else None,
                "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else None,
                "roc_auc": float(roc_auc_score(group_y, group_scores)),
            }
    return breakdown


def extract_features_from(detector, mono: np.ndarray, sample_rate: int) -> np.ndarray:
    """Feature extraction for evaluation uses the SAME shared path as inference."""
    from app.services.features import extract_features
    from app.services.features_enhanced import extract_enhanced_features

    if getattr(detector, "_feature_kind", "mfcc_baseline") == "mfcc_enhanced":
        return extract_enhanced_features(mono, sample_rate, detector._config)  # noqa: SLF001
    return extract_features(mono, sample_rate, detector._config)  # noqa: SLF001


if __name__ == "__main__":
    raise SystemExit(main())
