"""Shared evaluation metrics for training and held-out evaluation.

Used by scripts/train_detector.py (validation analysis) and
scripts/evaluate_detector.py (final metrics) so both report identical,
correctly-computed numbers.
"""
from __future__ import annotations

import numpy as np


def roc_auc_score_safe(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    """ROC-AUC when both classes are present, else None (never fabricated)."""
    import numpy as _np

    if _np.unique(y_true).size < 2:
        return None
    try:
        from sklearn.metrics import roc_auc_score

        value = float(roc_auc_score(y_true, scores))
        return value if 0.0 <= value <= 1.0 and _np.isfinite(value) else None
    except ValueError:
        return None


def equal_error_rate(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    """Equal error rate: the operating point where FAR == FRR on the ROC.

    Computed by finding the crossing of the (linearly interpolated) ROC
    curve with the line TPR = 1 - FPR, i.e. solving
    fpr + tpr(fpr) - 1 = 0 with Brent's method. This handles flat ROC
    segments (ties) correctly, unlike argmin-based approximations which
    can pick a wrong bracketing point when |fpr - fnr| has two equal minima.

    Returns a fraction in [0, 1], or None when undefined (single class or
    degenerate scores).
    """
    from scipy.optimize import brentq
    from sklearn.metrics import roc_curve

    y_true = np.asarray(y_true)
    if set(np.unique(y_true).tolist()) != {0, 1}:
        return None

    fpr, tpr, _ = roc_curve(y_true, scores)

    def crossing(x: float) -> float:
        # tpr interpolated at fpr=x; zero where fpr == fnr (= 1 - tpr).
        return float(np.interp(x, fpr, tpr) + x - 1.0)

    try:
        eer = brentq(crossing, 0.0, 1.0)
    except (ValueError, RuntimeError):
        # Degenerate score distributions have no unique crossing.
        return None
    if not np.isfinite(eer) or not 0.0 <= eer <= 1.0:
        return None
    return float(eer)


def best_youden_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float] | None:
    """Threshold maximizing Youden's J (TPR - FPR) — validation-only tooling."""
    from sklearn.metrics import roc_curve

    if set(np.unique(y_true).tolist()) != {0, 1}:
        return None
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    j = tpr - fpr
    best = int(np.argmax(j))
    return float(thresholds[best]), float(j[best])


def compute_binary_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.5,
    positive_class: str = "synthetic",
    max_curve_points: int = 60,
) -> dict:
    """Complete, honestly-computed binary metrics from real predictions.

    Positive class: synthetic/spoofed. ``scores`` are raw model scores; the
    threshold is applied for confusion-matrix metrics. Curve metrics that
    are undefined (single class, NaNs) come back as None — never invented.
    """

    import numpy as np

    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if y_true.size == 0:
        raise ValueError("no evaluation samples")
    finite = np.isfinite(scores)
    if not finite.all():
        raise ValueError("non-finite scores in evaluation output")

    preds = (scores >= threshold).astype(int)
    tp = int(np.sum((preds == 1) & (y_true == 1)))
    fp = int(np.sum((preds == 1) & (y_true == 0)))
    fn = int(np.sum((preds == 0) & (y_true == 1)))
    tn = int(np.sum((preds == 0) & (y_true == 0)))
    n_pos, n_neg = tp + fn, tn + fp

    def _safe_div(a: float, b: float) -> float | None:
        return float(a / b) if b else None

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = (
        float(2 * precision * recall / (precision + recall))
        if precision and recall and (precision + recall)
        else None
    )
    spec = _safe_div(tn, tn + fp)
    acc = _safe_div(tp + tn, y_true.size)

    both_classes = len(np.unique(y_true)) == 2
    roc_auc = roc_auc_score_safe(y_true, scores) if both_classes else None
    eer = equal_error_rate(y_true, scores) if both_classes else None
    average_precision = None
    if both_classes:
        try:
            from sklearn.metrics import average_precision_score

            average_precision = float(average_precision_score(y_true, scores))
        except ValueError:
            average_precision = None

    roc_curve_points, pr_curve_points = [], []
    if both_classes:
        from sklearn.metrics import precision_recall_curve, roc_curve

        fpr, tpr, _ = roc_curve(y_true, scores)
        step = max(1, len(fpr) // max_curve_points)
        roc_curve_points = [
            {"fpr": float(a), "tpr": float(b)} for a, b in zip(fpr[::step], tpr[::step], strict=False)
        ]
        pr, rc, _ = precision_recall_curve(y_true, scores)
        step = max(1, len(pr) // max_curve_points)
        pr_curve_points = [
            {"recall": float(a), "precision": float(b)} for a, b in zip(rc[::step], pr[::step], strict=False)
        ]

    return {
        "positive_class": positive_class,
        "score_semantics": "raw uncalibrated model score",
        "threshold": threshold,
        "n": int(y_true.size),
        "class_distribution": {"real": int(n_neg), "synthetic": int(n_pos)},
        "accuracy": acc,
        "precision_synthetic": precision,
        "recall_synthetic": recall,
        "f1_synthetic": f1,
        "specificity_real": spec,
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "false_positive_rate": _safe_div(fp, n_neg),
        "false_negative_rate": _safe_div(fn, n_pos),
        "roc_auc": roc_auc,
        "eer": eer,
        "average_precision": average_precision,
        "roc_curve": roc_curve_points,
        "pr_curve": pr_curve_points,
        "all_metrics_defined": all(
            v is not None for v in (roc_auc, eer, precision, recall, f1, acc)
        ),
    }
