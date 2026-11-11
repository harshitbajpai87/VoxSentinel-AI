"""Artifact loading for the trained detector.

SECURITY: a joblib artifact is a pickle. Loading one can execute arbitrary
code. This loader therefore only ever reads the single server-configured path
(`VOXSENTINEL_DETECTOR_MODEL_PATH`) and expects artifacts produced by this
project's `scripts/train_detector.py`. It is NEVER pointed at user uploads.

Every structural expectation is validated before the detector is reported as
ready: format tag, artifact version, label classes, fitted objects, feature
configuration compatibility and feature dimension. Anything unexpected means
"no model" — the API then serves the honest 503 model-unavailable state.
"""
from __future__ import annotations

import logging
from pathlib import Path

from app.services.artifact_compat import (
    ArtifactLoadError,
    dependency_fingerprint,
    load_artifact,
)
from app.services.detector import TrainedVoiceDetector
from app.services.features import FeatureConfig, feature_names
from app.services.features_enhanced import EnhancedFeatureConfig, enhanced_feature_names

logger = logging.getLogger("voxsentinel.model_loader")

ARTIFACT_FORMAT = "voxsentinel-detector"
SUPPORTED_ARTIFACT_VERSION = 1
# kind -> feature-configuration class. New kinds extend this mapping; the
# baseline kind stays byte-compatible with all previously shipped artifacts.
KIND_CONFIGS = {
    "mfcc_baseline": FeatureConfig,
    "mfcc_enhanced": EnhancedFeatureConfig,
}
# Backward-compat alias: the baseline trainer writes (and older code reads)
# this constant. The baseline kind remains the default production artifact kind.
SUPPORTED_KIND = "mfcc_baseline"

KIND_FEATURE_NAMES = {
    "mfcc_baseline": feature_names,
    "mfcc_enhanced": enhanced_feature_names,
}

REQUIRED_KEYS = (
    "format",
    "artifact_version",
    "kind",
    "model_name",
    "model_version",
    "feature_config",
    "classes",
    "scaler",
    "classifier",
)


def _invalid(reason: str, path: Path) -> None:
    """Log why an artifact was refused. The API falls back to NullVoiceDetector."""
    logger.warning("Detector artifact %s rejected: %s", path, reason)
    _LAST_STATUS.update({"path": str(path), "loaded": False, "error": reason})


# Result of the last load attempt, reported by /health so an operator sees the
# real reason instead of inferring it from a degraded model.
_LAST_STATUS: dict[str, object] = {"path": None, "loaded": False, "error": None}


def artifact_status() -> dict:
    """The last artifact load attempt, plus the versions that decide loading.

    The fingerprint makes an ImportError actionable: comparing these numbers
    with the environment that trained the artifact names the mismatch.
    """
    status = dict(_LAST_STATUS)
    status["dependencies"] = dependency_fingerprint()
    return status


def reset_artifact_status() -> None:
    """Forget the previous attempt (tests and explicit reloads)."""
    _LAST_STATUS.update({"path": None, "loaded": False, "error": None})


def _validate_fitted(obj: object, name: str, expected_features: int) -> bool:
    """A fitted sklearn estimator always exposes n_features_in_.

    CalibratedClassifierCV wrappers may not expose it directly; in that case
    the wrapped estimators must.
    """
    fitted_dim = getattr(obj, "n_features_in_", None)
    if fitted_dim is None:
        inner = getattr(obj, "calibrated_classifiers_", None) or []
        inner_estimators = [
            getattr(c, "estimator", None) for c in inner
        ] + [getattr(e, "base_estimator", None) for e in inner if hasattr(e, "base_estimator")]
        dims = [getattr(e, "n_features_in_", None) for e in inner_estimators if e is not None]
        if dims and all(d == expected_features for d in dims):
            return True
        return False
    if fitted_dim is None:
        _invalid(f"{name} is not fitted (no n_features_in_)", Path("(memory)"))
        return False
    if int(fitted_dim) != expected_features:
        _invalid(
            f"{name} expects {fitted_dim} features, config provides {expected_features}",
            Path("(memory)"),
        )
        return False
    if not hasattr(obj, "transform" if name == "scaler" else "predict_proba"):
        _invalid(f"{name} lacks the expected API", Path("(memory)"))
        return False
    return True


def _logistic_explanation_fn(classifier, feature_config: FeatureConfig, names: list[str]):
    """Explain predictions via standardized logistic-regression contributions.

    Honest scope: this lists the features that pushed the score up or down.
    It is a linear model's own weights — it is NOT causal proof and not a
    deep-learning saliency method.
    """

    def explain(features, scaled) -> str:
        import numpy as np

        coefficients = getattr(classifier, "coef_", None)
        if coefficients is None:
            raise RuntimeError("classifier has no coef_")
        row = np.asarray(scaled).reshape(-1)
        contributions = np.asarray(coefficients).reshape(-1) * row
        order = np.argsort(contributions)[::-1]
        top = [f"{names[i]} ({contributions[i]:+.2f})" for i in order[:3]]
        bottom = [f"{names[i]} ({contributions[i]:+.2f})" for i in order[-2:]]
        return (
            "Top standardized features raising the score: " + ", ".join(top)
            + ". Strongest reducing: " + ", ".join(bottom)
            + ". Linear-model contributions, not causal evidence."
        )

    return explain


def load_detector_from_path(path: str | Path) -> TrainedVoiceDetector | None:
    """Return a validated TrainedVoiceDetector, or None (never a half-model)."""
    artifact_path = Path(path)
    if not artifact_path.is_file():
        logger.info("No detector artifact at %s; model stays unavailable.", artifact_path)
        return None

    try:
        artifact = load_artifact(artifact_path)
    except ArtifactLoadError as exc:
        # Version / missing-module problems: report the actionable detail, not
        # just the exception class.
        _invalid(f"could not load artifact: {exc.detail}", artifact_path)
        return None
    except Exception as exc:  # noqa: BLE001 - any failure means "no model"
        _invalid(
            f"could not load artifact ({type(exc).__name__}: {exc})", artifact_path
        )
        return None

    if not isinstance(artifact, dict):
        _invalid("artifact is not a dictionary", artifact_path)
        return None

    missing = [key for key in REQUIRED_KEYS if key not in artifact]
    if missing:
        _invalid(f"missing keys {missing}", artifact_path)
        return None
    if artifact["format"] != ARTIFACT_FORMAT:
        _invalid(f"unexpected format {artifact['format']!r}", artifact_path)
        return None
    if artifact["artifact_version"] != SUPPORTED_ARTIFACT_VERSION:
        _invalid(f"unsupported artifact version {artifact['artifact_version']!r}", artifact_path)
        return None
    kind = artifact["kind"]
    if kind not in KIND_CONFIGS:
        _invalid(f"unsupported detector kind {kind!r}", artifact_path)
        return None

    classes = artifact["classes"]
    if not isinstance(classes, (list, tuple)) or list(classes) != ["real", "synthetic"]:
        _invalid(f"classes must be ['real', 'synthetic'], got {classes!r}", artifact_path)
        return None
    # The training pipeline encodes: real=0, synthetic=1 (classes_ is [0, 1]).
    synthetic_index = list(classes).index("synthetic")

    try:
        config = KIND_CONFIGS[kind].from_dict(dict(artifact["feature_config"]))
    except Exception as exc:  # noqa: BLE001
        _invalid(f"invalid feature configuration ({exc})", artifact_path)
        return None

    scaler = artifact["scaler"]
    classifier = artifact["classifier"]
    if not _validate_fitted(scaler, "scaler", config.feature_dim):
        _invalid("scaler is not a fitted transformer matching the feature config", artifact_path)
        return None
    if not _validate_fitted(classifier, "classifier", config.feature_dim):
        _invalid("classifier is not fitted or does not match the feature config", artifact_path)
        return None
    model_classes = list(getattr(classifier, "classes_", []))
    try:
        model_classes = [int(c) for c in model_classes]
    except (TypeError, ValueError):
        model_classes = []
    if model_classes != [0, 1]:
        _invalid(
            f"classifier classes {model_classes!r} do not match the training encoding [0, 1]",
            artifact_path,
        )
        return None

    names_fn = KIND_FEATURE_NAMES[kind]
    explanation_fn = (
        _logistic_explanation_fn(classifier, config, names_fn(config))
        if hasattr(classifier, "coef_")
        else None
    )

    logger.info(
        "Loaded detector artifact %s (%s v%s).",
        artifact_path,
        artifact["model_name"],
        artifact["model_version"],
    )
    _LAST_STATUS.update({"path": str(artifact_path), "loaded": True, "error": None})
    return TrainedVoiceDetector(
        classifier=classifier,
        scaler=scaler,
        feature_config=config,
        model_name=str(artifact["model_name"]),
        model_version=str(artifact["model_version"]),
        synthetic_index=synthetic_index,
        model_mode=str(artifact.get("model_mode", "baseline")),
        explanation_fn=explanation_fn,
        feature_kind=kind,
        calibrated=bool(artifact.get("calibrated", False)),
    )
