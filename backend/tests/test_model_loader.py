"""Artifact loading: strict validation before any model is reported ready."""
from __future__ import annotations

import joblib
import numpy as np
import pytest

from app.services.model_loader import ARTIFACT_FORMAT, load_detector_from_path


def valid_artifact_skeleton() -> dict:
    """A minimal structurally-valid artifact shell (fitted objects added by tests)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    x = np.random.default_rng(0).normal(size=(10, 40))
    y = np.array([0, 1] * 5)
    scaler = StandardScaler().fit(x)
    classifier = LogisticRegression().fit(scaler.transform(x), y)
    return {
        "format": ARTIFACT_FORMAT,
        "artifact_version": 1,
        "kind": "mfcc_baseline",
        "model_name": "test-artifact",
        "model_version": "9.9",
        "classes": ["real", "synthetic"],
        "feature_config": {
            "target_sample_rate": 16000,
            "segment_seconds": 4.0,
            "n_mfcc": 20,
            "n_fft": 1024,
            "hop_length": 256,
            "n_mels": 128,
            "fmin": 0.0,
            "fmax": 8000.0,
        },
        "scaler": scaler,
        "classifier": classifier,
    }


@pytest.fixture
def artifact_dir(tmp_path):
    return tmp_path


def dump(artifact_dir, name, payload):
    path = artifact_dir / name
    joblib.dump(payload, path)
    return path


def test_missing_artifact_returns_none(tmp_path):
    assert load_detector_from_path(tmp_path / "absent.joblib") is None


def test_non_dict_artifact_returns_none(artifact_dir):
    path = dump(artifact_dir, "list.joblib", [1, 2, 3])
    assert load_detector_from_path(path) is None


def test_missing_keys_returns_none(artifact_dir):
    for missing_key in ("format", "classes", "classifier", "feature_config"):
        artifact = valid_artifact_skeleton()
        del artifact[missing_key]
        path = dump(artifact_dir, f"no_{missing_key}.joblib", artifact)
        assert load_detector_from_path(path) is None, missing_key


def test_wrong_format_version_or_kind_returns_none(artifact_dir):
    for key, bad_value in (
        ("format", "something-else"),
        ("artifact_version", 99),
        ("kind", "unheard-of"),
    ):
        artifact = valid_artifact_skeleton()
        artifact[key] = bad_value
        path = dump(artifact_dir, f"bad_{key}.joblib", artifact)
        assert load_detector_from_path(path) is None, key


def test_wrong_classes_returns_none(artifact_dir):
    artifact = valid_artifact_skeleton()
    artifact["classes"] = ["fake", "real"]
    path = dump(artifact_dir, "bad_classes.joblib", artifact)
    assert load_detector_from_path(path) is None

    artifact = valid_artifact_skeleton()
    artifact["classes"] = ["real", "synthetic", "maybe"]
    path = dump(artifact_dir, "bad_classes2.joblib", artifact)
    assert load_detector_from_path(path) is None


def test_feature_config_incompatibility_returns_none(artifact_dir):
    artifact = valid_artifact_skeleton()
    artifact["feature_config"] = {"n_mfcc": 13, "unknown_key": True}
    path = dump(artifact_dir, "bad_config.joblib", artifact)
    assert load_detector_from_path(path) is None


def test_dimension_mismatch_between_scaler_and_config_returns_none(artifact_dir):
    artifact = valid_artifact_skeleton()
    artifact["feature_config"]["n_mfcc"] = 10  # dim 20 != scaler's 40
    path = dump(artifact_dir, "dim_mismatch.joblib", artifact)
    assert load_detector_from_path(path) is None


def test_unfitted_estimator_returns_none(artifact_dir):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    artifact = valid_artifact_skeleton()
    artifact["scaler"] = StandardScaler()  # never fitted
    path = dump(artifact_dir, "unfitted.joblib", artifact)
    assert load_detector_from_path(path) is None

    artifact = valid_artifact_skeleton()
    artifact["classifier"] = LogisticRegression()
    path = dump(artifact_dir, "unfitted_clf.joblib", artifact)
    assert load_detector_from_path(path) is None


def test_valid_artifact_loads_and_infers(artifact_dir):

    path = dump(artifact_dir, "good.joblib", valid_artifact_skeleton())
    detector = load_detector_from_path(path)
    assert detector is not None
    assert detector.is_ready() is True

    t = np.arange(16000 * 3) / 16000
    audio = (0.5 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    result = detector.analyze(audio, 16000, 1)
    assert 0.0 <= result.raw_score <= 1.0
    assert np.isfinite(result.raw_score)
    assert result.model_name == "test-artifact"
    assert result.explanation is not None and "mfcc" in result.explanation


def test_trained_script_artifact_loads(trained_model_dir):
    """The artifact produced by the real training CLI passes validation."""
    detector = load_detector_from_path(trained_model_dir / "detector.joblib")
    assert detector is not None
    description = detector.describe()
    assert description.loaded is True
    assert description.name == "VoxSentinel MFCC Baseline"
    assert "uncalibrated" in description.description.lower()
