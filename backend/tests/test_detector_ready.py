"""Backend behavior with a real (fixture-trained) artifact installed.

Every detector here comes from the smoke fixture: tones vs noise. Scores are
real model outputs on those signals, but nothing in this file says anything
about detecting real voice cloning.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from fastapi.testclient import TestClient

from app.main import create_app
from app.services.detector import TrainedVoiceDetector
from tests.conftest import make_wav


def client_with_artifact(settings, trained_model_dir):
    settings = replace(settings, detector_model_path=str(trained_model_dir / "detector.joblib"))
    app = create_app(settings=settings)
    return TestClient(app)


def test_health_and_models_report_ready(settings, trained_model_dir):
    with client_with_artifact(settings, trained_model_dir) as client:
        health = client.get("/health").json()
        assert health["model_ready"] is True

        models = client.get("/api/v1/models").json()["models"]
        detection = next(m for m in models if m["kind"] == "detection")
        assert detection["loaded"] is True
        assert detection["name"] == "VoxSentinel MFCC Baseline"
        assert detection["version"].startswith("0.1.")


def test_analyze_returns_real_detector_scores(settings, trained_model_dir):
    with client_with_artifact(settings, trained_model_dir) as client:
        response = client.post(
            "/api/v1/analyze",
            files={"file": ("clip.wav", make_wav(3.0), "audio/wav")},
            data={"consent": "true"},
        )
        assert response.status_code == 200
        body = response.json()

        scores = body["scores"]
        assert scores["raw_score"] is not None
        assert np.isfinite(scores["raw_score"])
        assert 0.0 <= scores["raw_score"] <= 1.0
        assert scores["calibrated_probability"] is None  # never invented
        assert scores["uncertainty"] is None
        assert body["model"]["name"] == "VoxSentinel MFCC Baseline"
        assert body["decision"]["category"] in {
            "low_observed_risk",
            "elevated_risk",
            "high_risk",
            "inconclusive",
        }
        # Explanation is present because the logistic baseline supports it.
        assert body["evidence"]["model_explanation"] is not None
        assert "mfcc" in body["evidence"]["model_explanation"]
        # Result persisted.
        listing = client.get("/api/v1/analyses").json()
        assert listing["total"] == 1


def test_history_and_detail_round_trip_with_real_scores(settings, trained_model_dir):
    with client_with_artifact(settings, trained_model_dir) as client:
        created = client.post(
            "/api/v1/analyze",
            files={"file": ("clip.wav", make_wav(3.0), "audio/wav")},
            data={"consent": "true"},
        ).json()
        fetched = client.get(f"/api/v1/analyses/{created['id']}").json()
        assert fetched["scores"]["raw_score"] == created["scores"]["raw_score"]


def test_invalid_artifact_falls_back_to_honest_503(settings, tmp_path):
    """A corrupt artifact must NEVER produce a ready model or a score."""

    bad_path = tmp_path / "detector.joblib"
    bad_path.write_bytes(b"this is not a joblib artifact")
    settings = replace(settings, detector_model_path=str(bad_path))

    app = create_app(settings=settings)
    with TestClient(app) as client:
        assert client.get("/health").json()["model_ready"] is False
        response = client.post(
            "/api/v1/analyze",
            files={"file": ("clip.wav", make_wav(3.0), "audio/wav")},
            data={"consent": "true"},
        )
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "model_unavailable"
        assert client.get("/api/v1/analyses").json()["total"] == 0


class _NanClassifier:
    """TEST DOUBLE: a classifier that emits an invalid (NaN) probability."""

    classes_ = [0, 1]
    n_features_in_ = 40

    def predict_proba(self, _x):
        return np.array([[np.nan, np.nan]])


class _IdentityScaler:
    n_features_in_ = 40

    def transform(self, x):
        return x


def test_invalid_inference_output_is_refused(settings, tmp_path):
    """A model emitting NaN is refused: 503, no record, no clamped score."""
    from app.services.features import FeatureConfig

    detector = TrainedVoiceDetector(
        classifier=_NanClassifier(),
        scaler=_IdentityScaler(),
        feature_config=FeatureConfig(),
        model_name="nan-test-double",
        model_version="0.0",
        synthetic_index=1,
    )
    app = create_app(settings=settings, detector=detector)
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/analyze",
            files={"file": ("clip.wav", make_wav(3.0), "audio/wav")},
            data={"consent": "true"},
        )
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "model_unavailable"
        assert "nan" not in response.text.lower()
        assert client.get("/api/v1/analyses").json()["total"] == 0
