"""The /api/v1/analyze endpoint: honesty rules and the full pipeline."""
from __future__ import annotations

import uuid

from tests.conftest import make_wav, post_analyze

CONTRACT_KEYS = {"id", "created_at", "input", "model", "scores", "decision", "evidence", "latency_ms"}


def test_missing_consent_is_rejected(client):
    response = post_analyze(client, make_wav(3.0), consent="false")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "missing_consent"


def test_valid_audio_without_detector_returns_503_not_fake_scores(client):
    response = post_analyze(client, make_wav(3.0))
    assert response.status_code == 503
    error = response.json()["error"]
    assert error["code"] == "model_unavailable"
    assert "no analysis was performed" in error["message"].lower()
    # Absolutely no fabricated score or verdict anywhere in the payload.
    body_text = response.text.lower()
    for banned in ("raw_score", "probability", "verdict", "synthetic speech detected"):
        assert banned not in body_text


def test_nothing_is_persisted_when_model_unavailable(client):
    post_analyze(client, make_wav(3.0))
    listing = client.get("/api/v1/analyses").json()
    assert listing["total"] == 0
    assert listing["analyses"] == []


def test_successful_pipeline_with_test_double_follows_contract(client_with_detector, mock_detector):
    response = post_analyze(client_with_detector, make_wav(3.0, channels=2))
    assert response.status_code == 200
    body = response.json()

    assert set(body.keys()) >= CONTRACT_KEYS
    assert uuid.UUID(body["id"]).hex == body["id"]
    assert body["created_at"].endswith("Z")
    assert body["input"]["display_name"] == "clip.wav"
    assert body["input"]["sample_rate_hz"] == 8000
    assert body["input"]["channels"] == 2
    assert body["model"]["name"] == "mock-detector"
    assert body["model"]["version"] == "0.0-test"
    assert body["scores"]["raw_score"] == 0.42
    assert body["scores"]["calibrated_probability"] is None
    assert body["decision"]["category"] == "elevated_risk"  # 0.42 between thresholds
    assert len(body["decision"]["recommendations"]) >= 1
    assert body["latency_ms"] >= 0
    assert mock_detector.calls == 1

    evidence = body["evidence"]
    assert evidence["model_explanation"] is None
    names = {f["name"] for f in evidence["measured_features"]}
    assert {"duration", "sample_rate", "channels", "peak_level", "rms_level"} <= names
    assert any("provisional" in limitation.lower() for limitation in evidence["limitations"])


def test_quality_warnings_reach_the_record(client_with_detector):
    response = post_analyze(client_with_detector, make_wav(1.0))  # below 2 s
    assert response.status_code == 200
    assert any("very short" in w for w in response.json()["input"]["quality_warnings"])


def test_untrusted_filename_is_never_used_as_path(client_with_detector, tmp_path):
    before = {p.name for p in tmp_path.glob("*")}
    response = post_analyze(client_with_detector, make_wav(3.0), name="../../etc/attack.wav")
    assert response.status_code == 200
    assert response.json()["input"]["display_name"] == "attack.wav"
    assert {p.name for p in tmp_path.glob("*")} == before


def test_unexpected_detector_crash_returns_safe_500(client_with_broken_detector):
    response = post_analyze(client_with_broken_detector, make_wav(3.0))
    assert response.status_code == 500
    error = response.json()["error"]
    assert error["code"] == "internal_error"
    message = error["message"]
    assert "traceback" not in message.lower()
    assert "/" not in message or message.count("/") == 0
    # No Python internals leak into the client payload.
    assert "RuntimeError" not in response.text
    assert "simulated detector crash" not in response.text


def test_records_are_persisted_and_listed(client_with_detector):
    post_analyze(client_with_detector, make_wav(3.0), name="one.wav")
    post_analyze(client_with_detector, make_wav(3.0), name="two.wav")
    listing = client_with_detector.get("/api/v1/analyses").json()
    assert listing["total"] == 2
    assert [a["input"]["display_name"] for a in listing["analyses"]] == ["two.wav", "one.wav"]
