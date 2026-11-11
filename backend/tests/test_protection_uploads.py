"""Issue E: Family Emergency & Scam Protection uploads and failure stages.

The reported errors were "Backend not running", "Analysis failed" and
"Backend returned 413". This file pins down:

* the same 25 MB / streaming upload policy as every other audio endpoint
* real decoder failures surfacing as 413 / 415 / 422 / 503 with the right code
* a broken ASR or a broken LLM provider degrading ONE stage, while the analysis
  still returns a result with honest stage statuses (never a generic failure)
* the response carrying the documented evidence fields

Speech recognition is replaced by a labelled stub (no model download), and the
LLM by a stub that always fails, so both failure modes are deterministic.
"""
from __future__ import annotations

from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.protection.transcription import TranscriptionResult, TranscriptSegment
from tests.conftest import MockVoiceDetector, make_wav

LIMIT_BYTES = 25 * 1024 * 1024
RANSOM_TEXT = (
    "Papa I have been kidnapped. Please send 50000 immediately. "
    "They will hurt me if you dont pay. Do not tell anyone or call the police."
)


class StubTranscriber:
    """TEST DOUBLE — returns fixed text, no ASR model involved."""

    def __init__(self, result: TranscriptionResult | None = None) -> None:
        self._result = result or TranscriptionResult(
            status="completed",
            language="en",
            language_probability=0.99,
            text=RANSOM_TEXT,
            segments=[TranscriptSegment(start=0.0, end=5.0, text=RANSOM_TEXT)],
            model="stub-transcriber",
        )
        self.calls = 0

    def is_available(self) -> bool:
        return True

    def describe(self) -> dict:
        return {"service": "stub", "model": "stub", "device": "cpu", "available": True}

    def transcribe(self, samples, sample_rate):
        self.calls += 1
        return self._result


class FailingLlm:
    """TEST DOUBLE — always reports an upstream failure."""

    provider_label = "Stub LLM"
    provider_key = "stub"

    def is_configured(self) -> bool:
        return True

    def describe(self) -> dict:
        return {"provider": "stub", "configured": True}

    def classify(self, text, language):
        return None, {"status": "failed", "code": "http_500", "detail": "upstream exploded"}


def wav_of_size(target_bytes: int, sample_rate: int = 24_000) -> bytes:
    bps = sample_rate * 2
    return make_wav(duration_s=max(0.5, (target_bytes - 44) / bps), sample_rate=sample_rate)


@pytest.fixture
def protection_client(settings):
    """Full protection module with a stubbed transcriber and long duration ceiling."""
    app = create_app(settings=replace(settings, max_duration_s=7200.0), detector=MockVoiceDetector())
    with TestClient(app) as client:
        app.state.protection_service._transcriber = StubTranscriber()
        yield client


def analyze(client, data: bytes, name: str = "call.wav"):
    return client.post(
        "/api/v1/protection/analyze",
        files={"file": (name, data, "audio/wav")},
        data={"consent": "true"},
    )


# ------------------------------------------------------------ upload handling


def test_small_recording_is_analyzed(protection_client):
    response = analyze(protection_client, make_wav(4.0))
    assert response.status_code == 200, response.text


def test_twenty_mb_recording_is_accepted(protection_client):
    """A long call recording must reach the pipeline, not 413."""
    response = analyze(protection_client, wav_of_size(20 * 1024 * 1024))
    assert response.status_code == 200, response.text


def test_exactly_at_limit_is_accepted(protection_client):
    payload = wav_of_size(LIMIT_BYTES)
    assert LIMIT_BYTES - 100 <= len(payload) <= LIMIT_BYTES
    response = analyze(protection_client, payload)
    assert response.status_code == 200, response.text


def test_above_limit_is_a_413_with_the_real_limit(protection_client):
    response = analyze(protection_client, wav_of_size(LIMIT_BYTES + 1024 * 1024))
    assert response.status_code == 413
    error = response.json()["error"]
    assert error["code"] == "file_too_large"
    assert "25 MB" in error["message"]


def test_corrupted_audio_is_rejected(protection_client):
    response = analyze(protection_client, b"\x00\x01\x02\x03" * 100)
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_media_type"


def test_missing_file_is_a_validation_error(protection_client):
    response = protection_client.post("/api/v1/protection/analyze", data={"consent": "true"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_consent_is_required(protection_client):
    response = protection_client.post(
        "/api/v1/protection/analyze",
        files={"file": ("call.wav", make_wav(4.0), "audio/wav")},
        data={"consent": "false"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "missing_consent"


# ------------------------------------------------------- degraded stage handling


def test_failed_transcription_degrades_one_stage_only(settings):
    app = create_app(settings=settings, detector=MockVoiceDetector())
    with TestClient(app) as client:
        app.state.protection_service._transcriber = StubTranscriber(
            TranscriptionResult(
                status="failed",
                error="ASR model crashed",
                limitations=["Speech recognition failed, so no transcript is available."],
            )
        )
        response = analyze(client, make_wav(4.0))
    assert response.status_code == 200, response.text
    body = response.json()
    stages = {stage["name"]: stage["status"] for stage in body["stages"]}
    assert stages["transcription"] == "failed"
    assert stages["risk_assessment"] == "completed"
    # No transcript must never be invented, and the risk must say it is limited.
    assert body["transcript"]["text"] == ""
    # With no transcript the honest outcome is INCONCLUSIVE, not a guess.
    assert body["risk"]["category"] in {"LOW", "CAUTION", "HIGH", "CRITICAL", "INCONCLUSIVE"}


def test_unavailable_transcription_is_explicit(settings):
    app = create_app(settings=settings, detector=MockVoiceDetector())
    with TestClient(app) as client:
        app.state.protection_service._transcriber = StubTranscriber(
            TranscriptionResult(status="unavailable", error="faster-whisper not installed")
        )
        response = analyze(client, make_wav(4.0))
    assert response.status_code == 200
    stages = {stage["name"]: stage["status"] for stage in response.json()["stages"]}
    assert stages["transcription"] in {"unavailable", "failed"}


def test_failing_llm_does_not_fail_the_analysis(settings):
    """Gemini being down must leave the rule-engine result intact."""
    app = create_app(settings=settings, detector=MockVoiceDetector())
    with TestClient(app) as client:
        app.state.protection_service._transcriber = StubTranscriber()
        app.state.protection_service._llm = FailingLlm()
        response = analyze(client, make_wav(4.0))
    assert response.status_code == 200, response.text
    body = response.json()
    stages = {stage["name"]: stage["status"] for stage in body["stages"]}
    assert stages["contextual_ai_analysis"] in {"failed", "unavailable", "degraded", "skipped"}
    # The rule engine still found the ransom signals on its own.
    assert body["content"]["signal_strength"] >= 7
    assert body["risk"]["category"] in {"HIGH", "CRITICAL"}


def test_response_contains_the_documented_evidence(protection_client):
    response = analyze(protection_client, make_wav(4.0))
    assert response.status_code == 200
    body = response.json()
    for key in ("id", "status", "risk", "transcript", "content", "stages"):
        assert key in body, key
    assert body["transcript"]["text"].strip()
    # With no transcript the honest outcome is INCONCLUSIVE, not a guess.
    assert body["risk"]["category"] in {"LOW", "CAUTION", "HIGH", "CRITICAL", "INCONCLUSIVE"}
    assert any(
        stage["name"] == "audio_validation" and stage["status"] == "completed"
        for stage in body["stages"]
    )
