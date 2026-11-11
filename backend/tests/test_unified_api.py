"""POST /api/analyze: the unified multi-model contract.

These tests exercise the real pipeline with a TEST DOUBLE detector and a stub
transcriber, so no heavy model is loaded and no metric here is presented as
detection performance. The contract itself (sections, badges, explanation,
persistence, consent, text-only input) is what is under test.
"""

from __future__ import annotations

import pytest

from app.protection.transcription import TranscriptionResult, TranscriptSegment
from app.protection.unified import _speaker_verification
from tests.conftest import build_client, make_wav


class StubTranscriber:
    """TEST DOUBLE transcriber: canned transcript in the multi-engine shape."""

    def __init__(
        self,
        text: str = "Papa, I had an accident, send 1 lakh rupees in 10 minutes.",
        status: str = "completed",
    ):
        self._text = text
        self._status = status

    def transcribe(self, samples, sample_rate) -> TranscriptionResult:
        return TranscriptionResult(
            status=self._status,
            language="en",
            language_probability=None,
            language_source="script_detection",
            text=self._text,
            segments=[TranscriptSegment(0.0, 2.0, self._text)],
            model="stub-asr",
            engines=[
                {"engine": "whisper_small", "model": "openai/whisper-small",
                 "status": "completed", "characters": len(self._text), "latency_ms": 5},
            ],
        )

    def describe(self) -> dict:
        return {"service": "stub-asr", "available": True, "engines": []}


@pytest.fixture
def unified_client(settings, mock_detector, monkeypatch):
    # Keep the suite fast: the encoder ensemble has its own real-model test.
    monkeypatch.setenv("VOXSENTINEL_TEXT_ENSEMBLE", "false")
    client = build_client(settings, mock_detector)
    client.app.state.protection_service._transcriber = StubTranscriber()
    with client:
        yield client


def post_audio(client, data: bytes | None = None, name: str = "clip.wav", **form):
    payload = {"consent": "true", **form}
    files = {"file": (name, data if data is not None else make_wav(), "audio/wav")}
    return client.post("/api/analyze", files=files, data=payload)


# ------------------------------------------------------------------ text path
def test_text_analysis_returns_every_documented_section(unified_client):
    response = unified_client.post(
        "/api/analyze",
        json={
            "text": "Papa, it's me. I had an accident, send 1 lakh rupees within 10 minutes. "
            "Don't tell anyone, don't call the police.",
            "consent": True,
        },
    )
    assert response.status_code == 200
    body = response.json()

    assert body["schema_version"] == "1.0"
    assert body["input_kind"] == "text"
    for section in (
        "transcription",
        "speaker_verification",
        "acoustic_forensics",
        "scam_risk",
        "llm_reasoning",
        "risk",
        "risk_badges",
        "explanation",
    ):
        assert section in body, section

    assert body["transcription"]["text"].startswith("Papa")
    assert any(
        badge["id"] == "transcription" and badge["label"] == "Transcript supplied by client"
        for badge in body["risk_badges"]
    )
    assert body["speaker_verification"]["status"] == "skipped"
    assert body["acoustic_forensics"]["fusion"]["score"] is None
    assert body["risk"]["category"] in {"MEDIUM", "HIGH", "CRITICAL"}
    assert body["scam_risk"]["score"] is not None
    assert body["evidence_fusion"]["confidence"] is None

    badges = {badge["id"]: badge for badge in body["risk_badges"]}
    assert {"risk", "voice_authenticity", "speaker_match", "scam_intent", "llm_reasoning"} <= set(badges)
    assert badges["risk"]["tone"] in {"critical", "warn", "ok", "info"}
    assert badges["voice_authenticity"]["label"] == "No audio to screen"


def test_text_analysis_explanation_is_structured(unified_client):
    response = unified_client.post(
        "/api/analyze",
        json={
            "text": "Sir, your KYC is expiring, share the OTP to keep your account active.",
            "consent": True,
        },
    )
    explanation = response.json()["explanation"]
    assert explanation["verdict"] == response.json()["risk"]["category"]
    assert isinstance(explanation["evidence"], list)
    assert isinstance(explanation["recommended_actions"], list)
    assert "why_not_conclusive" in explanation
    assert explanation["summary"]


def test_text_analysis_is_persisted_and_listed(unified_client):
    response = unified_client.post(
        "/api/analyze", json={"text": "Send the OTP now or your account will be blocked.", "consent": True}
    )
    record_id = response.json()["id"]
    listing = unified_client.get("/api/v1/protection/analyses").json()
    assert any(item["id"] == record_id for item in listing["analyses"])

    stored = unified_client.get(f"/api/v1/protection/analyses/{record_id}").json()
    assert stored["schema_version"] == "1.0"
    assert stored["explanation"]["verdict"]


def test_missing_input_is_rejected(unified_client):
    response = unified_client.post("/api/analyze", json={"consent": True})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "no_input"


def test_explicit_refusal_of_consent_is_rejected(unified_client):
    response = unified_client.post("/api/analyze", json={"text": "hello", "consent": False})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "missing_consent"


def test_blank_text_is_rejected(unified_client):
    response = unified_client.post("/api/analyze", json={"text": "   ", "consent": True})
    assert response.status_code == 400


# ----------------------------------------------------------------- audio path
def test_audio_analysis_reports_every_layer(unified_client):
    response = post_audio(unified_client)
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["input_kind"] == "audio"
    assert body["transcription"]["engine"] == "whisper_small"
    assert body["transcription"]["model"] == "stub-asr"
    assert body["transcription"]["engines"][0]["status"] == "completed"

    # Detector TEST DOUBLE reported 0.42, so no synthetic verdict is claimed.
    assert body["acoustic_forensics"]["detector"]["score"] == pytest.approx(0.42, abs=0.2)
    assert body["acoustic_forensics"]["fusion"]["components_used"]

    badges = {badge["id"]: badge for badge in body["risk_badges"]}
    assert badges["transcription"]["label"].startswith("Transcribed")
    assert badges["speaker_match"]["label"] == "Speaker not verified"


@pytest.mark.parametrize("asr_status,asr_text", [("unavailable", ""), ("completed", "   ")])
def test_missing_or_empty_transcript_is_not_treated_as_low_scam_risk(
    unified_client, asr_status, asr_text
):
    unified_client.app.state.protection_service._transcriber = StubTranscriber(
        text=asr_text, status=asr_status
    )
    body = post_audio(unified_client).json()
    assert body["scam_risk"]["status"] == "unavailable"
    assert body["scam_risk"]["score"] is None
    assert body["risk"]["category"] == "INCONCLUSIVE"
    badge = next(item for item in body["risk_badges"] if item["id"] == "scam_intent")
    assert badge["label"] == "Scam intent not assessed"
    assert "not evidence of safety" in badge["detail"]


def test_audio_analysis_rejects_an_undecodable_file(unified_client):
    response = post_audio(unified_client, data=b"not audio at all", name="broken.wav")
    assert response.status_code >= 400
    assert response.json()["error"]["code"] in {
        "unsupported_format",
        "unsupported_media_type",
        "decode_failed",
        "invalid_audio",
        "decoder_unavailable",
        "empty_file",
    }


def test_audio_analysis_requires_consent(unified_client):
    response = unified_client.post(
        "/api/analyze",
        files={"file": ("clip.wav", make_wav(), "audio/wav")},
        data={"consent": "false"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "missing_consent"


# ------------------------------------------------------------- capabilities
def test_capabilities_names_every_model_family(unified_client):
    body = unified_client.get("/api/capabilities").json()
    models = body["models"]
    expected = {"transcription", "speaker_verification", "acoustic_forensics", "scam_text", "llm_reasoning"}
    assert expected <= set(models)
    assert models["speaker_verification"]["backend"] == "none"
    assert models["speaker_verification"]["embedding_dim"] is None
    assert body["input_kinds"] == ["audio", "text"]
    assert body["endpoint"] == "POST /api/analyze"


def test_empty_upload_is_reported_as_no_input(unified_client):
    response = unified_client.post("/api/analyze", data={"consent": "true"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "no_input"


def test_speaker_mapper_normalizes_outcome_and_only_reports_executed_model():
    completed = _speaker_verification({
        "speaker_comparison": {
            "status": "consistent_with_reference",
            "outcome": "consistent_with_reference",
            "similarity": 0.82,
            "threshold": 0.75,
            "backend": "ecapa",
            "embedding_model": "speechbrain/spkrec-ecapa-voxceleb",
            "embedding_dim": 192,
        },
        "pipeline": {"speaker_verifier": "ecapa"},
    })
    assert completed["match"] is True
    assert completed["outcome"] == "likely_same_speaker"
    assert completed["model"] == "speechbrain/spkrec-ecapa-voxceleb"
    assert completed["embedding_dim"] == 192

    skipped = _speaker_verification({
        "speaker_comparison": {"status": "reference_unavailable", "outcome": "reference_unavailable"},
        "pipeline": {"speaker_verifier": "unavailable"},
    })
    assert skipped["match"] is None
    assert skipped["outcome"] == "reference_unavailable"
    assert skipped["model"] is None
    assert skipped["embedding_dim"] is None
