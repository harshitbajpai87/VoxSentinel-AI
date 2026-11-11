"""Tests for the additive Family Emergency & Scam Protection module.

Covers the multilingual rule engine, the independent risk policy (all four
required scenarios), and the HTTP surface with a clearly-labelled test double
for speech recognition (so tests never depend on a downloaded ASR model).
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.protection import risk
from app.protection.rules import analyze_transcript
from app.protection.transcription import TranscriptionResult, TranscriptSegment
from tests.conftest import make_wav

RANSOM_EN = (
    "Papa I have been kidnapped. Please send 50000 immediately. "
    "They will hurt me if you dont pay. Do not tell anyone or call the police."
)
RANSOM_HINGLISH = (
    "Papa, main musibat mein hoon. Mujhe kidnap kar liya hai. Abhi paise transfer karo. "
    "Kisi ko mat batana. Police ko phone mat karna. Warna mere saath kuch ho jayega."
)
OTP_SCAM = (
    "This is your bank officer. Your account will be blocked. "
    "Tell me the OTP to verify your account."
)
AWARENESS = (
    "In this scam awareness video we discuss how kidnap scams work. Fraudsters ask for OTPs "
    "and tell victims not to call the police. Beware of such calls."
)


# --------------------------------------------------------------------- rules


def test_ransom_english_detected():
    result = analyze_transcript(RANSOM_EN)
    categories = {f.category for f in result.categories}
    assert "ransom_or_kidnapping" in categories
    assert "family_emergency_impersonation" in categories
    assert "emergency_money_request" in categories
    assert result.flags["payment_demand"] is True
    assert result.flags["secrecy_or_urgency"] is True
    assert result.signal_strength >= 7


def test_ransom_hinglish_detected():
    result = analyze_transcript(RANSOM_HINGLISH)
    categories = {f.category for f in result.categories}
    assert "ransom_or_kidnapping" in categories
    assert result.flags["payment_demand"] is True
    assert result.signal_strength >= 7


def test_otp_bank_scam_detected():
    result = analyze_transcript(OTP_SCAM)
    categories = {f.category for f in result.categories}
    assert "otp_or_credential_theft" in categories
    assert "bank_account_or_card_fraud" in categories
    assert result.flags["otp_or_credentials"] is True


def test_awareness_discussion_is_not_flagged():
    """Scenario C: discussing scams must not be classified as a scam."""
    result = analyze_transcript(AWARENESS)
    assert result.discussion_context is True
    assert result.signal_strength == 0


def test_neutral_conversation_is_clean():
    result = analyze_transcript("Let us meet for lunch tomorrow at the usual place.")
    assert result.signal_strength == 0
    assert result.primary_category == "legitimate_or_no_clear_scam_evidence"


def test_empty_transcript_is_insufficient_evidence():
    result = analyze_transcript("   ")
    assert result.primary_category == "insufficient_evidence"
    assert result.signal_strength == 0


def test_matches_carry_exact_spans():
    result = analyze_transcript(RANSOM_EN)
    ransom = next(f for f in result.categories if f.category == "ransom_or_kidnapping")
    phrase = ransom.matches[0]["phrase"]
    start = ransom.matches[0]["start"]
    assert RANSOM_EN.lower()[start : start + len(phrase)].strip() == phrase


# ---------------------------------------------------------------------- risk


def _assess(transcript: str, synthetic: bool | None, speaker: str | None = "not_consistent_with_reference"):
    content = analyze_transcript(transcript)
    return risk.assess(
        risk.SignalInput(
            synthetic_suspected=synthetic,
            synthetic_note=None,
            speaker_outcome=speaker,
            signal_strength=content.signal_strength,
            flags=content.flags,
            transcription_status="completed",
            discussion_context=content.discussion_context,
        )
    )


def test_scenario_a_cloned_ransom_call_is_critical():
    assessment = _assess(RANSOM_EN, synthetic=True)
    assert assessment.category == "CRITICAL"
    assert any("ransom" in r.lower() or "serious" in r.lower() for r in assessment.reasoning)
    assert any("verify" in a.lower() for a in assessment.recommended_actions)


def test_scenario_b_real_voice_otp_scam_still_high():
    assessment = _assess(OTP_SCAM, synthetic=False)
    assert assessment.category == "HIGH"


def test_scenario_c_awareness_is_low():
    assessment = _assess(AWARENESS, synthetic=False)
    assert assessment.category == "LOW"


def test_scenario_d_poor_quality_is_inconclusive():
    assessment = risk.assess(
        risk.SignalInput(
            synthetic_suspected=None,
            synthetic_note=None,
            speaker_outcome=None,
            signal_strength=0,
            flags={},
            transcription_status="unavailable",
            discussion_context=False,
        )
    )
    assert assessment.category == "INCONCLUSIVE"
    assert any("no conclusion" in lim.lower() for lim in assessment.limitations)


def test_high_risk_includes_emergency_guidance():
    assessment = _assess(RANSOM_EN, synthetic=True)
    joined = " ".join(assessment.recommended_actions).lower()
    assert "otp" in joined
    assert "police" in joined or "emergency" in joined


# ------------------------------------------------------------------ HTTP API


class _StubTranscriber:
    """Test double for ASR: deterministic transcript, clearly not a real model."""

    def __init__(self, text: str) -> None:
        self._text = text

    def transcribe(self, samples, sample_rate):  # noqa: ANN001 - matches protocol
        return TranscriptionResult(
            status="completed",
            language="en",
            language_probability=0.99,
            text=self._text,
            segments=[TranscriptSegment(start=0.0, end=3.0, text=self._text)],
            model="stub-transcriber",
        )

    def describe(self):
        return {"service": "stub", "model": "stub", "device": "none", "available": True}


@pytest.fixture
def protection_client(settings):
    app = create_app(settings=settings)
    with TestClient(app) as client:
        app.state.protection_service._transcriber = _StubTranscriber(RANSOM_EN)
        yield client


def test_analyze_requires_consent(protection_client):
    response = protection_client.post(
        "/api/v1/protection/analyze",
        files={"file": ("call.wav", make_wav(), "audio/wav")},
        data={"consent": "false"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "missing_consent"


def test_analyze_returns_evidence_and_persists(protection_client):
    response = protection_client.post(
        "/api/v1/protection/analyze",
        files={"file": ("call.wav", make_wav(), "audio/wav")},
        data={"consent": "true"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["risk"]["category"] in {"HIGH", "CRITICAL"}
    assert "kidnap" in body["transcript"]["text"].lower()
    assert body["content"]["signal_strength"] >= 7
    analysis_id = body["id"]

    listing = protection_client.get("/api/v1/protection/analyses").json()
    assert any(a["id"] == analysis_id for a in listing["analyses"])

    detail = protection_client.get(f"/api/v1/protection/analyses/{analysis_id}")
    assert detail.status_code == 200
    assert detail.json()["id"] == analysis_id

    report = protection_client.get(f"/api/v1/protection/analyses/{analysis_id}/report")
    assert report.status_code == 200
    assert "attachment" in report.headers["content-disposition"]

    deleted = protection_client.delete(f"/api/v1/protection/analyses/{analysis_id}")
    assert deleted.status_code == 200
    assert protection_client.get(f"/api/v1/protection/analyses/{analysis_id}").status_code == 404


def test_invalid_audio_is_rejected(protection_client):
    response = protection_client.post(
        "/api/v1/protection/analyze",
        files={"file": ("bad.wav", b"not-audio-at-all", "audio/wav")},
        data={"consent": "true"},
    )
    # Unified with every other audio endpoint: bytes that contain no recognizable
    # audio container are an unsupported media type (415). This endpoint used to
    # collapse every validation failure into a generic 400, which is exactly why
    # the UI could not explain what went wrong.
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_media_type"


def test_family_profile_requires_consent(protection_client):
    response = protection_client.post(
        "/api/v1/protection/family-voices",
        files={"reference": ("ref.wav", make_wav(duration_s=4.0), "audio/wav")},
        data={"name": "Aarav", "relationship": "son", "consent": "false"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "missing_authorization"


def test_family_profile_create_list_delete(protection_client):
    created = protection_client.post(
        "/api/v1/protection/family-voices",
        files={"reference": ("ref.wav", make_wav(duration_s=4.0), "audio/wav")},
        data={"name": "Aarav", "relationship": "son", "consent": "true"},
    )
    assert created.status_code == 200, created.text
    profile = created.json()
    assert profile["consent"] is True

    listing = protection_client.get("/api/v1/protection/family-voices").json()
    assert any(p["id"] == profile["id"] for p in listing["profiles"])

    assert protection_client.delete(f"/api/v1/protection/family-voices/{profile['id']}").status_code == 200
    assert protection_client.get("/api/v1/protection/family-voices").json()["count"] == 0


def test_capabilities_endpoint(protection_client):
    body = protection_client.get("/api/v1/protection/capabilities").json()
    assert "transcription" in body
    assert "speaker_verifier_ready" in body
    assert "detector_ready" in body
