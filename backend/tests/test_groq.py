"""Groq adapter tests — mocked transport (offline) + optional live check."""
from __future__ import annotations

import json
import os

import httpx
import pytest

from app.protection.gemini import _category_from_text
from app.protection.groq import DEFAULT_GROQ_MODEL, GroqConfig, GroqScamClassifier

FULL = {
    "classification": "LIKELY_SCAM",
    "risk_level": "CRITICAL",
    "scam_categories": ["KIDNAPPING_OR_RANSOM"],
    "summary": "Caller claims a kidnapping and demands immediate money.",
    "indicators": [
        {
            "category": "RANSOM_DEMAND",
            "evidence_quote": "50000 rupees turant bhejo",
            "explanation": "Urgent payment demand.",
            "severity": "CRITICAL",
        }
    ],
    "manipulation_tactics": ["URGENCY"],
    "requested_actions": ["Send money"],
    "protective_recommendations": ["Verify via a saved number."],
    "uncertainty_reasons": [],
    "transcript_quality": "ADEQUATE",
}

# The flat shape open models actually return (observed live on gpt-oss-120b).
FLAT = {
    "classification": "LIKELY_SCAM",
    "risk_level": "HIGH",
    "scam_categories": ["Kidnap/Extortion"],
    "evidence_quotes": ["50000 rupees turant bhejo"],
    "transcript_quality": "FULL",
}

TRANSCRIPT = "Papa mujhe kidnap kar liya hai, 50000 rupees turant bhejo warna maar denge."


def _ok(content) -> httpx.MockTransport:
    text = content if isinstance(content, str) else json.dumps(content)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})

    return httpx.MockTransport(handler)


def _classifier(transport, key: str = "gsk_test") -> GroqScamClassifier:
    return GroqScamClassifier(GroqConfig(api_key=key, timeout_s=5.0), transport=transport)


def test_defaults_and_env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_x")
    cfg = GroqConfig.from_env()
    assert cfg.model == DEFAULT_GROQ_MODEL
    assert cfg.configured is True
    assert cfg.describe()["provider"] == "groq"
    assert "gsk_x" not in json.dumps(cfg.describe())


def test_canonical_response_parses():
    result, status = _classifier(_ok(FULL)).classify(TRANSCRIPT, "hi")
    assert status["status"] == "completed"
    assert status["provider"] == "groq"
    assert result["classification"] == "LIKELY_SCAM"


def test_flat_response_is_mapped_to_schema():
    """evidence_quotes + FULL quality + free-text category must not fail."""
    result, status = _classifier(_ok(FLAT)).classify(TRANSCRIPT, "hi")
    assert status["status"] == "completed", status
    assert result["indicators"][0]["evidence_quote"] == "50000 rupees turant bhejo"
    assert result["transcript_quality"] in {"ADEQUATE", "POOR", "UNUSABLE"}
    assert result["summary"]


def test_groq_endpoint_is_used():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(FULL)}}]}
        )

    _classifier(httpx.MockTransport(handler)).classify(TRANSCRIPT)
    assert seen["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert seen["auth"] == "Bearer gsk_test"


def test_missing_key_is_unavailable():
    result, status = GroqScamClassifier(GroqConfig(api_key=None)).classify(TRANSCRIPT)
    assert result is None
    assert status["status"] == "unavailable"


def test_fabricated_quote_is_dropped():
    inflated = dict(FULL)
    inflated["indicators"] = FULL["indicators"] + [
        {
            "category": "WALLET_DEMAND",
            "evidence_quote": "send bitcoin to wallet",
            "explanation": "Not actually present in the transcript.",
            "severity": "HIGH",
        }
    ]
    result, status = _classifier(_ok(inflated)).classify(TRANSCRIPT)
    quotes = [i["evidence_quote"] for i in result["indicators"]]
    assert "send bitcoin to wallet" not in quotes
    assert any("could not be matched" in n for n in status["notes"])


@pytest.mark.parametrize(
    "label,expected",
    [
        ("Kidnap/Extortion", "KIDNAPPING_OR_RANSOM"),
        ("Bank fraud", "BANKING_FRAUD"),
        ("OTP theft", "OTP_OR_CREDENTIAL_THEFT"),
        ("nonsense", None),
    ],
)
def test_free_text_category_mapping(label, expected):
    mapped = _category_from_text(label.upper().replace(" ", "_").replace("/", "_"))
    assert (mapped.value if mapped else None) == expected


@pytest.mark.skipif(not os.environ.get("GROQ_API_KEY"), reason="GROQ_API_KEY not configured")
def test_live_groq_classifies_ransom():  # pragma: no cover - opt-in, real network
    result, status = GroqScamClassifier().classify(TRANSCRIPT, "hi")
    assert status["status"] == "completed", status
    assert result["classification"] in {"LIKELY_SCAM", "SUSPICIOUS"}
    assert result["risk_level"] in {"HIGH", "CRITICAL"}
