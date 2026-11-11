"""Mistral AI adapter tests — mocked transport (offline) + optional live check."""
from __future__ import annotations

import json
import os

import httpx
import pytest

from app.protection.mistral import (
    DEFAULT_MISTRAL_MODEL,
    MistralConfig,
    MistralScamClassifier,
)

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

TRANSCRIPT = "Papa mujhe kidnap kar liya hai, 50000 rupees turant bhejo warna maar denge."


def _ok(content) -> httpx.MockTransport:
    text = content if isinstance(content, str) else json.dumps(content)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"model": "mistral-small-latest", "choices": [{"message": {"content": text}}]},
        )

    return httpx.MockTransport(handler)


def _classifier(transport, key: str = "mistral-test-key") -> MistralScamClassifier:
    return MistralScamClassifier(MistralConfig(api_key=key, timeout_s=5.0), transport=transport)


def test_defaults_and_env(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "mistral-test-key")
    monkeypatch.delenv("MISTRAL_MODEL", raising=False)
    cfg = MistralConfig.from_env()
    assert cfg.model == DEFAULT_MISTRAL_MODEL
    assert cfg.configured is True
    described = cfg.describe()
    assert described["provider"] == "mistral"
    assert described["label"] == "Mistral AI"
    assert "mistral-test-key" not in json.dumps(described)


def test_canonical_response_parses():
    result, status = _classifier(_ok(FULL)).classify(TRANSCRIPT, "hi")
    assert status["status"] == "completed"
    assert status["provider"] == "mistral"
    assert result["classification"] == "LIKELY_SCAM"


def test_endpoint_and_auth_header():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(FULL)}}]})

    _classifier(httpx.MockTransport(handler)).classify(TRANSCRIPT)
    assert seen["url"] == "https://api.mistral.ai/v1/chat/completions"
    assert seen["auth"] == "Bearer mistral-test-key"


def test_missing_key_is_unavailable_with_mistral_code():
    result, status = MistralScamClassifier(MistralConfig(api_key=None)).classify(TRANSCRIPT)
    assert result is None
    assert status["status"] == "unavailable"
    assert status["code"] == "mistral_not_configured"
    assert status["provider"] == "mistral"


def test_schema_invalid_is_reported_as_mistral_failure():
    result, status = _classifier(_ok({"classification": "NOT_A_LABEL"})).classify(TRANSCRIPT)
    assert result is None
    assert status["code"] == "mistral_schema_invalid"


def test_latency_and_model_reported():
    _, status = _classifier(_ok(FULL)).classify(TRANSCRIPT)
    assert status["model"] == "mistral-small-latest"
    assert isinstance(status["latency_ms"], int)


@pytest.mark.skipif(not os.environ.get("MISTRAL_API_KEY"), reason="MISTRAL_API_KEY not configured")
def test_live_mistral_classifies_ransom():  # pragma: no cover - opt-in, real network
    result, status = MistralScamClassifier().classify(TRANSCRIPT, "hi")
    assert status["status"] == "completed", status
    assert result["classification"] in {"LIKELY_SCAM", "SUSPICIOUS"}
