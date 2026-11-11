"""A busy primary Gemini model must not disable the contextual layer.

Gemini answers 503 for a model that is briefly saturated. Before the fallback
chain existed, that turned into "contextual analysis unavailable" on every call
made during the window, which is how a scam transcript ended up with rule-only
output.
"""

from __future__ import annotations

import httpx

from app.protection.gemini import GeminiConfig, GeminiScamClassifier

TRANSCRIPT = (
    "Papa mujhe kidnap kar liya hai, 50000 abhi bhejo warna hum nahi chhodenge, "
    "kisi ko batana nahi."
)

VALID = {
    "classification": "LIKELY_SCAM",
    "scam_categories": ["KIDNAPPING_OR_RANSOM"],
    "risk_level": "CRITICAL",
    "confidence": 0.93,
    "evidence_quotes": ["Papa mujhe kidnap kar liya hai"],
    "explanation": "Ransom demand with secrecy instruction.",
    "recommended_actions": ["Do not pay", "Contact police"],
    "limitations": [],
}


def _payload(result: dict) -> dict:
    import json

    return {
        "candidates": [
            {"content": {"parts": [{"text": json.dumps(result)}]}, "finishReason": "STOP"}
        ]
    }


def _classifier(handler, monkeypatch):
    monkeypatch.setattr("app.protection.gemini.time.sleep", lambda _s: None)
    config = GeminiConfig(api_key="test-key", model="primary-model", timeout_s=5.0)
    return GeminiScamClassifier(config=config, transport=httpx.MockTransport(handler))


def test_fallback_model_answers_when_primary_is_busy(monkeypatch):
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if "primary-model" in request.url.path:
            return httpx.Response(503, json={"error": "overloaded"})
        return httpx.Response(200, json=_payload(VALID))

    result, status = _classifier(handler, monkeypatch).classify(TRANSCRIPT, "hi-en")
    assert status["status"] == "completed"
    assert result["classification"] == "LIKELY_SCAM"
    assert any("primary-model" in path for path in seen)
    assert any("primary-model" not in path for path in seen), "fallback model was never tried"


def test_bad_credentials_stop_immediately(monkeypatch):
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(401, json={"error": "invalid key"})

    result, status = _classifier(handler, monkeypatch).classify(TRANSCRIPT, "hi-en")
    assert result is None
    assert status["code"] == "gemini_auth_failed"
    assert len(calls) == 1, "an auth error must not be retried across models"


def test_timeout_reports_timeout_after_fallbacks(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    result, status = _classifier(handler, monkeypatch).classify(TRANSCRIPT, "hi-en")
    assert result is None
    assert status["code"] == "gemini_timeout"
