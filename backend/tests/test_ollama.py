"""Ollama (local model) adapter tests — mocked transport, no daemon required."""
from __future__ import annotations

import json

import httpx
import pytest

from app.protection.grok import GrokScamClassifier
from app.protection.ollama import (
    DEFAULT_OLLAMA_MODEL,
    OllamaConfig,
    OllamaConfigError,
    OllamaScamClassifier,
    _normalise_base_url,
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
            200, json={"model": "voxsentinel-mistral", "choices": [{"message": {"content": text}}]}
        )

    return httpx.MockTransport(handler)


def _classifier(transport) -> OllamaScamClassifier:
    return OllamaScamClassifier(
        OllamaConfig(base_url="http://localhost:11434", enabled=True, timeout_s=5.0),
        transport=transport,
    )


def test_disabled_by_default(monkeypatch):
    for name in ("VOXSENTINEL_OLLAMA_ENABLED", "PROTECTION_LLM_PROVIDER", "OLLAMA_BASE_URL", "OLLAMA_HOST"):
        monkeypatch.delenv(name, raising=False)
    cfg = OllamaConfig.from_env()
    assert cfg.configured is False
    assert cfg.model == DEFAULT_OLLAMA_MODEL
    assert cfg.describe()["enabled"] is False


def test_explicit_selection_enables_it(monkeypatch):
    monkeypatch.setenv("PROTECTION_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:11500")
    cfg = OllamaConfig.from_env()
    assert cfg.configured is True
    assert cfg.base_url == "http://127.0.0.1:11500"


def test_enable_flag_without_selection(monkeypatch):
    monkeypatch.setenv("VOXSENTINEL_OLLAMA_ENABLED", "true")
    monkeypatch.delenv("PROTECTION_LLM_PROVIDER", raising=False)
    assert OllamaConfig.from_env().configured is True


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("http://localhost:11434", "http://localhost:11434"),
        ("localhost:11434", "http://localhost:11434"),
        ("http://127.0.0.1:11434/", "http://127.0.0.1:11434"),
        ("https://gpu-box.example:443", "https://gpu-box.example:443"),
    ],
)
def test_base_url_normalisation(raw, expected):
    assert _normalise_base_url(raw) == expected


@pytest.mark.parametrize("raw", ["http://gpu-box.example:11434", "ftp://localhost:11434", "http://"])
def test_insecure_or_malformed_base_url_is_rejected(raw):
    with pytest.raises(OllamaConfigError):
        _normalise_base_url(raw)


def test_canonical_response_parses():
    result, status = _classifier(_ok(FULL)).classify(TRANSCRIPT, "hi")
    assert status["status"] == "completed"
    assert status["provider"] == "ollama"
    assert result["classification"] == "LIKELY_SCAM"


def test_local_request_has_no_auth_and_no_response_format():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(FULL)}}]})

    _classifier(httpx.MockTransport(handler)).classify(TRANSCRIPT)
    assert seen["url"] == "http://localhost:11434/v1/chat/completions"
    assert seen["auth"] is None
    assert "response_format" not in seen["body"]
    assert seen["body"]["stream"] is False


def test_api_key_is_sent_when_configured():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(FULL)}}]})

    classifier = OllamaScamClassifier(
        OllamaConfig(base_url="http://localhost:11434", enabled=True, api_key="local-proxy-token"),
        transport=httpx.MockTransport(handler),
    )
    classifier.classify(TRANSCRIPT)
    assert seen["auth"] == "Bearer local-proxy-token"


def test_disabled_classifier_reports_unavailable():
    result, status = OllamaScamClassifier(OllamaConfig(enabled=False)).classify(TRANSCRIPT)
    assert result is None
    assert status["status"] == "unavailable"
    assert status["code"] == "ollama_not_configured"
    assert status["provider"] == "ollama"


def test_daemon_unreachable_is_reported_not_raised():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    result, status = _classifier(httpx.MockTransport(handler)).classify(TRANSCRIPT)
    assert result is None
    assert status["status"] == "failed"
    assert status["code"] == "ollama_unreachable"
    assert status["provider"] == "ollama"


def test_attempt_budget_is_two():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503, json={"error": "loading model"})

    _classifier(httpx.MockTransport(handler)).classify(TRANSCRIPT)
    assert calls["n"] == 2


def test_health_lists_models():
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url).endswith("/api/tags")
        return httpx.Response(
            200,
            json={"models": [{"name": "mistral:latest"}, {"name": "voxsentinel-mistral:latest"}]},
        )

    health = _classifier(httpx.MockTransport(handler)).health()
    assert health["reachable"] is True
    assert "voxsentinel-mistral:latest" in health["models"]


def test_health_failure_is_reported():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    health = _classifier(httpx.MockTransport(handler)).health()
    assert health["reachable"] is False
    assert health["models"] == []


def test_grok_codes_unchanged_by_refactor():
    """The provider-generic base must not rename the existing xAI codes."""
    result, status = GrokScamClassifier().classify(TRANSCRIPT)
    if result is None and not GrokScamClassifier().configured:
        assert status["code"] == "grok_not_configured"
    assert GrokScamClassifier.code_prefix == "grok"
    assert GrokScamClassifier.max_attempts == 3


def test_unsafe_remote_url_falls_back_to_disabled_provider(monkeypatch):
    """A plain-http remote daemon disables Ollama instead of leaking transcripts."""
    from app.protection.service import _build_llm

    monkeypatch.setenv("PROTECTION_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://gpu-box.example:11434")
    llm = _build_llm()
    assert isinstance(llm, OllamaScamClassifier)
    assert llm.describe()["enabled"] is False


def test_factory_selects_mistral_when_asked(monkeypatch):
    from app.protection.mistral import MistralScamClassifier
    from app.protection.service import _build_llm

    monkeypatch.setenv("PROTECTION_LLM_PROVIDER", "mistral")
    monkeypatch.setenv("MISTRAL_API_KEY", "mistral-test-key")
    assert isinstance(_build_llm(), MistralScamClassifier)


def test_factory_does_not_pick_disabled_ollama_by_accident(monkeypatch):
    from app.protection.service import _build_llm

    for name in ("VOXSENTINEL_OLLAMA_ENABLED", "XAI_API_KEY", "GROQ_API_KEY", "MISTRAL_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("PROTECTION_LLM_PROVIDER", raising=False)
    assert not isinstance(_build_llm(), OllamaScamClassifier)


def test_health_matches_untagged_model_name():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [{"name": "mistral:latest"}]})

    health = _classifier(httpx.MockTransport(handler)).health()
    assert health["model_present"] is True
