"""Capabilities and health must never overstate what this host can actually do.

Three separate lies were possible before these tests existed:
  * a gated Hugging Face repo advertised as a working ASR engine,
  * a configured-but-dead Ollama daemon reported as ready reasoning,
  * the active reasoning provider misreported as the legacy hosted one.
"""

from __future__ import annotations

import pytest

from app.api import routes as api_routes
from app.protection import llm_router
from app.services import hf_models


# ------------------------------------------------------- gated HF repositories
def test_gated_repo_is_unavailable_without_a_token(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setenv("VOXSENTINEL_HF_DOWNLOAD", "true")
    monkeypatch.setattr(hf_models, "HF_CACHE_SUBDIR", tmp_path)
    monkeypatch.setattr(hf_models, "transformers_available", lambda: True)

    source = hf_models.resolve_model("ai4bharat/indicwav2vec-hindi", "indicwav2vec-hindi")

    assert source.available is False
    assert "gated" in (source.reason or "")
    assert "HF_TOKEN" in (source.reason or "")


def test_gated_repo_is_available_with_a_token(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    monkeypatch.setenv("VOXSENTINEL_HF_DOWNLOAD", "true")
    monkeypatch.setattr(hf_models, "HF_CACHE_SUBDIR", tmp_path)
    monkeypatch.setattr(hf_models, "transformers_available", lambda: True)

    source = hf_models.resolve_model("ai4bharat/indicwav2vec-hindi", "indicwav2vec-hindi")

    assert source.available is True
    assert source.local is False


def test_ungated_repo_still_resolves_without_a_token(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setenv("VOXSENTINEL_HF_DOWNLOAD", "true")
    monkeypatch.setattr(hf_models, "HF_CACHE_SUBDIR", tmp_path)
    monkeypatch.setattr(hf_models, "transformers_available", lambda: True)

    assert hf_models.resolve_model("openai/whisper-small", "whisper-small").available is True


# ---------------------------------------------------------- reasoning readiness
def test_local_probe_is_cached(monkeypatch):
    calls = {"n": 0}

    class FakeResponse:
        status_code = 200

    def fake_get(url, timeout=None):
        calls["n"] += 1
        return FakeResponse()

    monkeypatch.setattr("httpx.get", fake_get)
    llm_router._PROBE_CACHE.clear()

    assert llm_router._probe_local("http://probe-test:11434")[0] is True
    assert llm_router._probe_local("http://probe-test:11434")[0] is True
    assert calls["n"] == 1, "the probe must not hit the daemon on every describe()"


def test_local_probe_reports_a_dead_daemon(monkeypatch):
    def fake_get(url, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr("httpx.get", fake_get)
    llm_router._PROBE_CACHE.clear()

    reachable, detail = llm_router._probe_local("http://dead-test:11434")
    assert reachable is False
    assert detail == "OSError"


# ------------------------------------------------------------- health reporting
@pytest.mark.parametrize(
    ("contextual", "expected"),
    [
        ({}, False),
        ({"configured": False, "resolved_chain": [{"configured": True, "reachable": True}]}, False),
        (
            {"configured": True, "resolved_chain": [{"configured": True, "reachable": False}]},
            False,
        ),
        (
            {"configured": True, "resolved_chain": [{"configured": True, "reachable": True}]},
            True,
        ),
        (
            {
                "configured": True,
                "resolved_chain": [
                    {"configured": True, "reachable": False},
                    {"configured": True, "reachable": True},
                ],
            },
            True,
        ),
        ({"configured": True}, True),
    ],
)
def test_contextual_ready_requires_a_chain_member_that_can_answer(contextual, expected):
    assert api_routes._contextual_ready(contextual) is expected
