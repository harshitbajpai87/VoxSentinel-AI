"""API tests for /api/v1/synthesize and /api/v1/providers.

A FakeGnaniClient (in-memory, no network) is injected through create_app, so
these tests exercise the real HTTP layer of this API without any real secret.
"""
from __future__ import annotations

from dataclasses import replace

from fastapi.testclient import TestClient

from app.main import create_app
from app.services.gnani_client import (
    GnaniAuthError,
    GnaniRateLimitError,
    GnaniTimeoutError,
    GnaniValidationError,
    SynthesizedAudio,
    TTSRequest,
)
from tests.conftest import settings as settings_fixture  # noqa: F401


class FakeGnaniClient:
    """TEST DOUBLE: implements the GnaniClient surface in memory."""

    def __init__(self, audio: bytes = b"RIFF-fake", fail_with: Exception | None = None) -> None:
        self.audio_bytes = audio
        self.fail_with = fail_with
        self.calls: list[TTSRequest] = []
        self.closed = False

    async def synthesize(self, request: TTSRequest) -> SynthesizedAudio:
        self.calls.append(request)
        if self.fail_with is not None:
            raise self.fail_with
        return SynthesizedAudio(
            audio=self.audio_bytes,
            content_type="audio/wav",
            container=request.container,
            model=request.model,
            voice=request.voice,
            language=request.language,
        )

    async def aclose(self) -> None:
        self.closed = True


def make_client(settings, gnani_client=None) -> TestClient:
    app = create_app(settings=settings, gnani_client=gnani_client)
    return TestClient(app, raise_server_exceptions=False)


def gnani_settings(base_settings) -> object:
    return replace(base_settings, gnani_enabled=True, gnani_api_key="test-key")


BODY = {
    "text": "This is a test of synthetic speech generation.",
    "voice": "Nalini",
    "model": "timbre-v2.5",
    "language": "hi-IN",
}


def test_synthesize_is_disabled_by_default(settings):
    client = make_client(settings)
    response = client.post("/api/v1/synthesize", json=BODY)
    assert response.status_code == 503
    body = response.json()
    assert set(body.keys()) == {"error"}
    assert body["error"]["code"] == "provider_disabled"


def test_synthesize_enabled_returns_audio_with_provenance(settings):
    fake = FakeGnaniClient(audio=b"RIFF-test-audio")
    client = make_client(gnani_settings(settings), gnani_client=fake)
    with client:
        response = client.post("/api/v1/synthesize", json=BODY)
    assert response.status_code == 200
    assert response.content == b"RIFF-test-audio"
    assert response.headers["content-type"].startswith("audio/wav")
    assert response.headers["X-Synthetic-Content"] == "true"
    assert response.headers["X-Content-Provenance"] == "synthetic"
    assert response.headers["X-Provider"] == "gnani"
    assert response.headers["X-Provider-Model"] == "timbre-v2.5"
    assert response.headers["X-Provider-Voice"] == "Nalini"
    assert fake.closed is True  # lifespan closes the provider client


def test_synthesize_validates_payload(settings):
    client = make_client(gnani_settings(settings), gnani_client=FakeGnaniClient())
    bad_model = client.post("/api/v1/synthesize", json={**BODY, "model": "gpt-4o"})
    assert bad_model.status_code == 422
    bad_speed = client.post("/api/v1/synthesize", json={**BODY, "speed": 3.0})
    assert bad_speed.status_code == 422
    bad_container = client.post("/api/v1/synthesize", json={**BODY, "container": "flac"})
    assert bad_container.status_code == 422
    empty_text = client.post("/api/v1/synthesize", json={**BODY, "text": ""})
    assert empty_text.status_code == 422


def test_synthesize_enforces_local_text_limit(settings):
    limited = replace(gnani_settings(settings), gnani_max_text_chars=10)
    client = make_client(limited, gnani_client=FakeGnaniClient())
    response = client.post("/api/v1/synthesize", json=BODY)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_synthesize_provider_auth_failure_is_safe(settings):
    client = make_client(gnani_settings(settings),
                         gnani_client=FakeGnaniClient(fail_with=GnaniAuthError("nope")))
    response = client.post("/api/v1/synthesize", json=BODY)
    assert response.status_code == 503
    body = response.json()
    assert body["error"]["code"] == "provider_auth_failed"
    # The failure detail never leaks the provider's message or any key.
    assert "nope" not in body["error"]["message"]
    assert "test-key" not in response.text


def test_synthesize_provider_rate_limit_maps_to_429(settings):
    client = make_client(gnani_settings(settings),
                         gnani_client=FakeGnaniClient(fail_with=GnaniRateLimitError("slow")))
    response = client.post("/api/v1/synthesize", json=BODY)
    assert response.status_code == 429
    assert response.json()["error"]["code"] == "rate_limited"


def test_synthesize_provider_timeout_maps_to_503(settings):
    client = make_client(gnani_settings(settings),
                         gnani_client=FakeGnaniClient(fail_with=GnaniTimeoutError("t")))
    response = client.post("/api/v1/synthesize", json=BODY)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "provider_unavailable"


def test_synthesize_local_rate_limit(settings):
    tight = replace(gnani_settings(settings), gnani_rate_limit_per_minute=1)
    client = make_client(tight, gnani_client=FakeGnaniClient())
    with client:
        first = client.post("/api/v1/synthesize", json=BODY)
        second = client.post("/api/v1/synthesize", json=BODY)
    assert first.status_code == 200
    assert second.status_code == 429
    assert second.json()["error"]["code"] == "rate_limited"


def test_synthesize_daily_quota(settings):
    tight = replace(gnani_settings(settings), gnani_daily_quota=1)
    client = make_client(tight, gnani_client=FakeGnaniClient())
    with client:
        first = client.post("/api/v1/synthesize", json=BODY)
        second = client.post("/api/v1/synthesize", json=BODY)
    assert first.status_code == 200
    assert second.status_code == 429


def test_synthesize_validation_error_from_client(settings):
    client = make_client(gnani_settings(settings),
                         gnani_client=FakeGnaniClient(
                             fail_with=GnaniValidationError("Text exceeds the 1000-character limit.")
                         ))
    response = client.post("/api/v1/synthesize", json=BODY)
    assert response.status_code == 422


def test_synthesize_audit_records_attribution_not_text(settings):
    fake = FakeGnaniClient()
    client = make_client(gnani_settings(settings), gnani_client=fake)
    with client:
        client.post("/api/v1/synthesize", json=BODY)
        audit = client.get("/api/v1/audit").json()
    events = audit.get("events", audit if isinstance(audit, list) else [])
    synthesize_events = [e for e in events if e.get("action") == "synthesize"]
    assert synthesize_events, "expected a synthesize audit event"
    metadata = synthesize_events[0].get("metadata") or {}
    assert metadata.get("synthetic") is True
    assert metadata.get("provider") == "gnani"
    assert metadata.get("model") == "timbre-v2.5"
    # The synthesized text must NOT be recorded in the audit log.
    assert BODY["text"] not in str(events)


def test_generated_storage_is_off_by_default(settings, tmp_path):
    generated_dir = tmp_path / "generated"
    enabled = replace(gnani_settings(settings), gnani_store_generated=True,
                      gnani_generated_dir=str(generated_dir))
    client = make_client(enabled, gnani_client=FakeGnaniClient())
    with client:
        response = client.post("/api/v1/synthesize", json=BODY)
    assert response.status_code == 200
    # When explicitly enabled, storage writes audio plus a provenance manifest.
    manifest = generated_dir / "provenance.jsonl"
    assert manifest.is_file()
    rows = manifest.read_text().strip().splitlines()
    import json

    row = json.loads(rows[-1])
    assert row["label"] == "synthetic"
    assert row["partition"] == "generated"
    assert row["generator_id"] == "gnani:timbre-v2.5:Nalini"


def test_providers_reports_honest_capabilities(settings):
    client = make_client(gnani_settings(settings), gnani_client=FakeGnaniClient())
    with client:
        response = client.get("/api/v1/providers")
    assert response.status_code == 200
    body = response.json()
    assert body["detection_note"].startswith("Detection is performed by VoxSentinel")
    providers = {p["provider"]: p for p in body["generations"]}
    gnani = providers["gnani"]
    assert gnani["generation_supported"] is True
    assert gnani["detection_supported"] is False
    assert "synthesis" in gnani["note"] or "synthetic" in gnani["note"]


def test_providers_without_provider_reports_disabled(settings):
    """Not enabled: the capability report is honest — TTS is a documented
    provider capability, but this deployment has not enabled it."""
    client = make_client(settings)
    response = client.get("/api/v1/providers")
    assert response.status_code == 200
    providers = {p["provider"]: p for p in response.json()["generations"]}
    gnani = providers["gnani"]
    assert gnani["generation_enabled"] is False
    assert gnani["generation_supported"] is True
    assert gnani["detection_supported"] is False


def test_no_secret_in_any_response(settings):
    secret_marker = "test-key"
    client = make_client(gnani_settings(settings), gnani_client=FakeGnaniClient())
    with client:
        for path in ("/health", "/api/v1/providers", "/api/v1/models"):
            response = client.get(path)
            assert secret_marker not in response.text
        synthesize = client.post("/api/v1/synthesize", json=BODY)
        assert secret_marker not in synthesize.text
        assert secret_marker not in str(synthesize.headers)
