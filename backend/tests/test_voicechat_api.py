"""API tests for POST /api/v1/voicechat (NemotronLabs VoiceChat 11B).

A FakeVoiceChatClient (scripted turn, no network, no GPU) is injected through
create_app, so these tests exercise the real HTTP layer of this API without
any real provider.
"""
from __future__ import annotations

import base64
import io
import wave
from dataclasses import replace

from fastapi.testclient import TestClient

from app.main import create_app
from app.services.voicechat_client import (
    VoiceChatConnectionError,
    VoiceChatServerError,
    VoiceChatTurn,
    VoiceChatValidationError,
)
from tests.conftest import make_wav


class FakeVoiceChatClient:
    """TEST DOUBLE: implements the VoiceChatClient surface in memory."""

    def __init__(self, turn: VoiceChatTurn | None = None,
                 fail_with: Exception | None = None) -> None:
        self.turn = turn or VoiceChatTurn(
            agent_audio_pcm=b"\x00\x01" * 240,  # 120 ms of near-silence PCM
            sample_rate=24000,
            agent_transcript="Hello, this is a synthetic agent.",
            user_transcript="Hello agent.",
        )
        self.fail_with = fail_with
        self.calls: list[dict] = []

    async def converse(self, samples, sample_rate, channels, *, instructions=None):
        self.calls.append({
            "samples": samples, "sample_rate": sample_rate,
            "channels": channels, "instructions": instructions,
        })
        if self.fail_with is not None:
            raise self.fail_with
        return self.turn


def make_app(settings, voicechat_client=None) -> TestClient:
    app = create_app(settings=settings, voicechat_client=voicechat_client)
    return TestClient(app, raise_server_exceptions=False)


def vc_settings(base_settings) -> object:
    return replace(
        base_settings,
        voicechat_enabled=True,
        voicechat_ws_url="ws://localhost:9000/v1/realtime",
    )


def post_voicechat(client: TestClient, audio: bytes, name: str = "speech.wav"):
    return client.post(
        "/api/v1/voicechat",
        files={"file": (name, audio, "audio/wav")},
        data={},
    )


def test_voicechat_disabled_by_default(settings):
    client = make_app(settings)
    response = post_voicechat(client, make_wav(3.0))
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "provider_disabled"


def test_voicechat_turn_returns_synthetic_labelled_json(settings):
    client = make_app(vc_settings(settings), voicechat_client=FakeVoiceChatClient())
    with client:
        response = post_voicechat(client, make_wav(3.0))
    assert response.status_code == 200
    body = response.json()
    assert body["synthetic"] is True
    assert body["provider"] == "voicechat"
    assert body["generator_id"] == "voicechat:nemotron-voicechat"
    assert body["agent_transcript"].startswith("Hello")
    assert body["user_transcript"] == "Hello agent."
    # The audio is a real WAV we can decode.
    wav_bytes = base64.b64decode(body["audio_b64"])
    assert wav_bytes[:4] == b"RIFF"
    with wave.open(io.BytesIO(wav_bytes)) as handle:
        assert handle.getframerate() == 24000
        assert handle.getnchannels() == 1


def test_voicechat_passes_instructions(settings):
    fake = FakeVoiceChatClient()
    client = make_app(vc_settings(settings), voicechat_client=fake)
    with client:
        response = client.post(
            "/api/v1/voicechat",
            files={"file": ("speech.wav", make_wav(3.0), "audio/wav")},
            data={"instructions": "Speak like a news anchor."},
        )
    assert response.status_code == 200
    assert fake.calls[0]["instructions"] == "Speak like a news anchor."


def test_voicechat_rejects_oversized_instructions(settings):
    client = make_app(vc_settings(settings), voicechat_client=FakeVoiceChatClient())
    with client:
        response = client.post(
            "/api/v1/voicechat",
            files={"file": ("speech.wav", make_wav(3.0), "audio/wav")},
            data={"instructions": "x" * 5000},
        )
    assert response.status_code == 422


def test_voicechat_rejects_bad_audio(settings):
    client = make_app(vc_settings(settings), voicechat_client=FakeVoiceChatClient())
    with client:
        silent = post_voicechat(client, make_wav(3.0, amplitude=0.0))
        corrupt = post_voicechat(client, b"plain text, not audio at all" * 20)
    assert silent.status_code == 422
    assert corrupt.status_code == 415


def test_voicechat_local_validation_error(settings):
    client = make_app(
        vc_settings(settings),
        voicechat_client=FakeVoiceChatClient(
            fail_with=VoiceChatValidationError("Input audio exceeds the 60s limit.")
        ),
    )
    with client:
        response = post_voicechat(client, make_wav(3.0))
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_voicechat_connection_error_maps_to_504(settings):
    client = make_app(
        vc_settings(settings),
        voicechat_client=FakeVoiceChatClient(fail_with=VoiceChatConnectionError("refused")),
    )
    with client:
        response = post_voicechat(client, make_wav(3.0))
    assert response.status_code == 504
    assert response.json()["error"]["code"] == "provider_timeout"


def test_voicechat_server_error_maps_to_502(settings):
    client = make_app(
        vc_settings(settings),
        voicechat_client=FakeVoiceChatClient(
            fail_with=VoiceChatServerError("inference_error", "boom")
        ),
    )
    with client:
        response = post_voicechat(client, make_wav(3.0))
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "provider_error"
    assert "boom" not in response.text


def test_voicechat_local_rate_limit(settings):
    tight = replace(vc_settings(settings), voicechat_rate_limit_per_minute=1)
    client = make_app(tight, voicechat_client=FakeVoiceChatClient())
    with client:
        first = post_voicechat(client, make_wav(3.0))
        second = post_voicechat(client, make_wav(3.0))
    assert first.status_code == 200
    assert second.status_code == 429
    assert second.json()["error"]["code"] == "rate_limited"


def test_voicechat_audit_records_metadata_not_transcripts(settings):
    client = make_app(vc_settings(settings), voicechat_client=FakeVoiceChatClient())
    with client:
        post_voicechat(client, make_wav(3.0))
        audit = client.get("/api/v1/audit").json()
    events = [e for e in audit["events"] if e.get("action") == "voicechat_turn"]
    assert events, "expected a voicechat_turn audit event"
    metadata = events[0].get("metadata") or {}
    assert metadata.get("synthetic") is True
    assert metadata.get("provider") == "voicechat"
    assert metadata.get("agent_transcript_chars") == len("Hello, this is a synthetic agent.")
    # Transcripts themselves are content: they must NOT be stored.
    assert "Hello, this is a synthetic agent." not in str(audit["events"])


def test_voicechat_storage_writes_provenance(settings, tmp_path):
    generated = tmp_path / "vc_generated"
    enabled = replace(
        vc_settings(settings),
        voicechat_store_generated=True,
        voicechat_generated_dir=str(generated),
    )
    client = make_app(enabled, voicechat_client=FakeVoiceChatClient())
    with client:
        response = post_voicechat(client, make_wav(3.0))
    assert response.status_code == 200
    manifest = generated / "provenance.jsonl"
    assert manifest.is_file()
    import json as _json

    row = _json.loads(manifest.read_text().strip().splitlines()[-1])
    assert row["label"] == "synthetic"
    assert row["partition"] == "generated"
    assert row["source"] == "voicechat_s2s"
    assert row["generator_id"] == "voicechat:nemotron-voicechat"
    # And the stored artifact really is the WAV bytes we returned.
    stored = generated / row["path"]
    assert stored.is_file()
    assert stored.read_bytes()[:4] == b"RIFF"


def test_providers_lists_both_generation_providers(settings):
    client = make_app(
        replace(settings, gnani_enabled=True, gnani_api_key="test-key"),
        voicechat_client=FakeVoiceChatClient(),
    )
    with client:
        response = client.get("/api/v1/providers")
    assert response.status_code == 200
    body = response.json()
    providers = {p["provider"]: p for p in body["generations"]}
    assert set(providers) == {"gnani", "voicechat"}
    for name, cap in providers.items():
        assert cap["detection_supported"] is False, name
    assert providers["gnani"]["generation_enabled"] is True
    assert providers["voicechat"]["generation_supported"] is True
    assert "NVIDIA" in providers["voicechat"]["api_name"]


def test_providers_with_neither_still_lists_capabilities(settings):
    client = make_app(settings)
    with client:
        response = client.get("/api/v1/providers")
    body = response.json()
    assert len(body["generations"]) == 2
    assert all(p["generation_enabled"] is False for p in body["generations"])
    assert body["detection_note"].startswith("Detection is performed by VoxSentinel")
