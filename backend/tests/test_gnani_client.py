"""Unit tests for the Gnani/Vach provider adapter.

All HTTP is mocked with httpx.MockTransport: no test here performs a real
network request, and the fake key below is a TEST VALUE, never a secret.
"""
from __future__ import annotations

import json

import httpx
import pytest

from app.services.gnani_client import (
    TTS_INFERENCE_PATH,
    GnaniAuthError,
    GnaniBadRequestError,
    GnaniClient,
    GnaniConnectionError,
    GnaniMalformedResponseError,
    GnaniProviderError,
    GnaniRateLimitError,
    GnaniTimeoutError,
    GnaniUnavailableError,
    GnaniValidationError,
    TTSRequest,
)

FAKE_KEY = "test-key-do-not-use"
WAV_BYTES = b"RIFF-fake-audio-bytes"


def make_client(handler) -> GnaniClient:
    transport = httpx.MockTransport(handler)
    return GnaniClient(
        FAKE_KEY,
        transport_factory=lambda: httpx.AsyncClient(transport=transport),
    )


def ok_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, content=WAV_BYTES, headers={"Content-Type": "audio/wav"})


def test_tts_request_validates_fields_locally():
    with pytest.raises(GnaniValidationError):
        TTSRequest(text="", model="timbre-v2.5", voice="Nalini", language="hi-IN")
    with pytest.raises(GnaniValidationError):
        TTSRequest(text="hi", model="not-a-model", voice="Nalini", language="hi-IN")
    with pytest.raises(GnaniValidationError):
        TTSRequest(text="hi", model="timbre-v2.5", voice="Nalini", language="xx-XX")
    with pytest.raises(GnaniValidationError):
        TTSRequest(text="hi", model="timbre-v2.5", voice="Nalini", language="hi-IN", speed=2.0)
    with pytest.raises(GnaniValidationError):
        TTSRequest(text="hi", model="timbre-v2.5", voice="Nalini", language="hi-IN",
                   sample_rate=12345)
    with pytest.raises(GnaniValidationError):
        TTSRequest(text="hi", model="timbre-v2.5", voice="Nalini", language="hi-IN",
                   container="flac")


def test_tts_payload_matches_documented_contract():
    request = TTSRequest(text="hello", model="timbre-v2.5", voice="Nalini", language="hi-IN")
    payload = request.to_payload(max_text_chars=100)
    assert payload["text"] == "hello"
    assert payload["voice"] == "Nalini"
    assert payload["model"] == "timbre-v2.5"
    assert payload["language"] == "hi-IN"
    assert payload["audio_config"]["container"] == "wav"
    assert payload["audio_config"]["sample_rate"] == 48000
    assert payload["audio_config"]["num_channels"] == 1
    assert payload["audio_config"]["sample_width"] == 2
    # mp3 omits the encoding field (verified against the TTS reference).
    mp3 = TTSRequest(text="hi", model="timbre-v2.5", voice="Nalini", language="hi-IN",
                     container="mp3")
    assert "encoding" not in mp3.to_payload(100)["audio_config"]


def test_tts_payload_enforces_local_text_limit():
    request = TTSRequest(text="x" * 50, model="timbre-v2.5", voice="Nalini", language="hi-IN")
    with pytest.raises(GnaniValidationError):
        request.to_payload(max_text_chars=10)


@pytest.mark.anyio
async def test_synthesize_success_sends_verified_auth_and_path():
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["auth"] = request.headers.get("X-API-Key-ID")
        seen["body"] = json.loads(request.read())
        return httpx.Response(200, content=WAV_BYTES)

    client = make_client(handler)
    request = TTSRequest(text="hello", model="timbre-v2.5", voice="Nalini", language="hi-IN")
    audio = await client.synthesize(request)
    await client.aclose()

    assert seen["path"] == TTS_INFERENCE_PATH
    assert seen["auth"] == FAKE_KEY
    assert audio.audio == WAV_BYTES
    assert audio.container == "wav"
    assert audio.provider == "gnani"
    assert audio.model == "timbre-v2.5"
    assert audio.voice == "Nalini"
    assert audio.generator_id == "gnani:timbre-v2.5:Nalini"
    assert audio.is_synthetic is True


@pytest.mark.parametrize("status,exc", [
    (401, GnaniAuthError),
    (403, GnaniAuthError),
    (400, GnaniBadRequestError),
    (429, GnaniRateLimitError),
    (500, GnaniProviderError),
    (503, GnaniUnavailableError),
    (502, GnaniProviderError),
])
@pytest.mark.anyio
async def test_provider_error_statuses_map_to_typed_exceptions(status, exc):
    client = make_client(lambda request: httpx.Response(status, json={"detail": "boom"}))
    request = TTSRequest(text="hello", model="timbre-v2.5", voice="Nalini", language="hi-IN")
    with pytest.raises(exc):
        await client.synthesize(request)
    await client.aclose()


@pytest.mark.anyio
async def test_timeout_maps_to_typed_exception():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out")

    client = make_client(handler)
    request = TTSRequest(text="hello", model="timbre-v2.5", voice="Nalini", language="hi-IN")
    with pytest.raises(GnaniTimeoutError):
        await client.synthesize(request)
    await client.aclose()


@pytest.mark.anyio
async def test_connection_failure_maps_to_typed_exception():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("dns failure")

    client = make_client(handler)
    request = TTSRequest(text="hello", model="timbre-v2.5", voice="Nalini", language="hi-IN")
    with pytest.raises(GnaniConnectionError):
        await client.synthesize(request)
    await client.aclose()


@pytest.mark.anyio
async def test_empty_success_body_is_malformed():
    client = make_client(lambda request: httpx.Response(200, content=b""))
    request = TTSRequest(text="hello", model="timbre-v2.5", voice="Nalini", language="hi-IN")
    with pytest.raises(GnaniMalformedResponseError):
        await client.synthesize(request)
    await client.aclose()


def test_https_is_enforced_for_base_url():
    with pytest.raises(GnaniValidationError):
        GnaniClient(FAKE_KEY, base_url="http://api.vachana.ai")
    with pytest.raises(GnaniValidationError):
        GnaniClient("")


def test_oversized_response_is_rejected():
    big = b"a" * 1000
    client = GnaniClient(
        FAKE_KEY,
        max_response_bytes=100,
        transport_factory=lambda: httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, content=big))
        ),
    )
    import asyncio

    request = TTSRequest(text="hello", model="timbre-v2.5", voice="Nalini", language="hi-IN")
    with pytest.raises(GnaniMalformedResponseError):
        asyncio.run(client.synthesize(request))
