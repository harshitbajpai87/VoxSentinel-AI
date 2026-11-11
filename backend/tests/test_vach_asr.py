"""Tests for the verified Vach (Gnani) speech-to-text adapter.

Every test runs against an injected httpx.MockTransport, so no request touches
the network and no real credential is ever used. The important properties:

  * the provider's documented request shape is what we actually send
    (X-API-Key-ID header, ``audio_file`` + ``language_code`` multipart fields);
  * the 30-second audio cap is respected by chunking longer calls;
  * provider failures become typed errors and never leak the key;
  * nothing is fabricated when the provider is unavailable.
"""

from __future__ import annotations

import json

import httpx
import numpy as np
import pytest

from app.services import vach_asr
from app.services.vach_asr import (
    AUTO_LANGUAGE_CODE,
    MAX_CHUNK_SECONDS,
    VachAuthError,
    VachInvalidRequestError,
    VachRateLimitError,
    VachSttClient,
    build_vach_stt_client,
    plan_chunks,
    wav_bytes,
)

TEST_KEY = "vach_test_key_value_for_unit_tests_only"

SAMPLE_OK = {
    "success": True,
    "request_id": "req-1",
    "transcript": "Hello beta, this is your uncle. Send money urgently.",
    "model": "prisma-v3.0",
    "detected_language_code": "en-IN",
    "processing_time": 0.2,
}


def _client(handler, **kwargs) -> VachSttClient:
    return VachSttClient(
        TEST_KEY,
        base_url="https://api.vachana.ai",
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


def _tone(seconds: float, sample_rate: int = 16000) -> np.ndarray:
    """A deterministic non-silent signal of a known duration."""
    t = np.arange(int(seconds * sample_rate), dtype=np.float32) / sample_rate
    return (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


# ------------------------------------------------------------------ request shape
def test_request_matches_verified_provider_contract():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["url"] = str(request.url)
        seen["key_header"] = request.headers.get("X-API-Key-ID")
        body = request.read()
        seen["body"] = body
        return httpx.Response(200, json=SAMPLE_OK)

    client = _client(handler)
    result = client.transcribe(_tone(3.0), 16000)

    assert result.status == "completed"
    assert seen["method"] == "POST"
    assert seen["url"] == "https://api.vachana.ai/stt/v3"
    assert seen["key_header"] == TEST_KEY
    # multipart field names the provider actually requires
    assert b'name="audio_file"' in seen["body"]
    assert b'name="language_code"' in seen["body"]
    assert b"indic-auto" in seen["body"]
    # RIFF/WAVE payload, not raw PCM
    assert b"RIFF" in seen["body"]


def test_transcript_and_detected_language_are_reported():
    client = _client(lambda r: httpx.Response(200, json=SAMPLE_OK))
    result = client.transcribe(_tone(3.0), 16000)

    assert result.text == SAMPLE_OK["transcript"]
    assert result.language == "en-IN"
    assert result.model == "prisma-v3.0"
    assert result.chunk_count == 1
    assert result.segments and result.segments[0].start == 0.0
    assert "vach_" not in json.dumps(result.chunks)


def test_unsupported_language_code_falls_back_to_auto_detect():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.read()
        captured["body"] = body
        return httpx.Response(200, json=SAMPLE_OK)

    client = _client(handler)
    # "auto" and "hi-en" are rejected by the live provider; never send them.
    client.transcribe(_tone(2.0), 16000, language_code="hi-en")
    assert b"indic-auto" in captured["body"]
    assert b"hi-en" not in captured["body"]

    client.transcribe(_tone(2.0), 16000, language_code="hi-IN")
    assert b"hi-IN" in captured["body"]


# ----------------------------------------------------------------------- chunking
def test_long_audio_is_chunked_under_the_provider_cap():
    windows = plan_chunks(_tone(75.0), 16000)
    assert len(windows) >= 3
    for start, end in windows:
        assert 0 <= start < end <= 75 * 16000
        # never above the provider's verified 30 s hard limit
        assert (end - start) / 16000 <= MAX_CHUNK_SECONDS + 1e-6
    # contiguous coverage: no dropped audio
    assert windows[0][0] == 0
    assert windows[-1][1] == 75 * 16000
    for (_, prev_end), (next_start, _) in zip(windows, windows[1:]):
        assert prev_end == next_start


def test_each_chunk_is_a_separate_request_and_text_is_joined():
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(len(request.read()))
        payload = dict(SAMPLE_OK)
        payload["transcript"] = f"part {len(calls)}"
        return httpx.Response(200, json=payload)

    client = _client(handler)
    result = client.transcribe(_tone(70.0), 16000)

    assert len(calls) == result.chunk_count >= 3
    assert result.status == "completed"
    assert result.text == " ".join(f"part {i}" for i in range(1, len(calls) + 1))
    # segments carry absolute offsets, so the timeline is preserved
    assert result.segments[0].start == 0.0
    assert result.segments[-1].end > result.segments[0].end


def test_chunk_boundary_prefers_a_pause():
    """A silent gap before the cut point should become the boundary."""
    sample_rate = 16000
    # 26.5 s of signal, then a 1 s pause, then more signal. The ideal cut is
    # 28 s, which would land mid-word; the pause is the better boundary.
    loud = _tone(26.5, sample_rate)
    pause = np.zeros(int(1.0 * sample_rate), dtype=np.float32)
    audio = np.concatenate([loud, pause, _tone(20.0, sample_rate)])
    windows = plan_chunks(audio, sample_rate)

    assert len(windows) == 2
    cut = windows[0][1]
    assert int(26.5 * sample_rate) <= cut <= int(27.6 * sample_rate)


# --------------------------------------------------------------- error handling
def test_http_401_raises_auth_error_without_leaking_the_key():
    client = _client(lambda r: httpx.Response(401, json={"detail": "bad key"}))
    with pytest.raises(VachAuthError) as excinfo:
        client._post(_tone(2.0), "indic-auto")
    assert TEST_KEY not in str(excinfo.value)
    assert "vach_test" not in str(excinfo.value)


def test_http_429_raises_rate_limit_error():
    client = _client(lambda r: httpx.Response(429, json={"detail": "slow down"}))
    with pytest.raises(VachRateLimitError):
        client._post(_tone(2.0), "indic-auto")


def test_success_false_body_is_a_failure_not_a_transcript():
    body = {
        "success": False,
        "error": {"type": "AUDIO_CONVERSION_ERROR", "message": "Decoding failed."},
    }
    client = _client(lambda r: httpx.Response(400, json=body))
    with pytest.raises(VachInvalidRequestError):
        client._post(_tone(2.0), "indic-auto")


def test_provider_errors_never_crash_transcribe():
    """The protection pipeline must survive any provider failure."""
    for status, body in (
        (401, {"detail": "bad key"}),
        (429, {"detail": "slow down"}),
        (400, {"success": False, "error": {"type": "AUDIO_CONVERSION_ERROR"}}),
        (500, {"detail": "boom"}),
    ):
        client = _client(lambda r, s=status, b=body: httpx.Response(s, json=b))
        result = client.transcribe(_tone(2.0), 16000)
        assert result.status == "failed"
        assert result.text == ""
        assert result.error
        assert TEST_KEY not in json.dumps(result.chunks)


def test_partial_chunk_failure_is_reported_honestly():
    state = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["n"] += 1
        if state["n"] == 1:
            return httpx.Response(200, json=SAMPLE_OK)
        return httpx.Response(503, json={"detail": "unavailable"})

    client = _client(handler)
    result = client.transcribe(_tone(70.0), 16000)

    assert result.status == "completed"
    assert result.text == SAMPLE_OK["transcript"]
    assert any("partial" in line.lower() for line in result.limitations)


def test_all_chunks_failing_returns_failed_status():
    client = _client(lambda r: httpx.Response(500, json={"detail": "boom"}))
    result = client.transcribe(_tone(70.0), 16000)
    assert result.status == "failed"
    assert result.text == ""


def test_empty_audio_is_rejected_without_a_request():
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("no request should be sent for empty audio")

    client = _client(handler)
    result = client.transcribe(np.zeros(0, dtype=np.float32), 16000)
    assert result.status == "failed"


# --------------------------------------------------------------------- security
def test_describe_never_contains_the_key():
    client = _client(lambda r: httpx.Response(200, json=SAMPLE_OK))
    described = client.describe()
    assert described["configured"] is True
    assert described["provider"] == "vach"
    serialized = json.dumps(described)
    assert TEST_KEY not in serialized
    assert "vach_test" not in serialized


def test_without_a_key_the_client_reports_unavailable(monkeypatch):
    for name in (
        "VOXSENTINEL_VACH_API_KEY",
        "VOXSENTINEL_GNANI_API_KEY",
        "API_KEY",
        "GNANI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)

    assert build_vach_stt_client() is None

    client = VachSttClient("")
    result = client.transcribe(_tone(2.0), 16000)
    assert result.status == "unavailable"
    assert client.describe()["configured"] is False


def test_shared_api_key_is_used_when_no_explicit_vach_key(monkeypatch):
    for name in ("VOXSENTINEL_VACH_API_KEY", "VOXSENTINEL_GNANI_API_KEY", "GNANI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("API_KEY", "vach_from_shared_secret")

    client = build_vach_stt_client()
    assert client is not None
    assert client.is_configured


def test_explicit_vach_key_wins_over_shared_api_key(monkeypatch):
    monkeypatch.setenv("API_KEY", "vach_shared")
    monkeypatch.setenv("VOXSENTINEL_VACH_API_KEY", "vach_explicit")

    client = build_vach_stt_client()
    assert client is not None
    # the explicit value must be the one actually sent
    sent: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        sent["key"] = request.headers.get("X-API-Key-ID")
        return httpx.Response(200, json=SAMPLE_OK)

    client._transport = httpx.MockTransport(handler)
    client.transcribe(_tone(1.0), 16000)
    assert sent["key"] == "vach_explicit"


def test_non_https_base_url_is_refused():
    with pytest.raises(VachInvalidRequestError):
        VachSttClient(TEST_KEY, base_url="http://api.vachana.ai")


def test_wav_bytes_are_mono_16_bit():
    import io
    import wave

    payload = wav_bytes(_tone(1.0))
    with wave.open(io.BytesIO(payload), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getframerate() == 16000


def test_default_language_code_is_provider_auto_detect():
    assert AUTO_LANGUAGE_CODE == "indic-auto"
    assert "indic-auto" in vach_asr.SUPPORTED_LANGUAGE_CODES
    # values the live provider rejects must not be in our accepted list
    assert "auto" not in vach_asr.SUPPORTED_LANGUAGE_CODES
    assert "hi-en" not in vach_asr.SUPPORTED_LANGUAGE_CODES
