"""Upload-size policy tests: one ceiling, enforced at every layer.

These are real HTTP uploads through the FULL FastAPI stack (multipart parsing,
the shared ingestion layer, validation, decode and the detector), using the
clearly-labelled MockVoiceDetector test double from conftest. No real model is
involved and no fixture is presented as detection performance.

The point of this file is the size boundary itself:
  * 1 / 5.5 / 10 / 20 / 24 MB and exactly 25 MB must be ACCEPTED
  * above 25 MB must be a 413 that states the real configured limit
  * a 5.5 MB file must never be rejected by an unintended smaller limit
  * a tighter hosting limit must be reported honestly, not silently swallowed
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from app.core.config import Settings
from app.services.ingestion import read_upload_limited
from tests.conftest import MockVoiceDetector, build_client, make_wav, post_analyze

LIMIT_BYTES = 25 * 1024 * 1024
# 24 kHz mono PCM16 = 96 000 bytes/s, so the duration for a target size is exact.
BYTES_PER_SECOND = 24_000 * 2


def wav_of_size(target_bytes: int, sample_rate: int = 24_000) -> bytes:
    """A real, decodable WAV of approximately ``target_bytes`` (header included)."""
    bps = sample_rate * 2
    # 44-byte RIFF/fmt/data header counts toward the upload size.
    duration = max(0.5, (target_bytes - 44) / bps)
    return make_wav(duration_s=duration, sample_rate=sample_rate, freq=180.0)


@pytest.fixture
def long_ok_settings(settings) -> Settings:
    """Same deployment, but the duration ceiling does not mask the size ceiling."""
    return replace(settings, max_duration_s=7200.0)


@pytest.mark.parametrize(
    "target_mb",
    [1, 5.5, 10, 20, 24],
    ids=["1MB", "5.5MB", "10MB", "20MB", "24MB"],
)
def test_files_below_limit_are_accepted(long_ok_settings, target_mb):
    payload = wav_of_size(int(target_mb * 1024 * 1024))
    assert len(payload) <= LIMIT_BYTES
    with build_client(long_ok_settings, MockVoiceDetector()) as client:
        response = post_analyze(client, payload, f"clip_{target_mb}mb.wav")
    assert response.status_code == 200, response.text
    assert response.json()["input"]["sample_rate_hz"] == 24_000


def test_five_point_five_mb_is_never_rejected(long_ok_settings):
    """The reported bug: a 5.5 MB clip was refused as too large."""
    payload = wav_of_size(int(5.5 * 1024 * 1024))
    assert len(payload) < LIMIT_BYTES
    with build_client(long_ok_settings, MockVoiceDetector()) as client:
        response = post_analyze(client, payload, "five_and_a_half.wav")
    assert response.status_code == 200, response.text


def test_exactly_at_limit_is_accepted(long_ok_settings):
    """Exactly 25 MB: the multipart envelope must not count against the file."""
    payload = wav_of_size(LIMIT_BYTES)
    assert LIMIT_BYTES - 100 <= len(payload) <= LIMIT_BYTES
    with build_client(long_ok_settings, MockVoiceDetector()) as client:
        response = post_analyze(client, payload, "exactly_25mb.wav")
    assert response.status_code == 200, response.text


def test_just_below_limit_is_accepted(long_ok_settings):
    payload = wav_of_size(LIMIT_BYTES - 4096)
    with build_client(long_ok_settings, MockVoiceDetector()) as client:
        response = post_analyze(client, payload, "just_below.wav")
    assert response.status_code == 200, response.text


def test_above_limit_returns_413_with_the_real_limit(long_ok_settings):
    payload = wav_of_size(LIMIT_BYTES + 1024 * 1024)
    assert len(payload) > LIMIT_BYTES
    with build_client(long_ok_settings, MockVoiceDetector()) as client:
        response = post_analyze(client, payload, "too_big.wav")
    assert response.status_code == 413
    body = response.json()["error"]
    assert body["code"] == "file_too_large"
    # The response must state the actual configured ceiling, not a vague "too large".
    assert "25 MB" in body["message"]


def test_hosting_limit_is_reported_honestly(settings):
    """A platform body cap below our own limit must be named, not hidden."""
    capped = replace(
        settings,
        max_duration_s=7200.0,
        hosting_body_limit_bytes=4_500_000,
    )
    payload = wav_of_size(int(5.5 * 1024 * 1024))
    with build_client(capped, MockVoiceDetector()) as client:
        response = post_analyze(client, payload, "five_and_a_half.wav")
    assert response.status_code == 413
    message = response.json()["error"]["message"]
    assert "4.3 MB" in message or "4.2 MB" in message
    assert "25 MB" in message
    assert "hosting platform" in message


def test_effective_limit_is_the_minimum(settings):
    capped = replace(settings, hosting_body_limit_bytes=1_000_000)
    assert capped.effective_upload_limit_bytes() == 1_000_000
    uncapped = replace(settings, hosting_body_limit_bytes=None)
    assert uncapped.effective_upload_limit_bytes() == settings.max_upload_bytes


def test_empty_upload_is_rejected(settings):
    with build_client(settings, MockVoiceDetector()) as client:
        response = post_analyze(client, b"", "empty.wav")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_audio"


class _FakeUpload:
    """Minimal UploadFile stand-in that records how much was actually read."""

    def __init__(self, total_bytes: int, chunk: int = 1024 * 1024) -> None:
        self.size = None  # streaming uploads report no size
        self.filename = "stream.wav"
        self.content_type = "audio/wav"
        self._total = total_bytes
        self._chunk = chunk
        self._sent = 0
        self.bytes_read = 0
        self.reads = 0

    async def read(self, n: int = -1) -> bytes:
        self.reads += 1
        amount = min(self._chunk if n < 0 else n, self._total - self._sent)
        if amount <= 0:
            return b""
        self._sent += amount
        self.bytes_read += amount
        return b"\x00" * amount


@pytest.mark.anyio
async def test_streaming_reader_stops_after_the_limit(settings):
    """An oversized body must be abandoned, not buffered in full."""
    upload = _FakeUpload(total_bytes=200 * 1024 * 1024)
    with pytest.raises(Exception) as exc:
        await read_upload_limited(upload, settings, "audio")
    status = getattr(exc.value, "status_code", None)
    assert status == 413
    # Stops within one chunk of the ceiling instead of reading 200 MB.
    assert upload.bytes_read <= settings.max_upload_bytes + 1024 * 1024
    assert upload.bytes_read < upload._total


@pytest.mark.anyio
async def test_streaming_reader_accepts_a_body_within_the_limit(settings):
    upload = _FakeUpload(total_bytes=5 * 1024 * 1024)
    data = await read_upload_limited(upload, settings, "audio")
    assert len(data) == 5 * 1024 * 1024
