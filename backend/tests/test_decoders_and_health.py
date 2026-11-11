"""Decoders, diagnostics, connectivity and the error contract.

Covers the "Backend not running / Backend returned 413 / ffmpeg not installed"
class of reports:

* /health must describe decoders, the REAL upload ceiling and each optional
  service separately, so the UI can distinguish "offline" from "no decoder"
* a missing ffmpeg must produce a 503 `decoder_unavailable` naming the formats
  that still work, not a 415 that blames the file
* WAV/MP3/FLAC/OGG must keep working with ffmpeg absent (libsndfile coverage)
* CORS must let the app's own origin through, and every response must carry an
  X-Request-ID for log correlation
* every rejection must use the documented {error:{code,message}} envelope
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from app.services import ffmpeg_decoder
from app.services.ingestion import clear_decoder_cache
from tests.conftest import MockVoiceDetector, build_client, make_wav, post_analyze

FAKE_FFMPEG = "/nonexistent/ffmpeg-binary-for-tests"
FAKE_FFPROBE = "/nonexistent/ffprobe-binary-for-tests"


@pytest.fixture
def no_ffmpeg_settings(settings):
    """A deployment without ffmpeg/ffprobe — the slim serverless bundle."""
    clear_decoder_cache()
    ffmpeg_decoder._AVAILABILITY.clear()
    yield replace(settings, ffmpeg_binary=FAKE_FFMPEG, ffprobe_binary=FAKE_FFPROBE)
    ffmpeg_decoder._AVAILABILITY.clear()
    clear_decoder_cache()


def _m4a_bytes(tmp_path) -> bytes | None:
    """A real M4A/AAC clip, or None when ffmpeg is unavailable to make one."""
    import subprocess

    src = tmp_path / "src.wav"
    src.write_bytes(make_wav(3.0))
    out = tmp_path / "clip.m4a"
    try:
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-f", "ipod", "-c:a", "aac", str(out)],
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.read_bytes()


# ------------------------------------------------------------------- /health


def test_health_reports_decoders_and_the_real_upload_limit(client):
    body = client.get("/health").json()
    audio = body["audio"]
    assert audio["decoders"]["ffmpeg"] is True  # this machine has ffmpeg
    assert audio["max_upload_bytes"] == 25 * 1024 * 1024
    assert audio["effective_max_upload_bytes"] == 25 * 1024 * 1024
    assert "wav" in audio["decodable_extensions"]
    assert body["audio_decoder_ready"] is True
    assert body["environment"]["platform"]


def test_health_reports_each_optional_service_separately(client):
    body = client.get("/health").json()
    services = body["services"]
    assert set(services) >= {"detector", "speaker_verifier", "transcription", "contextual_ai"}
    # Optional providers are reported as false, never as failures.
    assert services["contextual_ai"] is False
    assert body["status"] in {"ok", "degraded"}


def test_health_survives_a_missing_decoder(no_ffmpeg_settings):
    with build_client(no_ffmpeg_settings, MockVoiceDetector()) as client:
        body = client.get("/health").json()
    assert body["audio"]["decoders"]["ffmpeg"] is False
    assert body["audio_decoder_ready"] is False
    assert any("ffmpeg" in note for note in body["audio"]["decoders"]["notes"])
    # A missing optional decoder must not make the API look down.
    assert body["backend_ready"] is True
    assert body["status"] == "ok"


def test_health_does_not_leak_environment_or_paths(client):
    text = client.get("/health").text
    for token in ("VOXSENTINEL_", "/app/", "API_KEY", "secret"):
        assert token not in text


# ------------------------------------------- behaviour without ffmpeg installed


def test_m4a_without_ffmpeg_is_a_named_503(no_ffmpeg_settings, tmp_path):
    data = _m4a_bytes(tmp_path)
    if data is None:
        pytest.skip("ffmpeg not available to build the M4A fixture")
    with build_client(no_ffmpeg_settings, MockVoiceDetector()) as client:
        response = post_analyze(client, data, "clip.m4a")
    assert response.status_code == 503
    payload = response.json()["error"]
    assert payload["code"] == "decoder_unavailable"
    assert "WAV" in payload["message"]
    assert "ffmpeg" in payload["message"].lower()


def test_wav_still_works_without_ffmpeg(no_ffmpeg_settings):
    """libsndfile coverage must never depend on the ffmpeg binary."""
    with build_client(no_ffmpeg_settings, MockVoiceDetector()) as client:
        response = post_analyze(client, make_wav(3.0), "clip.wav")
    assert response.status_code == 200, response.text


def test_upload_policy_reports_the_smaller_hosting_limit(settings):
    capped = replace(settings, hosting_body_limit_bytes=4_500_000)
    with build_client(capped, MockVoiceDetector()) as client:
        body = client.get("/health").json()
    assert body["max_upload_bytes"] == 25 * 1024 * 1024
    assert body["effective_max_upload_bytes"] == 4_500_000
    assert body["audio"]["hosting_body_limit_bytes"] == 4_500_000


# --------------------------------------------------- connectivity and contract


def test_cors_allows_the_app_origin(client):
    response = client.options(
        "/api/v1/analyze",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.status_code in (200, 204)
    assert response.headers.get("access-control-allow-origin") in {"*", "http://localhost:5173"}


def test_responses_carry_a_request_id(client):
    response = client.get("/health")
    assert response.headers.get("x-request-id")
    # A caller-supplied id is echoed so a support report can be traced end to end.
    echoed = client.get("/health", headers={"x-request-id": "abc123"})
    assert echoed.headers["x-request-id"] == "abc123"


@pytest.mark.parametrize(
    "payload,name,expected_status,expected_code",
    [
        (b"", "empty.wav", 422, "invalid_audio"),
        (b"not audio at all" * 20, "fake.wav", 415, "unsupported_media_type"),
        (make_wav(3.0), "clip.txt", 415, "unsupported_media_type"),
    ],
)
def test_error_envelope_is_documented(client, payload, name, expected_status, expected_code):
    response = post_analyze(client, payload, name)
    assert response.status_code == expected_status
    error = response.json()["error"]
    assert set(error) == {"code", "message"}
    assert error["code"] == expected_code
    assert error["message"]
