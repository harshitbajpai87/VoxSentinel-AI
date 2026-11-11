"""Issue C/D: Speaker-verification uploads, sizes and decoding.

Reproduces the two reported failures against the real endpoint:

* C — "This server cannot decode M4A files: ffmpeg is not installed" (and a
  genuine HTTP 500 when the embedding stack was missing)
* D — HTTP 413 on the comparison endpoint, caused by the COMBINED multipart body
  being measured instead of each file

Every file is size-checked and decoded independently, and the error must name
which recording failed. Model-free paths are exercised with the Null verifier
(HTTP 200 + explicit `unavailable`) and the decode paths run for real.
"""
from __future__ import annotations

import subprocess
from dataclasses import replace

import pytest

from app.services import ffmpeg_decoder
from app.services.ingestion import clear_decoder_cache
from tests.conftest import build_client, make_wav

LIMIT_BYTES = 25 * 1024 * 1024
FAKE_FFMPEG = "/nonexistent/ffmpeg-binary-for-tests"
FAKE_FFPROBE = "/nonexistent/ffprobe-binary-for-tests"


def post_verify(
    client,
    reference: bytes,
    test: bytes,
    *,
    consent="true",
    authorized="true",
    ref_name="ref.wav",
    test_name="test.wav",
):
    return client.post(
        "/api/v1/speaker-verify",
        files={
            "reference": (ref_name, reference, "audio/wav"),
            "test": (test_name, test, "audio/wav"),
        },
        data={"consent": consent, "reference_authorized": authorized},
    )


def wav_of_size(target_bytes: int, sample_rate: int = 24_000) -> bytes:
    bps = sample_rate * 2
    return make_wav(duration_s=max(0.5, (target_bytes - 44) / bps), sample_rate=sample_rate)


@pytest.fixture
def client_long(settings):
    """Deployment whose duration ceiling does not mask size behaviour."""
    with build_client(replace(settings, max_duration_s=7200.0)) as client:
        yield client


def _encode(tmp_path, args: list[str], suffix: str) -> bytes | None:
    src = tmp_path / "src.wav"
    src.write_bytes(make_wav(4.0))
    out = tmp_path / f"clip{suffix}"
    try:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), *args, str(out)], check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.read_bytes()


# ---------------------------------------------------------------- happy paths


def test_two_wavs_are_compared(client_long):
    response = post_verify(client_long, make_wav(3.0), make_wav(3.0))
    assert response.status_code == 200, response.text
    body = response.json()
    # No model installed here: explicit unavailable, never a fabricated number.
    assert body["status"] in {"completed", "unavailable", "partial"}
    if body["status"] == "unavailable":
        assert body["similarity"] is None


def test_m4a_reference_is_decoded(client_long, tmp_path):
    data = _encode(tmp_path, ["-f", "ipod", "-c:a", "aac"], ".m4a")
    if data is None:
        pytest.skip("ffmpeg not available to build the M4A fixture")
    response = client_long.post(
        "/api/v1/speaker-verify",
        files={
            "reference": ("ref.m4a", data, "audio/x-m4a"),
            "test": ("test.wav", make_wav(3.0), "audio/wav"),
        },
        data={"consent": "true", "reference_authorized": "true"},
    )
    assert response.status_code == 200, response.text
    assert "ffmpeg is not installed" not in response.text


def test_both_recordings_may_be_m4a(client_long, tmp_path):
    data = _encode(tmp_path, ["-f", "ipod", "-c:a", "aac"], ".m4a")
    if data is None:
        pytest.skip("ffmpeg not available to build the M4A fixture")
    response = client_long.post(
        "/api/v1/speaker-verify",
        files={
            "reference": ("ref.m4a", data, "audio/mp4"),
            "test": ("test.m4a", data, "audio/mp4"),
        },
        data={"consent": "true", "reference_authorized": "true"},
    )
    assert response.status_code == 200, response.text


# -------------------------------------------------------------- size handling


def test_twenty_mb_reference_is_accepted(client_long):
    response = post_verify(client_long, wav_of_size(20 * 1024 * 1024), make_wav(3.0))
    assert response.status_code == 200, response.text


def test_combined_request_may_exceed_the_per_file_limit(client_long):
    """Two 20 MB files = a 40 MB body; each file is still under the 25 MB rule."""
    response = post_verify(
        client_long,
        wav_of_size(20 * 1024 * 1024),
        wav_of_size(20 * 1024 * 1024),
    )
    assert response.status_code == 200, response.text


def test_oversized_reference_is_named(client_long):
    response = post_verify(client_long, wav_of_size(LIMIT_BYTES + 1024 * 1024), make_wav(3.0))
    assert response.status_code == 413
    error = response.json()["error"]
    assert error["code"] == "file_too_large"
    assert "reference" in error["message"].lower()
    assert "25 MB" in error["message"]


def test_oversized_test_recording_is_named(client_long):
    response = post_verify(client_long, make_wav(3.0), wav_of_size(LIMIT_BYTES + 1024 * 1024))
    assert response.status_code == 413
    assert "test" in response.json()["error"]["message"].lower()


# ------------------------------------------------------ validation and consent


def test_corrupted_reference_is_rejected(client_long):
    response = post_verify(client_long, b"\x00\x11\x22\x33" * 200, make_wav(3.0))
    assert response.status_code in (415, 422)
    assert response.json()["error"]["code"] in {"invalid_audio", "unsupported_media_type"}


def test_missing_test_recording_is_a_validation_error(client_long):
    response = client_long.post(
        "/api/v1/speaker-verify",
        files={"reference": ("ref.wav", make_wav(3.0), "audio/wav")},
        data={"consent": "true", "reference_authorized": "true"},
    )
    assert response.status_code == 422
    assert "test" in response.text.lower()


def test_consent_is_required(client_long):
    response = post_verify(client_long, make_wav(3.0), make_wav(3.0), consent="false")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "missing_consent"


def test_reference_authorization_is_required(client_long):
    response = post_verify(client_long, make_wav(3.0), make_wav(3.0), authorized="false")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "missing_authorization"


def test_silent_reference_is_rejected(client_long):
    response = post_verify(client_long, make_wav(3.0, amplitude=0.0), make_wav(3.0))
    assert response.status_code == 422


# --------------------------------------------------- deployment without ffmpeg


def test_m4a_without_ffmpeg_is_a_named_503(settings, tmp_path):
    data = _encode(tmp_path, ["-f", "ipod", "-c:a", "aac"], ".m4a")
    if data is None:
        pytest.skip("ffmpeg not available to build the M4A fixture")
    clear_decoder_cache()
    ffmpeg_decoder._AVAILABILITY.clear()
    bare = replace(settings, ffmpeg_binary=FAKE_FFMPEG, ffprobe_binary=FAKE_FFPROBE)
    try:
        with build_client(bare) as client:
            response = client.post(
                "/api/v1/speaker-verify",
                files={
                    "reference": ("ref.m4a", data, "audio/x-m4a"),
                    "test": ("test.wav", make_wav(3.0), "audio/wav"),
                },
                data={"consent": "true", "reference_authorized": "true"},
            )
    finally:
        ffmpeg_decoder._AVAILABILITY.clear()
        clear_decoder_cache()
    assert response.status_code == 503
    error = response.json()["error"]
    assert error["code"] == "decoder_unavailable"
    assert "reference" in error["message"].lower() or "M4A" in error["message"]


def test_no_model_never_returns_500(client_long):
    """The reported white-screen case: missing stack must be unavailable, not 500."""
    response = post_verify(client_long, make_wav(3.0), make_wav(3.0))
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"completed", "unavailable", "partial"}
    assert "internal_error" not in response.text
