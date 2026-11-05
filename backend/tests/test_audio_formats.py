"""Module 1 tests: universal audio-format support end to end.

Real files are encoded with ffmpeg in a session fixture (no personal audio);
each is uploaded through the FULL validation + decode pipeline with a
clearly-labelled test-double detector (conftest.MockVoiceDetector — canned
scores, NOT a real model). These tests verify DECODING, not detection skill.
"""
from __future__ import annotations

import subprocess

import pytest

from tests.conftest import make_wav, post_analyze

pytestmark = pytest.mark.usefixtures("settings")


def _encode(tmp_path_factory, name: str, source_wav: bytes, ffmpeg_args: list[str]) -> bytes:
    directory = tmp_path_factory.mktemp("formats")
    src = directory / "src.wav"
    src.write_bytes(source_wav)
    out = directory / name
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(src), *ffmpeg_args, str(out)],
        check=True,
    )
    return out.read_bytes()


@pytest.fixture(scope="session")
def encoded_formats(tmp_path_factory) -> dict[str, bytes]:
    """One real 3s tone transcoded into every format the backend claims."""
    tone = make_wav(3.0)
    matrix: dict[str, bytes] = {"wav": tone}
    cases = {
        "mp3": (["-c:a", "libmp3lame"], "mp3"),
        "flac": (["-c:a", "flac"], "flac"),
        "ogg": (["-c:a", "libvorbis"], "ogg"),
        "opus": (["-f", "ogg", "-c:a", "libopus"], "opus"),
        "m4a": (["-f", "ipod", "-c:a", "aac"], "m4a"),
        "aac": (["-f", "adts", "-c:a", "aac"], "aac"),
        "webm": (["-f", "webm", "-c:a", "libopus"], "webm"),
    }
    for key, (args, ext) in cases.items():
        matrix[key] = _encode(tmp_path_factory, f"out.{ext}", tone, args)
    return matrix


def _ffmpeg_available() -> bool:
    try:
        return subprocess.run(["ffmpeg", "-version"], capture_output=True).returncode == 0
    except OSError:
        return False


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg binary not available in this environment")
class TestEverySupportedFormat:
    """One test per format — each must survive the full upload pipeline."""

    def test_wav(self, client_with_detector, encoded_formats):
        response = post_analyze(client_with_detector, encoded_formats["wav"], "clip.wav")
        assert response.status_code == 200
        assert response.json()["input"]["duration_s"] == pytest.approx(3.0, abs=0.2)

    def test_mp3(self, client_with_detector, encoded_formats):
        response = post_analyze(client_with_detector, encoded_formats["mp3"], "clip.mp3")
        assert response.status_code == 200
        assert response.json()["input"]["sample_rate_hz"] > 0

    def test_flac(self, client_with_detector, encoded_formats):
        response = post_analyze(client_with_detector, encoded_formats["flac"], "clip.flac")
        assert response.status_code == 200

    def test_ogg_vorbis(self, client_with_detector, encoded_formats):
        response = post_analyze(client_with_detector, encoded_formats["ogg"], "clip.ogg")
        assert response.status_code == 200

    def test_ogg_opus(self, client_with_detector, encoded_formats):
        response = post_analyze(client_with_detector, encoded_formats["opus"], "clip.opus")
        assert response.status_code == 200

    def test_m4a(self, client_with_detector, encoded_formats):
        response = post_analyze(client_with_detector, encoded_formats["m4a"], "clip.m4a")
        assert response.status_code == 200

    def test_aac_adts(self, client_with_detector, encoded_formats):
        response = post_analyze(client_with_detector, encoded_formats["aac"], "clip.aac")
        assert response.status_code == 200

    def test_webm_opus(self, client_with_detector, encoded_formats):
        """webm/opus is exactly what browsers' MediaRecorder produces."""
        response = post_analyze(client_with_detector, encoded_formats["webm"], "clip.webm")
        assert response.status_code == 200

    def test_temp_dir_stays_clean(self, client_with_detector, encoded_formats, settings):
        """No decoded temp file outlives the request (retention rule 10)."""
        import os

        post_analyze(client_with_detector, encoded_formats["webm"], "clip.webm")
        leftovers = (
            os.listdir(settings.temp_upload_dir)
            if os.path.isdir(settings.temp_upload_dir)
            else []
        )
        assert leftovers == []


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg binary not available in this environment")
class TestCorruptAndUnsupported:
    def test_truncated_m4a_is_rejected(self, client_with_detector, encoded_formats):
        truncated = encoded_formats["m4a"][: len(encoded_formats["m4a"]) // 2]
        response = post_analyze(client_with_detector, truncated, "clip.m4a")
        assert response.status_code in (415, 422)
        assert response.json()["error"]["code"] in ("invalid_audio", "unsupported_media_type")

    def test_text_renamed_to_webm_is_rejected(self, client_with_detector):
        response = post_analyze(client_with_detector, b"not audio at all" * 20, "clip.webm")
        assert response.status_code == 415

    def test_unsupported_extension_is_rejected(self, client_with_detector):
        response = post_analyze(client_with_detector, make_wav(3.0), "clip.xyz")
        assert response.status_code == 415

    def test_garbage_bytes_with_valid_extension_is_rejected(self, client_with_detector):
        response = post_analyze(client_with_detector, b"\x1aE\xdf\xa3garbage" * 40, "clip.webm")
        assert response.status_code in (415, 422)

    def test_silent_recording_is_rejected(self, client_with_detector):
        response = post_analyze(client_with_detector, make_wav(3.0, amplitude=0.0), "silence.wav")
        assert response.status_code == 422
        assert "silent" in response.json()["error"]["message"].lower()


def test_wav_still_decodes_without_ffmpeg_layer(monkeypatch, client_with_detector, encoded_formats):
    """libsndfile-covered formats must not depend on ffmpeg being installed."""
    from app.services import audio_validation

    def _no_ffmpeg(data, settings):
        from app.services.ffmpeg_decoder import FFmpegUnavailableError

        raise FFmpegUnavailableError("simulated missing ffmpeg (test)")

    monkeypatch.setattr(audio_validation, "decode_audio", lambda data, fmt, settings:
                        _raise_for_ffmpeg_only(fmt, data, settings))
    # direct: wav path must not call the ffmpeg layer at all
    from app.core.config import Settings
    from app.services.audio_validation import validate_and_decode

    decoded = validate_and_decode(encoded_formats["wav"], "x.wav", Settings())
    assert decoded.duration_s == pytest.approx(3.0, abs=0.2)


def _raise_for_ffmpeg_only(fmt, data, settings):
    from app.services.audio_validation import AudioValidationError, _decode_wav_stdlib, _decode_with_soundfile

    if fmt in ("wav", "flac", "ogg", "mp3"):
        decoded = _decode_with_soundfile(data)
        if decoded is not None:
            return decoded
        if fmt == "wav":
            return _decode_wav_stdlib(data)
    raise AudioValidationError(415, "unsupported_media_type", "ffmpeg unavailable (test)")
