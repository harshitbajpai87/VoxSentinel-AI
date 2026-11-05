"""Server-side audio validation rules (unit + endpoint level)."""
from __future__ import annotations

import numpy as np
import pytest

from app.core.config import Settings
from app.services.audio_validation import (
    AudioValidationError,
    assert_finite,
    extension_of,
    measure,
    sniff_format,
    strip_untrusted_path,
    validate_and_decode,
)
from tests.conftest import (
    build_client,
    make_corrupt_wav,
    make_silent_wav,
    make_wav,
    post_analyze,
)

LIMITS = Settings()


# ---------------------------------------------------------------- helpers


def test_extension_of_handles_missing_and_upper_case():
    assert extension_of("clip.WAV") == "wav"
    assert extension_of("clip") == ""
    assert extension_of("archive.tar.gz") == "gz"


def test_strip_untrusted_path_removes_path_components():
    assert strip_untrusted_path("../../etc/passwd") == "passwd"
    assert strip_untrusted_path(r"C:\Windows\system32\evil.wav") == "evil.wav"
    assert strip_untrusted_path(None) == "audio"
    assert strip_untrusted_path("") == "audio"


def test_strip_untrusted_path_drops_control_characters():
    assert "\n" not in strip_untrusted_path("bad\nname.wav")


# ---------------------------------------------------------------- sniffing


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (make_wav(0.5), "wav"),
        (b"fLaC" + b"\x00" * 32, "flac"),
        (b"OggS" + b"\x00" * 32, "ogg"),
        (b"ID3\x04\x00\x00" + b"\x00" * 16, "mp3"),
        (b"\x00\x00\x00\x20ftypM4A " + b"\x00" * 8, "m4a"),
        (b"not audio at all", None),
        (b"", None),
    ],
)
def test_sniff_format_detects_containers(data, expected):
    assert sniff_format(data) == expected


# ---------------------------------------------------------------- finite samples


def test_assert_finite_rejects_nan_and_inf():
    with pytest.raises(AudioValidationError) as nan_error:
        assert_finite(np.array([0.1, np.nan], dtype=np.float32))
    assert nan_error.value.status_code == 422

    with pytest.raises(AudioValidationError):
        assert_finite(np.array([np.inf], dtype=np.float32))


def test_assert_finite_rejects_empty_arrays():
    with pytest.raises(AudioValidationError):
        assert_finite(np.array([], dtype=np.float32))


# ---------------------------------------------------------------- measure()


def test_measure_extracts_metadata_and_flags_short_audio():
    samples = (np.sin(np.arange(8000) / 10.0) * 0.5).astype(np.float32)
    decoded = measure(samples, 8000, 1, LIMITS)
    assert decoded.duration_s == pytest.approx(1.0)
    assert decoded.sample_rate == 8000
    assert decoded.channels == 1
    assert 0.4 < decoded.peak <= 0.5
    assert decoded.rms > 0.2
    assert any("very short" in w.lower() for w in decoded.warnings)


def test_measure_rejects_silence():
    with pytest.raises(AudioValidationError) as exc:
        measure(np.zeros(8000, dtype=np.float32), 8000, 1, LIMITS)
    assert exc.value.code == "invalid_audio"
    assert "silent" in exc.value.message.lower()


def test_measure_flags_quiet_and_clipped_audio():
    quiet = measure(np.full(8000, 0.007, dtype=np.float32), 8000, 1, LIMITS)
    assert any("extremely low" in w for w in quiet.warnings)

    clipped = measure(np.full(8000, 0.999, dtype=np.float32), 8000, 1, LIMITS)
    assert any("clipped" in w for w in clipped.warnings)


# ---------------------------------------------------------------- validate()


def test_validate_accepts_valid_wav():
    decoded = validate_and_decode(make_wav(3.0), "clip.wav", LIMITS)
    assert decoded.duration_s == pytest.approx(3.0, abs=0.01)
    assert decoded.sample_rate == 8000


def test_validate_rejects_empty_file():
    with pytest.raises(AudioValidationError) as exc:
        validate_and_decode(b"", "clip.wav", LIMITS)
    assert exc.value.status_code == 422


def test_validate_rejects_unsupported_extension():
    with pytest.raises(AudioValidationError) as exc:
        validate_and_decode(make_wav(2.0), "notes.txt", LIMITS)
    assert exc.value.status_code == 415


def test_validate_rejects_corrupt_m4a_with_clear_message():
    """M4A is SUPPORTED now (ffmpeg layer); a corrupt file must fail as
    invalid audio (422), not as unsupported format (415)."""
    with pytest.raises(AudioValidationError) as exc:
        validate_and_decode(b"\x00\x00\x00\x20ftypM4A " + b"\x00" * 64, "clip.m4a", LIMITS)
    assert exc.value.status_code == 422
    assert "decode" in exc.value.message.lower()


def test_validate_rejects_genuinely_unsupported_format():
    """An unknown extension is still rejected as unsupported (415)."""
    with pytest.raises(AudioValidationError) as exc:
        validate_and_decode(b"some binary blob" * 20, "clip.xyz", LIMITS)
    assert exc.value.status_code == 415


def test_validate_rejects_renamed_non_audio_file():
    """A .wav extension must not be enough: content is inspected too."""
    with pytest.raises(AudioValidationError) as exc:
        validate_and_decode(b"this is plain text, not audio" * 10, "fake.wav", LIMITS)
    assert exc.value.status_code == 415


def test_validate_rejects_corrupt_audio():
    with pytest.raises(AudioValidationError) as exc:
        validate_and_decode(make_corrupt_wav(), "corrupt.wav", LIMITS)
    assert exc.value.status_code == 422


def test_validate_rejects_silent_audio():
    with pytest.raises(AudioValidationError) as exc:
        validate_and_decode(make_silent_wav(3.0), "silent.wav", LIMITS)
    assert exc.value.status_code == 422


def test_validate_rejects_excessive_duration():
    short_limits = Settings(max_duration_s=2.0)
    with pytest.raises(AudioValidationError) as exc:
        validate_and_decode(make_wav(3.0), "long.wav", short_limits)
    assert exc.value.status_code == 422
    assert "longer than" in exc.value.message


def test_validate_enforces_size_limit():
    tiny_limits = Settings(max_upload_bytes=1000)
    with pytest.raises(AudioValidationError) as exc:
        validate_and_decode(make_wav(3.0), "big.wav", tiny_limits)
    assert exc.value.status_code == 413


# ---------------------------------------------------------------- endpoints


def test_endpoint_rejects_oversized_upload(settings, small_limit_settings):
    with build_client(small_limit_settings) as client:
        response = post_analyze(client, make_wav(5.0), "big.wav")
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "file_too_large"


def test_endpoint_rejects_unsupported_format(client):
    response = post_analyze(client, make_wav(3.0), "notes.txt")
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_media_type"


def test_endpoint_rejects_empty_upload(client):
    response = post_analyze(client, b"", "empty.wav")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_audio"


def test_endpoint_rejects_excessive_duration(short_limit_settings):
    with build_client(short_limit_settings) as client:
        response = post_analyze(client, make_wav(3.0), "long.wav")
        assert response.status_code == 422
        assert "longer than" in response.json()["error"]["message"]


def test_endpoint_rejects_missing_file(client):
    response = client.post("/api/v1/analyze", data={"consent": "true"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
