"""Server-side audio validation and decoding.

This module is the backend's own security boundary: the frontend's checks are
a convenience for the user, never trusted here. Uploaded bytes are validated,
decoded in memory, measured, and then discarded. Nothing is written to disk,
nothing is executed, and no audio is sent to any external service.

Supported containers depend on the installed libsndfile build. This project
verifies them at runtime (`soundfile.available_formats()`) and documents the
result instead of promising formats we cannot decode.
"""
from __future__ import annotations

import io
import wave
from dataclasses import dataclass, field

import numpy as np

from app.core.config import Settings

# Formats this service will attempt to decode. The first group is decoded by
# libsndfile (or the stdlib WAV reader); the second needs the ffmpeg fallback.
NATIVE_EXTENSIONS = ("wav", "flac", "ogg", "mp3")
SUPPORTED_EXTENSIONS = NATIVE_EXTENSIONS + ("m4a", "m4b", "mp4", "aac", "webm", "opus", "oga", "mkv")
# Containers only the ffmpeg fallback layer can decode (see ffmpeg_decoder).
FFMPEG_ONLY_EXTENSIONS = ("m4a", "m4b", "mp4", "aac", "webm", "opus", "oga", "mkv")

#: Human label for the formats decodable on THIS deployment (ffmpeg-aware).
def supported_formats_label(settings: Settings) -> str:
    from app.services import ffmpeg_decoder

    native = "WAV, MP3, FLAC, OGG"
    if ffmpeg_decoder.ffmpeg_available(settings):
        return f"{native}, M4A, MP4, AAC, WEBM, OPUS"
    return native


def _human_mb(num_bytes: int) -> str:
    megabytes = num_bytes / (1024 * 1024)
    return f"{megabytes:.0f} MB" if megabytes >= 10 else f"{megabytes:.1f} MB"
# Accepted by the frontend's preview but not decodable by this backend build
# (libsndfile has no AAC/M4A support). Rejected with a clear 415.
# Everything else a browser might send that we still cannot decode honestly.
KNOWN_UNSUPPORTED_EXTENSIONS = ()

_MAGIC_READ_SIZE = 16


class AudioValidationError(Exception):
    """A rejected upload. Carries the HTTP status the API should return."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


@dataclass
class DecodedAudio:
    """Decoded, measured audio. `samples` are mono float32 in [-1, 1]."""

    samples: np.ndarray
    sample_rate: int
    channels: int
    duration_s: float
    peak: float
    rms: float
    warnings: list[str] = field(default_factory=list)


def strip_untrusted_path(filename: str | None) -> str:
    """Reduce a client-supplied filename to a safe display name.

    Filenames are untrusted input. We never use them as filesystem paths;
    this only produces a short label for the UI, with any path components
    (including Windows separators) removed.
    """
    if not filename:
        return "audio"
    normalised = filename.replace("\\", "/").split("/")[-1].strip()
    normalised = "".join(ch for ch in normalised if ch.isprintable())
    return normalised[:255] or "audio"


def extension_of(filename: str) -> str:
    lowered = filename.lower()
    if "." not in lowered:
        return ""
    return lowered.rsplit(".", 1)[-1]


def sniff_format(data: bytes) -> str | None:
    """Best-effort content sniffing so a renamed .txt is not treated as audio."""
    if len(data) < 4:
        return None
    head = data[:_MAGIC_READ_SIZE]
    if head.startswith(b"RIFF") and data[8:12] == b"WAVE":
        return "wav"
    if head.startswith(b"fLaC"):
        return "flac"
    if head.startswith(b"OggS"):
        return "ogg"
    if head.startswith(b"ID3"):
        return "mp3"
    # ADTS AAC before the generic MPEG sync (0xFF F1/F8/F9 collide with it);
    # even a rare mis-sniff is harmless: the ffmpeg layer decodes either.
    if len(head) >= 2 and head[0] == 0xFF and head[1] in (0xF1, 0xF9, 0xF8):
        return "aac"
    if len(head) >= 2 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0:
        return "mp3"
    if data[4:8] == b"ftyp":
        return "m4a"
    # WebM / Matroska start with the EBML magic 0x1A45DFA3.
    if head.startswith(b"\x1aE\xdf\xa3"):
        return "webm"
    return None


def assert_finite(samples: np.ndarray) -> None:
    """Reject NaN/Inf samples, which would poison every downstream statistic."""
    if samples.size == 0:
        raise AudioValidationError(422, "invalid_audio", "The audio contains no samples.")
    if not bool(np.isfinite(samples).all()):
        raise AudioValidationError(
            422,
            "invalid_audio",
            "The audio contains non-finite sample values and cannot be analyzed.",
        )


def _to_mono_float32(data: np.ndarray) -> np.ndarray:
    array = np.asarray(data, dtype=np.float32)
    if array.ndim == 2:
        array = array.mean(axis=1)
    return np.clip(array.reshape(-1), -1.0, 1.0)


def _decode_with_soundfile(data: bytes) -> tuple[np.ndarray, int, int] | None:
    try:
        import soundfile as sf  # imported lazily so the API also runs without it
    except Exception:  # noqa: BLE001 - optional dependency
        return None
    try:
        with sf.SoundFile(io.BytesIO(data)) as handle:
            sample_rate = int(handle.samplerate)
            channels = int(handle.channels)
            frames = int(handle.frames)
            if frames <= 0:
                raise AudioValidationError(422, "invalid_audio", "The audio contains no samples.")
            array = handle.read(dtype="float32", always_2d=True)
    except AudioValidationError:
        raise
    except Exception as exc:  # noqa: BLE001 - any decoder failure is a 422
        raise AudioValidationError(
            422,
            "invalid_audio",
            "The audio could not be decoded. The file may be corrupted or use an unsupported encoding.",
        ) from exc
    return _to_mono_float32(array), sample_rate, channels


def _decode_wav_stdlib(data: bytes) -> tuple[np.ndarray, int, int]:
    """Fallback for PCM WAV when soundfile is unavailable."""
    try:
        with wave.open(io.BytesIO(data), "rb") as handle:
            sample_rate = handle.getframerate()
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            frames = handle.readframes(handle.getnframes())
    except Exception as exc:  # noqa: BLE001
        raise AudioValidationError(
            422,
            "invalid_audio",
            "The audio could not be decoded. The file may be corrupted or use an unsupported encoding.",
        ) from exc

    if width == 1:
        samples = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        samples = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise AudioValidationError(
            415, "unsupported_media_type", "Only 8, 16 and 32-bit PCM WAV files are supported."
        )

    if channels > 1:
        samples = samples.reshape(-1, channels)
    return _to_mono_float32(samples), sample_rate, channels


def decode_audio(data: bytes, fmt: str, settings: Settings) -> tuple[np.ndarray, int, int]:
    """Decode bytes into (mono samples, sample_rate, channels).

    Layer 1: libsndfile via soundfile (WAV/FLAC/OGG/MP3 and friends).
    Layer 2: stdlib wave for PCM WAV when libsndfile is unavailable.
    Layer 3: the ffmpeg fallback (M4A/AAC/WebM/Opus and libsndfile failures).
    The user is never told "supported" unless a layer actually decoded.
    """
    if fmt in ("wav", "flac", "ogg", "mp3"):
        decoded = _decode_with_soundfile(data)
        if decoded is not None:
            return decoded
        if fmt == "wav":
            try:
                return _decode_wav_stdlib(data)
            except AudioValidationError:
                pass  # fall through to the ffmpeg layer below

    # ffmpeg fallback: any container ffmpeg understands.
    from app.services import ffmpeg_decoder

    try:
        samples, sample_rate, channels = ffmpeg_decoder.decode_to_mono_f32(data, settings)
    except ffmpeg_decoder.FFmpegUnavailableError:
        # 503, not 415: the format IS supported by the application, the
        # deployed runtime is missing a required processing service. The client
        # can convert the clip locally and retry (see code decoder_unavailable).
        raise AudioValidationError(
            503,
            "decoder_unavailable",
            f"This deployment cannot decode {fmt.upper()} files: the ffmpeg decoder is not "
            "installed or not executable on this server. Formats decodable here: "
            f"{supported_formats_label(settings)}. Install ffmpeg (e.g. `apt-get install "
            "ffmpeg`) and restart the backend, or convert the file to WAV/MP3/FLAC/OGG first.",
        ) from None
    except TimeoutError as exc:
        raise AudioValidationError(
            422,
            "invalid_audio",
            "Audio decoding timed out on this server. The recording may be too long or "
            "the file may be malformed.",
        ) from exc
    except ffmpeg_decoder.AudioTooLongError as exc:
        raise AudioValidationError(422, "invalid_audio", str(exc)) from exc
    except ValueError:
        raise AudioValidationError(
            422,
            "invalid_audio",
            "The audio could not be decoded; it may be corrupted, truncated or use an "
            "unsupported codec.",
        ) from None
    return samples, sample_rate, channels


def measure(samples: np.ndarray, sample_rate: int, channels: int, settings: Settings) -> DecodedAudio:
    """Compute honest, reproducible measurements and quality findings."""
    assert_finite(samples)

    duration_s = float(samples.size) / float(sample_rate) if sample_rate > 0 else 0.0
    if sample_rate <= 0:
        raise AudioValidationError(422, "invalid_audio", "The audio reports an invalid sample rate.")

    peak = float(np.max(np.abs(samples)))
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))

    warnings: list[str] = []
    # Duration checks come first so the message names the real problem. A 0.4 s
    # clip or a truncated file is not "silent" — telling the user that sends them
    # looking for the wrong fix.
    if duration_s <= 0.2:
        raise AudioValidationError(
            422,
            "invalid_audio",
            "The file contains no decodable audio frames. It is probably truncated, "
            "corrupted or not really an audio file.",
        )
    if rms < settings.silence_rms:
        if duration_s < settings.min_duration_s:
            raise AudioValidationError(
                422,
                "invalid_audio",
                f"The recording is only {duration_s:.1f} s and contains no usable speech. "
                f"Upload at least {settings.min_duration_s:.0f} s of audio.",
            )
        raise AudioValidationError(
            422,
            "invalid_audio",
            "The recording appears to be silent, so there is no usable audio to analyze.",
        )
    if duration_s > settings.max_duration_s:
        raise AudioValidationError(
            422,
            "invalid_audio",
            f"The recording is longer than the {settings.max_duration_s:.0f} second limit.",
        )
    if duration_s < settings.min_duration_s:
        warnings.append(
            f"The recording is very short ({duration_s:.1f} s). Detection results may be unreliable."
        )
    if rms < settings.quiet_rms:
        warnings.append("The audio level is extremely low, which can make detection unreliable.")
    if peak >= settings.clipping_peak:
        warnings.append(
            "Samples reach maximum level; the recording may be clipped, which can distort detection features."
        )

    return DecodedAudio(
        samples=samples,
        sample_rate=sample_rate,
        channels=channels,
        duration_s=duration_s,
        peak=peak,
        rms=rms,
        warnings=warnings,
    )


def validate_and_decode(
    data: bytes,
    filename: str,
    settings: Settings,
    content_type: str | None = None,
) -> DecodedAudio:
    """Full server-side validation pipeline for one uploaded file.

    The declared ``content_type`` and the filename are HINTS: the container is
    identified from the bytes (``sniff_format``) and only a successful decode
    proves the upload is really audio.
    """
    if len(data) == 0:
        raise AudioValidationError(422, "invalid_audio", "The uploaded file is empty (0 bytes).")

    limit = settings.effective_upload_limit_bytes()
    if len(data) > limit:
        raise AudioValidationError(
            413,
            "file_too_large",
            f"The uploaded file is {_human_mb(len(data))}, above this deployment's "
            f"{_human_mb(limit)} upload limit.",
        )

    extension = extension_of(filename)
    detected = sniff_format(data)

    if detected is None:
        # Unknown signature. If the name claims audio we say the bytes are not
        # decodable audio; otherwise the format simply is not supported here.
        if extension in SUPPORTED_EXTENSIONS:
            raise AudioValidationError(
                415,
                "unsupported_media_type",
                "These bytes are not recognizable audio (no decodable container signature was "
                "found), so the file cannot be analyzed. If it came from a browser recording, "
                "re-record or convert it to WAV first.",
            )
        raise AudioValidationError(
            415,
            "unsupported_media_type",
            f"“{extension or filename}” is not a supported audio format. "
            f"Supported formats on this deployment: {supported_formats_label(settings)}.",
        )

    if extension and extension not in SUPPORTED_EXTENSIONS and extension not in FFMPEG_ONLY_EXTENSIONS:
        raise AudioValidationError(
            415,
            "unsupported_media_type",
            f"The file name extension “.{extension}” is not an accepted audio type, even though "
            f"its content looks like {detected.upper()}. Accepted extensions: "
            f"{supported_formats_label(settings)}.",
        )

    samples, sample_rate, channels = decode_audio(data, detected, settings)
    return measure(samples, sample_rate, channels, settings)
