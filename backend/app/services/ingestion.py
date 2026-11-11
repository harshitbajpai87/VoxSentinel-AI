"""Shared audio-ingestion layer.

Every endpoint that accepts audio (``/api/v1/analyze``, ``/api/v1/speaker-verify``,
``/api/v1/protection/analyze``, ``/api/v1/protection/family-voices``) goes through
this module so one upload policy, one size ceiling and one set of error codes are
enforced everywhere.

Responsibilities
----------------
* bounded, chunk-by-chunk upload reading (never ``await file.read()`` unbounded)
* the *effective* size ceiling: ``min(configured, what the hosting platform can
  actually deliver)`` — a serverless host that caps request bodies below our
  configured limit must not make us claim 25 MB and then fail opaquely
* extension + signature + decode validation through ``audio_validation``
* decoder capability reporting (ffmpeg / ffprobe / libsndfile) for ``/health``
* one mapping from internal failures to the documented HTTP status codes

Nothing here trusts the client: the browser's ``Content-Type`` and filename are
hints only; the bytes are sniffed and decoded before they are used.
"""
from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
from dataclasses import dataclass, field

from fastapi import HTTPException, UploadFile

from app.core.config import Settings
from app.services.audio_validation import (
    AudioValidationError,
    DecodedAudio,
    strip_untrusted_path,
    validate_and_decode,
)

logger = logging.getLogger("voxsentinel.ingest")

#: Containers only ffmpeg can decode (libsndfile cannot).
FFMPEG_CONTAINERS = ("m4a", "mp4", "m4b", "aac", "webm", "opus", "oga", "mkv", "amr", "3gp", "wma")

#: One read chunk. 1 MiB keeps peak memory bounded without a syscall per byte.
READ_CHUNK_BYTES = 1024 * 1024

_DECODER_CACHE: dict[str, DecoderStatus] = {}
_DECODER_LOCK = threading.Lock()


def human_mb(num_bytes: int) -> str:
    """Format a byte count the way the API messages and the UI show it."""
    megabytes = num_bytes / (1024 * 1024)
    if megabytes >= 10:
        return f"{megabytes:.0f} MB"
    return f"{megabytes:.1f} MB"


@dataclass(frozen=True)
class DecoderStatus:
    """What this process can actually decode right now."""

    ffmpeg: bool
    ffprobe: bool
    ffmpeg_version: str | None = None
    ffprobe_version: str | None = None
    libsndfile: str | None = None
    libsndfile_formats: tuple[str, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)

    def supports(self, extension: str) -> bool:
        ext = (extension or "").lower()
        if ext in self.libsndfile_formats:
            return True
        if ext in FFMPEG_CONTAINERS and self.ffmpeg:
            return True
        return False

    def as_dict(self) -> dict:
        """Public, secret-free view for /health."""
        return {
            "ffmpeg": self.ffmpeg,
            "ffprobe": self.ffprobe,
            "ffmpeg_version": self.ffmpeg_version,
            "ffprobe_version": self.ffprobe_version,
            "libsndfile": self.libsndfile,
            "libsndfile_formats": list(self.libsndfile_formats),
            "m4a_aac_webm_opus": bool(self.ffmpeg),
            "notes": list(self.notes),
        }


def _binary_version(binary: str, settings: Settings) -> str | None:
    try:
        result = subprocess.run(
            [binary, "-version"],
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    first_line = result.stdout.decode("utf-8", "replace").splitlines()
    if first_line and first_line[0].strip():
        return first_line[0].strip()[:120]
    return settings.ffmpeg_binary


def probe_decoders(settings: Settings, *, use_cache: bool = True) -> DecoderStatus:
    """Detect ffmpeg, ffprobe and libsndfile coverage.

    Cached briefly so /health and every request do not spawn subprocesses; the
    cache key includes the configured binary names so a settings override in a
    test (or a config change) is honoured immediately.
    """
    key = f"{settings.ffmpeg_binary}|{settings.ffprobe_binary}"
    if use_cache:
        with _DECODER_LOCK:
            cached = _DECODER_CACHE.get(key)
        if cached is not None:
            return cached

    notes: list[str] = []
    ffmpeg_version = _binary_version(settings.ffmpeg_binary, settings)
    ffprobe_version = _binary_version(settings.ffprobe_binary, settings)
    if ffmpeg_version is None:
        notes.append(
            f"ffmpeg is not executable at '{settings.ffmpeg_binary}'; "
            "M4A/AAC/WebM/Opus uploads cannot be decoded by this process."
        )
    if ffprobe_version is None:
        notes.append(
            f"ffprobe is not executable at '{settings.ffprobe_binary}'; "
            "container metadata (sample rate, channels) is taken from the decode step instead."
        )

    libsndfile_version: str | None = None
    formats: tuple[str, ...] = ()
    try:
        import soundfile as sf

        libsndfile_version = sf.__libsndfile_version__
        formats = tuple(sorted(fmt.lower() for fmt in sf.available_formats()))
    except Exception as exc:  # noqa: BLE001 - optional dependency
        notes.append(f"soundfile/libsndfile unavailable ({type(exc).__name__}); native decoding is limited.")

    status = DecoderStatus(
        ffmpeg=ffmpeg_version is not None,
        ffprobe=ffprobe_version is not None,
        ffmpeg_version=ffmpeg_version,
        ffprobe_version=ffprobe_version,
        libsndfile=libsndfile_version,
        libsndfile_formats=formats,
        notes=tuple(notes),
    )
    if use_cache:
        with _DECODER_LOCK:
            _DECODER_CACHE[key] = status
    return status


def clear_decoder_cache() -> None:
    with _DECODER_LOCK:
        _DECODER_CACHE.clear()


def effective_upload_limit(settings: Settings) -> int:
    """The largest upload this deployment can genuinely accept."""
    return settings.effective_upload_limit_bytes()


def limit_message(settings: Settings, size_bytes: int | None = None, label: str | None = None) -> str:
    """Explain the real ceiling, including a tighter hosting limit.

    Requirement: the error must state the actual configured maximum instead of a
    hard-coded number, and must not blame the file for a platform restriction.
    """
    effective = effective_upload_limit(settings)
    subject = f"The {label} file" if label else "The uploaded file"
    if size_bytes is not None:
        message = (
            f"{subject} is {human_mb(size_bytes)}, above this deployment's "
            f"{human_mb(effective)} upload limit."
        )
    else:
        message = f"{subject} exceeds this deployment's {human_mb(effective)} upload limit."
    hosting = settings.hosting_body_limit_bytes
    if hosting and hosting < settings.max_upload_bytes:
        message += (
            f" The application is configured for {human_mb(settings.max_upload_bytes)}, but this "
            f"hosting platform rejects request bodies above {human_mb(hosting)} before the API "
            "sees them, so the smaller value applies. Deploy the container backend "
            "(deploy/docker) to accept the full 25 MB."
        )
    return message


async def read_upload_limited(
    file: UploadFile,
    settings: Settings,
    label: str = "audio",
) -> bytes:
    """Read an upload with a hard, streaming size ceiling.

    Reads in 1 MiB chunks and stops at the limit + 1 byte, so an arbitrarily
    large body is never buffered in full. ``UploadFile.size`` is used only as a
    fast pre-check; it is client-influenced, so the byte count is authoritative.
    """
    effective = effective_upload_limit(settings)
    if file.size is not None and file.size > effective:
        raise HTTPException(
            status_code=413,
            detail={"code": "file_too_large", "message": limit_message(settings, file.size, label)},
        )

    buffer = bytearray()
    while True:
        chunk = await file.read(READ_CHUNK_BYTES)
        if not chunk:
            break
        buffer.extend(chunk)
        if len(buffer) > effective:
            raise HTTPException(
                status_code=413,
                detail={"code": "file_too_large", "message": limit_message(settings, len(buffer), label)},
            )

    if not buffer:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_audio",
                "message": f"The {label} file is empty (0 bytes), so there is nothing to analyze.",
            },
        )
    return bytes(buffer)


def raise_for_audio_error(exc: AudioValidationError) -> None:
    """Map an internal validation failure onto the documented API contract."""
    raise HTTPException(
        status_code=exc.status_code,
        detail={"code": exc.code, "message": exc.message},
    )


async def ingest_audio(
    file: UploadFile,
    settings: Settings,
    label: str = "audio",
) -> tuple[DecodedAudio, str, int]:
    """Bounded read + full validation/decode for one upload.

    Returns ``(decoded, display_name, size_bytes)``. Raises ``HTTPException``
    with the contract error shape for every rejection.
    """
    data = await read_upload_limited(file, settings, label)
    display_name = strip_untrusted_path(file.filename)
    started = time.perf_counter()
    try:
        decoded = validate_and_decode(
            data,
            display_name,
            settings,
            content_type=file.content_type,
        )
    except AudioValidationError as exc:
        logger.info(
            "ingest rejected | endpoint_field=%s name=%s bytes=%d content_type=%s status=%d code=%s",
            label,
            display_name,
            len(data),
            (file.content_type or "unknown")[:60],
            exc.status_code,
            exc.code,
        )
        raise_for_audio_error(exc)
        raise  # unreachable; keeps type checkers happy
    logger.info(
        "ingest ok | field=%s name=%s bytes=%d type=%s rate=%d channels=%d "
        "duration_s=%.1f decode_ms=%d",
        label,
        display_name,
        len(data),
        (file.content_type or "unknown")[:60],
        decoded.sample_rate,
        decoded.channels,
        decoded.duration_s,
        int((time.perf_counter() - started) * 1000),
    )
    return decoded, display_name, len(data)


def audio_capabilities(settings: Settings) -> dict:
    """Secret-free capability block shared by /health and /protection/capabilities."""
    decoders = probe_decoders(settings)
    return {
        "decoders": decoders.as_dict(),
        "decodable_extensions": sorted(
            set(decoders.libsndfile_formats) | (set(FFMPEG_CONTAINERS) if decoders.ffmpeg else set())
        ),
        "max_upload_bytes": settings.max_upload_bytes,
        "effective_max_upload_bytes": effective_upload_limit(settings),
        "hosting_body_limit_bytes": settings.hosting_body_limit_bytes,
        "ffmpeg_binary": settings.ffmpeg_binary,
        "ffprobe_binary": settings.ffprobe_binary,
    }


def describe_environment() -> dict:
    """Non-secret facts about the runtime that matter for audio processing."""
    import os
    import sys

    return {
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "serverless": bool(os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME")),
    }


def json_log(event: str, **fields: object) -> None:
    """Structured one-line log entry (safe fields only, never audio/key data)."""
    payload = {"event": event, **fields}
    try:
        logger.info(json.dumps(payload, default=str, ensure_ascii=False))
    except Exception:  # noqa: BLE001 - logging must never raise
        logger.info("%s %s", event, fields)
