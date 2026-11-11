"""FFmpeg-based fallback decoding for containers libsndfile cannot read.

Security model (this module is reachable with arbitrary user-uploaded bytes):

* Uploads are written to a server-generated temporary file name inside a
  configured directory. The user-supplied filename is NEVER used as a path.
* ffmpeg / ffprobe are invoked as fixed argv lists (no shell, no user-supplied
  strings anywhere in the command line) with a hard timeout.
* Decoded PCM is read from stdout and parsed as float32; nothing is executed.
* The temporary file is ALWAYS removed, success or failure, and a stale-file
  sweeper removes files older than ``settings.temp_max_age_seconds`` for
  crash resilience.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.core.config import Settings

logger = logging.getLogger("voxsentinel.ffmpeg")

# Containers that only the ffmpeg layer can decode (everything libsndfile
# covers stays on the soundfile path).
FFMPEG_EXTENSIONS = ("m4a", "aac", "webm", "opus", "oga", "mp4", "mkv")


@dataclass(frozen=True)
class ProbedAudio:
    sample_rate: int
    channels: int
    duration_s: float
    codec: str


class FFmpegUnavailableError(RuntimeError):
    """Raised when the ffmpeg binary is not installed / not usable."""


class AudioTooLongError(ValueError):
    """Raised before decoding when a container exceeds the duration ceiling."""


_AVAILABILITY: dict[str, bool] = {}


def _ffmpeg_available(binary: str) -> bool:
    if binary in _AVAILABILITY:
        return _AVAILABILITY[binary]
    try:
        result = subprocess.run(
            [binary, "-version"],
            capture_output=True,
            timeout=10,
            check=False,
        )
        available = result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        available = False
    _AVAILABILITY[binary] = available
    return available


def ffmpeg_available(settings: Settings, *, refresh: bool = False) -> bool:
    """True when the configured ffmpeg binary can be executed.

    Cached: the answer only changes if the runtime does. ``refresh=True`` is
    used by diagnostics/tests that deliberately change PATH.
    """
    if refresh:
        _AVAILABILITY.pop(settings.ffmpeg_binary, None)
    return _ffmpeg_available(settings.ffmpeg_binary)


def probe_audio(data: bytes, settings: Settings) -> ProbedAudio:
    """ffprobe the raw upload; raises FFmpegUnavailableError if no binary."""
    binary = settings.ffprobe_binary
    path = _write_temp(data, settings)
    try:
        result = _run(
            [binary, "-v", "error", "-show_streams", "-show_format",
             "-of", "json", path],
            timeout=settings.ffmpeg_timeout_seconds,
        )
        info = json.loads(result.stdout.decode("utf-8", "replace") or "{}")
        streams = [s for s in info.get("streams", []) if s.get("codec_type") == "audio"]
        if not streams:
            raise ValueError("no audio stream found")
        stream = streams[0]
        fmt = info.get("format", {})
        try:
            duration = float(fmt.get("duration") or stream.get("duration") or 0.0)
        except (TypeError, ValueError):
            duration = 0.0
        return ProbedAudio(
            sample_rate=int(stream.get("sample_rate", 0) or 0),
            channels=int(stream.get("channels", 0) or 0),
            duration_s=duration,
            codec=str(stream.get("codec_name", "unknown")),
        )
    finally:
        _remove_temp(path)


def decode_to_mono_f32(data: bytes, settings: Settings) -> tuple[np.ndarray, int, int]:
    """Decode any ffmpeg-supported container to mono float32 at native rate.

    Returns (samples, sample_rate, channels_original). Raises
    FFmpegUnavailableError when ffmpeg is missing and a ValueError-family
    error is surfaced by the caller as an honest 415/422.
    """
    binary = settings.ffmpeg_binary
    path = _write_temp(data, settings)
    try:
        # Probe BEFORE decoding: a 10-minute 48 kHz stereo clip expands to
        # >100 MB of raw PCM, so the duration ceiling is enforced first and the
        # decoder is given an explicit upper time bound as a belt-and-braces
        # guard. Refusing early also stops a hostile upload from burning CPU.
        probed = _run(
            [settings.ffprobe_binary, "-v", "error", "-show_streams",
             "-select_streams", "a:0", "-of", "json", path],
            timeout=settings.ffmpeg_timeout_seconds,
        )
        info = json.loads(probed.stdout.decode("utf-8", "replace") or "{}")
        streams = info.get("streams") or []
        if not streams:
            raise ValueError("no audio stream found")
        sample_rate = int(streams[0].get("sample_rate", 0) or 0)
        channels = int(streams[0].get("channels", 0) or 0)
        if sample_rate <= 0 or channels <= 0:
            raise ValueError("ffprobe reported an invalid stream")

        duration = 0.0
        try:
            duration = float(streams[0].get("duration") or 0.0)
        except (TypeError, ValueError):
            duration = 0.0
        if duration and duration > settings.max_duration_s:
            raise AudioTooLongError(
                f"The recording is {duration:.0f} s, longer than the "
                f"{settings.max_duration_s:.0f} second limit."
            )
        bound = max(1.0, settings.max_duration_s + 1.0)

        result = _run(
            [binary, "-v", "error", "-nostdin",
             "-i", path, "-vn", "-map", "a:0",
             "-ac", "1", "-t", f"{bound:.3f}",
             "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1"],
            timeout=settings.ffmpeg_timeout_seconds,
        )
        if not result.stdout:
            raise ValueError("ffmpeg produced no audio samples")
        samples = np.frombuffer(result.stdout, dtype="<f4").astype(np.float32, copy=True)
        return samples, sample_rate, channels
    finally:
        _remove_temp(path)


def cleanup_stale_temp_files(settings: Settings, now: float | None = None) -> int:
    """Remove leftover temp uploads older than the configured age.

    Returns the number of removed files. Crash-safe and idempotent: the
    happy path always removes its own temp file, so this sweeper only ever
    sees files from hard crashes / kills.
    """
    now = now if now is not None else time.time()
    directory = Path(settings.temp_upload_dir)
    if not directory.is_dir():
        return 0
    removed = 0
    prefix = "voxsentinel_upload_"
    for entry in directory.iterdir():
        if not entry.is_file() or not entry.name.startswith(prefix):
            continue
        try:
            age = now - entry.stat().st_mtime
        except OSError:
            continue
        if age > settings.temp_max_age_seconds:
            try:
                entry.unlink()
                removed += 1
            except OSError:
                logger.warning("Could not remove stale temp file %s", entry.name)
    return removed


def _write_temp(data: bytes, settings: Settings) -> str:
    directory = Path(settings.temp_upload_dir)
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError:
        # Read-only bundle filesystem (serverless hosts). The upload itself is
        # fine; only the scratch copy cannot live where it was configured, so
        # fall back to the platform temp directory instead of failing the user.
        logger.info("configured temp dir is not writable; using the system temp dir")
        directory = Path(tempfile.gettempdir())
    fd, path = tempfile.mkstemp(prefix="voxsentinel_upload_", dir=str(directory))
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
    return path


def _remove_temp(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        logger.warning("Failed to remove temp file during cleanup", exc_info=True)


def _run(argv: list[str], timeout: int) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(argv, capture_output=True, timeout=timeout, check=False)
    except FileNotFoundError as exc:
        raise FFmpegUnavailableError(f"decoder binary not found: {argv[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError("audio decoding timed out") from exc
    if result.returncode != 0:
        raise ValueError("decoder failed on this file")
    return result
