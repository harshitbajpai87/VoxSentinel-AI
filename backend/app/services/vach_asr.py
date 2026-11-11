"""Verified Vach (Gnani) speech-to-text adapter — server-side only.

This is the real ASR client for the credential the deployment provides. The
contract below was verified against the live provider on 2026-11-11, not
guessed from the key format:

    POST https://api.vachana.ai/stt/v3
    header: X-API-Key-ID: <key>
    multipart: audio_file=<wav>, language_code=<code>
    200 -> {"success": true, "transcript": "...", "model": "prisma-v3.0",
            "detected_language_code": "en-IN", ...}
    400 -> {"success": false, "error": {"type": "...", "message": "..."}}

Verified provider facts that shape this module:

  * the audio cap is **30 seconds** per request (``MAX_AUDIO_DURATION_EXCEEDED``
    above it), so longer calls are split into chunks here and rejoined;
  * accepted language codes are ``as-IN, bn-IN, en-IN, gu-IN, hi-IN,
    indic-auto, kn-IN, ml-IN, mr-IN, pa-IN, ta-IN, te-IN`` — ``auto`` and
    ``hi-en`` are rejected, and ``indic-auto`` is the auto-detect option;
  * ``indic-auto`` returns ``detected_language_code`` and punctuated text;
  * a 200 response with ``success: false`` is still a failure.

Design rules, matching the rest of the codebase:

  * the key is read from the backend environment only. It is never logged,
    echoed in an exception message, returned in a status payload, or put in a
    frontend bundle. ``describe()`` reports ``configured`` as a boolean.
  * the HTTP transport is injectable so tests never touch the network or a real
    secret.
  * nothing is fabricated: a failed chunk is reported as failed, and a partial
    transcript says so instead of presenting silence as speech.
"""

from __future__ import annotations

import io
import logging
import time
import wave
from dataclasses import dataclass, field
from typing import Any

import httpx
import numpy as np

logger = logging.getLogger("voxsentinel.vach_asr")

PROVIDER_ID = "vach"
PROVIDER_LABEL = "Vach (Gnani) speech-to-text"
DEFAULT_BASE_URL = "https://api.vachana.ai"
STT_PATH = "/stt/v3"
AUTH_HEADER = "X-API-Key-ID"

#: Verified accepted by /stt/v3. Anything else is rejected with HTTP 400.
SUPPORTED_LANGUAGE_CODES: tuple[str, ...] = (
    "as-IN", "bn-IN", "en-IN", "gu-IN", "hi-IN", "indic-auto",
    "kn-IN", "ml-IN", "mr-IN", "pa-IN", "ta-IN", "te-IN",
)
#: Server-side auto detection; returns detected_language_code.
AUTO_LANGUAGE_CODE = "indic-auto"
DEFAULT_LANGUAGE_CODE = AUTO_LANGUAGE_CODE

#: Provider hard limit is 30 s. Stay under it so a boundary rounding error
#: cannot turn a whole call into a 400.
MAX_CHUNK_SECONDS = 28.0
MIN_CHUNK_SECONDS = 1.0
#: How far back to look for a quiet point so a chunk boundary does not slice
#: a word in half.
SILENCE_SEARCH_SECONDS = 2.0
FRAME_SECONDS = 0.02

TARGET_SAMPLE_RATE = 16000


# ----------------------------------------------------------------- exceptions
class VachSttError(RuntimeError):
    """Base class. Messages are safe to surface: no key, URL or body inside."""


class VachAuthError(VachSttError):
    """The provider rejected the configured credential (401/403)."""


class VachRateLimitError(VachSttError):
    """Provider rate limit or quota exceeded (429)."""


class VachInvalidRequestError(VachSttError):
    """Locally invalid request, or a 400 the provider explained."""


class VachServiceError(VachSttError):
    """Provider-side failure (5xx)."""


class VachConnectionError(VachSttError):
    """The endpoint could not be reached (DNS/TLS/timeout)."""


class VachMalformedResponseError(VachSttError):
    """A success status with a body that carries no usable transcript."""


# ------------------------------------------------------------------ results
@dataclass
class VachSegment:
    start: float
    end: float
    text: str


@dataclass
class VachTranscript:
    """One call's transcript, with the evidence needed to trust it."""

    status: str  # completed | failed | unavailable
    text: str = ""
    language: str | None = None
    model: str = ""
    segments: list[VachSegment] = field(default_factory=list)
    chunk_count: int = 0
    latency_ms: int | None = None
    error: str | None = None
    #: Per-chunk telemetry so a partial failure is visible, never hidden.
    chunks: list[dict] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)


# ------------------------------------------------------------------- helpers
def _mono_float32(samples: np.ndarray) -> np.ndarray:
    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)


def _resample(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate == TARGET_SAMPLE_RATE:
        return audio
    try:
        import librosa

        return librosa.resample(
            audio, orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE
        ).astype(np.float32)
    except Exception:  # noqa: BLE001 - librosa is optional on serverless hosts
        duration = audio.size / float(sample_rate)
        n_out = max(1, int(round(duration * TARGET_SAMPLE_RATE)))
        positions = np.linspace(0.0, max(audio.size - 1, 0), n_out)
        low = np.floor(positions).astype(np.int64)
        high = np.minimum(low + 1, max(audio.size - 1, 0))
        frac = (positions - low).astype(np.float32)
        return (audio[low] * (1.0 - frac) + audio[high] * frac).astype(np.float32)


def _frame_rms(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    size = max(1, int(FRAME_SECONDS * sample_rate))
    usable = (audio.size // size) * size
    if usable < size:
        return np.array([float(np.sqrt(np.mean(audio**2)))] if audio.size else [0.0])
    frames = audio[:usable].reshape(-1, size)
    return np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))


def _quiet_boundary(audio: np.ndarray, sample_rate: int, start: int, ideal_end: int) -> int:
    """Move a hard chunk boundary onto the quietest nearby frame.

    Splitting mid-word costs accuracy on exactly the words that matter (names,
    amounts, account digits), so prefer a low-energy point within the last
    ``SILENCE_SEARCH_SECONDS`` of the chunk when one is clearly quieter than the
    chunk average. Falls back to the ideal boundary.
    """
    search = int(SILENCE_SEARCH_SECONDS * sample_rate)
    window_start = max(start, ideal_end - search)
    segment = audio[window_start:ideal_end]
    if segment.size == 0:
        return ideal_end

    rms = _frame_rms(segment, sample_rate)
    if rms.size == 0:
        return ideal_end

    frame_size = max(1, int(FRAME_SECONDS * sample_rate))
    quietest = int(np.argmin(rms))
    average = float(np.mean(rms))
    if average <= 0.0:
        return ideal_end
    if float(rms[quietest]) > 0.35 * average:
        # No clear pause here; a hard cut is more honest than a random one.
        return ideal_end

    candidate = window_start + (quietest + 1) * frame_size
    if candidate - start < int(MIN_CHUNK_SECONDS * sample_rate):
        return ideal_end
    return min(candidate, ideal_end)


def plan_chunks(audio: np.ndarray, sample_rate: int) -> list[tuple[int, int]]:
    """Split a call into provider-sized windows, preferring pauses."""
    total = int(audio.size)
    if total <= 0:
        return []
    max_samples = max(1, int(MAX_CHUNK_SECONDS * sample_rate))
    min_samples = max(1, int(MIN_CHUNK_SECONDS * sample_rate))
    if total <= max_samples:
        return [(0, total)]

    windows: list[tuple[int, int]] = []
    start = 0
    while start < total:
        ideal_end = min(start + max_samples, total)
        if ideal_end >= total:
            end = total
        else:
            end = _quiet_boundary(audio, sample_rate, start, ideal_end)
        if end - start < min_samples:
            end = min(start + max_samples, total)
        windows.append((start, end))
        start = end

    # A trailing sliver is not worth a separate request; fold it into the last
    # window unless that would push the window past the provider's hard cap.
    if len(windows) >= 2 and windows[-1][1] - windows[-1][0] < min_samples:
        tail_start, tail_end = windows.pop()
        prev_start, _ = windows[-1]
        if tail_end - prev_start <= int(MAX_CHUNK_SECONDS * sample_rate):
            windows[-1] = (prev_start, tail_end)
        else:
            windows.append((tail_start, tail_end))
    return windows


def _plan_with_cap(
    audio: np.ndarray, sample_rate: int, cap_seconds: float
) -> list[tuple[int, int]]:
    """plan_chunks with a caller-narrowed maximum window size."""
    cap = max(int(MIN_CHUNK_SECONDS * sample_rate), int(cap_seconds * sample_rate))
    if audio.size <= cap:
        return [(0, int(audio.size))]
    windows: list[tuple[int, int]] = []
    start = 0
    while start < audio.size:
        ideal_end = min(start + cap, int(audio.size))
        end = ideal_end if ideal_end >= audio.size else _quiet_boundary(
            audio, sample_rate, start, ideal_end
        )
        if end - start < int(MIN_CHUNK_SECONDS * sample_rate):
            end = ideal_end
        windows.append((start, end))
        start = end
    return windows


def wav_bytes(audio: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> bytes:
    """16-bit mono PCM WAV in memory. Never a temp file, never a URL."""
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()


def _error_detail(payload: Any) -> tuple[str | None, str | None]:
    """(error_type, message) from the provider's documented error envelope."""
    if not isinstance(payload, dict):
        return None, None
    error = payload.get("error")
    if isinstance(error, dict):
        return (
            str(error.get("type") or "") or None,
            str(error.get("message") or "") or None,
        )
    if isinstance(error, str):
        return None, error
    return None, None


def _extract_text(payload: dict) -> str:
    """The transcript, across the response shapes the provider documents."""
    for candidate in (
        payload.get("transcript"),
        (payload.get("output") or {}).get("literal")
        if isinstance(payload.get("output"), dict)
        else None,
        (payload.get("output") or {}).get("text")
        if isinstance(payload.get("output"), dict)
        else None,
        payload.get("text"),
    ):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return ""


class VachSttClient:
    """Synchronous, bounded Vach ``/stt/v3`` client with call chunking."""

    provider_key = PROVIDER_ID

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout_s: float = 120.0,
        max_chunk_seconds: float = MAX_CHUNK_SECONDS,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        key = (api_key or "").strip()
        if key and any(ch.isspace() for ch in key):
            # A pasted key with whitespace/newlines is the usual cause of a 401.
            key = "".join(key.split())
        self._api_key = key
        url_base = (base_url or DEFAULT_BASE_URL).strip().rstrip("/")
        if not url_base.startswith("https://"):
            raise VachInvalidRequestError("The speech-to-text base URL must be https://.")
        self.base_url = url_base
        self.endpoint = f"{url_base}{STT_PATH}"
        self.timeout_s = float(timeout_s)
        self.max_chunk_seconds = max(MIN_CHUNK_SECONDS, float(max_chunk_seconds))
        self._transport = transport

    # -- status -----------------------------------------------------------
    @property
    def is_configured(self) -> bool:
        return bool(self._api_key)

    def describe(self) -> dict:
        """Public status. Never includes the key or any part of it."""
        return {
            "provider": PROVIDER_ID,
            "label": PROVIDER_LABEL,
            "configured": self.is_configured,
            "model": "prisma-v3.0",
            "endpoint": "provider-hosted" if self.is_configured else None,
            "language_codes": list(SUPPORTED_LANGUAGE_CODES),
            "auto_detect": True,
            "max_audio_seconds": self.max_chunk_seconds,
            "chunked": True,
        }

    # -- request ----------------------------------------------------------
    def _headers(self) -> dict[str, str]:
        return {AUTH_HEADER: self._api_key}

    def _post(self, clip: np.ndarray, language_code: str) -> dict:
        if not self.is_configured:
            raise VachAuthError("No Vach API key is configured on the server.")

        code = (language_code or DEFAULT_LANGUAGE_CODE).strip()
        if code not in SUPPORTED_LANGUAGE_CODES:
            code = DEFAULT_LANGUAGE_CODE

        files = {
            "audio_file": ("audio.wav", wav_bytes(clip), "audio/wav"),
        }
        data = {"language_code": code}

        client_kwargs: dict[str, Any] = {"timeout": self.timeout_s}
        if self._transport is not None:
            client_kwargs["transport"] = self._transport

        try:
            with httpx.Client(**client_kwargs) as client:
                response = client.post(
                    self.endpoint, headers=self._headers(), files=files, data=data
                )
        except httpx.TimeoutException as exc:
            raise VachConnectionError("The speech-to-text service timed out.") from exc
        except httpx.HTTPError as exc:
            raise VachConnectionError(
                "The speech-to-text service could not be reached."
            ) from exc

        if response.status_code in (401, 403):
            raise VachAuthError(
                "The speech-to-text service rejected the configured credential."
            )
        if response.status_code == 429:
            raise VachRateLimitError(
                "The speech-to-text service rate limit was reached. Try again shortly."
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise VachMalformedResponseError(
                "The speech-to-text service returned a non-JSON response."
            ) from exc

        ok = response.status_code < 400 and not (
            isinstance(payload, dict) and payload.get("success") is False
        )
        if not ok:
            error_type, message = _error_detail(payload)
            if response.status_code == 400 or error_type:
                raise VachInvalidRequestError(
                    message
                    or f"The speech-to-text request was rejected ({error_type or 'bad_request'})."
                )
            raise VachServiceError(
                f"The speech-to-text service returned HTTP {response.status_code}."
            )

        if not isinstance(payload, dict):
            raise VachMalformedResponseError(
                "The speech-to-text service returned an unexpected body."
            )
        return payload

    # -- transcription ----------------------------------------------------
    def transcribe(
        self,
        samples: np.ndarray,
        sample_rate: int,
        language_code: str = DEFAULT_LANGUAGE_CODE,
    ) -> VachTranscript:
        """Transcribe a whole call, chunking past the provider's 30 s cap."""
        if not self.is_configured:
            return VachTranscript(
                status="unavailable",
                error="No Vach API key is configured on the server.",
            )

        audio = _resample(_mono_float32(samples), int(sample_rate or TARGET_SAMPLE_RATE))
        if audio.size == 0:
            return VachTranscript(
                status="failed", error="No audio samples were supplied."
            )

        # Chunk to the provider's verified 30-second audio cap, preferring
        # pause-aligned boundaries so words are not sliced in half.
        cap = min(self.max_chunk_seconds, MAX_CHUNK_SECONDS)
        if cap < MAX_CHUNK_SECONDS:
            windows = _plan_with_cap(audio, TARGET_SAMPLE_RATE, cap)
        else:
            windows = plan_chunks(audio, TARGET_SAMPLE_RATE)

        started = time.perf_counter()
        parts: list[str] = []
        segments: list[VachSegment] = []
        chunk_reports: list[dict] = []
        model = ""
        language: str | None = None
        failures = 0

        for index, (c_start, c_end) in enumerate(windows):
            clip = audio[c_start:c_end]
            offset = c_start / TARGET_SAMPLE_RATE
            duration = (c_end - c_start) / TARGET_SAMPLE_RATE
            try:
                payload = self._post(clip, language_code)
            except VachSttError as exc:
                failures += 1
                chunk_reports.append(
                    {
                        "index": index,
                        "start": round(offset, 2),
                        "end": round(offset + duration, 2),
                        "status": "failed",
                        "detail": str(exc),
                    }
                )
                logger.warning("Vach STT chunk %s failed: %s", index, type(exc).__name__)
                continue

            text = _extract_text(payload)
            chunk_model = str(payload.get("model") or "")
            detected = payload.get("detected_language_code")
            if chunk_model:
                model = chunk_model
            if isinstance(detected, str) and detected.strip():
                language = detected.strip()

            if text:
                parts.append(text)
                segments.append(
                    VachSegment(start=round(offset, 2), end=round(offset + duration, 2), text=text)
                )
            chunk_reports.append(
                {
                    "index": index,
                    "start": round(offset, 2),
                    "end": round(offset + duration, 2),
                    "status": "completed",
                    "characters": len(text),
                }
            )

        latency_ms = int((time.perf_counter() - started) * 1000)
        text_out = " ".join(parts).strip()

        limitations: list[str] = [
            "Automatic transcription can mishear names, numbers and accents; "
            "verify important wording against the recording.",
        ]
        if len(windows) > 1:
            limitations.append(
                f"Audio was split into {len(windows)} windows to respect the provider's "
                "30-second limit; a word on a boundary may be repeated or clipped."
            )
        if failures and parts:
            limitations.append(
                f"{failures} of {len(windows)} audio windows could not be transcribed; "
                "the transcript is partial."
            )
        if not text_out:
            limitations.append("No speech was recognized in this recording.")

        if failures and not parts:
            return VachTranscript(
                status="failed",
                model=model or "vach/stt-v3",
                language=language,
                chunk_count=len(windows),
                latency_ms=latency_ms,
                error="Every audio window failed to transcribe.",
                chunks=chunk_reports,
                limitations=limitations,
            )

        return VachTranscript(
            status="completed" if parts else "failed",
            text=text_out,
            language=language,
            model=model or "vach/stt-v3",
            segments=segments,
            chunk_count=len(windows),
            latency_ms=latency_ms,
            error=None if parts else "No speech was recognized.",
            chunks=chunk_reports,
            limitations=limitations,
        )


def build_vach_stt_client(
    api_key: str | None = None,
    base_url: str | None = None,
    timeout_s: float | None = None,
) -> VachSttClient | None:
    """Build a client from server-side configuration, or None when no key.

    Resolution goes through :func:`app.core.config.get_settings`, so the whole
    backend agrees on one credential: an explicit argument, then
    ``VOXSENTINEL_VACH_API_KEY``, then the deployment-wide ``API_KEY``, then the
    legacy ``GNANI_API_KEY``. The same value drives text-to-speech. No key ever
    reaches the browser.
    """
    resolved_key = (api_key or "").strip()
    resolved_base = (base_url or "").strip()
    resolved_timeout = timeout_s

    if not resolved_key or not resolved_base or resolved_timeout is None:
        try:
            from app.core.config import get_settings

            settings = get_settings()
        except Exception:  # noqa: BLE001 - config must never block transcription
            settings = None
        if settings is not None:
            resolved_key = resolved_key or (settings.vach_api_key or "").strip()
            resolved_base = resolved_base or (settings.vach_base_url or "").strip()
            if resolved_timeout is None:
                resolved_timeout = float(settings.vach_stt_timeout_s or 120.0)

    if not resolved_key:
        return None

    resolved_base = resolved_base or DEFAULT_BASE_URL
    if resolved_timeout is None:
        resolved_timeout = 120.0

    try:
        return VachSttClient(
            resolved_key, base_url=resolved_base, timeout_s=resolved_timeout
        )
    except VachInvalidRequestError as exc:
        logger.warning("Vach STT disabled: %s", exc)
        return None
