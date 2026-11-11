"""Cloud speech-to-text for hosts that cannot run Whisper locally.

Why this exists
---------------
`transcription.py` runs faster-whisper on the machine. That is the preferred
path: private, free, and offline. But a serverless host (Vercel, for example)
cannot install torch/ctranslate2 or keep a model in memory, so there the local
transcriber is honestly "unavailable" and every suspicious-call upload dies
before the scam analysis can say anything.

This module adds a remote transcriber with the exact same contract
(`transcribe(samples, sample_rate) -> TranscriptionResult`), so the pipeline,
the rule engine and the contextual classifier stay unchanged:

  * audio is sent as 16-bit PCM WAV, built in memory (stdlib `wave`), never a
    temp file and never a URL the provider keeps
  * the provider is Gemini's audio-understanding endpoint, because a key is
    already part of this project's provider set (`GEMINI_API_KEY`)
  * the model's answer is parsed defensively: only text/segments that survive
    validation are used, and a bad or empty answer never turns into invented
    speech
  * every failure returns `status="failed"` / `"unavailable"` with a reason, so
    the UI shows a limitation instead of a made-up transcript

Privacy note: audio leaves the machine when this path is used. `describe()`
reports the service as `gemini-audio` (never `faster-whisper`) so the UI can
say so, and the deployment's capabilities response reflects reality.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import wave
from typing import Any

import httpx
import numpy as np

from app.protection.transcription import TranscriptionResult, TranscriptSegment

logger = logging.getLogger("voxsentinel.protection.transcription_remote")

GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_MODEL = os.environ.get("VOXSENTINEL_ASR_MODEL", "gemini-3.5-flash")
# The provider returns 503/429 while a model is under load. Transcription is a
# best-effort step, so a busy primary model should not cost the whole analysis:
# these are tried in order, and the model that actually answered is reported.
FALLBACK_MODELS: tuple[str, ...] = tuple(
    m.strip()
    for m in os.environ.get("VOXSENTINEL_ASR_FALLBACKS", "gemini-3.1-flash-lite").split(",")
    if m.strip()
)
REQUEST_TIMEOUT_S = float(os.environ.get("VOXSENTINEL_ASR_TIMEOUT_S", "120"))
# Gemini accepts large inline payloads, but keeping the clip small keeps the
# request fast and under every provider's inline-data ceiling.
MAX_INLINE_BYTES = 18 * 1024 * 1024
TARGET_SAMPLE_RATE = 16000

_PROMPT = (
    "Transcribe the speech in this audio verbatim, in the language actually spoken. "
    "Preserve numbers, names, amounts, account digits and threats exactly as said; "
    "do not translate, summarise or correct grammar. "
    "Respond with JSON only, matching this shape: "
    '{"language": "<BCP-47 code or null>", "text": "<full transcript>", '
    '"segments": [{"start": <seconds>, "end": <seconds>, "text": "<segment text>"}]}. '
    "If there is no intelligible speech, return an empty text and an empty segments array. "
    "No commentary, no markdown."
)


def _to_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    """16 kHz mono 16-bit PCM WAV, built in memory.

    Linear resampling is deliberate: it is a few lines of numpy, has no
    dependencies, and speech recognition tolerates it. The alternative on a
    serverless host is no transcription at all.
    """
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    if not np.all(np.isfinite(audio)):
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    if sample_rate and sample_rate != TARGET_SAMPLE_RATE and audio.size:
        duration = audio.size / float(sample_rate)
        target_n = max(1, int(round(duration * TARGET_SAMPLE_RATE)))
        positions = np.linspace(0.0, audio.size - 1, num=target_n, dtype=np.float64)
        audio = np.interp(positions, np.arange(audio.size), audio).astype(np.float32)
    pcm = np.clip(audio, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(TARGET_SAMPLE_RATE)
        handle.writeframes(pcm16.tobytes())
    return buffer.getvalue()


def _segments_from_payload(payload: dict[str, Any], text: str) -> list[TranscriptSegment]:
    segments: list[TranscriptSegment] = []
    raw = payload.get("segments")
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            chunk = str(item.get("text") or "").strip()
            if not chunk:
                continue
            try:
                start = float(item.get("start") or 0.0)
                end = float(item.get("end") or start)
            except (TypeError, ValueError):
                continue
            segments.append(TranscriptSegment(start=max(0.0, start), end=max(start, end), text=chunk))
    if not segments and text:
        # Timestamps are optional evidence; a transcript without them is still
        # real transcript, so it is kept and flagged rather than discarded.
        segments.append(TranscriptSegment(start=0.0, end=0.0, text=text))
    return segments


class GeminiTranscriber:
    """Remote transcription through Gemini's audio understanding."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout_s: float | None = None,
    ) -> None:
        self._api_key = (api_key if api_key is not None else os.environ.get("GEMINI_API_KEY", "")).strip()
        self._model = model or DEFAULT_MODEL
        self._timeout_s = timeout_s or REQUEST_TIMEOUT_S

    # -- capability --------------------------------------------------------
    def is_available(self) -> bool:
        if not self._api_key:
            return False
        return True

    def describe(self) -> dict:
        return {
            "service": "gemini-audio",
            "model": self._model,
            "device": "cloud",
            "available": self.is_available(),
            "note": (
                "Audio is transcribed by Gemini because this host cannot run Whisper locally."
                if self.is_available()
                else "Set GEMINI_API_KEY to enable cloud transcription on this host."
            ),
        }

    # -- transcription -----------------------------------------------------
    def transcribe(self, samples: np.ndarray, sample_rate: int) -> TranscriptionResult:
        if not self.is_available():
            return TranscriptionResult(
                status="unavailable",
                error=(
                    "No speech-to-text is available on this host: local Whisper is not installed "
                    "and GEMINI_API_KEY is not configured."
                ),
            )

        try:
            audio_bytes = _to_wav_bytes(samples, sample_rate)
        except Exception as exc:  # noqa: BLE001
            return TranscriptionResult(
                status="failed",
                model=f"gemini:{self._model}",
                error=f"Audio could not be prepared for transcription: {type(exc).__name__}",
            )

        if len(audio_bytes) > MAX_INLINE_BYTES:
            return TranscriptionResult(
                status="failed",
                model=f"gemini:{self._model}",
                error="The recording is too long for cloud transcription; trim it and retry.",
            )

        body = {
            "contents": [
                {
                    "parts": [
                        {"text": _PROMPT},
                        {
                            "inline_data": {
                                "mime_type": "audio/wav",
                                "data": base64.b64encode(audio_bytes).decode("ascii"),
                            }
                        },
                    ]
                }
            ],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
        }

        response = None
        used_model = self._model
        last_status = None
        for candidate in (self._model, *FALLBACK_MODELS):
            try:
                with httpx.Client(timeout=self._timeout_s) as client:
                    attempt = client.post(
                        f"{GEMINI_ENDPOINT}/{candidate}:generateContent",
                        headers={"x-goog-api-key": self._api_key, "Content-Type": "application/json"},
                        json=body,
                    )
            except Exception as exc:  # noqa: BLE001 - network failure stays honest
                logger.warning("Cloud transcription request failed: %s", type(exc).__name__)
                return TranscriptionResult(
                    status="failed",
                    model=f"gemini:{candidate}",
                    error=f"Cloud transcription request failed: {type(exc).__name__}",
                )
            if attempt.status_code == 200:
                response, used_model = attempt, candidate
                break
            last_status = attempt.status_code
            logger.warning("Cloud transcription HTTP %s on %s", attempt.status_code, candidate)
            if attempt.status_code not in (404, 429, 500, 503):
                break

        if response is None:
            return TranscriptionResult(
                status="failed",
                model=f"gemini:{used_model}",
                error=f"Cloud transcription was rejected by the provider (HTTP {last_status}).",
            )

        try:
            payload = response.json()
            text_parts: list[str] = []
            for candidate in payload.get("candidates") or []:
                for part in (candidate.get("content") or {}).get("parts") or []:
                    if isinstance(part.get("text"), str):
                        text_parts.append(part["text"])
            answer = "\n".join(text_parts).strip()
            if answer.startswith("```"):
                answer = answer.strip("`")
                answer = answer.split("\n", 1)[1] if "\n" in answer else answer
            parsed = json.loads(answer) if answer else {}
        except Exception:  # noqa: BLE001 - unparseable answer is a failure, not text
            return TranscriptionResult(
                status="failed",
                model=f"gemini:{self._model}",
                error="Cloud transcription returned an unreadable answer.",
            )

        text = str(parsed.get("text") or "").strip()
        language = parsed.get("language") if isinstance(parsed.get("language"), str) else None
        segments = _segments_from_payload(parsed, text)

        limitations = [
            "Transcription can mishear names, numbers and accents; verify important "
            "wording against the recording.",
        ]
        if self.is_available():
            limitations.append(
                "Audio was transcribed off-device by the configured cloud provider."
            )
        if not any(segment.start or segment.end for segment in segments) and text:
            limitations.append("Segment timestamps were not reported for this recording.")
        if not text:
            limitations.append("No speech was recognized in this recording.")

        return TranscriptionResult(
            status="completed",
            language=language,
            language_probability=None,
            text=text,
            segments=segments,
            model=f"gemini:{used_model}",
            limitations=limitations,
        )


def build_remote_transcriber() -> GeminiTranscriber | None:
    """Remote transcriber when a provider key exists, otherwise None."""
    candidate = GeminiTranscriber()
    return candidate if candidate.is_available() else None
