"""Real multilingual speech transcription (faster-whisper / Whisper).

The model runs locally on CPU. Nothing is fabricated: when the library or the
model file is unavailable, transcription returns an explicit `unavailable`
status and the rest of the pipeline continues without invented text.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger("voxsentinel.protection.transcription")

DEFAULT_MODEL_SIZE = os.environ.get("VOXSENTINEL_WHISPER_MODEL", "tiny")


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    status: str  # completed | unavailable | failed
    language: str | None = None
    language_probability: float | None = None
    text: str = ""
    segments: list[TranscriptSegment] = field(default_factory=list)
    model: str = ""
    error: str | None = None
    limitations: list[str] = field(default_factory=list)
    #: How the language was determined: engine | script_detection | request_hint.
    language_source: str | None = None
    #: Per-engine telemetry from the multi-engine chain (empty for one engine).
    engines: list[dict] = field(default_factory=list)
    #: Engine whose transcript was selected after any fallback/comparison.
    selected_engine: str | None = None


class WhisperTranscriber:
    """Lazy, thread-safe wrapper around faster-whisper.

    The first call downloads/loads the model (cached under the HF cache dir).
    A load or runtime failure degrades to an honest status, never fake text.
    """

    def __init__(
        self, model_size: str | None = None, device: str = "cpu", compute_type: str = "int8"
    ) -> None:
        self._model_size = model_size or DEFAULT_MODEL_SIZE
        self._device = device
        self._compute_type = compute_type
        self._model = None
        self._lock = threading.Lock()
        self._unavailable_reason: str | None = None

    def is_available(self) -> bool:
        try:
            import faster_whisper  # noqa: F401
        except Exception:  # pragma: no cover - environment dependent
            return False
        return True

    def _load(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            from faster_whisper import WhisperModel

            self._model = WhisperModel(self._model_size, device=self._device, compute_type=self._compute_type)
            logger.info("Whisper model loaded: %s (%s)", self._model_size, self._device)
            return self._model

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> TranscriptionResult:
        if not self.is_available():
            self._unavailable_reason = (
                "faster-whisper is not installed on this server, so no speech-to-text model is "
                "available. Install it (`pip install faster-whisper`) to enable transcription."
            )
            return TranscriptionResult(status="unavailable", error=self._unavailable_reason)
        try:
            model = self._load()
            audio = np.asarray(samples, dtype=np.float32)
            if sample_rate != 16000:
                # faster-whisper expects 16 kHz mono float32.
                import librosa

                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
            segments_iter, info = model.transcribe(
                audio,
                beam_size=1,
                vad_filter=True,
                condition_on_previous_text=False,
            )
            segments = [
                TranscriptSegment(start=float(s.start), end=float(s.end), text=(s.text or "").strip())
                for s in segments_iter
            ]
            text = " ".join(s.text for s in segments).strip()
            limitations = [
                "Automatic transcription can mishear names, numbers and accents; "
                "verify important wording against the recording.",
            ]
            if not text:
                limitations.append("No speech was recognized in this recording.")
            return TranscriptionResult(
                status="completed",
                language=getattr(info, "language", None),
                language_probability=(
                    float(info.language_probability)
                    if getattr(info, "language_probability", None) is not None
                    else None
                ),
                text=text,
                segments=segments,
                model=f"faster-whisper:{self._model_size}",
                limitations=limitations,
            )
        except Exception as exc:  # noqa: BLE001 - surface as honest failure
            logger.warning("Transcription failed: %s", exc)
            return TranscriptionResult(
                status="failed",
                model=f"faster-whisper:{self._model_size}",
                error=f"Speech recognition failed: {type(exc).__name__}",
            )

    def describe(self) -> dict:
        return {
            "service": "faster-whisper",
            "model": self._model_size,
            "device": self._device,
            "available": self.is_available(),
        }

def default_transcriber():
    """Pick the best transcriber available on this machine.

    Preference order:
      1. local faster-whisper (private, offline, free)
      2. a remote provider when a GEMINI_API_KEY is configured (serverless hosts)
      3. the local transcriber anyway, so `describe()` reports the real reason
    """
    local = WhisperTranscriber()
    if local.is_available():
        return local
    try:
        from app.protection.transcription_remote import build_remote_transcriber

        remote = build_remote_transcriber()
    except Exception:  # noqa: BLE001 - a broken optional import must not break the API
        remote = None
    return remote or local
