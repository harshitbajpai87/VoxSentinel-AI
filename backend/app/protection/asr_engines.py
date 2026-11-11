"""Multi-engine speech-to-text for the unified protection pipeline.

Engines (each is real, lazy, and reports its own status; none fabricates text):

  * ``whisper_small``   openai/whisper-small        multilingual, English + Hinglish
  * ``indic_hindi``     ai4bharat/indicwav2vec-hindi dedicated Hindi CTC model
  * ``faster_whisper``  local CTranslate2 Whisper   fast fallback
  * ``gnani_asr``       Gnani/remote ASR endpoint   only when GNANI_ASR_ENDPOINT is set

``MultiEngineTranscriber`` is a drop-in replacement for
``app.protection.transcription.WhisperTranscriber``: same ``transcribe()`` and
``describe()`` surface, plus per-engine telemetry so the dashboard can say
exactly which model produced the text.

Routing rules, in order:
  1. run the primary engine (VOXSENTINEL_ASR_ENGINE, default ``auto``)
  2. when the language is Hindi/Devanagari and the dedicated Hindi model is
     available, run it as well and keep the better transcript
  3. if every local engine is unavailable, try the configured remote ASR
  4. report ``unavailable`` with the full reason list instead of guessing
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from app.protection.transcription import (
    TranscriptionResult,
    TranscriptSegment,
    WhisperTranscriber,
)
from app.services.hf_models import BACKEND_DIR, resolve_model

logger = logging.getLogger("voxsentinel.protection.asr")

TARGET_SR = 16000
WHISPER_SMALL_ID = "openai/whisper-small"
WHISPER_LARGE_TURBO_ID = "openai/whisper-large-v3-turbo"
FASTER_WHISPER_LARGE_ID = "Systran/faster-whisper-large-v3"
MODELS_DIR = BACKEND_DIR / "models"
INDIC_HINDI_ID = "ai4bharat/indicwav2vec-hindi"
_DEVANAGARI = re.compile(r"[\u0900-\u097F]")
#: Hinglish is Latin script with Hindi function words; used only to pick the
#: right engine, never to claim a language probability.
_HINGLISH_MARKERS = (
    "bhai", "hai", "nahi", "kya", "aap", "mera", "paisa", "rupaye", "abhi",
    "jaldi", "batao", "sun", "kar", "ho", "gaya", "tum", "hum",
)


@dataclass
class EngineTranscript:
    """One engine's honest attempt."""

    engine: str
    model: str
    status: str  # completed | unavailable | failed | skipped
    text: str = ""
    language: str | None = None
    language_probability: float | None = None
    segments: list[TranscriptSegment] = field(default_factory=list)
    latency_ms: int | None = None
    detail: str | None = None

    def as_dict(self) -> dict:
        return {
            "engine": self.engine,
            "model": self.model,
            "status": self.status,
            "characters": len(self.text or ""),
            "language": self.language,
            "language_probability": self.language_probability,
            "segments": len(self.segments),
            "latency_ms": self.latency_ms,
            "detail": self.detail,
        }


def _mono16k(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate == TARGET_SR:
        return audio
    try:
        import librosa

        return librosa.resample(audio, orig_sr=sample_rate, target_sr=TARGET_SR)
    except Exception:  # pragma: no cover - librosa is present in the container image
        duration = audio.size / float(sample_rate)
        n_out = max(1, int(round(duration * TARGET_SR)))
        positions = np.linspace(0.0, max(audio.size - 1, 0), n_out)
        low = np.floor(positions).astype(np.int64)
        high = np.minimum(low + 1, max(audio.size - 1, 0))
        frac = (positions - low).astype(np.float32)
        return (audio[low] * (1.0 - frac) + audio[high] * frac).astype(np.float32)


def is_hindi_text(text: str) -> bool:
    """True when the transcript is predominantly Devanagari."""
    if not text:
        return False
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    return len(_DEVANAGARI.findall(text)) / len(letters) >= 0.4


def looks_hinglish(text: str) -> bool:
    """Latin-script Hindi/English mix heuristic (engine routing only)."""
    if not text:
        return False
    words = re.findall(r"[a-zA-Z']+", text.lower())
    if len(words) < 4:
        return False
    hits = sum(1 for w in words if w in _HINGLISH_MARKERS)
    return hits >= 2 and hits / len(words) >= 0.08


def _coerce_segments(chunks: list | None) -> list[TranscriptSegment]:
    segments: list[TranscriptSegment] = []
    for chunk in chunks or []:
        if not isinstance(chunk, (list, tuple)) or len(chunk) != 2:
            continue
        stamp, text = chunk
        if not isinstance(stamp, (list, tuple)) or len(stamp) != 2:
            continue
        start, end = stamp
        if start is None or end is None:
            continue
        segments.append(TranscriptSegment(float(start), float(end), str(text or "").strip()))
    return segments


class WhisperSmallEngine:
    """openai/whisper-small through the Transformers ASR pipeline."""

    name = "whisper_small"

    def __init__(self) -> None:
        self._pipe = None
        self._lock = threading.Lock()
        self._unavailable: str | None = None
        self._source = resolve_model(WHISPER_SMALL_ID, "whisper-small")

    def is_available(self) -> bool:
        return self._source.available

    def _load(self):
        if self._pipe is not None:
            return self._pipe
        with self._lock:
            if self._pipe is not None:
                return self._pipe
            import torch
            from transformers import pipeline

            started = time.perf_counter()
            self._pipe = pipeline(
                "automatic-speech-recognition",
                model=self._source.source,
                device=-1,
                torch_dtype=torch.float32,
                chunk_length_s=30,
                stride_length_s=5,
            )
            logger.info(
                "whisper-small loaded from %s in %.1fs",
                self._source.source,
                time.perf_counter() - started,
            )
            return self._pipe

    def transcribe(
        self, samples: np.ndarray, sample_rate: int, language_hint: str | None = None
    ) -> EngineTranscript:
        if not self.is_available():
            return EngineTranscript(
                engine=self.name,
                model=WHISPER_SMALL_ID,
                status="unavailable",
                detail=self._source.reason,
            )
        started = time.perf_counter()
        audio = _mono16k(samples, sample_rate)
        try:
            pipe = self._load()
            kwargs: dict = {"return_timestamps": True}
            if language_hint:
                kwargs["generate_kwargs"] = {"language": language_hint}
            output = pipe({"raw": audio, "sampling_rate": TARGET_SR}, **kwargs)
            if isinstance(output, list):  # chunks were returned separately
                text = " ".join(str(part.get("text", "")) for part in output).strip()
                chunks = [c for part in output for c in (part.get("chunks") or [])]
            else:
                text = str(output.get("text") or "").strip()
                chunks = output.get("chunks") or []
            return EngineTranscript(
                engine=self.name,
                model=WHISPER_SMALL_ID,
                status="completed",
                text=text,
                segments=_coerce_segments(chunks),
                latency_ms=int((time.perf_counter() - started) * 1000),
                detail=None if text else "The model produced no text for this audio.",
            )
        except Exception as exc:  # noqa: BLE001 - honest, logged failure
            logger.warning("whisper-small failed: %s", exc)
            return EngineTranscript(
                engine=self.name,
                model=WHISPER_SMALL_ID,
                status="failed",
                latency_ms=int((time.perf_counter() - started) * 1000),
                detail=f"whisper-small failed: {type(exc).__name__}",
            )


class IndicHindiEngine:
    """ai4bharat/indicwav2vec-hindi: CTC acoustic model, greedy decode."""

    name = "indic_hindi"

    def __init__(self) -> None:
        self._model = None
        self._processor = None
        self._lock = threading.Lock()
        self._source = resolve_model(INDIC_HINDI_ID, "indicwav2vec-hindi")

    def is_available(self) -> bool:
        return self._source.available

    def _load(self):
        if self._model is not None:
            return self._model, self._processor
        with self._lock:
            if self._model is not None:
                return self._model, self._processor
            from transformers import AutoProcessor, Wav2Vec2ForCTC

            self._processor = AutoProcessor.from_pretrained(self._source.source)
            self._model = Wav2Vec2ForCTC.from_pretrained(self._source.source).eval()
            logger.info("indicwav2vec-hindi loaded from %s", self._source.source)
            return self._model, self._processor

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> EngineTranscript:
        if not self.is_available():
            return EngineTranscript(
                engine=self.name,
                model=INDIC_HINDI_ID,
                status="unavailable",
                detail=self._source.reason,
            )
        started = time.perf_counter()
        audio = _mono16k(samples, sample_rate)
        try:
            import torch

            model, processor = self._load()
            inputs = processor(audio, sampling_rate=TARGET_SR, return_tensors="pt")
            with torch.no_grad():
                logits = model(**inputs).logits
            ids = torch.argmax(logits, dim=-1)
            text = processor.batch_decode(ids)[0].strip()
            return EngineTranscript(
                engine=self.name,
                model=INDIC_HINDI_ID,
                status="completed",
                text=text,
                language="hi",
                # CTC token-frame confidence is not a language-ID probability;
                # blank-dominated frames can make it misleadingly high.
                language_probability=None,
                latency_ms=int((time.perf_counter() - started) * 1000),
                detail=None if text else "The Hindi model produced no text for this audio.",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("indicwav2vec-hindi failed: %s", exc)
            detail = f"Hindi ASR failed: {type(exc).__name__}"
            text_of = str(exc)
            if "GatedRepo" in type(exc).__name__ or "401" in text_of or "gated" in text_of.lower():
                detail = (
                    "ai4bharat/indicwav2vec-hindi is a gated Hub repository: accept its terms "
                    "on huggingface.co and set HF_TOKEN, then run "
                    "`python -m scripts.download_hf_models indicwav2vec-hindi`."
                )
            return EngineTranscript(
                engine=self.name,
                model=INDIC_HINDI_ID,
                status="failed",
                latency_ms=int((time.perf_counter() - started) * 1000),
                detail=detail,
            )


class FasterWhisperEngine:
    """The project's existing local Whisper (CTranslate2) as an engine."""

    name = "faster_whisper"

    def __init__(self, wrapper: WhisperTranscriber | None = None) -> None:
        self._wrapper = wrapper or WhisperTranscriber()

    def is_available(self) -> bool:
        return self._wrapper.is_available()

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> EngineTranscript:
        started = time.perf_counter()
        result = self._wrapper.transcribe(samples, sample_rate)
        return EngineTranscript(
            engine=self.name,
            model=getattr(self._wrapper, "_model_size", "whisper"),
            status=result.status,
            text=result.text or "",
            language=result.language,
            language_probability=result.language_probability,
            segments=list(result.segments),
            latency_ms=int((time.perf_counter() - started) * 1000),
            detail=result.error,
        )


class GnaniAsrEngine:
    """Remote ASR through Vach (Gnani) ``/stt/v3``.

    This is the cloud engine for the deployment credential: one ``API_KEY``
    enables both speech-to-text here and text-to-speech in
    ``app/services/gnani_client.py``. The contract was verified live on
    2026-11-11 (see ``app/services/vach_asr.py``): ``X-API-Key-ID`` auth,
    multipart ``audio_file`` + ``language_code``, ``indic-auto`` for language
    detection, and a 30-second per-request audio cap that this engine handles by
    chunking longer calls.

    The engine name is kept as ``gnani_asr`` so existing capability payloads,
    saved analyses and dashboards keep resolving. It reports ``unavailable``
    (with the real reason) when no key is configured, never a fabricated
    transcript.
    """

    name = "gnani_asr"

    def __init__(self) -> None:
        try:
            from app.services.vach_asr import build_vach_stt_client

            self._client = build_vach_stt_client()
        except Exception:  # noqa: BLE001 - a broken optional import must not break ASR
            logger.warning("Vach ASR client could not be built", exc_info=True)
            self._client = None

    def is_available(self) -> bool:
        return bool(self._client is not None and self._client.is_configured)

    def _reason(self) -> str:
        if self._client is None:
            return (
                "The Vach speech-to-text client is not configured for this host "
                "(check the backend API_KEY value)."
            )
        return "No Vach API key is configured on the server (API_KEY)."

    def transcribe(
        self,
        samples: np.ndarray,
        sample_rate: int,
        language_hint: str | None = None,
    ) -> EngineTranscript:
        if self._client is None:
            return EngineTranscript(
                engine=self.name,
                model="vach/stt-v3",
                status="unavailable",
                detail=self._reason(),
            )
        if not self._client.is_configured:
            return EngineTranscript(
                engine=self.name,
                model="vach/stt-v3",
                status="unavailable",
                detail=self._reason(),
            )

        # A request hint narrows the provider's own detection; otherwise let
        # ``indic-auto`` detect the language and report ``detected_language_code``.
        from app.services.vach_asr import AUTO_LANGUAGE_CODE

        hint = (language_hint or "").strip().lower()
        language_code = AUTO_LANGUAGE_CODE
        if hint in {"hi", "hi-in"}:
            language_code = "hi-IN"
        elif hint in {"en", "en-in"}:
            language_code = "en-IN"

        result = self._client.transcribe(samples, sample_rate, language_code=language_code)

        if result.status != "completed":
            return EngineTranscript(
                engine=self.name,
                model=result.model or "vach/stt-v3",
                status=result.status,
                latency_ms=result.latency_ms,
                detail=result.error or "Vach speech-to-text failed.",
            )

        segments = [
            TranscriptSegment(start=seg.start, end=seg.end, text=seg.text)
            for seg in result.segments
        ]
        return EngineTranscript(
            engine=self.name,
            model=result.model or "vach/stt-v3",
            status="completed",
            text=result.text,
            language=(result.language or "").split("-")[0] or None,
            language_probability=None,
            segments=segments,
            latency_ms=result.latency_ms,
            detail="; ".join(result.limitations) or None,
        )


class WhisperLargeTurboEngine:
    """openai/whisper-large-v3-turbo: the accuracy tier, 809M params."""

    name = "whisper_large_turbo"

    def __init__(self) -> None:
        self._pipe = None
        self._lock = threading.Lock()
        self._source = resolve_model(WHISPER_LARGE_TURBO_ID, "whisper-large-v3-turbo")

    def is_available(self) -> bool:
        return self._source.available

    def _load(self):
        if self._pipe is not None:
            return self._pipe
        with self._lock:
            if self._pipe is None:
                import torch
                from transformers import pipeline

                started = time.perf_counter()
                self._pipe = pipeline(
                    "automatic-speech-recognition",
                    model=self._source.source,
                    device=-1,
                    torch_dtype=torch.float32,
                    chunk_length_s=30,
                    stride_length_s=5,
                )
                logger.info(
                    "whisper-large-v3-turbo loaded from %s in %.1fs",
                    self._source.source,
                    time.perf_counter() - started,
                )
            return self._pipe

    def transcribe(
        self, samples: np.ndarray, sample_rate: int, language_hint: str | None = None
    ) -> EngineTranscript:
        if not self.is_available():
            return EngineTranscript(
                engine=self.name,
                model=WHISPER_LARGE_TURBO_ID,
                status="unavailable",
                detail=self._source.reason,
            )
        started = time.perf_counter()
        audio = _mono16k(samples, sample_rate)
        try:
            pipe = self._load()
            kwargs: dict = {"return_timestamps": True}
            if language_hint:
                kwargs["generate_kwargs"] = {"language": language_hint}
            output = pipe({"raw": audio, "sampling_rate": TARGET_SR}, **kwargs)
            if isinstance(output, list):
                text = " ".join(str(part.get("text", "")) for part in output).strip()
                chunks = [c for part in output for c in (part.get("chunks") or [])]
            else:
                text = str(output.get("text") or "").strip()
                chunks = output.get("chunks") or []
            return EngineTranscript(
                engine=self.name,
                model=WHISPER_LARGE_TURBO_ID,
                status="completed",
                text=text,
                segments=_coerce_segments(chunks),
                latency_ms=int((time.perf_counter() - started) * 1000),
                detail=None if text else "The model produced no text for this audio.",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("whisper-large-v3-turbo failed: %s", exc)
            return EngineTranscript(
                engine=self.name,
                model=WHISPER_LARGE_TURBO_ID,
                status="failed",
                latency_ms=int((time.perf_counter() - started) * 1000),
                detail=f"whisper-large-v3-turbo failed: {type(exc).__name__}",
            )


class FasterWhisperLargeEngine:
    """Systran/faster-whisper-large-v3 through CTranslate2 (int8 on CPU)."""

    name = "faster_whisper_large"

    def __init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()
        self._error: str | None = None
        self._dir = MODELS_DIR / "hf" / "faster-whisper-large-v3"

    def is_available(self) -> bool:
        return self._dir.is_dir() and any(self._dir.glob("*.bin"))

    def _load(self):
        if self._model is not None or self._error is not None:
            return self._model
        with self._lock:
            if self._model is not None or self._error is not None:
                return self._model
            try:
                from faster_whisper import WhisperModel

                self._model = WhisperModel(
                    str(self._dir), device="cpu", compute_type="int8", local_files_only=True
                )
            except Exception as exc:  # noqa: BLE001
                self._error = f"{type(exc).__name__}: {str(exc)[:120]}"
            return self._model

    def transcribe(
        self, samples: np.ndarray, sample_rate: int, language_hint: str | None = None
    ) -> EngineTranscript:
        if not self.is_available():
            return EngineTranscript(
                engine=self.name,
                model=FASTER_WHISPER_LARGE_ID,
                status="unavailable",
                detail=f"No CTranslate2 weights at {self._dir}.",
            )
        started = time.perf_counter()
        audio = _mono16k(samples, sample_rate)
        model = self._load()
        if model is None:
            return EngineTranscript(
                engine=self.name,
                model=FASTER_WHISPER_LARGE_ID,
                status="failed",
                detail=self._error,
            )
        try:
            segments, info = model.transcribe(
                audio, language=language_hint, vad_filter=True, beam_size=5
            )
            collected = [
                TranscriptSegment(float(seg.start), float(seg.end), seg.text.strip())
                for seg in segments
            ]
            text = " ".join(seg.text for seg in collected).strip()
            return EngineTranscript(
                engine=self.name,
                model=FASTER_WHISPER_LARGE_ID,
                status="completed",
                text=text,
                language=getattr(info, "language", None),
                language_probability=getattr(info, "language_probability", None),
                segments=collected,
                latency_ms=int((time.perf_counter() - started) * 1000),
                detail=None if text else "The model produced no text for this audio.",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("faster-whisper-large-v3 failed: %s", exc)
            return EngineTranscript(
                engine=self.name,
                model=FASTER_WHISPER_LARGE_ID,
                status="failed",
                latency_ms=int((time.perf_counter() - started) * 1000),
                detail=f"faster-whisper-large-v3 failed: {type(exc).__name__}",
            )


class MultiEngineTranscriber:
    """Drop-in transcriber that runs the best engine chain for the audio."""

    def __init__(
        self,
        engines: list | None = None,
        preferred: str | None = None,
        language_hint: str | None = None,
    ) -> None:
        self._engines = engines or [
            WhisperLargeTurboEngine(),
            GnaniAsrEngine(),
            FasterWhisperLargeEngine(),
            WhisperSmallEngine(),
            IndicHindiEngine(),
            FasterWhisperEngine(),
        ]
        self._preferred = (
            preferred or os.environ.get("VOXSENTINEL_ASR_ENGINE") or "auto"
        ).strip().lower()
        self._fallbacks_enabled = os.environ.get("VOXSENTINEL_ASR_FALLBACKS", "true").strip().lower() not in {
            "0", "false", "no", "off"
        }
        self._language_hint = language_hint

    # -- registry ---------------------------------------------------------
    def _engine(self, name: str):
        for engine in self._engines:
            if engine.name == name:
                return engine
        return None

    def engine_status(self) -> list[dict]:
        report: list[dict] = []
        for engine in self._engines:
            entry = {"engine": engine.name, "available": bool(engine.is_available())}
            source = getattr(engine, "_source", None)
            if source is not None:
                entry["model_id"] = source.model_id
                entry["reason"] = source.reason
            if engine.name == "gnani_asr":
                entry["reason"] = None if engine.is_available() else engine._reason()
                client = getattr(engine, "_client", None)
                if client is not None:
                    described = client.describe()
                    entry["provider"] = described.get("provider")
                    entry["label"] = described.get("label")
                    entry["model"] = described.get("model")
                    entry["chunked"] = described.get("chunked")
                    entry["max_audio_seconds"] = described.get("max_audio_seconds")
            report.append(entry)
        return report

    def _order(self) -> list[str]:
        # Accuracy-first: the large local model, then the hosted Indic model
        # (strong on Hindi/Hinglish and the only option on a serverless host),
        # then progressively smaller local models.
        automatic = [
            "whisper_large_turbo",
            "gnani_asr",
            "faster_whisper_large",
            "whisper_small",
            "indic_hindi",
            "faster_whisper",
        ]
        if self._preferred in {"auto", ""}:
            return automatic
        if not self._fallbacks_enabled:
            return [self._preferred]
        return [self._preferred, *(name for name in automatic if name != self._preferred)]

    def _primary(self):
        order = self._order()
        for name in order:
            engine = self._engine(name)
            if engine is not None and engine.is_available():
                return engine
        return None

    def _run(self, engine, samples: np.ndarray, sample_rate: int) -> EngineTranscript:
        if engine.name in {"whisper_small", "whisper_large_turbo", "faster_whisper_large", "gnani_asr"}:
            return engine.transcribe(samples, sample_rate, language_hint=self._language_hint)
        return engine.transcribe(samples, sample_rate)

    def _run_safe(self, engine, samples: np.ndarray, sample_rate: int) -> EngineTranscript:
        try:
            return self._run(engine, samples, sample_rate)
        except Exception as exc:  # noqa: BLE001 - retain adapter failure in telemetry
            return EngineTranscript(
                engine=engine.name,
                model=getattr(getattr(engine, "_source", None), "model_id", engine.name),
                status="failed",
                detail=type(exc).__name__,
            )

    def _hindi_engine(self):
        engine = self._engine("indic_hindi")
        if engine is not None and engine.is_available():
            return engine
        return None

    # -- transcription ----------------------------------------------------
    def transcribe(self, samples: np.ndarray, sample_rate: int) -> TranscriptionResult:
        attempts: list[EngineTranscript] = []
        hint = self._language_hint
        primary = self._primary()

        if primary is None:
            # Report every engine's real reason instead of a single vague error.
            diagnostic_engines = (
                [engine for engine in [self._engine(self._preferred)] if engine is not None]
                if not self._fallbacks_enabled and self._preferred not in {"auto", ""}
                else self._engines
            )
            for engine in diagnostic_engines:
                try:
                    attempts.append(
                        self._run(engine, samples, sample_rate)
                    )
                except Exception as exc:  # noqa: BLE001 - defensive
                    attempts.append(
                        EngineTranscript(
                            engine=engine.name,
                            model=getattr(getattr(engine, "_source", None), "model_id", engine.name),
                            status="failed",
                            detail=type(exc).__name__,
                        )
                    )
            detail = "No ASR engine is available on this host."
            reasons = [a.detail for a in attempts if a.detail]
            return TranscriptionResult(
                status="unavailable",
                error=" ".join([detail, *reasons]) if reasons else detail,
                engines=[a.as_dict() for a in attempts],
            )

        result = self._run_safe(primary, samples, sample_rate)
        attempts.append(result)

        # A model may be present but fail while loading or decoding. When the
        # fallback policy is enabled, try the remaining available engines rather
        # than turning a recoverable provider failure into a failed transcript.
        if (result.status != "completed" or not result.text.strip()) and self._fallbacks_enabled:
            for name in self._order():
                if name == primary.name:
                    continue
                engine = self._engine(name)
                if engine is None or not engine.is_available():
                    continue
                candidate = self._run_safe(engine, samples, sample_rate)
                attempts.append(candidate)
                if candidate.status == "completed" and candidate.text.strip():
                    result = candidate
                    break

        # Hindi path: when the text looks Hindi/Hinglish, run the dedicated
        # Hindi acoustic model too and keep the better transcript. Both are
        # real models, so the choice is evidence-based, not aesthetic.
        hindi = self._hindi_engine()
        if (
            hindi is not None
            and hindi is not primary
            and result.status == "completed"
            and (
                result.language in {"hi", "hi-en"}
                or is_hindi_text(result.text)
                or looks_hinglish(result.text)
                or (hint == "hi")
            )
        ):
            hindi_result = hindi.transcribe(samples, sample_rate)
            attempts.append(hindi_result)
            if hindi_result.status == "completed" and _prefer_hindi(result.text, hindi_result.text):
                result = hindi_result

        language, confidence, source = _resolve_language(result, hint)

        if result.status != "completed":
            return TranscriptionResult(
                status=result.status,
                error=result.detail or "Speech recognition failed.",
                language=language,
                language_probability=confidence,
                language_source=source,
                selected_engine=result.engine,
                engines=[a.as_dict() for a in attempts],
            )

        limitations = [
            "Automatic transcription can mishear names, numbers and accents; "
            "verify important wording against the recording.",
        ]
        if is_hindi_text(result.text):
            limitations.append(
                "Hindi text was produced by a Devanagari ASR model; code-mixed Hinglish "
                "words may be transliterated."
            )
        if not result.text:
            limitations.append("No speech was recognized in this recording.")
        return TranscriptionResult(
            status="completed",
            language=language,
            language_probability=confidence,
            language_source=source,
            selected_engine=result.engine,
            text=result.text,
            segments=result.segments,
            model=result.model,
            limitations=limitations,
            engines=[a.as_dict() for a in attempts],
        )

    def describe(self) -> dict:
        return {
            "service": "multi-engine-asr",
            "preferred": self._preferred,
            "fallbacks_enabled": self._fallbacks_enabled,
            "engines": self.engine_status(),
            "available": any(entry["available"] for entry in self.engine_status()),
            "language_hint": self._language_hint,
        }


def _prefer_hindi(primary_text: str, hindi_text: str) -> bool:
    """Keep the Hindi model's text when it is clearly the better transcript."""
    if not hindi_text.strip():
        return False
    primary_devanagari = len(_DEVANAGARI.findall(primary_text or ""))
    hindi_devanagari = len(_DEVANAGARI.findall(hindi_text))
    if hindi_devanagari == 0:
        return False
    if primary_devanagari == 0:
        return True
    return hindi_devanagari >= primary_devanagari and len(hindi_text) >= len(primary_text) * 0.6


def _resolve_language(
    result: EngineTranscript, hint: str | None
) -> tuple[str | None, float | None, str]:
    if hint:
        return hint, result.language_probability, "request_hint"
    if result.language:
        return result.language, result.language_probability, "engine"
    text = result.text or ""
    if is_hindi_text(text):
        return "hi", None, "script_detection"
    if looks_hinglish(text):
        return "hi-en", None, "script_detection"
    if text.strip():
        return "en", None, "script_detection"
    return None, None, "unknown"


def build_transcriber(language_hint: str | None = None) -> MultiEngineTranscriber:
    """Factory used by the app; keeps the engine chain in one place."""
    return MultiEngineTranscriber(language_hint=language_hint)
