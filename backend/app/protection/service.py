"""Orchestrates one protection analysis end to end.

Reuses the existing, unmodified building blocks:
  - `validate_and_decode` for audio validation/decoding
  - the existing synthetic-speech detector service (never altered)
  - the existing ECAPA speaker-verification service for family comparison
and adds the new transcription + content-analysis + risk layers.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from datetime import UTC, datetime

import numpy as np

from app.protection import content_classifier, risk, rules
from app.protection.asr_engines import build_transcriber
from app.protection.gemini import GeminiScamClassifier
from app.protection.grok import GrokScamClassifier
from app.protection.groq import GroqScamClassifier
from app.protection.llm_router import build_router, to_verdict
from app.protection.mistral import MistralScamClassifier
from app.protection.ollama import OllamaConfig, OllamaConfigError, OllamaScamClassifier
from app.protection.quality import assess_quality, qualify_voice
from app.protection.reasoner import ReasonerReview, ReasoningReviewer
from app.protection.text_ensemble import TextEnsemble
from app.protection.transcription import TranscriptionResult, WhisperTranscriber
from app.services.audio_validation import validate_and_decode
from app.services.detector import DetectorResult, ModelNotReadyError
from app.services.speaker import speaker_backend_name

logger = logging.getLogger(__name__)


def _now() -> tuple[str, float]:
    now = datetime.now(UTC)
    return now.isoformat(), now.timestamp()


#: How long a provider is skipped after it rejects the configured credentials.
PROVIDER_COOLDOWN_S = 900.0

#: Provider error codes that mean "this configuration will not work today".
_PROVIDER_FATAL_CODES = ("auth_failed", "forbidden", "bad_request", "model_unavailable")


def _stage(name: str, status: str, detail: str | None = None) -> dict:
    return {"name": name, "status": status, "detail": detail}


def _primary_category(content, gemini_result, gemini_status) -> str:
    """Headline scam category: rules first, then the contextual model's own label."""
    if content.primary_category and content.primary_category != "legitimate_or_no_clear_scam_evidence":
        return content.primary_category
    if gemini_status.get("status") == "completed" and gemini_result:
        categories = gemini_result.get("scam_categories") or []
        classification = str(gemini_result.get("classification") or "").upper()
        if categories and classification not in {"", "LIKELY_LEGITIMATE"}:
            return str(categories[0])
    return content.primary_category


def _build_llm(override=None):
    """Pick the contextual classifier.

    Selection order: PROTECTION_LLM_PROVIDER (xai | groq | mistral | gemini |
    ollama), then whichever of those is configured, so a key alone is enough to
    enable a hosted provider. Ollama is never auto-picked by accident: it must be
    enabled with VOXSENTINEL_OLLAMA_ENABLED or selected explicitly.
    """
    if override is not None:
        return override
    provider = (os.environ.get("PROTECTION_LLM_PROVIDER") or "").strip().lower()
    if provider == "gemini":
        return GeminiScamClassifier()
    if provider == "groq":
        return GroqScamClassifier()
    if provider == "xai":
        return GrokScamClassifier()
    if provider == "mistral":
        return MistralScamClassifier()
    if provider == "ollama":
        try:
            return OllamaScamClassifier()
        except OllamaConfigError as exc:
            # Fail closed but keep the pipeline alive: an unsafe base URL disables
            # the local provider instead of shipping transcripts in clear text or
            # crashing startup. Rule-based analysis still runs.
            logger.warning("ollama disabled: %s", exc)
            return OllamaScamClassifier(OllamaConfig(enabled=False))

    # No explicit choice: prefer the local-first router (voxisentinel-mistral on
    # Ollama, hosted Mistral as fallback) when it is configured at all.
    try:
        # Local-first: the Ollama model counts as configured even when no hosted
        # fallback key exists, which is the intended default deployment.
        router = build_router()
        if router.describe().get("configured"):
            return router
    except Exception:  # pragma: no cover - defensive, never block startup
        logger.warning("llm router could not be built", exc_info=True)

    # Otherwise use the first configured hosted provider.
    # Order matters: a configured-but-broken key (xAI rejects the configured
    # model with HTTP 400) would otherwise win every automatic selection and
    # waste a request per upload. Providers are tried by likelihood of working.
    for candidate in (
        MistralScamClassifier,
        GroqScamClassifier,
        GeminiScamClassifier,
        GrokScamClassifier,
        OllamaScamClassifier,
    ):
        try:
            instance = candidate()
        except Exception:  # pragma: no cover - defensive
            continue
        if getattr(instance, "_config", None) is not None and instance._config.configured:
            return instance
    return GrokScamClassifier()


class ProtectionService:
    def __init__(
        self,
        settings,
        detector_service,
        speaker_service,
        transcriber: WhisperTranscriber | None = None,
        gemini: object | None = None,
        forensics_detector: object | None = None,
        text_ensemble: TextEnsemble | None = None,
        forensics_engine: object | None = None,
        language_hint: str | None = None,
    ) -> None:
        self._settings = settings
        self._forensics = forensics_detector
        self._detector = detector_service
        self._speaker = speaker_service
        # Multi-engine ASR by default: whisper-small, indicative Hindi model,
        # faster-whisper and the configured remote ASR, in that order.
        self._transcriber = transcriber or build_transcriber(language_hint)
        self._text_ensemble = text_ensemble
        self._forensics_engine = forensics_engine
        self._llm = _build_llm(gemini)
        self._reviewer = ReasoningReviewer()
        # Provider circuit breaker: a rejected key (401/403) or a bad model
        # (400) is not worth retrying on every upload, so the provider is
        # skipped for PROVIDER_COOLDOWN_S after such a failure. The pipeline
        # keeps working on the rule engine and the trained local classifier.
        self._llm_down_until: dict[str, float] = {}

    # ------------------------------------------------------------------ voice
    def _voice_component(
        self, samples: np.ndarray, sample_rate: int, channels: int
    ) -> tuple[dict, bool | None]:
        try:
            result: DetectorResult = self._detector.analyze(samples, sample_rate, channels)
        except ModelNotReadyError:
            return (
                {
                    "status": "unavailable",
                    "detail": "No synthetic-speech detector is loaded on this server.",
                    "score": None,
                    "score_definition": None,
                    "calibrated_probability": None,
                    "model": None,
                },
                None,
            )
        except Exception as exc:  # noqa: BLE001 - honest failure, no fabricated score
            return (
                {"status": "failed", "detail": f"Voice analysis failed: {type(exc).__name__}", "score": None},
                None,
            )

        # Reuse the project's EXISTING risk policy thresholds (unmodified) so the
        # protection module interprets detector output exactly like the core app.
        from app.services.risk_engine import RiskThresholds

        thresholds = RiskThresholds()
        calibrated = result.calibrated_probability
        reference = calibrated if calibrated is not None else result.raw_score
        # A detector whose score lives on a different scale (the torch-free
        # acoustic screen) declares its own elevated threshold; otherwise the
        # project's existing risk policy band applies unchanged.
        inner = getattr(self._detector, "_detector", self._detector)
        declared = getattr(inner, "elevated_threshold", None)
        threshold = float(declared) if declared is not None else float(thresholds.elevated)
        suspected = reference >= threshold
        # An ambiguity band keeps a borderline score from flipping to a
        # confident verdict. Modern neural TTS and a genuine recording can land
        # in the same range, so the honest answer there is "not determinable".
        band = getattr(inner, "ambiguity_band", None)
        ambiguous = band is not None and abs(reference - threshold) <= float(band)
        suspected = None if ambiguous else suspected
        return (
            {
                "status": "completed",
                # Deliberately NOT called probability: uncalibrated raw scores
                # are model outputs, and calibrated values say so in the label.
                "score": float(result.raw_score),
                "evidence": result.explanation,
                "ambiguous": bool(ambiguous),
                "reliability": {
                    "level": "model",
                    "note": (
                        "Acoustic cues only (bandwidth, digital silence, pitch smoothness, "
                        "loudness). Modern neural voice cloning is not reliably separable from "
                        "a real recording by these cues or by open anti-spoofing models, so a "
                        "borderline score means undetermined, not innocent."
                    ),
                },
                "score_label": "raw model score",
                "score_definition": result.raw_score_definition,
                "calibrated_probability": calibrated,
                "threshold": threshold,
                "suspected_synthetic": suspected,
                "model": f"{result.model_name} {result.model_version}",
                "mode": result.model_mode,
            },
            suspected,
        )

    @property
    def _gemini(self):  # legacy alias, kept for backward compatibility
        return self._llm

    @_gemini.setter
    def _gemini(self, value):
        self._llm = value

    # -------------------------------------------------------------- speakers
    def _speaker_component(
        self, reference: np.ndarray, reference_sr: int, test: np.ndarray, test_sr: int
    ) -> dict:
        if not self._speaker.is_ready():
            return {
                "status": "reference_unavailable",
                "outcome": "reference_unavailable",
                "similarity": None,
                "detail": "No speaker-embedding model is installed on this server.",
            }
        try:
            res = self._speaker.compare(reference, reference_sr, test, test_sr)
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "verification_failed",
                "outcome": "verification_failed",
                "similarity": None,
                "detail": type(exc).__name__,
            }
        outcome = getattr(res, "outcome", None) or "inconclusive"
        mapping = {
            "likely_same_speaker": "consistent_with_reference",
            "likely_different_speaker": "not_consistent_with_reference",
            "unavailable": "reference_unavailable",
            "inconclusive": "inconclusive",
        }
        return {
            "status": mapping.get(outcome, "inconclusive"),
            "outcome": mapping.get(outcome, "inconclusive"),
            "similarity": getattr(res, "similarity", None),
            "threshold": getattr(res, "threshold", None),
            "backend": speaker_backend_name(self._speaker),
            "embedding_model": getattr(res, "embedding_model", None),
            "embedding_dim": 192 if speaker_backend_name(self._speaker) == "ecapa" else None,
            "warnings": list(getattr(res, "warnings", []) or []),
        }

    # ---------------------------------------------------------------- analyze
    def analyze(
        self,
        data: bytes,
        filename: str,
        profile: dict | None = None,
        reference_audio: bytes | None = None,
    ) -> dict:
        started = time.perf_counter()
        stages: list[dict] = []
        warnings: list[str] = []

        decoded = validate_and_decode(data, filename, self._settings)
        warnings.extend(decoded.warnings)
        stages.append(
            _stage("audio_validation", "completed", f"{decoded.duration_s:.1f}s @ {decoded.sample_rate} Hz")
        )

        # --- transcription -------------------------------------------------
        transcription = self._transcriber.transcribe(decoded.samples, decoded.sample_rate)
        transcript_usable = transcription.status == "completed" and bool(
            (transcription.text or "").strip()
        )
        stages.append(
            _stage(
                "transcription",
                transcription.status,
                transcription.error if transcription.status != "completed" else transcription.model,
            )
        )
        stages.append(
            _stage(
                "language_identification",
                "completed" if transcription.language else "skipped",
                f"{transcription.language} ({transcription.language_probability:.2f})"
                if transcription.language and transcription.language_probability is not None
                else transcription.language,
            )
        )

        # --- voice authenticity (existing detector, unchanged) -------------
        voice, synthetic_suspected = self._voice_component(
            decoded.samples, decoded.sample_rate, decoded.channels
        )
        # --- audio quality (independent of the detector) ------------------
        quality = assess_quality(
            decoded.samples,
            decoded.sample_rate,
            decoded.channels,
            len((transcription.text or "").split()),
        )
        voice = qualify_voice(voice, quality)
        # Map the qualified verdict, never the pre-quality threshold result.
        synthetic_suspected = voice.get("suspected_synthetic")
        if voice.get("verdict") in {"INCONCLUSIVE", "UNAVAILABLE"}:
            synthetic_suspected = None
        # Optional heuristics never enter SignalInput or detector readiness.
        if voice.get("status") == "unavailable" and self._forensics is not None:
            try:
                forensic_result = self._forensics.analyze(
                    decoded.samples, decoded.sample_rate, decoded.channels
                )
                voice["forensics"] = {
                    "status": "completed", "kind": "heuristic_evidence_only",
                    "evidence": forensic_result.explanation, "used_for_risk": False,
                    "detail": "Acoustic heuristics are not a trained detector or proof of synthesis.",
                }
            except Exception as exc:  # optional evidence must not break analysis
                voice["forensics"] = {
                    "status": "failed", "kind": "heuristic_evidence_only",
                    "used_for_risk": False, "detail": type(exc).__name__,
                }
        if quality.verdict == "insufficient":
            # A score from unusable audio must not drive the risk decision.
            synthetic_suspected = None
            warnings.append(
                "Audio quality is insufficient, so the synthetic-speech score is "
                "reported as inconclusive and was not used for risk scoring."
            )
        elif quality.verdict == "degraded":
            warnings.append(
                "Audio quality is degraded ("
                + ", ".join(r.rstrip(".") for r in quality.reasons)
                + "). The synthetic-speech score is qualified and does not override "
                "the transcript-level evidence."
            )
        stages.append(_stage("synthetic_voice_analysis", voice["status"], voice.get("detail")))
        stages.append(
            _stage(
                "audio_quality_assessment",
                quality.verdict,
                "; ".join(quality.reasons)[:160]
                or f"{quality.rms_dbfs:.1f} dBFS, SNR {quality.snr_db:.1f} dB",
            )
        )

        # --- family voice comparison --------------------------------------
        speaker = None
        if profile and reference_audio:
            ref = validate_and_decode(reference_audio, "reference.wav", self._settings)
            speaker = self._speaker_component(
                ref.samples, ref.sample_rate, decoded.samples, decoded.sample_rate
            )
            stages.append(_stage("family_voice_comparison", speaker["status"]))
        else:
            stages.append(
                _stage(
                    "family_voice_comparison",
                    "skipped" if not profile else "reference_unavailable",
                    None if not profile else "The selected profile has no usable reference audio.",
                )
            )

        # --- content analysis ---------------------------------------------
        content, model_signal, model_raised, text_ensemble_block = self._content_layers(
            transcription.text,
            transcription.language,
            stages,
            transcript_ok=transcript_usable,
        )

        # --- contextual threat reasoning (local Mistral -> hosted fallback) --
        gemini_result, gemini_status = self._llm_layers(
            transcription.text,
            transcription.language,
            transcript_ok=transcript_usable,
            stages=stages,
            warnings=warnings,
        )

        # --- acoustic forensics: detector + handcrafted cues + XLSR head ----
        forensics_block = self._forensics_layers(
            decoded.samples, decoded.sample_rate, decoded.channels, voice, stages
        )

        # --- risk ----------------------------------------------------------
        assessment, review = self._risk_layers(
            content=content,
            gemini_result=gemini_result,
            gemini_status=gemini_status,
            synthetic_suspected=synthetic_suspected,
            voice=voice,
            speaker=speaker,
            profile=profile,
            transcript_ok=transcript_usable,
            transcript=transcription.text,
            stages=stages,
            warnings=warnings,
        )
        stages.append(_stage("report_generation", "completed"))

        llm_block = self._llm_block(gemini_result, gemini_status, assessment, review)

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        created_at, created_epoch = _now()
        record_id = uuid.uuid4().hex[:16]

        return {
            "id": record_id,
            "created_at": created_at,
            "created_at_epoch": created_epoch,
            "input_kind": "audio",
            "status": "completed" if transcript_usable else "partial",
            "display_name": filename,
            "size_bytes": len(data),
            "duration_s": decoded.duration_s,
            "sample_rate": decoded.sample_rate,
            "channels": decoded.channels,
            "profile_id": profile["id"] if profile else None,
            "profile_name": profile["name"] if profile else None,
            "stages": stages,
            "transcript": _transcript_block(transcription),
            "voice": voice,
            "audio_quality": quality.as_dict(),
            "speaker_comparison": speaker,
            "acoustic_forensics": forensics_block,
            "content": _content_block(
                content, model_signal, model_raised, text_ensemble_block, gemini_result, gemini_status
            ),
            "ai": {
                "llm": llm_block,
                "gemini": llm_block,
                "provider": llm_block.get("provider"),
                "provider_label": llm_block.get("provider_label"),
                "role": llm_block.get("role"),
            },
            "risk": _risk_block(assessment),
            "warnings": warnings,
            "processing_ms": elapsed_ms,
            "pipeline": self._pipeline_block(voice),
        }

    # ------------------------------------------------------------- text-only
    def analyze_text(
        self,
        text: str,
        language: str | None = None,
        display_name: str = "pasted transcript",
    ) -> dict:
        """Analyze supplied text with no audio: same content, LLM and risk layers.

        The audio-authenticity layers cannot run without a recording, so they are
        reported as ``skipped`` (not as "clean") and never influence the risk
        category. Everything else - rules, TF-IDF classifier, MuRIL/DeBERTa
        ensemble, threat reasoning, risk - is identical to the audio path.
        """
        started = time.perf_counter()
        stages: list[dict] = []
        warnings: list[str] = []
        clean = (text or "").strip()
        if not clean:
            raise ValueError("text is empty")
        resolved_language = (language or "").strip() or _guess_language(clean)
        warnings.append(
            "Text-only analysis: voice authenticity, speaker verification and audio quality "
            "cannot be assessed without a recording."
        )
        stages.append(_stage("text_input", "completed", f"{len(clean)} characters"))
        stages.append(_stage("audio_validation", "skipped", "Text-only request: no audio was uploaded."))
        stages.append(_stage("transcription", "skipped", "The transcript was supplied by the client."))
        stages.append(_stage("language_identification", "completed", resolved_language))
        stages.append(
            _stage("synthetic_voice_analysis", "skipped", "No audio to screen for synthetic speech.")
        )
        stages.append(_stage("audio_quality_assessment", "skipped", "No audio to measure."))
        stages.append(_stage("family_voice_comparison", "skipped", "No audio to compare."))

        voice = {
            "status": "skipped",
            "detail": "Text-only request: no audio authenticity score exists.",
            "score": None,
            "score_definition": None,
            "calibrated_probability": None,
            "suspected_synthetic": None,
            "verdict": "NOT_ASSESSED",
            "model": None,
            "mode": None,
        }

        content, model_signal, model_raised, text_ensemble_block = self._content_layers(
            clean, resolved_language, stages, transcript_ok=True
        )
        gemini_result, gemini_status = self._llm_layers(
            clean, resolved_language, transcript_ok=True, stages=stages, warnings=warnings
        )
        assessment, review = self._risk_layers(
            content=content,
            gemini_result=gemini_result,
            gemini_status=gemini_status,
            synthetic_suspected=None,
            voice=voice,
            speaker=None,
            profile=None,
            transcript_ok=True,
            transcript=clean,
            stages=stages,
            warnings=warnings,
        )
        stages.append(_stage("acoustic_forensics", "skipped", "No audio to screen."))
        stages.append(_stage("report_generation", "completed"))
        llm_block = self._llm_block(gemini_result, gemini_status, assessment, review)

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        created_at, created_epoch = _now()
        record_id = uuid.uuid4().hex[:16]
        transcription = TranscriptionResult(
            status="completed",
            language=resolved_language,
            text=clean,
            model="client-supplied transcript",
            language_source="client",
            limitations=[
                "The transcript was supplied by the client, so its accuracy is not verified "
                "against the original audio.",
            ],
        )

        return {
            "id": record_id,
            "created_at": created_at,
            "created_at_epoch": created_epoch,
            "input_kind": "text",
            "status": "completed",
            "display_name": display_name,
            "size_bytes": len(clean.encode("utf-8")),
            "duration_s": None,
            "sample_rate": None,
            "channels": None,
            "profile_id": None,
            "profile_name": None,
            "stages": stages,
            "transcript": _transcript_block(transcription),
            "voice": voice,
            "audio_quality": None,
            "speaker_comparison": None,
            "acoustic_forensics": {
                "status": "skipped",
                "detail": "Text-only request: no acoustic evidence exists.",
                "detector": {"status": "skipped"},
                "handcrafted": {"status": "skipped", "signals": [], "evidence": {}},
                "wav2vec2_head": {"status": "skipped"},
                "fusion": {"score": None, "verdict": "not_assessed", "components_used": []},
                "explanation": (
                    "No audio was supplied, so authenticity was not screened. The content "
                    "evidence below stands on its own."
                ),
            },
            "content": _content_block(
                content, model_signal, model_raised, text_ensemble_block, gemini_result, gemini_status
            ),
            "ai": {
                "llm": llm_block,
                "gemini": llm_block,
                "provider": llm_block.get("provider"),
                "provider_label": llm_block.get("provider_label"),
                "role": llm_block.get("role"),
            },
            "risk": _risk_block(assessment),
            "warnings": warnings,
            "processing_ms": elapsed_ms,
            "pipeline": self._pipeline_block(voice),
        }

    # --------------------------------------------------------------- layers
    def _content_layers(
        self, text: str, language: str | None, stages: list[dict], transcript_ok: bool
    ) -> tuple:
        content = rules.analyze_transcript(text)

        # Trained classifier: an independent, key-free supporting signal. It can
        # raise a weak rule verdict when the wording is unfamiliar; it can never
        # veto strong rule evidence, and the report always names the layer.
        if transcript_ok and (text or "").strip():
            model_signal = content_classifier.classify_text(text)
        else:
            model_signal = content_classifier.Classification(
                status="unavailable",
                detail="No transcript was produced, so the trained classifier was skipped.",
            )
        model_raised = False
        if (
            model_signal.status == "completed"
            and model_signal.label == "scam"
            and not content.discussion_context
            and 0 < content.signal_strength < 4
        ):
            content.signal_strength = min(10, content.signal_strength + 3)
            model_raised = True
            content.limitations = [
                *content.limitations,
                (
                    "A trained text classifier rated this transcript as likely fraudulent "
                    f"(score {model_signal.confidence}), which raised the rule-based signal. "
                    "That score is a model output on labelled training text, not a real-call probability."
                ),
            ]
        stages.append(
            _stage(
                "scam_content_analysis",
                "completed" if transcript_ok else "unavailable",
                content.summary,
            )
        )
        stages.append(
            _stage(
                "trained_content_classifier",
                model_signal.status,
                (
                    f"{model_signal.label} ({model_signal.confidence})"
                    if model_signal.status == "completed"
                    else model_signal.detail or model_signal.status
                ),
            )
        )

        # --- MuRIL / DeBERTa encoder ensemble --------------------------------
        ensemble: dict = {
            "status": "unavailable",
            "detail": "The transformer ensemble is not wired into this service.",
            "branches": [],
            "score": None,
            "label": None,
        }
        if self._text_ensemble is not None:
            try:
                ensemble = self._text_ensemble.classify(text or "", language)
            except Exception as exc:  # noqa: BLE001 - an optional branch must not break analysis
                ensemble = {
                    "status": "failed",
                    "detail": f"Text ensemble failed: {type(exc).__name__}",
                    "branches": [],
                    "score": None,
                    "label": None,
                }
        ensemble_raised = False
        if (
            ensemble.get("status") == "completed"
            and ensemble.get("label") == "scam"
            and not content.discussion_context
            and content.signal_strength < 4
        ):
            content.signal_strength = min(10, content.signal_strength + 2)
            ensemble_raised = True
            content.limitations = [
                *content.limitations,
                (
                    "The MuRIL/DeBERTa encoder ensemble also rated this wording as likely "
                    f"fraudulent (score {ensemble.get('score')}); it raised the rule-based "
                    "signal by two points. Encoder scores are model outputs on labelled text, "
                    "not a probability that this call is fraud."
                ),
            ]
        ensemble["raised_signal"] = ensemble_raised
        stages.append(
            _stage(
                "transformer_text_ensemble",
                str(ensemble.get("status") or "unavailable"),
                (
                    f"{ensemble.get('label')} ({ensemble.get('score')})"
                    if ensemble.get("status") == "completed"
                    else ensemble.get("detail")
                ),
            )
        )
        return content, model_signal, model_raised, ensemble

    def _llm_layers(
        self,
        text: str,
        language: str | None,
        transcript_ok: bool,
        stages: list[dict],
        warnings: list[str],
    ) -> tuple:
        provider_label = getattr(self._llm, "provider_label", "Contextual AI")
        provider_key = str(getattr(self._llm, "provider_key", "llm"))
        cooling = self._llm_down_until.get(provider_key, 0.0) > time.monotonic()
        if transcript_ok and (text or "").strip() and not cooling:
            gemini_result, gemini_status = self._llm.classify(text, language)
            code = str(gemini_status.get("code") or "")
            if any(marker in code for marker in _PROVIDER_FATAL_CODES):
                self._llm_down_until[provider_key] = time.monotonic() + PROVIDER_COOLDOWN_S
                logger.warning(
                    "provider %s rejected (%s); skipping it for %ss",
                    provider_key,
                    code,
                    int(PROVIDER_COOLDOWN_S),
                )
        elif cooling:
            gemini_result, gemini_status = (
                None,
                {
                    "provider": provider_key,
                    "status": "unavailable",
                    "code": "provider_cooling_down",
                    "detail": (
                        f"{provider_label} rejected the configured credentials; it is "
                        "skipped for a while and this result uses the rule-based baseline."
                    ),
                },
            )
        else:
            gemini_result, gemini_status = (
                None,
                {
                    "provider": "llm",
                    "status": "unavailable",
                    "code": "no_transcript",
                    "detail": "No transcript was produced, so contextual analysis was skipped.",
                },
            )
        provider_label = gemini_status.get("provider_label") or provider_label
        stages.append(
            _stage(
                "contextual_ai_analysis",
                gemini_status.get("status", "failed"),
                (
                    gemini_result.get("classification")
                    if gemini_result
                    else gemini_status.get("detail") or f"{provider_label} analysis unavailable"
                ),
            )
        )
        if gemini_status.get("status") == "failed":
            warnings.append(
                f"{provider_label} contextual analysis failed; this result uses the "
                "rule-based baseline only."
            )
        elif gemini_status.get("status") == "unavailable":
            warnings.append(
                f"{provider_label} contextual analysis is not configured on this server; "
                "this result uses the rule-based baseline only."
            )
        return gemini_result, gemini_status

    def _forensics_layers(
        self, samples, sample_rate: int, channels: int, voice: dict, stages: list[dict]
    ) -> dict | None:
        if self._forensics_engine is None:
            return None
        try:
            block = self._forensics_engine.analyze(samples, sample_rate, channels, voice)
        except Exception as exc:  # noqa: BLE001 - optional evidence only
            block = {
                "status": "failed",
                "detail": f"Acoustic forensics failed: {type(exc).__name__}",
                "fusion": {"score": None, "verdict": "unavailable", "components_used": []},
                "explanation": "The acoustic forensics layer failed; its evidence was not used.",
            }
        fusion = block.get("fusion") or {}
        stages.append(
            _stage(
                "acoustic_forensics",
                str(block.get("status") or "completed"),
                f"{fusion.get('verdict')} ({fusion.get('score')})"
                if fusion.get("score") is not None
                else block.get("detail"),
            )
        )
        return block

    def _risk_layers(
        self,
        *,
        content,
        gemini_result,
        gemini_status: dict,
        synthetic_suspected,
        voice: dict,
        speaker: dict | None,
        profile: dict | None,
        transcript_ok: bool,
        transcript: str,
        stages: list[dict],
        warnings: list[str],
    ) -> tuple:
        provider_label = gemini_status.get("provider_label") or getattr(
            self._llm, "provider_label", "Contextual AI"
        )
        assessment = risk.assess(
            risk.SignalInput(
                synthetic_suspected=synthetic_suspected,
                synthetic_note=voice.get("detail"),
                speaker_outcome=speaker["outcome"]
                if speaker
                else ("reference_unavailable" if profile else None),
                signal_strength=content.signal_strength,
                flags=content.flags,
                transcription_status="completed" if transcript_ok else "unavailable",
                discussion_context=content.discussion_context,
                gemini_classification=(gemini_result["classification"] if gemini_result else None),
                gemini_risk=gemini_result["risk_level"] if gemini_result else None,
                gemini_categories=tuple(gemini_result["scam_categories"]) if gemini_result else (),
                gemini_status=gemini_status.get("status", "failed"),
                llm_label=provider_label,
            )
        )
        # A claimed family emergency plus a demand for money is a high-risk
        # pattern even when no explicit threat was spoken (voice-clone openers
        # typically lead with money). This never manufactures CRITICAL.
        if (
            assessment.category == "MEDIUM"
            and content.flags.get("family_claim")
            and content.flags.get("payment_demand")
        ):
            assessment = risk.RiskAssessment(
                "HIGH",
                assessment.reasoning
                + [
                    "A family emergency claim combined with a payment demand is a "
                    "high-risk pattern even without an explicit threat."
                ],
                assessment.recommended_actions,
                assessment.limitations,
            )
        # ---- second-pass reasoning review (Reasoner4All pattern, opt-in) ----
        review = ReasonerReview(status="disabled")
        if transcript_ok and transcript:
            review = self._reviewer.review(
                transcript=transcript,
                classification=(gemini_result["classification"] if gemini_result else None),
                categories=list(gemini_result["scam_categories"]) if gemini_result else [],
                rule_summary=content.summary,
            )
        if review.status in {"completed", "failed", "unavailable"}:
            stages.append(
                _stage(
                    "reasoning_review",
                    review.status,
                    review.detail or (review.verdict or None),
                )
            )
        if review.status == "completed":
            warnings.append(
                f"Reasoning review: {review.critique}"
                if review.critique
                else "A second reasoning pass reviewed this classification."
            )
            if review.overridden and assessment.category in {"HIGH", "CRITICAL"}:
                assessment = risk.RiskAssessment(
                    category="MEDIUM",
                    reasoning=assessment.reasoning
                    + [
                        "An independent reasoning pass read this transcript as "
                        f"{review.verdict.replace('_', ' ').lower()}, so the warning is "
                        "reduced while the evidence is still shown."
                    ],
                    recommended_actions=assessment.recommended_actions,
                    limitations=assessment.limitations
                    + ["Reasoning review softened the verdict; treat as context-dependent."],
                )
                stages.append(
                    _stage(
                        "reasoning_override",
                        "applied",
                        f"{assessment.category} after review",
                    )
                )
        stages.append(_stage("risk_assessment", "completed", assessment.category))
        return assessment, review

    def _llm_block(self, gemini_result, gemini_status: dict, assessment, review) -> dict:
        provider_label = gemini_status.get("provider_label") or getattr(
            self._llm, "provider_label", "Contextual AI"
        )
        return {
            "provider": gemini_status.get("provider") or getattr(self._llm, "provider_key", "xai"),
            "provider_label": provider_label,
            "status": gemini_status.get("status", "failed"),
            "code": gemini_status.get("code"),
            "detail": gemini_status.get("detail"),
            "latency_ms": gemini_status.get("latency_ms"),
            # Prefer the model the provider reports for this call, then config.
            "model": gemini_status.get("model")
            or gemini_status.get("config", {}).get("model")
            or getattr(getattr(self._llm, "_config", None), "model", None),
            "result": gemini_result,
            "configured": bool(
                getattr(self._llm, "_config", None)
                and getattr(self._llm._config, "configured", None)
            ),
            "fallback": None
            if gemini_status.get("status") == "completed"
            else "rules_only",
            "fallback_used": bool(gemini_status.get("fallback_used")),
            "fallback_reason": gemini_status.get("detail")
            if gemini_status.get("status") != "completed"
            else None,
            "quota_exhausted": bool(
                gemini_status.get("status") != "completed"
                and "429" in str(gemini_status.get("detail") or "")
            ),
            "dropped_quotes": gemini_status.get("dropped_quotes", 0),
            "notes": gemini_status.get("notes", []),
            "attempts": gemini_status.get("attempts", []),
            "reasoning": to_verdict(gemini_result, gemini_status).as_dict(),
            "role": "primary_contextual_classifier"
            if gemini_status.get("status") == "completed"
            else "unavailable_fallback_to_rules",
            "reasoner": {
                **review.as_dict(),
                "stage_one_classification": (
                    gemini_result["classification"] if gemini_result else None
                ),
                "stage_one_risk": (gemini_result["risk_level"] if gemini_result else None),
                "final_risk_category": assessment.category,
            },
        }

    def _pipeline_block(self, voice: dict) -> dict:
        return {
            "transcription": self._transcriber.describe(),
            "content_classifier": content_classifier.describe(),
            "detector": voice.get("model"),
            "speaker_verifier": speaker_backend_name(self._speaker),
            "contextual_ai": self._llm.describe(),
            "reasoning_review": self._reviewer.describe(),
            "text_ensemble": (
                self._text_ensemble.describe() if self._text_ensemble is not None else None
            ),
            "acoustic_forensics": (
                self._forensics_engine.describe() if self._forensics_engine is not None else None
            ),
            "diarization": self._diarization_status(),
        }

    @staticmethod
    def _diarization_status() -> dict | None:
        """Diarizer status without loading models; never raises."""
        try:
            from app.services.diarization import get_diarizer

            return get_diarizer().status()
        except Exception as exc:  # noqa: BLE001 - capability probe, never fatal
            return {"status": "failed", "detail": f"{type(exc).__name__}: {str(exc)[:160]}"}

    def describe(self) -> dict:
        llm = self._llm.describe()
        speaker_description = self._speaker.describe()
        speaker_backend = speaker_backend_name(self._speaker)
        return {
            "transcription": self._transcriber.describe(),
            "speaker_verifier_ready": self._speaker.is_ready(),
            "speaker_verifier": {
                "backend": speaker_backend,
                "name": speaker_description.name,
                "model": "speechbrain/spkrec-ecapa-voxceleb"
                if speaker_backend == "ecapa"
                else speaker_description.name,
                "embedding_dim": 192 if speaker_backend == "ecapa" else None,
                "ready": self._speaker.is_ready(),
                "supports_same_different_decision": bool(
                    getattr(self._speaker, "supports_same_different_decision", lambda: False)()
                ),
                "detail": speaker_description.description,
            },
            "detector_ready": self._detector.is_ready(),
            # Provider-neutral key (Grok today, Gemini if selected via env).
            "contextual_ai": llm,
            # Second-pass review: says plainly whether it is configured and ran.
            "reasoning_review": self._reviewer.describe(),
            # Legacy alias so older clients keep working.
            "gemini": llm,
            "text_ensemble": (
                self._text_ensemble.describe() if self._text_ensemble is not None else None
            ),
            "acoustic_forensics": (
                self._forensics_engine.describe() if self._forensics_engine is not None else None
            ),
        }


def _transcript_block(transcription: TranscriptionResult) -> dict:
    return {
        "status": transcription.status,
        "text": transcription.text,
        "language": transcription.language,
        "language_probability": transcription.language_probability,
        "language_source": transcription.language_source,
        "model": transcription.model,
        "engine": transcription.selected_engine,
        "segments": [
            {"start": s.start, "end": s.end, "text": s.text} for s in transcription.segments
        ],
        "engines": transcription.engines,
        "error": transcription.error,
        "limitations": transcription.limitations,
    }


def _content_block(
    content, model_signal, model_raised: bool, text_ensemble: dict, gemini_result, gemini_status: dict
) -> dict:
    return {
        "summary": content.summary,
        # When the rule baseline finds nothing but the contextual model
        # flagged a scam, the headline category must come from the model.
        # Otherwise the panel reads "no clear scam evidence" next to a
        # CRITICAL risk, which is contradictory and misleading.
        "primary_category": _primary_category(content, gemini_result, gemini_status),
        "signal_strength": content.signal_strength,
        "discussion_context": content.discussion_context,
        "flags": content.flags,
        "categories": [
            {"category": f.category, "label": f.label, "severity": f.severity, "matches": f.matches}
            for f in content.categories
        ],
        "evidence_quotes": getattr(content, "evidence_quotes", []),
        "limitations": content.limitations,
        "trained_classifier": model_signal.to_dict(),
        "trained_classifier_raised_signal": model_raised,
        "transformer_ensemble": text_ensemble,
    }


def _risk_block(assessment) -> dict:
    return {
        "category": assessment.category,
        "reasoning": assessment.reasoning,
        "recommended_actions": assessment.recommended_actions,
        "limitations": assessment.limitations,
    }


_HINGLISH_TOKENS = (
    "bhai", "hai", "nahi", "kya", "aap", "mera", "paisa", "abhi", "jaldi", "batao",
)


def _guess_language(text: str) -> str:
    """Cheap, honest language label for client-supplied text (no model claims)."""
    import re

    devanagari = re.findall(r"[\u0900-\u097F]", text)
    letters = [c for c in text if c.isalpha()]
    if letters and len(devanagari) / len(letters) >= 0.4:
        return "hi"
    words = [w.lower() for w in re.findall(r"[A-Za-z']+", text)]
    if words and sum(1 for w in words if w in _HINGLISH_TOKENS) >= 2:
        return "hi-en"
    return "en"
