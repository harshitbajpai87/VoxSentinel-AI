"""Orchestrates one protection analysis end to end.

Reuses the existing, unmodified building blocks:
  - `validate_and_decode` for audio validation/decoding
  - the existing synthetic-speech detector service (never altered)
  - the existing ECAPA speaker-verification service for family comparison
and adds the new transcription + content-analysis + risk layers.
"""

from __future__ import annotations

import os
import time
import uuid
from datetime import UTC, datetime

import numpy as np

from app.protection import risk, rules
from app.protection.gemini import GeminiScamClassifier
from app.protection.grok import GrokScamClassifier
from app.protection.groq import GroqScamClassifier
from app.protection.quality import assess_quality, qualify_voice
from app.protection.reasoner import ReasonerReview, ReasoningReviewer
from app.protection.transcription import WhisperTranscriber
from app.services.audio_validation import validate_and_decode
from app.services.detector import DetectorResult, ModelNotReadyError


def _now() -> tuple[str, float]:
    now = datetime.now(UTC)
    return now.isoformat(), now.timestamp()


def _stage(name: str, status: str, detail: str | None = None) -> dict:
    return {"name": name, "status": status, "detail": detail}


def _build_llm(override=None):
    """Pick the contextual classifier.

    Selection order: PROTECTION_LLM_PROVIDER (xai | groq | gemini), then whichever
    of those has an API key configured, so a key alone is enough to enable it.
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

    # No explicit choice: use the first configured provider.
    for candidate in (GrokScamClassifier, GroqScamClassifier, GeminiScamClassifier):
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
    ) -> None:
        self._settings = settings
        self._detector = detector_service
        self._speaker = speaker_service
        self._transcriber = transcriber or WhisperTranscriber()
        self._llm = _build_llm(gemini)
        self._reviewer = ReasoningReviewer()

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
        # "Suspected synthetic" mirrors the core app's 'elevated or high' band.
        suspected = reference >= float(thresholds.elevated)
        threshold = float(thresholds.elevated)
        return (
            {
                "status": "completed",
                # Deliberately NOT called probability: uncalibrated raw scores
                # are model outputs, and calibrated values say so in the label.
                "score": float(result.raw_score),
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
        content = rules.analyze_transcript(transcription.text)
        stages.append(
            _stage(
                "scam_content_analysis",
                "completed" if transcription.status == "completed" else "unavailable",
                content.summary,
            )
        )

        # --- Contextual LLM analysis (Grok by default; primary when configured)
        provider_label = getattr(self._llm, "provider_label", "Contextual AI")
        if transcription.status == "completed" and transcription.text.strip():
            gemini_result, gemini_status = self._llm.classify(
                transcription.text, transcription.language
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
                f"{provider_label} contextual analysis failed; this result uses the rule-based baseline only."
            )
        elif gemini_status.get("status") == "unavailable":
            warnings.append(
                f"{provider_label} contextual analysis is not configured on this server; "
                "this result uses the rule-based baseline only."
            )

        # --- risk ----------------------------------------------------------
        assessment = risk.assess(
            risk.SignalInput(
                synthetic_suspected=synthetic_suspected,
                synthetic_note=voice.get("detail"),
                speaker_outcome=speaker["outcome"]
                if speaker
                else ("reference_unavailable" if profile else None),
                signal_strength=content.signal_strength,
                flags=content.flags,
                transcription_status=transcription.status,
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
                assessment.actions,
                assessment.limitations,
            )
        # ---- second-pass reasoning review (Reasoner4All pattern, opt-in) ----
        review = ReasonerReview(status="disabled")
        if transcription.status == "completed" and transcription.text:
            review = self._reviewer.review(
                transcript=transcription.text,
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
        stages.append(_stage("report_generation", "completed"))

        llm_block = {
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
                and getattr(self._llm._config, "api_key", None)
            ),
            "fallback": None
            if gemini_status.get("status") == "completed"
            else "rules_only",
            "fallback_reason": gemini_status.get("detail")
            if gemini_status.get("status") != "completed"
            else None,
            "quota_exhausted": bool(
                gemini_status.get("status") != "completed"
                and "429" in str(gemini_status.get("detail") or "")
            ),
            "dropped_quotes": gemini_status.get("dropped_quotes", 0),
            "notes": gemini_status.get("notes", []),
            "reasoner": {
                **review.as_dict(),
                "stage_one_classification": (
                    gemini_result["classification"] if gemini_result else None
                ),
                "stage_one_risk": (gemini_result["risk_level"] if gemini_result else None),
                "final_risk_category": assessment.category,
            },
        }

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        created_at, created_epoch = _now()
        record_id = uuid.uuid4().hex[:16]

        return {
            "id": record_id,
            "created_at": created_at,
            "created_at_epoch": created_epoch,
            "status": "completed" if transcription.status == "completed" else "partial",
            "display_name": filename,
            "size_bytes": len(data),
            "duration_s": decoded.duration_s,
            "sample_rate": decoded.sample_rate,
            "channels": decoded.channels,
            "profile_id": profile["id"] if profile else None,
            "profile_name": profile["name"] if profile else None,
            "stages": stages,
            "transcript": {
                "status": transcription.status,
                "text": transcription.text,
                "language": transcription.language,
                "language_probability": transcription.language_probability,
                "model": transcription.model,
                "segments": [
                    {"start": s.start, "end": s.end, "text": s.text} for s in transcription.segments
                ],
                "error": transcription.error,
                "limitations": transcription.limitations,
            },
            "voice": voice,
            "audio_quality": quality.as_dict(),
            "speaker_comparison": speaker,
            "content": {
                "summary": content.summary,
                "primary_category": content.primary_category,
                "signal_strength": content.signal_strength,
                "discussion_context": content.discussion_context,
                "flags": content.flags,
                "categories": [
                    {"category": f.category, "label": f.label, "severity": f.severity, "matches": f.matches}
                    for f in content.categories
                ],
                "evidence_quotes": getattr(content, "evidence_quotes", []),
                "limitations": content.limitations,
            },
            "ai": {
                # Provider-neutral block; "gemini" kept as an alias so records
                # written before the Grok switch still render in the UI.
                "llm": llm_block,
                "gemini": llm_block,
                "provider": llm_block.get("provider"),
                "provider_label": provider_label,
                "role": "primary_contextual_classifier"
                if gemini_status.get("status") == "completed"
                else "unavailable_fallback_to_rules",
            },
            "risk": {
                "category": assessment.category,
                "reasoning": assessment.reasoning,
                "recommended_actions": assessment.recommended_actions,
                "limitations": assessment.limitations,
            },
            "warnings": warnings,
            "processing_ms": elapsed_ms,
            "pipeline": {
                "transcription": self._transcriber.describe(),
                "detector": voice.get("model"),
                "speaker_verifier": "ecapa" if self._speaker.is_ready() else "unavailable",
                "contextual_ai": self._llm.describe(),
                "reasoning_review": self._reviewer.describe(),
            },
        }

    def describe(self) -> dict:
        llm = self._llm.describe()
        return {
            "transcription": self._transcriber.describe(),
            "speaker_verifier_ready": self._speaker.is_ready(),
            "detector_ready": self._detector.is_ready(),
            # Provider-neutral key (Grok today, Gemini if selected via env).
            "contextual_ai": llm,
            # Second-pass review: says plainly whether it is configured and ran.
            "reasoning_review": self._reviewer.describe(),
            # Legacy alias so older clients keep working.
            "gemini": llm,
        }
