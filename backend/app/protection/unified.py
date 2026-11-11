"""Unified protection response (schema 1.0) for ``POST /api/analyze``.

One request, one document, five model families. The pipeline already computes
every signal; this module only *presents* them under the names the product and
the dashboard use, and never invents a value that the pipeline did not produce.

Sections
--------
transcription        which ASR engine ran (whisper-small / indicwav2vec-hindi /
                     faster-whisper / remote), the text, language and confidence
speaker_verification ECAPA-TDNN 192-dim comparison against a family profile
acoustic_forensics   detector.joblib + handcrafted cues + XLSR representation head
scam_risk            rule engine, TF-IDF classifier, MuRIL/DeBERTa ensemble
llm_reasoning        local voxisentinel-mistral first, mistral-small-latest second
risk                 the category and what to do about it
risk_badges          the small set of states the dashboard renders live
explanation          the structured JSON threat explanation
"""

from __future__ import annotations

from typing import Any

SCHEMA_VERSION = "1.0"

_TONE = {"CRITICAL": "critical", "HIGH": "critical", "MEDIUM": "warn", "LOW": "ok",
         "INCONCLUSIVE": "info"}

_CATEGORY_LABELS = {
    "CRITICAL": "Critical risk",
    "HIGH": "High risk",
    "MEDIUM": "Medium risk",
    "LOW": "Low risk",
    "INCONCLUSIVE": "Inconclusive",
}


def _badge(badge_id: str, label: str, tone: str, value: Any = None, detail: str | None = None) -> dict:
    return {"id": badge_id, "label": label, "tone": tone, "value": value, "detail": detail}


# --------------------------------------------------------------- sections
def _transcription(result: dict) -> dict:
    transcript = result.get("transcript") or {}
    engines = transcript.get("engines") or []
    used = [e for e in engines if e.get("status") == "completed"]
    primary = used[0] if used else None
    selected_engine = transcript.get("engine") or (primary or {}).get("engine")
    return {
        "status": transcript.get("status"),
        "text": transcript.get("text") or "",
        "language": transcript.get("language"),
        "language_probability": transcript.get("language_probability"),
        "language_source": transcript.get("language_source"),
        "engine": selected_engine,
        "model": transcript.get("model") or (primary or {}).get("model"),
        "segments": transcript.get("segments") or [],
        "engines": engines,
        "error": transcript.get("error"),
        "limitations": transcript.get("limitations") or [],
    }


def _speaker_verification(result: dict) -> dict:
    comparison = result.get("speaker_comparison") or {}
    pipeline = result.get("pipeline") or {}
    backend = comparison.get("backend") or pipeline.get("speaker_verifier") or "unavailable"
    status = comparison.get("status") or ("skipped" if not comparison else "inconclusive")
    raw_outcome = comparison.get("outcome") or status
    outcome = {
        "consistent_with_reference": "likely_same_speaker",
        "not_consistent_with_reference": "likely_different_speaker",
    }.get(raw_outcome, raw_outcome)
    similarity = comparison.get("similarity")
    threshold = comparison.get("threshold")
    return {
        "status": status,
        "outcome": outcome,
        "match": True if outcome == "likely_same_speaker" else False
        if outcome == "likely_different_speaker" else None,
        "similarity": similarity,
        "threshold": threshold,
        "profile_id": result.get("profile_id"),
        "profile_name": result.get("profile_name"),
        "backend": backend,
        "model": comparison.get("embedding_model") if similarity is not None else None,
        "embedding_dim": comparison.get("embedding_dim") if similarity is not None else None,
        "sample_rate_hz": 16000 if similarity is not None and backend == "ecapa" else None,
        "detail": comparison.get("detail")
        or (
            "No family voice profile was supplied, so no same/different-speaker verdict exists."
            if not comparison
            else None
        ),
        "warnings": comparison.get("warnings") or [],
    }


def _acoustic_forensics(result: dict) -> dict:
    block = result.get("acoustic_forensics")
    voice = result.get("voice") or {}
    if not block:
        return {
            "status": voice.get("status") or "unavailable",
            "detector": {
                "status": voice.get("status"),
                "model": voice.get("model"),
                "score": voice.get("score"),
                "calibrated_probability": voice.get("calibrated_probability"),
                "threshold": voice.get("threshold"),
                "verdict": voice.get("verdict"),
            },
            "handcrafted": {"status": "unavailable", "signals": [], "evidence": {}},
            "wav2vec2_head": {"status": "unavailable"},
            "fusion": {
                "score": voice.get("calibrated_probability") or voice.get("score"),
                "verdict": (voice.get("verdict") or "unavailable").lower(),
                "components_used": ["detector"] if voice.get("status") == "completed" else [],
            },
            "explanation": voice.get("evidence") or voice.get("detail"),
        }
    return block


def _scam_risk(result: dict) -> dict:
    content = result.get("content") or {}
    voice = result.get("voice") or {}
    transcript = result.get("transcript") or {}
    transcript_usable = transcript.get("status") == "completed" and bool(
        str(transcript.get("text") or "").strip()
    )
    strength = content.get("signal_strength")
    return {
        "status": "completed" if transcript_usable else "unavailable",
        "score": None
        if not transcript_usable or strength is None
        else round(float(strength) * 10, 1),
        "score_scale": "0-100 rule-and-model evidence strength, not a probability",
        "signal_strength": strength if transcript_usable else None,
        "signal_strength_max": 10,
        "primary_category": content.get("primary_category"),
        "summary": content.get("summary"),
        "categories": content.get("categories") or [],
        "flags": content.get("flags") or {},
        "discussion_context": content.get("discussion_context"),
        "evidence_quotes": content.get("evidence_quotes") or [],
        "trained_classifier": content.get("trained_classifier"),
        "transformer_ensemble": content.get("transformer_ensemble"),
        "detector": {
            "model": voice.get("model"),
            "score": voice.get("score"),
            "calibrated_probability": voice.get("calibrated_probability"),
            "suspected_synthetic": voice.get("suspected_synthetic"),
            "verdict": voice.get("verdict"),
        },
        "limitations": content.get("limitations") or [],
    }


def _llm_reasoning(result: dict) -> dict:
    ai = result.get("ai") or {}
    llm = ai.get("llm") or {}
    reasoning = llm.get("reasoning") or {}
    return {
        "provider": llm.get("provider"),
        "provider_label": llm.get("provider_label"),
        "model": llm.get("model"),
        "status": llm.get("status"),
        "fallback_used": bool(llm.get("fallback_used")),
        "fallback_reason": llm.get("fallback_reason"),
        "latency_ms": llm.get("latency_ms"),
        "attempts": llm.get("attempts") or [],
        "threat_type": reasoning.get("threat_type"),
        "severity": reasoning.get("severity"),
        "confidence": reasoning.get("confidence"),
        "explanation": reasoning.get("explanation"),
        "indicators": reasoning.get("indicators") or [],
        "manipulation_tactics": reasoning.get("manipulation_tactics") or [],
        "requested_actions": reasoning.get("requested_actions") or [],
        "recommended_actions": reasoning.get("recommended_actions") or [],
        "uncertainty_reasons": reasoning.get("uncertainty_reasons") or [],
    }


# ----------------------------------------------------------------- badges
def _badges(result: dict, sections: dict) -> list[dict]:
    risk_block = result.get("risk") or {}
    category = str(risk_block.get("category") or "INCONCLUSIVE")
    voice = result.get("voice") or {}
    forensics = sections["acoustic_forensics"]
    fusion = forensics.get("fusion") or {}
    speaker = sections["speaker_verification"]
    content = result.get("content") or {}
    transcription = sections["transcription"]
    reasoning = sections["llm_reasoning"]

    badges = [
        _badge(
            "risk",
            _CATEGORY_LABELS.get(category, category),
            _TONE.get(category, "info"),
            category,
            (risk_block.get("reasoning") or [""])[0],
        )
    ]

    synthetic = voice.get("suspected_synthetic")
    fused_verdict = str(fusion.get("verdict") or "").lower()
    if voice.get("status") == "skipped":
        badges.append(_badge("voice_authenticity", "No audio to screen", "info", None,
                             "Audio authenticity only applies to uploaded recordings."))
    elif fused_verdict == "inconclusive":
        # The fused screen exists and did not cross its threshold, so the badge
        # must not claim a synthetic voice just because the detector alone did.
        detail = fusion.get("explanation") or (
            "The acoustic screens did not agree strongly enough to call this recording "
            "synthetic or genuine."
        )
        if synthetic is True:
            detail = (
                "The trained detector scored above its own threshold, but the other acoustic "
                "cues did not corroborate it, so the fused verdict stays undetermined. "
                + detail
            )
        badges.append(
            _badge("voice_authenticity", "Voice authenticity inconclusive", "info",
                   fusion.get("score"), detail)
        )
    elif synthetic is True or fused_verdict == "elevated":
        badges.append(_badge(
            "voice_authenticity", "Synthetic voice suspected", "critical",
            fusion.get("score"),
            "Acoustic evidence is elevated. This is a screening signal, not proof of a deepfake.",
        ))
    elif synthetic is False or fused_verdict == "not_elevated":
        badges.append(_badge(
            "voice_authenticity", "No synthetic markers", "ok", fusion.get("score"),
            "No authenticity signal crossed its threshold; a clean real recording also looks like this.",
        ))
    else:
        badges.append(_badge(
            "voice_authenticity", "Voice authenticity inconclusive", "info", fusion.get("score"),
            voice.get("detail") or fusion.get("explanation"),
        ))

    outcome = speaker.get("outcome")
    if outcome == "likely_same_speaker":
        badges.append(_badge("speaker_match", "Voice matches profile", "ok",
                             speaker.get("similarity"), speaker.get("profile_name")))
    elif outcome == "likely_different_speaker":
        badges.append(_badge("speaker_match", "Voice does not match", "critical",
                             speaker.get("similarity"), speaker.get("profile_name")))
    elif outcome in {"inconclusive", "reference_unavailable", "skipped"} or not outcome:
        badges.append(_badge("speaker_match", "Speaker not verified", "info",
                             speaker.get("similarity"), speaker.get("detail")))
    else:
        badges.append(_badge("speaker_match", f"Speaker: {outcome.replace('_', ' ')}", "info",
                             speaker.get("similarity"), speaker.get("detail")))

    strength = content.get("signal_strength")
    usable_text = transcription.get("status") == "completed" and bool(
        str(transcription.get("text") or "").strip()
    )
    if not usable_text:
        badges.append(_badge(
            "scam_intent", "Scam intent not assessed", "info", None,
            "No usable transcript was available; the absence of scam wording is not evidence of safety.",
        ))
    elif isinstance(strength, (int, float)) and strength >= 6:
        badges.append(_badge("scam_intent", "Scam language detected", "critical", strength,
                             content.get("summary")))
    elif isinstance(strength, (int, float)) and strength > 0:
        badges.append(_badge("scam_intent", "Some scam wording", "warn", strength,
                             content.get("summary")))
    else:
        badges.append(_badge("scam_intent", "No scam wording found", "ok", strength,
                             content.get("summary")))

    if usable_text and result.get("input_kind") == "text":
        badges.append(_badge(
            "transcription", "Transcript supplied by client", "info",
            transcription.get("language"),
            "Text was not verified against an original recording.",
        ))
    elif usable_text:
        badges.append(_badge(
            "transcription",
            f"Transcribed ({transcription.get('engine') or transcription.get('model') or 'ASR'})",
            "ok",
            transcription.get("language"),
            transcription.get("model"),
        ))
    elif transcription.get("status") == "completed":
        badges.append(_badge(
            "transcription", "No speech recognized", "warn", None,
            "The ASR engine completed but returned no usable text.",
        ))
    else:
        badges.append(_badge(
            "transcription", "No transcript", "warn", None,
            transcription.get("error") or "Speech recognition did not run on this host.",
        ))

    if reasoning.get("status") == "completed":
        label = (
            f"{reasoning.get('provider_label') or reasoning.get('model') or 'Threat reasoning'}"
            + (" (fallback)" if reasoning.get("fallback_used") else "")
        )
        badges.append(_badge("llm_reasoning", label, "ok", reasoning.get("severity"),
                             reasoning.get("threat_type")))
    else:
        badges.append(_badge(
            "llm_reasoning", "Rules-only verdict", "info", None,
            reasoning.get("fallback_reason") or "No reasoning provider answered.",
        ))

    return badges


# ------------------------------------------------------------ explanation
def _explanation(result: dict, sections: dict) -> dict:
    risk_block = result.get("risk") or {}
    content = result.get("content") or {}
    reasoning = sections["llm_reasoning"]
    transcription = sections["transcription"]
    forensics = sections["acoustic_forensics"]
    speaker = sections["speaker_verification"]

    evidence: list[str] = []
    for item in reasoning.get("indicators") or []:
        quote = item.get("quote") or ""
        explanation = item.get("explanation") or ""
        if quote or explanation:
            evidence.append(f"{item.get('category', 'indicator')}: {explanation} (\"{quote}\")")
    for quote in content.get("evidence_quotes") or []:
        text = quote if isinstance(quote, str) else str(quote)
        if text and text not in evidence:
            evidence.append(f"Rule match: \"{text}\"")
    categories = content.get("categories") or []
    for item in categories[:3]:
        label = item.get("label") or item.get("category")
        if label:
            evidence.append(f"Category matched: {label} (severity {item.get('severity')})")

    why_not_conclusive: list[str] = []
    why_not_conclusive.extend(content.get("limitations") or [])
    why_not_conclusive.extend(risk_block.get("limitations") or [])
    if transcription.get("status") != "completed" or not str(transcription.get("text") or "").strip():
        why_not_conclusive.append("The recording could not be transcribed, so text evidence is missing.")
    elif result.get("input_kind") == "text":
        why_not_conclusive.append("The transcript was supplied by the client and was not checked against audio.")
    if forensics.get("status") in {"unavailable", "skipped", "failed"}:
        why_not_conclusive.append(
            "Acoustic authenticity was not assessed on this host, so a cloned voice "
            "cannot be confirmed or excluded."
        )
    if speaker.get("match") is None and result.get("input_kind") != "text":
        why_not_conclusive.append(
            "No family voice profile was supplied, so the speaker could not be verified."
        )

    return {
        "verdict": risk_block.get("category"),
        "threat_type": reasoning.get("threat_type") or content.get("primary_category"),
        "severity": reasoning.get("severity") or risk_block.get("category"),
        "confidence": reasoning.get("confidence"),
        "summary": reasoning.get("explanation") or content.get("summary"),
        "evidence": evidence[:12],
        "transcript_excerpt": (transcription.get("text") or "")[:600],
        "manipulation_tactics": reasoning.get("manipulation_tactics") or [],
        "requested_actions": reasoning.get("requested_actions") or [],
        "speaker": {
            "outcome": speaker.get("outcome"),
            "similarity": speaker.get("similarity"),
            "threshold": speaker.get("threshold"),
        },
        "authenticity": {
            "fused_score": (forensics.get("fusion") or {}).get("score"),
            "verdict": (forensics.get("fusion") or {}).get("verdict"),
            "components_used": (forensics.get("fusion") or {}).get("components_used"),
        },
        "why_not_conclusive": why_not_conclusive[:10],
        "recommended_actions": risk_block.get("recommended_actions") or [],
        "model_trail": {
            "transcription": transcription.get("model"),
            "speaker_verification": speaker.get("model"),
            "acoustic_forensics": [
                (forensics.get("detector") or {}).get("model"),
                (forensics.get("wav2vec2_head") or {}).get("model"),
            ],
            "scam_text": [
                (content.get("trained_classifier") or {}).get("model"),
                *[
                    branch.get("model")
                    for branch in ((content.get("transformer_ensemble") or {}).get("branches") or [])
                    if branch.get("status") == "completed"
                ],
            ],
            "llm_reasoning": reasoning.get("model"),
        },
    }


def build_unified(result: dict) -> dict:
    """Wrap a protection result in the documented unified contract.

    The legacy keys are preserved verbatim so existing clients and stored
    records keep rendering while the dashboard moves to the new sections.
    """
    sections = {
        "transcription": _transcription(result),
        "speaker_verification": _speaker_verification(result),
        "acoustic_forensics": _acoustic_forensics(result),
        "scam_risk": _scam_risk(result),
        "llm_reasoning": _llm_reasoning(result),
    }
    core = {
        "schema_version": SCHEMA_VERSION,
        "id": result.get("id"),
        "created_at": result.get("created_at"),
        "input_kind": result.get("input_kind") or "audio",
        "status": result.get("status"),
        "display_name": result.get("display_name"),
        "duration_s": result.get("duration_s"),
        "size_bytes": result.get("size_bytes"),
        "processing_ms": result.get("processing_ms"),
        "risk": result.get("risk"),
        "stages": result.get("stages"),
        "warnings": result.get("warnings"),
        "pipeline": result.get("pipeline"),
    }
    unified = {
        **core,
        **sections,
        "risk_badges": _badges(result, sections),
        "explanation": _explanation(result, sections),
    }
    unified["units"] = {
        "similarity": "cosine similarity of 192-dim ECAPA embeddings (higher = same speaker)",
        "forensics_score": "0-1 fused synthetic-speech screen (higher = more synthetic)",
        "scam_risk_score": "0-100 evidence strength from rules and text models",
    }
    # Legacy aliases: the same data under the keys older screens read.
    unified.update(
        {
            "transcript": result.get("transcript"),
            "voice": result.get("voice"),
            "audio_quality": result.get("audio_quality"),
            "speaker_comparison": result.get("speaker_comparison"),
            "content": result.get("content"),
            "ai": result.get("ai"),
            "profile_id": result.get("profile_id"),
            "profile_name": result.get("profile_name"),
        }
    )
    return unified
