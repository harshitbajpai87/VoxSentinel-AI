"""Second-pass reasoning review, adapted from the Reasoner4All pattern.

Reasoner4All (huggingface.co/spaces/McLoviniTtt/Reasoner4All) is a Chainlit
Docker demo that wraps *external* providers (OpenAI, Mistral, Groq, Gemini,
NVIDIA) and adds a separate reasoning model on top of a chat model. It ships no
downloadable weights, exposes no public inference API, and runs on a
``cpu-basic`` Space with user-supplied keys, so it cannot be "trained" or
imported as a model. What *is* portable is its architectural idea:

    classifier answer  ->  independent reasoning pass  ->  reconciled verdict

This module implements that idea on top of whichever contextual provider the
deployment already uses, so no second vendor is required:

    stage 1  Groq/Gemini/xAI classifies the transcript (structured JSON)
    stage 2  a reasoning pass critiques that answer for false positives
             (news, role-play, awareness campaigns, legitimate emergencies)

Design rules:

* Opt-in. ``PROTECTION_REASONER=enabled`` turns it on; default is off so the
  existing latency and API contract are unchanged.
* Never-raise. Any failure returns ``status="failed"`` with a detail string and
  the pipeline continues with the stage-1 answer untouched.
* No new probability. The reasoning pass can only *downgrade* a verdict toward
  caution or confirm it; it never invents a numeric score, and its disagreement
  is surfaced as a warning rather than silently overriding evidence.
* Backend-only. The key is read from the process environment; nothing is sent
  to the browser.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger("voxsentinel.protection.reasoner")

PROVIDER_ENDPOINTS = {
    "groq": "https://api.groq.com/openai/v1/chat/completions",
    "xai": "https://api.x.ai/v1/chat/completions",
}
PROVIDER_KEYS = {"groq": "GROQ_API_KEY", "xai": "XAI_API_KEY"}
PROVIDER_MODELS = {"groq": "openai/gpt-oss-120b", "xai": "grok-4.7"}
PROVIDER_LABELS = {"groq": "Groq (GPT-OSS)", "xai": "Grok"}

MAX_TRANSCRIPT_CHARS = 6000

_VERDICTS = {"LIKELY_SCAM", "SUSPICIOUS", "LIKELY_LEGITIMATE", "EDUCATIONAL_OR_DISCUSSION", "INCONCLUSIVE"}

SYSTEM_PROMPT = (
    "You are a careful reviewer of fraud-detection decisions. You receive a call "
    "transcript and the classification another model produced. Your job is to catch "
    "false positives and unsupported escalation. Consider whether the transcript is "
    "actually news reporting, a role-play or film scene, an awareness/training example, "
    "a hypothetical discussion, or a genuine emergency rather than an attack on the "
    "listener. Reply with STRICT JSON only:\n"
    '{"agrees": true|false, "verdict": "LIKELY_SCAM|SUSPICIOUS|LIKELY_LEGITIMATE|'
    'EDUCATIONAL_OR_DISCUSSION|INCONCLUSIVE", "critique": "<2 sentences max>", '
    '"context_signals": ["<short phrases from the transcript that decided it>"]}\n'
    "Do not quote text that is not in the transcript. Never state that a crime occurred."
)


@dataclass
class ReasonerReview:
    """Outcome of the second-pass review. Always serialisable."""

    status: str  # completed | unavailable | failed | disabled
    provider: str | None = None
    provider_label: str | None = None
    model: str | None = None
    agrees: bool | None = None
    verdict: str | None = None
    critique: str | None = None
    context_signals: list[str] = field(default_factory=list)
    overridden: bool = False
    evidence_verified: bool = False
    dropped_signals: int = 0
    latency_ms: int | None = None
    detail: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "provider": self.provider,
            "provider_label": self.provider_label,
            "model": self.model,
            "agrees": self.agrees,
            "verdict": self.verdict,
            "critique": self.critique,
            "context_signals": list(self.context_signals),
            "overridden": self.overridden,
            "evidence_verified": self.evidence_verified,
            "dropped_signals": self.dropped_signals,
            "latency_ms": self.latency_ms,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class ReasonerConfig:
    enabled: bool
    provider: str
    api_key: str | None
    model: str
    timeout_s: float = 45.0
    max_transcript_chars: int = MAX_TRANSCRIPT_CHARS

    @property
    def available(self) -> bool:
        return self.enabled and bool(self.api_key) and self.provider in PROVIDER_ENDPOINTS

    @classmethod
    def from_env(cls, provider: str | None = None) -> ReasonerConfig:
        resolved = (provider or os.environ.get("PROTECTION_LLM_PROVIDER") or "groq").strip().lower()
        if resolved not in PROVIDER_ENDPOINTS:
            # Gemini and unknown providers have no OpenAI-compatible chat endpoint
            # here; the review is simply reported as unavailable.
            resolved = "groq"
        enabled = (os.environ.get("PROTECTION_REASONER", "disabled") or "").strip().lower() in {
            "enabled",
            "on",
            "true",
            "1",
        }
        key_env = PROVIDER_KEYS.get(resolved, "GROQ_API_KEY")
        model = (
            os.environ.get("PROTECTION_REASONER_MODEL")
            or os.environ.get(f"{resolved.upper()}_REASONER_MODEL")
            or PROVIDER_MODELS.get(resolved, PROVIDER_MODELS["groq"])
        )
        return cls(
            enabled=enabled,
            provider=resolved,
            api_key=os.environ.get(key_env) or None,
            model=model,
            timeout_s=float(os.environ.get("PROTECTION_REASONER_TIMEOUT_S", 45.0)),
            max_transcript_chars=int(
                os.environ.get("PROTECTION_REASONER_MAX_CHARS", MAX_TRANSCRIPT_CHARS)
            ),
        )


def _extract_json(text: str) -> dict[str, Any] | None:
    """Parse the model's JSON, tolerating code fences and prose wrappers."""
    if not text:
        return None
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.S)
    if fence:
        cleaned = fence.group(1)
    else:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            return None
        cleaned = cleaned[start : end + 1]
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


class ReasoningReviewer:
    """Reviews a stage-1 classification with an independent reasoning pass."""

    def __init__(self, config: ReasonerConfig | None = None) -> None:
        self._config = config or ReasonerConfig.from_env()

    def describe(self) -> dict[str, Any]:
        cfg = self._config
        return {
            "enabled": cfg.enabled,
            "available": cfg.available,
            "provider": cfg.provider,
            "provider_label": PROVIDER_LABELS.get(cfg.provider),
            "model": cfg.model,
            "inspired_by": "huggingface.co/spaces/McLoviniTtt/Reasoner4All",
        }

    def review(
        self,
        transcript: str,
        classification: str | None,
        categories: list[str] | None = None,
        rule_summary: str | None = None,
    ) -> ReasonerReview:
        cfg = self._config
        if not cfg.enabled:
            return ReasonerReview(status="disabled", detail="Reasoning review is off.")
        if not cfg.available:
            return ReasonerReview(
                status="unavailable",
                provider=cfg.provider,
                detail="No API key configured for the reasoning review.",
            )
        if not transcript or not transcript.strip():
            return ReasonerReview(status="unavailable", detail="No transcript to review.")

        payload_text = transcript.strip()[: cfg.max_transcript_chars]
        user_prompt = (
            f"TRANSCRIPT:\n{payload_text}\n\n"
            f"STAGE-1 CLASSIFICATION: {classification or 'INCONCLUSIVE'}\n"
            f"STAGE-1 CATEGORIES: {', '.join(categories or []) or 'none'}\n"
            f"RULE SUMMARY: {rule_summary or 'none'}\n\n"
            "Review this decision and reply with the JSON object."
        )
        started = time.monotonic()
        try:
            with httpx.Client(timeout=cfg.timeout_s) as client:
                response = client.post(
                    PROVIDER_ENDPOINTS[cfg.provider],
                    headers={
                        "Authorization": f"Bearer {cfg.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": cfg.model,
                        "temperature": 0,
                        "max_tokens": 400,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                    },
                )
        except Exception as exc:  # noqa: BLE001 - never raise into the pipeline
            logger.warning("reasoner transport failed: %s", type(exc).__name__)
            return ReasonerReview(
                status="failed",
                provider=cfg.provider,
                provider_label=PROVIDER_LABELS.get(cfg.provider),
                model=cfg.model,
                detail=f"Reasoning request failed ({type(exc).__name__}).",
            )

        latency_ms = int((time.monotonic() - started) * 1000)
        if response.status_code != 200:
            return ReasonerReview(
                status="failed",
                provider=cfg.provider,
                provider_label=PROVIDER_LABELS.get(cfg.provider),
                model=cfg.model,
                latency_ms=latency_ms,
                detail=f"Reasoning provider returned {response.status_code}.",
            )

        try:
            body = response.json()
            content = body["choices"][0]["message"]["content"]
        except Exception:  # noqa: BLE001
            return ReasonerReview(
                status="failed",
                provider=cfg.provider,
                model=cfg.model,
                latency_ms=latency_ms,
                detail="Reasoning provider returned an unexpected body.",
            )

        data = _extract_json(content)
        if data is None:
            return ReasonerReview(
                status="failed",
                provider=cfg.provider,
                model=cfg.model,
                latency_ms=latency_ms,
                detail="Reasoning answer was not valid JSON.",
            )

        verdict = str(data.get("verdict") or "").strip().upper()
        if verdict not in _VERDICTS:
            verdict = None
        agrees = data.get("agrees")
        if not isinstance(agrees, bool):
            agrees = (verdict == classification) if verdict else None
        raw_signals = data.get("context_signals")
        raw_signals = [str(x).strip() for x in raw_signals][:8] if isinstance(raw_signals, list) else []
        # A signal only counts when it is a verbatim slice of the transcript.
        # This is what stops the reviewer inventing or rewording evidence.
        lowered = payload_text.lower()
        signals = [s[:120] for s in raw_signals if s and s.lower() in lowered][:5]
        dropped = len([s for s in raw_signals if s]) - len(signals)
        critique = str(data.get("critique") or "").strip()[:400] or None
        # Softening requires (a) a downgrade verdict and (b) at least one quoted,
        # verbatim signal from the transcript. A downgrade with no evidence is
        # logged but never applied, so a strong ransom/fraud warning survives.
        downgrade_verdict = classification in {"LIKELY_SCAM", "SUSPICIOUS"} and verdict in {
            "LIKELY_LEGITIMATE",
            "EDUCATIONAL_OR_DISCUSSION",
        }
        # Every quoted signal must be verbatim: one invented quote means the
        # review is not trustworthy enough to soften a protective warning.
        downgrade = downgrade_verdict and bool(signals) and dropped == 0
        detail = None
        if downgrade_verdict and (not signals or dropped):
            detail = (
                "Reviewer proposed softening the verdict without quotable "
                "transcript evidence; stage-one result kept."
            )
        return ReasonerReview(
            status="completed",
            provider=cfg.provider,
            provider_label=PROVIDER_LABELS.get(cfg.provider),
            model=cfg.model,
            agrees=agrees,
            verdict=verdict,
            critique=critique,
            context_signals=signals,
            overridden=downgrade,
            evidence_verified=bool(signals),
            dropped_signals=max(0, dropped),
            latency_ms=latency_ms,
            detail=detail,
        )
