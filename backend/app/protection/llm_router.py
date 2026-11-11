"""LLM threat-reasoning router: local Ollama first, Mistral AI cloud second.

The requested architecture is a private model that answers the reasoning step
on the operator's own machine, with a hosted fallback so the product still says
something useful when that machine is asleep, out of memory or missing the tag:

    primary   Ollama daemon  -> model ``voxisentinel-mistral``
              (fine-tune of unsloth/mistral-7b-instruct-v0.3, served from GGUF)
    fallback  Mistral AI La Plateforme -> ``mistral-small-latest``

Both adapters already exist and both return the same validated JSON schema, so
the router only decides which one answers and records what happened. Rules:

  * the fallback is used only when the primary does not produce a valid answer
  * a fallback answer is labelled, never presented as the local model
  * when neither answers, the pipeline keeps the rule-based verdict and says so
  * every attempt (status, code, latency, model) is returned for the dashboard
"""

from __future__ import annotations

import dataclasses
import logging
import os
from dataclasses import dataclass, field
from typing import Any

from app.protection.mistral import DEFAULT_MISTRAL_MODEL, MistralConfig, MistralScamClassifier
from app.protection.ollama import OllamaConfig, OllamaConfigError, OllamaScamClassifier

logger = logging.getLogger("voxsentinel.protection.llm_router")

LOCAL_MODEL = "voxisentinel-mistral"
_FALLBACK_BUILDERS = {"mistral", "groq", "gemini", "xai", "none"}


@dataclass(frozen=True)
class RouterConfig:
    """What the router is wired to. No keys are ever stored here."""

    primary: str = "ollama"
    fallback: str = "mistral"
    primary_model: str = LOCAL_MODEL
    fallback_model: str = DEFAULT_MISTRAL_MODEL
    configured: bool = False
    allow_remote_fallback: bool = True

    def describe(self) -> dict[str, Any]:
        return {
            "provider": "router",
            "label": "Local voxisentinel-mistral (Ollama) with Mistral AI fallback",
            "configured": self.configured,
            "model": self.primary_model,
            "enabled": self.configured,
            "chain": [self.primary, self.fallback] if self.allow_remote_fallback else [self.primary],
            "primary_model": self.primary_model,
            "fallback_model": self.fallback_model,
            "role": "primary_contextual_classifier" if self.configured else "unavailable",
        }


def _local_ollama_config() -> OllamaConfig:
    """Ollama config whose default tag is the project's fine-tune."""
    config = OllamaConfig.from_env()
    explicit = (os.environ.get("OLLAMA_MODEL") or os.environ.get("VOXSENTINEL_OLLAMA_MODEL") or "").strip()
    if explicit:
        return config
    model = (os.environ.get("VOXSENTINEL_OLLAMA_MODEL") or LOCAL_MODEL).strip() or LOCAL_MODEL
    return dataclasses.replace(config, model=model)


def build_fallback(name: str):
    """Build the configured hosted fallback, or None when unavailable."""
    name = (name or "none").strip().lower()
    if name == "none":
        return None
    if name == "mistral":
        config = MistralConfig.from_env()
        return MistralScamClassifier(config) if config.configured else None
    if name == "groq":
        from app.protection.groq import GroqConfig, GroqScamClassifier

        config = GroqConfig.from_env()
        return GroqScamClassifier(config) if config.configured else None
    if name == "gemini":
        from app.protection.gemini import GeminiConfig, GeminiScamClassifier

        config = GeminiConfig.from_env()
        return GeminiScamClassifier(config) if config.configured else None
    if name == "xai":
        from app.protection.grok import GrokConfig, GrokScamClassifier

        config = GrokConfig.from_env()
        return GrokScamClassifier(config) if config.configured else None
    return None


# --------------------------------------------------------------- reachability
_PROBE_TTL_S = 30.0
_PROBE_CACHE: dict[str, tuple[float, bool, str | None]] = {}


def _probe_local(base_url: str) -> tuple[bool, str | None]:
    """Is the local Ollama daemon answering? Cached, and fast enough for /health."""
    import time as _time

    import httpx

    now = _time.monotonic()
    cached = _PROBE_CACHE.get(base_url)
    if cached and now - cached[0] < _PROBE_TTL_S:
        return cached[1], cached[2]
    url = f"{base_url.rstrip('/')}/api/tags"
    try:
        response = httpx.get(url, timeout=0.6)
        reachable = response.status_code == 200
        detail = None if reachable else f"HTTP {response.status_code}"
    except Exception as exc:  # noqa: BLE001 - any failure means "not answering"
        reachable = False
        detail = type(exc).__name__
    _PROBE_CACHE[base_url] = (now, reachable, detail)
    return reachable, detail


class ThreatReasoningRouter:
    """Ordered provider chain with per-attempt telemetry."""

    provider_key = "router"

    def __init__(
        self,
        primary: Any | None = None,
        fallback: Any | None = None,
        fallback_name: str | None = None,
        force: bool = False,
    ) -> None:
        # ``force`` lets the unified pipeline enable the local model without the
        # operator having to set the legacy enable flag as well.
        if primary is None:
            try:
                config = _local_ollama_config()
                if force and not config.enabled:
                    config = dataclasses.replace(config, enabled=True)
                primary = OllamaScamClassifier(config)
            except OllamaConfigError as exc:  # unsafe base URL: disable, never crash
                logger.warning("ollama disabled for router: %s", exc)
                primary = OllamaScamClassifier(OllamaConfig(enabled=False))
        self._primary = primary
        name = (fallback_name or os.environ.get("VOXSENTINEL_LLM_FALLBACK") or "mistral").strip().lower()
        if name not in _FALLBACK_BUILDERS:
            name = "mistral"
        self._fallback_name = name
        self._fallback = fallback if fallback is not None else build_fallback(name)
        # A refused connection to a sleeping daemon costs a syscall per request,
        # so a failed primary is skipped briefly instead of being retried on
        # every upload. The hosted fallback still answers meanwhile.
        self._primary_down_until = 0.0
        self._primary_cooldown_s = float(os.environ.get("VOXSENTINEL_LLM_PRIMARY_COOLDOWN_S", "60"))
        primary_conf = getattr(self._primary, "_config", None)
        fallback_conf = getattr(self._fallback, "_config", None)
        self._config = RouterConfig(
            primary_model=str(getattr(primary_conf, "model", LOCAL_MODEL)),
            fallback_model=str(
                getattr(fallback_conf, "model", DEFAULT_MISTRAL_MODEL)
            ),
            configured=bool(
                getattr(primary_conf, "configured", False)
                or getattr(fallback_conf, "configured", False)
            ),
            allow_remote_fallback=self._fallback is not None,
            fallback=name if self._fallback is not None else "none",
        )

    # -- status -----------------------------------------------------------
    @property
    def provider_label(self) -> str:
        return "Local voxisentinel-mistral" if self._primary_configured() else "Threat reasoning (LLM)"

    def _primary_configured(self) -> bool:
        return bool(getattr(getattr(self._primary, "_config", None), "configured", False))

    def describe(self) -> dict[str, Any]:
        chain = []
        for name, provider in self._chain():
            conf = getattr(provider, "_config", None)
            configured = bool(getattr(conf, "configured", False))
            entry = {
                "provider": name,
                "configured": configured,
                "model": getattr(conf, "model", None),
                "label": getattr(provider, "provider_label", name),
            }
            if name == "ollama":
                # A configured local model is not a working one: probe the daemon
                # so /health and /api/capabilities never overstate readiness.
                reachable, detail = _probe_local(str(getattr(conf, "base_url", "")))
                entry["reachable"] = bool(reachable and configured)
                entry["detail"] = detail
            else:
                # Hosted providers can only be judged by their key being present.
                entry["reachable"] = configured
            chain.append(entry)
        return {**self._config.describe(), "resolved_chain": chain}

    def _chain(self) -> list[tuple[str, Any]]:
        chain: list[tuple[str, Any]] = []
        if self._primary is not None:
            chain.append(("ollama", self._primary))
        if self._fallback is not None and self._config.allow_remote_fallback:
            chain.append((self._fallback_name, self._fallback))
        return chain

    # -- inference --------------------------------------------------------
    def classify(
        self, transcript: str, language: str | None = None
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        import time as _time

        attempts: list[dict[str, Any]] = []
        for index, (name, provider) in enumerate(self._chain()):
            if index == 0 and self._primary_down_until > _time.monotonic():
                attempts.append(
                    {
                        "provider": name,
                        "status": "skipped",
                        "detail": "The local model failed recently; retrying shortly.",
                    }
                )
                continue
            config = getattr(provider, "_config", None)
            if config is not None and not getattr(config, "configured", False):
                attempts.append(
                    {
                        "provider": name,
                        "status": "unavailable",
                        "detail": f"{name} is not configured on this server.",
                    }
                )
                continue
            result, status = provider.classify(transcript, language)
            attempts.append(
                {
                    "provider": name,
                    "status": status.get("status"),
                    "code": status.get("code"),
                    "model": status.get("model") or getattr(config, "model", None),
                    "latency_ms": status.get("latency_ms"),
                    "detail": status.get("detail"),
                }
            )
            if result is not None:
                merged = dict(status)
                merged.update(
                    {
                        "provider": name,
                        "model": status.get("model") or getattr(config, "model", None),
                        "provider_label": getattr(provider, "provider_label", name),
                        "fallback_used": index > 0,
                        "primary_failed": index > 0,
                        "attempts": attempts,
                    }
                )
                if index > 0:
                    merged["detail"] = (
                        status.get("detail")
                        or "Answered by the hosted fallback because the local model did not answer."
                    )
                return result, merged
            logger.info("llm chain: %s did not answer (%s)", name, status.get("code"))
            if index == 0 and str(status.get("code") or "").startswith(
                ("ollama_connection", "ollama_timeout", "ollama_unavailable")
            ):
                self._primary_down_until = _time.monotonic() + self._primary_cooldown_s

        statuses = {a["status"] for a in attempts}
        if statuses == {"unavailable"}:
            overall = "unavailable"
            code = "no_provider"
            detail = (
                "No threat-reasoning provider is configured. The verdict comes from the rule "
                "engine and the local text classifiers."
            )
        else:
            overall = "failed"
            code = "all_providers_failed"
            detail = (
                "Neither the local voxisentinel-mistral model nor the hosted fallback returned "
                "a valid answer; the verdict comes from the rule engine and local classifiers."
            )
        return None, {
            "provider": "router",
            "status": overall,
            "code": code,
            "detail": detail,
            "model": self._config.primary_model,
            "attempts": attempts,
        }

    def health(self) -> dict[str, Any]:
        """Probe the local daemon when it is an Ollama adapter."""
        probe = getattr(self._primary, "health", None)
        if callable(probe):
            try:
                return {"ollama": probe()}
            except Exception as exc:  # noqa: BLE001 - reporting only
                return {"ollama": {"reachable": False, "detail": type(exc).__name__}}
        return {}


@dataclass
class ReasoningVerdict:
    """Normalised, structured threat reasoning for the unified response."""

    provider: str | None = None
    model: str | None = None
    status: str = "unavailable"
    threat_type: str | None = None
    severity: str | None = None
    confidence: float | None = None
    explanation: str | None = None
    indicators: list[dict] = field(default_factory=list)
    tactics: list[str] = field(default_factory=list)
    requested_actions: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    uncertainty: list[str] = field(default_factory=list)
    fallback_used: bool = False
    attempts: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "status": self.status,
            "fallback_used": self.fallback_used,
            "threat_type": self.threat_type,
            "severity": self.severity,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "indicators": self.indicators,
            "manipulation_tactics": self.tactics,
            "requested_actions": self.requested_actions,
            "recommended_actions": self.recommendations,
            "uncertainty_reasons": self.uncertainty,
            "attempts": self.attempts,
        }


def to_verdict(result: dict | None, status: dict | None) -> ReasoningVerdict:
    """Normalise a provider result into the structured reasoning block."""
    status = status or {}
    verdict = ReasoningVerdict(
        provider=status.get("provider"),
        model=status.get("model"),
        status=str(status.get("status") or "unavailable"),
        fallback_used=bool(status.get("fallback_used")),
        attempts=list(status.get("attempts") or []),
    )
    if not result:
        return verdict
    categories = [str(c) for c in (result.get("scam_categories") or [])]
    severity = str(result.get("risk_level") or "") or None
    verdict.threat_type = ", ".join(categories[:3]) if categories else None
    verdict.severity = severity
    # Provider schemas do not supply a validated confidence estimate. Severity
    # is categorical and must not be converted into a fabricated probability.
    raw_confidence = result.get("confidence")
    if isinstance(raw_confidence, (int, float)) and 0.0 <= float(raw_confidence) <= 1.0:
        verdict.confidence = float(raw_confidence)
    verdict.explanation = result.get("summary")
    verdict.indicators = [
        {
            "category": item.get("category"),
            "quote": item.get("evidence_quote"),
            "explanation": item.get("explanation"),
            "severity": item.get("severity"),
        }
        for item in (result.get("indicators") or [])
    ]
    verdict.tactics = [str(t) for t in (result.get("manipulation_tactics") or [])]
    verdict.requested_actions = [str(t) for t in (result.get("requested_actions") or [])]
    verdict.recommendations = [str(t) for t in (result.get("protective_recommendations") or [])]
    verdict.uncertainty = [str(t) for t in (result.get("uncertainty_reasons") or [])]
    return verdict


def build_router(force_local: bool | None = None) -> ThreatReasoningRouter:
    """Factory used by the unified pipeline.

    ``force_local`` defaults to VOXSENTINEL_LOCAL_LLM (true by default) so a
    developer who runs Ollama with the voxisentinel-mistral tag gets the private
    model without also setting the older enable flag.
    """
    if force_local is None:
        raw = os.environ.get("VOXSENTINEL_LOCAL_LLM")
        force_local = True if raw is None or raw.strip() == "" else raw.strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    return ThreatReasoningRouter(force=bool(force_local))
