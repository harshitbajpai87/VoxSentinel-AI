"""Ollama (local models) contextual scam classification.

Ollama runs models on the operator's own machine and exposes an
OpenAI-compatible endpoint, so this adapter reuses the same validated schema,
evidence-quote verification and never-raise contract as the hosted providers.

  POST $OLLAMA_BASE_URL/v1/chat/completions     (default port 11434)
  Authorization: none for a local daemon (Bearer only if OLLAMA_API_KEY is set)
  default model: mistral                        (override with OLLAMA_MODEL)

Enabling is explicit on purpose. A local daemon is assumed to be a deliberate
deployment choice, not something to fall back to silently:

    VOXSENTINEL_OLLAMA_ENABLED=true      # or...
    PROTECTION_LLM_PROVIDER=ollama       # selecting it counts as enabling it

Security note: a plain-http base URL is accepted only for loopback hosts
(localhost / 127.0.0.1 / ::1 / 0.0.0.0). Anything else must be https, so a
mis-set environment variable cannot send transcripts in clear text across a
network.

The model served here is usually the fine-tuned one produced by
`scripts/build_scam_finetune_dataset.py` + `scripts/create_ollama_model.py`
(see docs/MISTRAL_OLLAMA.md).
"""

from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx

from app.protection.gemini import MAX_TRANSCRIPT_CHARS
from app.protection.grok import GrokScamClassifier

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "mistral"
DEFAULT_TIMEOUT_S = 120.0
_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0", "[::1]"}


class OllamaConfigError(ValueError):
    """Raised when the configured Ollama base URL is unsafe or malformed."""


def _normalise_base_url(raw: str) -> str:
    value = (raw or "").strip().rstrip("/")
    if not value:
        value = DEFAULT_OLLAMA_HOST
    if "://" not in value:
        # OLLAMA_HOST is commonly written as host:port (no scheme).
        value = f"http://{value}"
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise OllamaConfigError("OLLAMA_BASE_URL must be an http(s) URL, e.g. http://localhost:11434.")
    host = parsed.hostname.lower()
    loopback = host in _LOOPBACK_HOSTS
    if not loopback:
        try:
            loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            loopback = False
    if parsed.scheme == "http" and not loopback:
        raise OllamaConfigError(
            "Plain-http OLLAMA_BASE_URL is allowed only for loopback hosts. "
            "Use https:// for a remote Ollama daemon."
        )
    return f"{parsed.scheme}://{parsed.netloc}"


def ollama_base_url() -> str:
    """Resolved, validated base URL shared by the classifier and the reasoner."""
    raw = (
        os.environ.get("OLLAMA_BASE_URL")
        or os.environ.get("VOXSENTINEL_OLLAMA_URL")
        or os.environ.get("OLLAMA_HOST")
        or DEFAULT_OLLAMA_HOST
    )
    return _normalise_base_url(raw)


@dataclass(frozen=True)
class OllamaConfig:
    base_url: str = DEFAULT_OLLAMA_HOST
    model: str = DEFAULT_OLLAMA_MODEL
    enabled: bool = False
    api_key: str | None = None
    timeout_s: float = DEFAULT_TIMEOUT_S
    max_transcript_chars: int = MAX_TRANSCRIPT_CHARS

    @property
    def configured(self) -> bool:
        return self.enabled

    @classmethod
    def from_env(cls) -> OllamaConfig:
        flag = (os.environ.get("VOXSENTINEL_OLLAMA_ENABLED", "") or "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        selected = (os.environ.get("PROTECTION_LLM_PROVIDER", "") or "").strip().lower() == "ollama"
        return cls(
            base_url=ollama_base_url(),
            model=os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL) or DEFAULT_OLLAMA_MODEL,
            enabled=flag or selected,
            api_key=os.environ.get("OLLAMA_API_KEY") or None,
            timeout_s=float(os.environ.get("OLLAMA_TIMEOUT_S", DEFAULT_TIMEOUT_S)),
            max_transcript_chars=int(
                os.environ.get("OLLAMA_MAX_TRANSCRIPT_CHARS", MAX_TRANSCRIPT_CHARS)
            ),
        )

    def describe(self) -> dict[str, Any]:
        return {
            "provider": "ollama",
            "label": "Ollama (local)",
            "configured": self.configured,
            "model": self.model,
            "enabled": self.configured,
            "base_url": self.base_url,
            "role": "primary_contextual_classifier" if self.configured else "unavailable",
        }


class OllamaScamClassifier(GrokScamClassifier):
    """Ollama adapter: identical pipeline, local OpenAI-compatible transport."""

    provider_key = "ollama"
    code_prefix = "ollama"
    # A cold local model has to be loaded into memory on the first request, so
    # one retry after a stall is more useful than three fast attempts.
    max_attempts = 2

    @property
    def provider_label(self) -> str:
        return "Ollama (local)"

    def __init__(
        self,
        config: OllamaConfig | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        super().__init__(config or OllamaConfig.from_env(), transport=transport)

    def _endpoint(self) -> str:
        return f"{self._config.base_url}/v1/chat/completions"

    def _payload(self, transcript: str, language: str | None) -> dict[str, Any]:
        payload = super()._payload(transcript, language)
        # Ollama honours `format: json` through its native API; the OpenAI shim
        # is version-dependent, so the JSON contract is enforced by parsing and
        # schema validation here instead of asking for a mode that may be
        # ignored.
        payload.pop("response_format", None)
        payload["stream"] = False
        return payload

    def _post(self, payload: dict[str, Any]) -> httpx.Response:
        headers = {"Content-Type": "application/json"}
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"
        with httpx.Client(timeout=self._config.timeout_s, transport=self._transport) as client:
            return client.post(self._endpoint(), headers=headers, json=payload)

    def health(self) -> dict[str, Any]:
        """Probe the daemon. Never raises; used by capability reporting."""
        url = f"{self._config.base_url}/api/tags"
        try:
            with httpx.Client(timeout=5.0, transport=self._transport) as client:
                response = client.get(url)
        except httpx.HTTPError as exc:
            return {"reachable": False, "detail": type(exc).__name__, "models": []}
        if response.status_code != 200:
            return {"reachable": False, "detail": f"HTTP {response.status_code}", "models": []}
        try:
            body = response.json()
        except ValueError:
            return {"reachable": False, "detail": "invalid_json", "models": []}
        models = [m.get("name") for m in (body.get("models") or []) if isinstance(m, dict)]
        names = [m for m in models if isinstance(m, str)]
        # Ollama tags carry an implicit ":latest"; compare base names so
        # OLLAMA_MODEL=mistral matches mistral:latest.
        wanted = self._config.model.split(":", 1)[0]
        present = any(name.split(":", 1)[0] == wanted for name in names)
        return {"reachable": True, "models": names, "model_present": present}
