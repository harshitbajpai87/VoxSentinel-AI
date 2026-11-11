"""Mistral AI contextual scam classification.

Mistral (mistral.ai) serves its own open-weight models over an
OpenAI-compatible chat-completions API, so this adapter reuses the same
validated schema, evidence-quote verification and never-raise contract as the
Groq/xAI adapters. Only transport details differ:

  POST https://api.mistral.ai/v1/chat/completions
  Authorization: Bearer $MISTRAL_API_KEY
  default model: mistral-small-latest   (override with MISTRAL_MODEL)

Set PROTECTION_LLM_PROVIDER=mistral to select it.

A locally fine-tuned Mistral served by Ollama is a different adapter
(`app/protection/ollama.py`); this one talks to the hosted La Plateforme API.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx

from app.protection.gemini import MAX_TRANSCRIPT_CHARS
from app.protection.grok import GrokScamClassifier

MISTRAL_ENDPOINT = "https://api.mistral.ai/v1/chat/completions"
DEFAULT_MISTRAL_MODEL = "mistral-small-latest"


@dataclass(frozen=True)
class MistralConfig:
    api_key: str | None
    model: str = DEFAULT_MISTRAL_MODEL
    timeout_s: float = 60.0
    max_transcript_chars: int = MAX_TRANSCRIPT_CHARS

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    @classmethod
    def from_env(cls) -> MistralConfig:
        return cls(
            api_key=os.environ.get("MISTRAL_API_KEY") or None,
            model=os.environ.get("MISTRAL_MODEL", DEFAULT_MISTRAL_MODEL)
            or DEFAULT_MISTRAL_MODEL,
            timeout_s=float(os.environ.get("MISTRAL_TIMEOUT_S", 60.0)),
            max_transcript_chars=int(
                os.environ.get("MISTRAL_MAX_TRANSCRIPT_CHARS", MAX_TRANSCRIPT_CHARS)
            ),
        )

    def describe(self) -> dict[str, Any]:
        return {
            "provider": "mistral",
            "label": "Mistral AI",
            "configured": self.configured,
            "model": self.model,
            "enabled": self.configured,
            "role": "primary_contextual_classifier" if self.configured else "unavailable",
        }


class MistralScamClassifier(GrokScamClassifier):
    """Mistral adapter: identical pipeline, OpenAI-compatible transport."""

    provider_key = "mistral"
    code_prefix = "mistral"

    @property
    def provider_label(self) -> str:
        return "Mistral AI"

    def __init__(
        self,
        config: MistralConfig | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        super().__init__(config or MistralConfig.from_env(), transport=transport)

    def _endpoint(self) -> str:
        return MISTRAL_ENDPOINT
