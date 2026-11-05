"""Groq contextual scam classification.

Groq (groq.com) serves open models over an OpenAI-compatible API. It is a
differently-named provider from xAI's Grok: keys start with ``gsk_`` and the
base URL is ``api.groq.com/openai/v1``. Both are supported here through the
same validated schema, evidence-quote verification and never-raise contract.

  POST https://api.groq.com/openai/v1/chat/completions
  Authorization: Bearer $GROQ_API_KEY
  default model: openai/gpt-oss-120b   (override with GROQ_MODEL)

Set PROTECTION_LLM_PROVIDER=groq to select it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx

from app.protection.gemini import MAX_TRANSCRIPT_CHARS
from app.protection.grok import GrokScamClassifier

GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"


@dataclass(frozen=True)
class GroqConfig:
    api_key: str | None
    model: str = DEFAULT_GROQ_MODEL
    timeout_s: float = 60.0
    max_transcript_chars: int = MAX_TRANSCRIPT_CHARS

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    @classmethod
    def from_env(cls) -> GroqConfig:
        return cls(
            api_key=os.environ.get("GROQ_API_KEY") or None,
            model=os.environ.get("GROQ_MODEL", DEFAULT_GROQ_MODEL) or DEFAULT_GROQ_MODEL,
            timeout_s=float(os.environ.get("GROQ_TIMEOUT_S", 60.0)),
            max_transcript_chars=int(
                os.environ.get("GROQ_MAX_TRANSCRIPT_CHARS", MAX_TRANSCRIPT_CHARS)
            ),
        )

    def describe(self) -> dict[str, Any]:
        return {
            "provider": "groq",
            "label": "Groq (GPT-OSS)",
            "configured": self.configured,
            "model": self.model,
            "enabled": self.configured,
            "role": "primary_contextual_classifier" if self.configured else "unavailable",
        }


class GroqScamClassifier(GrokScamClassifier):
    """Groq adapter: identical pipeline, OpenAI-compatible transport."""

    provider_key = "groq"

    @property
    def provider_label(self) -> str:
        return "Groq (GPT-OSS)"

    def __init__(
        self, config: GroqConfig | None = None, transport: httpx.BaseTransport | None = None
    ) -> None:
        super().__init__(config or GroqConfig.from_env(), transport=transport)

    def _endpoint(self) -> str:
        return GROQ_ENDPOINT

    def _describe_config(self) -> dict[str, Any]:
        return self._config.describe()

    def describe(self) -> dict[str, Any]:
        return self._config.describe()
