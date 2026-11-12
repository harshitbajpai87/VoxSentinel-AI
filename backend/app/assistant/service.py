"""Gemini-backed help assistant for the VoxSentinel AI web app.

Design notes that matter for the deployment:

* The API key lives only in the server environment (``GEMINI_API_KEY``). It is
  read here, never logged, and never included in a response.
* The model is configurable (``GEMINI_ASSISTANT_MODEL`` / ``GEMINI_MODEL``) so
  it can be upgraded without touching the UI.
* Every failure mode raises a typed error. The route turns that into a friendly
  message; the assistant never returns invented platform facts or results.
* Rate limiting is reused from :mod:`app.services.rate_limit`, the same bounded
  sliding window the provider-backed speech endpoints use.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.services.rate_limit import SlidingWindowLimiter

logger = logging.getLogger("voxsentinel.assistant")

GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models"
# Same default the contextual scam layer uses (live-verified Oct 2026); the
# deployment can pin a different one with GEMINI_ASSISTANT_MODEL.
DEFAULT_MODEL = "gemini-3.1-flash-lite"
DEFAULT_FALLBACK_MODELS = "gemini-3.8-flash"

# Conversation limits: enough for a real back-and-forth, small enough that a
# single request cannot burn the provider quota or the request budget.
MAX_MESSAGES = 12
MAX_MESSAGE_CHARS = 4000
MAX_TOTAL_CHARS = 14000
MAX_OUTPUT_TOKENS = 1200
REQUEST_TIMEOUT_S = 40.0
RETRY_STATUSES = {429, 500, 502, 503, 504}
PERMANENT_STATUSES = {400, 401, 403, 404}

SYSTEM_PROMPT = """You are VoxSentinel Assistant, the official AI guide for the VoxSentinel AI platform.

Your primary responsibility is to help new and existing users understand how to use the application, explain its functionalities, guide users through workflows and clarify AI-generated results.

You have knowledge of these platform modules:

1. Dashboard: Overview of the platform and analysis activity.
2. Analyze Audio: Upload audio files and analyze them for possible synthetic speech and suspicious content.
3. Live Recorder: Record audio through the browser and view provisional risk information.
4. Voice Detection: Explain synthetic speech detection using acoustic features and trained machine learning models.
5. Speaker Verification: Explain voice enrollment, consent and comparison using ECAPA-TDNN speaker embeddings.
6. Speech-to-Text: Explain transcription and multilingual speech recognition.
7. Risk Analysis: Explain scam indicators, suspicious phrases, contextual analysis, evidence and risk levels.
8. History: Help users find and review previous analyses.
9. Reports and CSV Export: Explain how to view and export available analysis results.
10. Evaluation: Explain model evaluation, precision, recall, F1-score, ROC-AUC and other available metrics.
11. Security and Audit: Explain consent, audit logs, data retention and privacy controls.
12. Settings: Explain available application and privacy settings.

IMPORTANT BEHAVIOUR:

- Explain features in simple language suitable for beginners.
- Provide step-by-step instructions when users ask how to use a feature.
- Explain technical terms such as MFCC, ECAPA-TDNN, Whisper, embeddings, synthetic speech, precision, recall and F1-score.
- When a user asks how to perform an action, explain exactly which page or button to open.
- When possible, provide contextual help based on the current page.
- If the user is on the Analyze Audio page, prioritize explaining audio uploads and analysis.
- If the user is on Speaker Verification, explain enrollment, consent and voice comparison.
- Never claim that the platform guarantees fraud detection or proves someone's identity.
- Explain that synthetic voice detection and scam detection are separate tasks.
- Never invent features, buttons, settings, model results or analysis outcomes.
- Do not fabricate access to the user's private analysis records.
- Never reveal API keys, credentials, internal secrets or sensitive user data.
- If a feature is unavailable, clearly explain that it is unavailable.
- If a question is outside VoxSentinel AI, politely answer briefly or redirect the user to the relevant project functionality.
- Only name buttons and controls that the CURRENT CONTEXT section describes. If a control is not described there, refer to it generically (for example "the upload area") rather than guessing a label.

Your personality should be friendly, intelligent, professional and supportive.

Keep normal responses concise, but provide detailed step-by-step guidance when requested."""

# Page context: route -> what the user is looking at right now. Only public,
# static descriptions are sent, never audio, transcripts or analysis records.
PAGE_CONTEXT: dict[str, dict[str, str]] = {
    "/dashboard": {
        "page": "Dashboard",
        "module": "Dashboard / Overview",
        "hint": (
            "The user is on the Dashboard. It summarises analysis activity: recent analyses, "
            "risk categories, quality warnings and backend readiness. Offer to explain the "
            "counters or how to start a first analysis."
        ),
    },
    "/analyze": {
        "page": "Analyze Audio",
        "module": "Analyze Audio + Voice Detection + Speech-to-Text",
        "hint": (
            'The page starts with an "Upload & inspect" step (drag-and-drop or a file picker, '
            'then an in-browser preview), followed by a "Start analysis" action. A finished '
            'analysis reports an outcome, a score, audio-quality warnings and an explanation. '
            'Offer to explain supported formats, size limits, or how to read the result.' 
        ),
    },
    "/record": {
        "page": "Live Recording",
        "module": "Live Recorder",
        "hint": (
            'The page records through the browser with Pause/Resume, shows a Mean score and a '
            'Provisional risk reading, and lists Checkpoints. Explain that a provisional reading '
            'is not a final verdict and that a short or quiet recording is less reliable.' 
        ),
    },
    "/verify": {
        "page": "Speaker Verification",
        "module": "Speaker Verification",
        "hint": (
            'The page takes a "Reference recording" and a "Recording to verify", requires the '
            'authorized-comparison consent, then reports Similarity, Threshold and Outcome. '
            'Explain enrollment, consent, and that a similarity score is not proof of identity.' 
        ),
    },
    "/history": {
        "page": "Analysis History",
        "module": "History + Reports + CSV Export",
        "hint": (
            'The page lists past analyses with timestamp, file, risk category, model, latency and '
            'outcome, and offers a CSV export of the history. Offer help reviewing a past '
            'analysis, opening a report, or exporting the history.' 
        ),
    },
    "/protection": {
        "page": "Family Emergency & Scam Protection",
        "module": "Risk Analysis",
        "hint": (
            'The page takes a call recording (or pasted transcript), and reports a transcription, '
            'Detected indicators, a risk category, Model recommendations and a Family voice '
            'comparison when a family voice profile is selected. Explain that scam detection and '
            'synthetic-voice detection are separate tasks.' 
        ),
    },
    "/evaluation": {
        "page": "Evaluation",
        "module": "Evaluation",
        "hint": (
            'Evaluation reports whether a run has executed; an empty result means no evaluation '
            'run has been executed yet. Offer to explain precision, recall, F1-score and ROC-AUC.' 
        ),
    },
    "/": {
        "page": "Landing page",
        "module": "Overview",
        "hint": "The user is on the landing page. Offer a short tour of what VoxSentinel AI does.",
    },
}


def normalise_api_key(raw: str | None) -> str | None:
    """Return a usable key, repairing a truncated new-format prefix.

    New-format Google keys look like ``AQ.Ab8...`` (50 characters after the
    prefix). If a secret is stored without the leading ``AQ.`` the key is
    rejected with ``API_KEY_INVALID`` even though it is valid, so the prefix is
    restored here. Legacy ``AIza...`` keys and anything already prefixed are
    passed through untouched.
    """
    key = (raw or "").strip().strip('"').strip("'")
    if not key:
        return None
    if key.startswith("AQ.") or key.startswith("AIza"):
        return key
    if "." not in key and 40 <= len(key) <= 70 and key[0].isalnum():
        return f"AQ.{key}"
    return key


class AssistantError(Exception):
    """Base error carrying a safe, user-facing message and a stable code."""

    code = "assistant_error"
    http_status = 500
    message = "The assistant is unavailable right now."

    def __init__(self, message: str | None = None, *, code: str | None = None) -> None:
        super().__init__(message or self.message)
        if message:
            self.message = message
        if code:
            self.code = code


class AssistantUnavailable(AssistantError):
    """Gemini could not answer (no key, timeout, provider error)."""

    code = "assistant_unavailable"
    http_status = 503
    message = "The assistant can't reach Gemini right now. Please try again in a moment."


class RateLimited(AssistantError):
    code = "assistant_rate_limited"
    http_status = 429
    message = "You're sending messages quickly. Give it a few seconds and try again."


class InvalidRequest(AssistantError):
    code = "invalid_request"
    http_status = 400
    message = "That message could not be sent."


@dataclass(frozen=True)
class AssistantConfig:
    api_key: str | None = None
    model: str = DEFAULT_MODEL
    fallback_models: tuple[str, ...] = ()
    timeout_s: float = REQUEST_TIMEOUT_S
    enabled: bool = True

    @property
    def configured(self) -> bool:
        return bool(self.api_key) and self.enabled

    @classmethod
    def from_env(cls) -> "AssistantConfig":
        enabled = (os.environ.get("VOXSENTINEL_ASSISTANT_ENABLED") or "true").lower() not in {
            "0",
            "false",
            "no",
        }
        model = (
            os.environ.get("GEMINI_ASSISTANT_MODEL")
            or os.environ.get("GEMINI_MODEL")
            or DEFAULT_MODEL
        ).strip()
        fallbacks = tuple(
            m.strip()
            for m in (
                os.environ.get("GEMINI_ASSISTANT_FALLBACK_MODELS") or DEFAULT_FALLBACK_MODELS
            ).split(",")
            if m.strip() and m.strip() != model
        )
        return cls(
            api_key=normalise_api_key(os.environ.get("GEMINI_API_KEY")),
            model=model,
            fallback_models=fallbacks,
            timeout_s=float(os.environ.get("GEMINI_ASSISTANT_TIMEOUT_S") or REQUEST_TIMEOUT_S),
            enabled=enabled,
        )

    def describe(self) -> dict[str, Any]:
        """Public capability block. Never includes the key."""
        return {
            "provider": "gemini",
            "model": self.model,
            "fallback_models": list(self.fallback_models),
            "enabled": self.enabled,
            "configured": self.configured,
            "limits": {
                "max_messages": MAX_MESSAGES,
                "max_message_chars": MAX_MESSAGE_CHARS,
                "max_total_chars": MAX_TOTAL_CHARS,
            },
        }


@dataclass
class _Attempt:
    ok: bool
    text: str | None = None
    model: str | None = None
    code: str | None = None
    detail: str | None = None
    latency_ms: int | None = None
    blocked: bool = False


def _clean(value: Any) -> str:
    return (value or "").strip() if isinstance(value, str) else ""


def normalise_messages(raw: Any) -> list[dict[str, str]]:
    """Validate the browser transcript. Raises :class:`InvalidRequest`."""
    if not isinstance(raw, list) or not raw:
        raise InvalidRequest("Send at least one message.", code="no_messages")

    cleaned: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        role = _clean(item.get("role")).lower()
        content = _clean(item.get("content") or item.get("text"))
        if role not in {"user", "assistant", "model"}:
            continue
        if not content:
            continue
        if len(content) > MAX_MESSAGE_CHARS:
            raise InvalidRequest(
                f"Each message must be under {MAX_MESSAGE_CHARS} characters.",
                code="message_too_long",
            )
        cleaned.append({"role": "model" if role == "model" else role, "content": content})

    # Keep the most recent turns; Gemini does not need the whole session.
    trimmed = cleaned[-MAX_MESSAGES:]
    while trimmed and trimmed[-1]["role"] != "user":
        trimmed.pop()
    if not trimmed:
        raise InvalidRequest("Send a question to the assistant.", code="no_user_message")
    total = sum(len(m["content"]) for m in trimmed)
    if total > MAX_TOTAL_CHARS:
        raise InvalidRequest(
            "This conversation is getting long. Clear it and ask again.",
            code="conversation_too_long",
        )
    return trimmed


def _page_block(path: str | None) -> dict[str, str]:
    route = _clean(path) or "/dashboard"
    if "?" in route:
        route = route.split("?", 1)[0]
    if route not in PAGE_CONTEXT and route.rstrip("/") in PAGE_CONTEXT:
        route = route.rstrip("/") or "/"
    block = PAGE_CONTEXT.get(route) or PAGE_CONTEXT["/dashboard"]
    return {"route": route, **block}


class AssistantService:
    """Thin, testable wrapper around Gemini's generateContent chat API."""

    def __init__(
        self,
        config: AssistantConfig | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._config = config or AssistantConfig.from_env()
        self._transport = transport
        self._limiter = SlidingWindowLimiter(per_minute=15, daily_quota=600)

    # ------------------------------------------------------------- capability
    def describe(self) -> dict[str, Any]:
        return {**self._config.describe(), "provider_label": "Google Gemini"}

    @property
    def configured(self) -> bool:
        return self._config.configured

    def check_rate(self, key: str) -> None:
        decision = self._limiter.check(key)
        if not decision.allowed:
            raise RateLimited(code=f"assistant_{decision.reason}_limit")

    def record_hit(self, key: str) -> None:
        self._limiter.record(key)

    # ----------------------------------------------------------------- request
    def _payload(
        self, messages: list[dict[str, str]], page: dict[str, str], sanitised: str
    ) -> dict[str, Any]:
        contents = [
            {
                "role": "model" if message["role"] == "assistant" else message["role"],
                "parts": [{"text": message["content"]}],
            }
            for message in messages
        ]
        system = (
            f"{SYSTEM_PROMPT}\n\nCURRENT CONTEXT (provided by the application, not by the user):\n"
            f"- Page: {page['page']}\n"
            f"- Route: {page['route']}\n"
            f"- Module: {page['module']}\n"
            f"- Guidance: {page['hint']}\n\n"
            "Use this context only to tailor your help. It contains no private user data.\n\n"
            f"CURRENT TIME (UTC): {sanitised}"
        )
        return {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": contents,
            "generationConfig": {
                "temperature": 0.6,
                "topP": 0.95,
                "maxOutputTokens": MAX_OUTPUT_TOKENS,
            },
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
            ],
        }

    def _post(self, payload: dict[str, Any], model: str) -> httpx.Response:
        headers = {
            "x-goog-api-key": self._config.api_key or "",
            "Content-Type": "application/json",
        }
        url = f"{GEMINI_ENDPOINT}/{model}:generateContent"
        with httpx.Client(timeout=self._config.timeout_s, transport=self._transport) as client:
            return client.post(url, headers=headers, json=payload)

    def _extract(self, body: Any) -> tuple[str | None, bool]:
        """Pull answer text out of a generateContent body. (text, blocked)."""
        if not isinstance(body, dict):
            return None, False
        candidates = body.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            blocked = bool((body.get("promptFeedback") or {}).get("blockReason"))
            return None, blocked
        candidate = candidates[0] or {}
        parts = ((candidate.get("content") or {}).get("parts")) or []
        chunks = [
            _clean(part.get("text"))
            for part in parts
            if isinstance(part, dict) and not part.get("thought") and _clean(part.get("text"))
        ]
        text = "\n\n".join(chunks).strip()
        if not text:
            finish = (candidate.get("finishReason") or "").upper()
            return None, finish in {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST"}
        return text, False

    def _attempt(self, payload: dict[str, Any], model: str) -> _Attempt:
        started = time.perf_counter()
        for attempt in range(2):
            try:
                response = self._post(payload, model)
            except httpx.TimeoutException:
                if attempt == 0:
                    time.sleep(0.4)
                    continue
                return _Attempt(False, code="gemini_timeout", detail="The Gemini request timed out.")
            except httpx.HTTPError as exc:
                if attempt == 0:
                    time.sleep(0.4)
                    continue
                return _Attempt(
                    False, code="gemini_unreachable", detail=f"{type(exc).__name__}"
                )

            status = response.status_code
            latency = int((time.perf_counter() - started) * 1000)
            if status in RETRY_STATUSES and attempt == 0:
                time.sleep(0.5)
                continue
            if status in RETRY_STATUSES:
                return _Attempt(
                    False,
                    code="gemini_rate_limited",
                    detail="Gemini is rate limiting or overloaded.",
                    latency_ms=latency,
                )
            if status in PERMANENT_STATUSES:
                return _Attempt(
                    False,
                    code="gemini_auth_failed" if status in {401, 403} else "gemini_bad_request",
                    detail=f"Gemini returned {status}.",
                    latency_ms=latency,
                )
            if status >= 400:
                return _Attempt(
                    False, code="gemini_error", detail=f"Gemini returned {status}.", latency_ms=latency
                )

            try:
                body = response.json()
            except ValueError:
                return _Attempt(
                    False, code="gemini_malformed", detail="Unreadable Gemini response.", latency_ms=latency
                )
            text, blocked = self._extract(body)
            if text:
                return _Attempt(True, text=text, model=model, latency_ms=latency)
            return _Attempt(
                False,
                code="gemini_blocked" if blocked else "gemini_empty",
                detail="Gemini returned no usable answer.",
                latency_ms=latency,
                blocked=blocked,
            )
        return _Attempt(False, code="gemini_error", detail="No attempt was made.")

    # -------------------------------------------------------------------- chat
    def chat(
        self,
        messages: Any,
        *,
        path: str | None = None,
        now_utc: str | None = None,
    ) -> dict[str, Any]:
        if not self._config.enabled:
            raise AssistantUnavailable(
                "The VoxSentinel Assistant is switched off on this deployment.",
                code="assistant_disabled",
            )
        if not self._config.api_key:
            raise AssistantUnavailable(
                "The assistant is not configured on this server yet (missing Gemini key).",
                code="assistant_not_configured",
            )

        cleaned = normalise_messages(messages)
        page = _page_block(path)
        payload = self._payload(cleaned, page, now_utc or time.strftime("%Y-%m-%d T%H:%M", time.gmtime()))

        last_code = "gemini_error"
        for model in (self._config.model, *self._config.fallback_models):
            result = self._attempt(payload, model)
            if result.ok and result.text:
                logger.info(
                    "assistant chat ok | model=%s latency_ms=%s page=%s turns=%s",
                    model,
                    result.latency_ms,
                    page["route"],
                    len(cleaned),
                )
                return {
                    "reply": result.text,
                    "provider": "gemini",
                    "model": result.model,
                    "latency_ms": result.latency_ms,
                    "page": page["page"],
                    "context": {"route": page["route"], "module": page["module"]},
                }
            last_code = result.code or last_code
            if result.code in {"gemini_auth_failed", "gemini_bad_request"}:
                break
            if result.blocked:
                raise AssistantUnavailable(
                    "I can't help with that request. Ask me something about VoxSentinel AI.",
                    code="gemini_blocked",
                )

        logger.warning("assistant chat failed | code=%s page=%s", last_code, page["route"])
        raise AssistantUnavailable(code=last_code)


_SERVICE: AssistantService | None = None
_LOCK = threading.Lock()


def get_assistant() -> AssistantService:
    global _SERVICE  # noqa: PLW0603 - process-wide, config is frozen per start
    if _SERVICE is None:
        with _LOCK:
            if _SERVICE is None:
                _SERVICE = AssistantService()
    return _SERVICE
