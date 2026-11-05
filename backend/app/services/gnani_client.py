"""Gnani.ai / Vach provider adapter (Phase 6).

Typed, asynchronous client for the OFFICIALLY DOCUMENTED Gnani Speech API
(docs.gnani.ai, base URL https://api.vachana.ai). The contract implemented
here was verified against the provider documentation on 2026-10-25; see
docs/GNANI_PROVIDER.md for the full verification record. Nothing in this
module is inferred from the API key format.

Design rules:
  - The API key is read from a backend-only environment variable and is
    never logged, echoed in errors, or returned to clients.
  - All requests use HTTPS with bounded connect/read/write/pool timeouts.
  - The provider's error statuses map to a typed exception taxonomy so the
    API layer can distinguish auth failures, rate limits, outages, and
    malformed responses.
  - The HTTP transport is injectable: tests pass an httpx.MockTransport (or
    any httpx.AsyncClient-compatible factory), so no test ever touches the
    network or a real secret.
  - This adapter is a SEPARATE component from the local detector. It never
    produces VoxSentinel detection scores; the provider's documented API
    does not offer synthetic-speech detection (verified 2026-10-25).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger("voxsentinel.gnani")

PROVIDER_ID = "gnani"
PROVIDER_API_NAME = "Gnani Speech API (Vach)"

# Verified against https://docs.gnani.ai/api/introduction/introduction.md
DEFAULT_BASE_URL = "https://api.vachana.ai"
AUTH_HEADER = "X-API-Key-ID"

# Verified against https://docs.gnani.ai/api/TTS/tts-inference.md
TTS_INFERENCE_PATH = "/api/v1/tts/inference"
TTS_MODELS = ("timbre-v2.5", "timbre-v3.0")
TTS_LANGUAGES = (
    "auto", "hi-IN", "en-IN", "ta-IN", "te-IN", "kn-IN", "ml-IN",
    "mr-IN", "pa-IN", "bn-IN", "gu-IN", "hi-en",
)
TTS_CONTAINERS = ("wav", "raw", "mp3", "ogg", "mulaw", "alaw")
TTS_ENCODINGS = ("linear_pcm", "pcm_s16le", "pcm_mulaw", "pcm_alaw", "oggopus")
TTS_SAMPLE_RATES = (8000, 16000, 22050, 24000, 44100, 48000)
TTS_SPEED_MIN = 0.85
TTS_SPEED_MAX = 1.15

CONTAINER_CONTENT_TYPES = {
    "wav": "audio/wav",
    "raw": "application/octet-stream",
    "mp3": "audio/mpeg",
    "ogg": "audio/ogg",
    "mulaw": "audio/basic",
    "alaw": "audio/alaw",
}


class GnaniError(RuntimeError):
    """Base class for all provider failures. Messages are safe for clients
    (no URLs, keys, or request bodies are embedded)."""


class GnaniValidationError(GnaniError):
    """Local request validation failed before any network call."""


class GnaniAuthError(GnaniError):
    """Provider rejected the configured credentials (HTTP 401/403)."""


class GnaniBadRequestError(GnaniError):
    """Provider rejected request parameters (HTTP 400)."""


class GnaniRateLimitError(GnaniError):
    """Provider rate limit exceeded (HTTP 429)."""


class GnaniProviderError(GnaniError):
    """Provider-side internal error (HTTP 5xx other than 503)."""


class GnaniUnavailableError(GnaniError):
    """Provider service is temporarily unavailable (HTTP 503)."""


class GnaniTimeoutError(GnaniError):
    """Provider did not answer within the bounded timeout."""


class GnaniConnectionError(GnaniError):
    """Provider endpoint could not be reached (network/DNS/TLS)."""


class GnaniMalformedResponseError(GnaniError):
    """Provider returned a success status with a body that is not usable audio."""


@dataclass(frozen=True)
class TTSRequest:
    """One text-to-speech request, validated locally before it is sent."""

    text: str
    model: str
    voice: str
    language: str
    speed: float = 1.0
    sample_rate: int = 48000
    container: str = "wav"
    encoding: str = "linear_pcm"

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise GnaniValidationError("Text must be a non-empty string.")
        if self.model not in TTS_MODELS:
            raise GnaniValidationError(f"Unsupported model. Supported: {', '.join(TTS_MODELS)}.")
        if not isinstance(self.voice, str) or not self.voice.strip():
            raise GnaniValidationError("A voice identifier is required.")
        if self.language not in TTS_LANGUAGES:
            raise GnaniValidationError(f"Unsupported language. Supported: {', '.join(TTS_LANGUAGES)}.")
        if not (TTS_SPEED_MIN <= float(self.speed) <= TTS_SPEED_MAX):
            raise GnaniValidationError(
                f"Speed must be between {TTS_SPEED_MIN} and {TTS_SPEED_MAX}."
            )
        if self.sample_rate not in TTS_SAMPLE_RATES:
            raise GnaniValidationError(
                f"Unsupported sample rate. Supported: {', '.join(str(r) for r in TTS_SAMPLE_RATES)}."
            )
        if self.container not in TTS_CONTAINERS:
            raise GnaniValidationError(f"Unsupported container. Supported: {', '.join(TTS_CONTAINERS)}.")
        if self.encoding not in TTS_ENCODINGS:
            raise GnaniValidationError(
                f"Unsupported encoding. Supported: {', '.join(TTS_ENCODINGS)}."
            )

    def to_payload(self, max_text_chars: int) -> dict[str, Any]:
        """Serialize to the documented request schema, enforcing the local
        bounded text limit before anything is sent."""
        if len(self.text) > max_text_chars:
            raise GnaniValidationError(
                f"Text exceeds the {max_text_chars}-character limit."
            )
        # mp3 does not take an encoding field (verified in the TTS reference).
        audio_config: dict[str, Any] = {
            "sample_rate": self.sample_rate,
            "num_channels": 1,
            "sample_width": 2,
            "container": self.container,
        }
        if self.container != "mp3":
            audio_config["encoding"] = self.encoding
        return {
            "text": self.text,
            "voice": self.voice,
            "model": self.model,
            "language": self.language,
            "speed": float(self.speed),
            "audio_config": audio_config,
        }


@dataclass(frozen=True)
class SynthesizedAudio:
    """Provider-generated audio with explicit provenance attribution.

    Everything generated through this type is SYNTHETIC by construction and
    must always be labelled as such when stored or used downstream.
    """

    audio: bytes
    content_type: str
    container: str
    provider: str = PROVIDER_ID
    model: str = field(default="")  # e.g. "timbre-v2.5"
    voice: str = field(default="")
    language: str = field(default="")

    @property
    def is_synthetic(self) -> bool:
        return True

    @property
    def generator_id(self) -> str:
        """Stable generator identifier for dataset manifests."""
        return f"{PROVIDER_ID}:{self.model}:{self.voice}"


def _status_to_exception(status_code: int) -> type[GnaniError]:
    if status_code in (401, 403):
        return GnaniAuthError
    if status_code == 400:
        return GnaniBadRequestError
    if status_code == 429:
        return GnaniRateLimitError
    if status_code == 503:
        return GnaniUnavailableError
    if status_code >= 500:
        return GnaniProviderError
    return GnaniProviderError  # any undocumented status is treated as provider-side


def _safe_detail(body: bytes) -> str:
    """Extract a short, non-sensitive error detail from a provider error body.

    Provider error bodies may echo request content, so only a trimmed prefix
    is used, and it is only logged server-side, never returned to clients.
    """
    try:
        text = body.decode("utf-8", errors="replace").strip()
    except Exception:  # noqa: BLE001
        return ""
    return " ".join(text.split())[:200]


class GnaniClient:
    """Asynchronous HTTP client for the verified Gnani Speech API contract.

    `transport_factory` is a callable returning an httpx.AsyncClient-compatible
    object; tests inject `lambda: httpx.AsyncClient(transport=httpx.MockTransport(...))`
    so unit tests never perform real network requests.
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        connect_timeout_s: float = 5.0,
        read_timeout_s: float = 30.0,
        max_text_chars: int = 1000,
        max_response_bytes: int = 25 * 1024 * 1024,
        transport_factory: Any = None,
    ) -> None:
        if not api_key or not api_key.strip():
            raise GnaniValidationError("A non-empty provider API key is required.")
        if not base_url.startswith("https://"):
            # Verified-host enforcement: the documented endpoint is HTTPS-only.
            raise GnaniValidationError("The provider base URL must use HTTPS.")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = httpx.Timeout(
            connect=connect_timeout_s, read=read_timeout_s, write=10.0, pool=connect_timeout_s
        )
        self._max_text_chars = max_text_chars
        self._max_response_bytes = max_response_bytes
        self._transport_factory = transport_factory
        self._client: httpx.AsyncClient | None = None

    @property
    def base_url(self) -> str:
        return self._base_url

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            if self._transport_factory is not None:
                self._client = self._transport_factory()
            else:
                self._client = httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()

    async def synthesize(self, request: TTSRequest) -> SynthesizedAudio:
        """Call the documented synchronous TTS inference endpoint.

        POST {base}/api/v1/tts/inference with JSON body and X-API-Key-ID auth;
        returns the full audio as binary in the requested container.
        """
        payload = request.to_payload(self._max_text_chars)
        url = f"{self._base_url}{TTS_INFERENCE_PATH}"
        try:
            client = await self._ensure_client()
            response = await client.post(
                url,
                json=payload,
                headers={
                    AUTH_HEADER: self._api_key,
                    "Accept": request.container and CONTAINER_CONTENT_TYPES[request.container],
                },
                timeout=self._timeout,
            )
        except httpx.TimeoutException as exc:
            raise GnaniTimeoutError("The speech provider did not respond in time.") from exc
        except httpx.HTTPError as exc:
            # Connection-level failure (DNS, TLS, refused). The exception text
            # can contain the full URL; only the class is surfaced.
            logger.info("provider connection failure: %s", type(exc).__name__)
            raise GnaniConnectionError("The speech provider could not be reached.") from exc

        if response.status_code != 200:
            detail = _safe_detail(response.content)
            # Logged server-side only; never includes the key or full request.
            logger.warning(
                "provider error | status=%s | detail=%s", response.status_code, detail
            )
            exc_class = _status_to_exception(response.status_code)
            if exc_class is GnaniAuthError:
                raise GnaniAuthError("The speech provider rejected the configured credentials.")
            if exc_class is GnaniBadRequestError:
                raise GnaniBadRequestError("The speech provider rejected the request parameters.")
            if exc_class is GnaniRateLimitError:
                raise GnaniRateLimitError("The speech provider rate limit was exceeded.")
            if exc_class is GnaniUnavailableError:
                raise GnaniUnavailableError("The speech provider is temporarily unavailable.")
            raise GnaniProviderError("The speech provider reported an internal error.")

        audio = response.content
        if not audio:
            raise GnaniMalformedResponseError("The speech provider returned empty audio.")
        if len(audio) > self._max_response_bytes:
            raise GnaniMalformedResponseError("The speech provider response exceeded the size bound.")
        return SynthesizedAudio(
            audio=audio,
            content_type=CONTAINER_CONTENT_TYPES[request.container],
            container=request.container,
            model=request.model,
            voice=request.voice,
            language=request.language,
        )


class GnaniSpeechService:
    """Application-level wrapper: availability, quotas, and honest capability
    reporting. Kept separate from the raw client so the API layer depends on
    this service, not on HTTP details."""

    def __init__(self, client: GnaniClient | None) -> None:
        self._client = client

    @property
    def enabled(self) -> bool:
        return self._client is not None

    async def synthesize(self, request: TTSRequest) -> SynthesizedAudio:
        if self._client is None:
            raise GnaniUnavailableError("The speech provider integration is not enabled.")
        return await self._client.synthesize(request)

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()

    def capabilities(self) -> dict[str, Any]:
        """Honest capability report. The provider's documented API does NOT
        offer synthetic-speech detection; detection stays with the local
        VoxSentinel classifier."""
        return {
            "provider": PROVIDER_ID,
            "api_name": PROVIDER_API_NAME,
            "generation_enabled": self.enabled,
            "generation_supported": True,  # TTS: verified 2026-10-25
            "detection_supported": False,   # not part of the documented API
            "note": (
                "The provider's documented API offers speech synthesis only. "
                "VoxSentinel's local classifier remains the sole detection source."
            ),
        }
