"""NVIDIA NemotronLabs VoiceChat 11B provider adapter (Phase 6C).

Typed asynchronous client for the SELF-HOSTED NemotronLabs VoiceChat realtime
container. The contract implemented here was verified against:

  - The official model card:
    https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B (fetched
    2026-10-25; OpenMDW 1.1 license; 11B full-duplex speech-to-speech model)
  - The official API reference shipped with the deployment container:
    voicechat_realtime_instructions/api-reference.md in
    github.com/NVIDIA-NeMo/Speech (branch nemotron-labs-voicechat)

Verified facts this adapter depends on:
  - The container exposes a WebSocket realtime API (OpenAI-Realtime-compatible
    wire protocol, JSON messages, base64 audio) at /v1/realtime.
  - HTTP health probe: GET /v1/realtime/health -> 200 ok / 503 not ready.
  - Audio: PCM 16-bit signed little-endian, mono, client in/out 24 kHz;
    recommended input chunk ~80 ms (3,840 bytes at 24 kHz PCM16).
  - Session lifecycle: session.created -> session.update -> session.updated
    -> input_audio_buffer.append* -> response.output_audio.delta* ->
    response.done -> session.close -> session.end.
  - Server `error` events carry {"error": {"code", "message"}} with known
    codes inference_timeout, inference_error, session_timeout, tools_not_set.

Honest scope:
  - VoiceChat is a conversational speech-to-speech model. It is NOT a
    synthetic-speech detector, and this adapter never produces detection
    scores. Detection stays with the local VoxSentinel classifier.
  - Running the model requires an NVIDIA GPU (A100/H100/H200/B100/B200/
    RTX-6000) with the NVIDIA inference container. This backend talks to that
    deployment over the network; it never bundles the 11B weights.
  - Like every provider result, generated agent audio is SYNTHETIC by
    construction and is always labelled as such.

Transport injection: `transport_factory` returns an awaitable that connects and
yields an object with async `send(str)`, `recv() -> str`, and `aclose()`
(websockets.WebSocketClientProtocol satisfies this). Tests inject a scripted
fake session, so no test performs a network connection.
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import io
import json
import logging
import wave
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx
import numpy as np

logger = logging.getLogger("voxsentinel.voicechat")

PROVIDER_ID = "voicechat"
PROVIDER_NAME = "NVIDIA NemotronLabs VoiceChat 11B"
MODEL_NAME = "nemotron-voicechat"

# Verified against the API reference (Audio Format section).
PCM_RATE = 24000
PCM_CHANNELS = 1
PCM_SAMPLE_WIDTH = 2
CHUNK_BYTES = 3840  # ~80 ms at 24 kHz PCM16
WS_PATH = "/v1/realtime"
HEALTH_PATH = "/v1/realtime/health"

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}

MAX_INPUT_SECONDS_DEFAULT = 60.0
IDLE_TIMEOUT_S_DEFAULT = 60.0


class VoiceChatError(RuntimeError):
    """Base class for provider failures. Client-safe messages only."""


class VoiceChatValidationError(VoiceChatError):
    """Local validation failed before any network call."""


class VoiceChatConnectionError(VoiceChatError):
    """The realtime endpoint could not be reached."""


class VoiceChatUnavailableError(VoiceChatError):
    """The health probe reports the service is not ready (HTTP 503)."""


class VoiceChatTimeoutError(VoiceChatError):
    """The provider did not produce a complete turn within the idle bound."""


class VoiceChatServerError(VoiceChatError):
    """The provider sent a fatal `error` event."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message or f"Provider error: {code}")
        self.code = code


class VoiceChatProtocolError(VoiceChatError):
    """The provider violated the documented session protocol."""


def resample_to_pcm16(
    samples: np.ndarray, sample_rate: int, channels: int
) -> bytes:
    """Convert decoded audio to the provider's wire format: mono 24 kHz PCM16.

    Linear-interpolation resampling (matches the project's lightweight CPU
    decoding style; the provider resamples internally anyway: model input is
    16 kHz, output 22.05 kHz, both carried at the 24 kHz client rate).
    """
    mono = samples
    if samples.ndim == 2:
        if channels < 1:
            raise VoiceChatValidationError("Invalid channel count.")
        mono = samples[:, : max(channels, 1)].mean(axis=1)
    mono = np.asarray(mono, dtype=np.float64).reshape(-1)

    if sample_rate == PCM_RATE:
        resampled = mono
    else:
        n_out = max(1, int(round(len(mono) * PCM_RATE / sample_rate)))
        if len(mono) < 2:
            resampled = np.zeros(n_out)
        else:
            step = (len(mono) - 1) / max(n_out - 1, 1)
            idx = np.arange(n_out) * step
            lo = np.floor(idx).astype(np.int64)
            hi = np.minimum(lo + 1, len(mono) - 1)
            frac = idx - lo
            resampled = mono[lo] * (1.0 - frac) + mono[hi] * frac

    pcm = np.clip(resampled, -1.0, 1.0)
    return (pcm * 32767.0).astype("<i2").tobytes()


def chunk_pcm(pcm: bytes, chunk_bytes: int = CHUNK_BYTES) -> list[bytes]:
    """Split PCM into ~80 ms chunks exactly as the reference recommends."""
    if chunk_bytes < 1:
        raise VoiceChatValidationError("Chunk size must be positive.")
    return [pcm[i : i + chunk_bytes] for i in range(0, len(pcm), chunk_bytes)]


def pcm16_to_wav(pcm: bytes, sample_rate: int = PCM_RATE) -> bytes:
    """Wrap raw PCM16 mono bytes in a RIFF/WAVE container for delivery."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(PCM_CHANNELS)
        handle.setsampwidth(PCM_SAMPLE_WIDTH)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    return buffer.getvalue()


@dataclass(frozen=True)
class VoiceChatTurn:
    """One completed speech-to-speech turn with explicit provenance."""

    agent_audio_pcm: bytes
    sample_rate: int
    agent_transcript: str = ""
    user_transcript: str = ""
    provider: str = PROVIDER_ID
    model: str = MODEL_NAME

    @property
    def is_synthetic(self) -> bool:
        return True

    def as_wav(self) -> bytes:
        return pcm16_to_wav(self.agent_audio_pcm, self.sample_rate)

    @property
    def generator_id(self) -> str:
        return f"{PROVIDER_ID}:{self.model}"


class BaseVoiceChatSession:
    """Minimal async session surface the client needs (send/recv/close)."""

    async def send(self, message: str) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    async def recv(self) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    async def aclose(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class WebSocketSession(BaseVoiceChatSession):
    """Real transport backed by the `websockets` library (lazy import)."""

    def __init__(self, ws_connection: Any) -> None:
        self._ws = ws_connection

    async def send(self, message: str) -> None:
        await self._ws.send(message)

    async def recv(self) -> str:
        return str(await self._ws.recv())

    async def aclose(self) -> None:
        await self._ws.close()


def default_transport_factory(ws_url: str, connect_timeout_s: float):
    """Build the production transport: connect via `websockets` over TLS.

    `websockets` is imported lazily so unit tests never require it.
    """

    async def connect() -> BaseVoiceChatSession:
        import websockets

        connection = await asyncio.wait_for(
            websockets.connect(ws_url, open_timeout=connect_timeout_s),
            timeout=connect_timeout_s + 1.0,
        )
        return WebSocketSession(connection)

    return connect


def validate_ws_url(ws_url: str) -> str:
    """Enforce verified-host transport: plain ws:// is allowed only for
    localhost deployments; everything else must use wss:// (TLS)."""
    if not ws_url or not ws_url.strip():
        raise VoiceChatValidationError("A realtime WebSocket URL is required.")
    parsed = urlparse(ws_url)
    if parsed.scheme == "wss":
        return ws_url
    if parsed.scheme == "ws" and (parsed.hostname or "") in LOCAL_HOSTS:
        return ws_url
    raise VoiceChatValidationError(
        "Non-localhost realtime URLs must use wss:// (TLS)."
    )


class VoiceChatClient:
    """Typed client for the VoiceChat realtime protocol.

    Parameters mirror Settings fields; `transport_factory(ws_url, timeout)`
    is injectable for tests (must return an awaitable producing a session).
    """

    def __init__(
        self,
        ws_url: str,
        *,
        health_url: str = "",
        connect_timeout_s: float = 5.0,
        idle_timeout_s: float = IDLE_TIMEOUT_S_DEFAULT,
        max_input_seconds: float = MAX_INPUT_SECONDS_DEFAULT,
        transport_factory: Callable[[str, float], Awaitable[BaseVoiceChatSession]] | None = None,
    ) -> None:
        self._ws_url = validate_ws_url(ws_url)
        self._health_url = health_url
        self._connect_timeout_s = connect_timeout_s
        self._idle_timeout_s = idle_timeout_s
        self._max_input_seconds = max_input_seconds
        factory = transport_factory or default_transport_factory
        self._connect = factory(self._ws_url, connect_timeout_s)

    async def check_health(self) -> dict[str, Any]:
        """Probe GET /v1/realtime/health. Raises on not-ready (503)."""
        if not self._health_url:
            # Without a configured HTTP endpoint we cannot honestly claim
            # health; connection-level checks still happen per turn.
            raise VoiceChatUnavailableError("No health endpoint configured.")
        try:
            async with httpx.AsyncClient(timeout=self._connect_timeout_s) as client:
                response = await client.get(self._health_url)
        except httpx.HTTPError as exc:
            logger.info("voicechat health probe failed: %s", type(exc).__name__)
            raise VoiceChatUnavailableError("The voice provider could not be reached.") from exc
        if response.status_code != 200:
            raise VoiceChatUnavailableError("The voice provider is not ready.")
        try:
            return response.json()
        except ValueError as exc:
            raise VoiceChatProtocolError("The health response was not valid JSON.") from exc

    async def converse(
        self,
        samples: np.ndarray,
        sample_rate: int,
        channels: int,
        *,
        instructions: str | None = None,
    ) -> VoiceChatTurn:
        """Run one full-duplex turn: stream user audio, collect agent audio.

        Implements the documented lifecycle strictly; any deviation from the
        documented event order raises VoiceChatProtocolError rather than
        being guessed around.
        """
        pcm = resample_to_pcm16(samples, sample_rate, channels)
        duration_s = len(pcm) / (PCM_RATE * PCM_CHANNELS * PCM_SAMPLE_WIDTH)
        if duration_s <= 0:
            raise VoiceChatValidationError("Input audio is empty.")
        if duration_s > self._max_input_seconds:
            raise VoiceChatValidationError(
                f"Input audio exceeds the {self._max_input_seconds:.0f}s limit."
            )

        try:
            session = await self._connect()
        except TimeoutError as exc:
            raise VoiceChatTimeoutError("Connecting to the voice provider timed out.") from exc
        except OSError as exc:
            logger.info("voicechat connection failure: %s", type(exc).__name__)
            raise VoiceChatConnectionError("The voice provider could not be reached.") from exc
        audio_chunks: list[bytes] = []
        agent_transcript = ""
        user_transcript = ""
        model_name = MODEL_NAME
        try:
            await self._expect(session, "session.created")
            model_name = self._last_model or MODEL_NAME

            await session.send(self._event("session.update", session={
                "audio": {
                    "input": {"format": {"type": "audio/pcm", "rate": PCM_RATE}},
                    "output": {"format": {"type": "audio/pcm", "rate": PCM_RATE}},
                },
                "instructions": instructions,
                "tools": [],
            }))
            await self._expect(session, "session.updated")

            for chunk in chunk_pcm(pcm):
                await session.send(self._event(
                    "input_audio_buffer.append",
                    audio=base64.b64encode(chunk).decode("ascii"),
                ))

            done = False
            while not done:
                message = await self._recv_bounded(session)
                event = self._parse(message)
                etype = event.get("type")

                if etype == "response.output_audio.delta":
                    delta = event.get("delta")
                    if not isinstance(delta, str):
                        raise VoiceChatProtocolError("Audio delta is not base64 text.")
                    try:
                        audio_chunks.append(base64.b64decode(delta, validate=True))
                    except (binascii.Error, ValueError) as exc:
                        raise VoiceChatProtocolError("Audio delta is not valid base64.") from exc
                elif etype == "response.output_audio_transcript.done":
                    agent_transcript = self._transcript(event, "transcript")
                elif etype == "conversation.item.input_audio_transcription.completed":
                    user_transcript = self._transcript(event, "transcript")
                elif etype == "response.done":
                    done = True
                elif etype == "error":
                    code, message_text = self._error_payload(event)
                    raise VoiceChatServerError(code, message_text)
                elif etype == "session.end":
                    # Server closed the session before completing a turn.
                    raise VoiceChatProtocolError("The session ended before the turn completed.")
                # Other documented events (speech_started/stopped,
                # response.created, item/content_part events, transcript
                # deltas) are informational and safely ignorable.
        except VoiceChatServerError as exc:
            if exc.code == "inference_timeout":
                raise VoiceChatTimeoutError("The provider timed out during inference.") from exc
            raise
        except TimeoutError as exc:
            raise VoiceChatTimeoutError("The provider did not finish the turn in time.") from exc
        finally:
            try:
                await session.send(self._event("session.close"))
                # Best-effort orderly shutdown: wait briefly for session.end.
                while True:
                    message = await self._recv_bounded(session, short=True)
                    if self._parse(message).get("type") == "session.end":
                        break
            except Exception:  # noqa: BLE001 — shutdown must never mask results
                pass
            await session.aclose()

        return VoiceChatTurn(
            agent_audio_pcm=b"".join(audio_chunks),
            sample_rate=PCM_RATE,
            agent_transcript=agent_transcript,
            user_transcript=user_transcript,
            model=model_name,
        )

    # -- protocol helpers ---------------------------------------------------

    _last_model: str | None = None

    @staticmethod
    def _event(event_type: str, **fields: Any) -> str:
        import uuid

        payload: dict[str, Any] = {"type": event_type, "event_id": uuid.uuid4().hex}
        payload.update(fields)
        return json.dumps(payload)

    async def _recv_bounded(
        self, session: BaseVoiceChatSession, short: bool = False
    ) -> str:
        try:
            return await asyncio.wait_for(
                session.recv(), timeout=1.0 if short else self._idle_timeout_s
            )
        except OSError as exc:
            # A socket that dies mid-turn is a connection failure, whatever
            # the exact stage of the documented lifecycle it happens at.
            logger.info("voicechat socket dropped mid-session")
            raise VoiceChatConnectionError("The voice provider connection was lost.") from exc

    def _parse(self, message: str) -> dict[str, Any]:
        try:
            event = json.loads(message)
        except json.JSONDecodeError as exc:
            raise VoiceChatProtocolError("The provider sent a non-JSON message.") from exc
        if not isinstance(event, dict) or not isinstance(event.get("type"), str):
            raise VoiceChatProtocolError("The provider sent a malformed event.")
        return event

    async def _expect(self, session: BaseVoiceChatSession, expected: str) -> dict[str, Any]:
        message = await self._recv_bounded(session)
        event = self._parse(message)
        if event.get("type") != expected:
            raise VoiceChatProtocolError(
                f"Expected '{expected}' but got '{event.get('type')}'."
            )
        if expected == "session.created":
            model = event.get("session", {}).get("model")
            self._last_model = model if isinstance(model, str) and model else None
        return event

    @staticmethod
    def _transcript(event: dict[str, Any], key: str) -> str:
        value = event.get(key, "")
        return value if isinstance(value, str) else ""

    @staticmethod
    def _error_payload(event: dict[str, Any]) -> tuple[str, str]:
        error = event.get("error", {})
        if not isinstance(error, dict):
            raise VoiceChatProtocolError("Malformed error event.")
        code = str(error.get("code", "unknown_error"))
        message = str(error.get("message", ""))[:200]
        return code, message


class VoiceChatService:
    """Application-level wrapper mirroring GnaniSpeechService: availability,
    honest capability reporting, and a single converse() entry point."""

    def __init__(self, client: VoiceChatClient | None) -> None:
        self._client = client

    @property
    def enabled(self) -> bool:
        return self._client is not None

    async def converse(
        self,
        samples: np.ndarray,
        sample_rate: int,
        channels: int,
        *,
        instructions: str | None = None,
    ) -> VoiceChatTurn:
        if self._client is None:
            raise VoiceChatUnavailableError("The voice provider integration is not enabled.")
        return await self._client.converse(
            samples, sample_rate, channels, instructions=instructions
        )

    async def aclose(self) -> None:
        return None  # sessions are per-turn; nothing persistent to close

    def capabilities(self) -> dict[str, Any]:
        """Honest capability report. VoiceChat is a speech-to-speech
        conversational generator; it performs NO synthetic-speech detection.
        Detection stays with the local VoxSentinel classifier."""
        return {
            "provider": PROVIDER_ID,
            "api_name": PROVIDER_NAME,
            "generation_enabled": self.enabled,
            "generation_supported": True,   # conversational S2S: verified 2026-10-25
            "detection_supported": False,    # not part of the documented API
            "note": (
                "Self-hosted full-duplex speech-to-speech model (requires an "
                "NVIDIA GPU deployment of the official inference container). "
                "The provider's documented API offers speech generation only; "
                "VoxSentinel's local classifier remains the sole detection source."
            ),
        }
