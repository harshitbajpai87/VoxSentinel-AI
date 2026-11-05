"""Unit tests for the NemotronLabs VoiceChat provider adapter.

All WebSocket traffic is a scripted FAKE session implementing the documented
protocol. No test performs a network connection and no real model or GPU is
involved.
"""
from __future__ import annotations

import base64
import json

import numpy as np
import pytest

from app.services.voicechat_client import (
    CHUNK_BYTES,
    PCM_RATE,
    VoiceChatClient,
    VoiceChatConnectionError,
    VoiceChatProtocolError,
    VoiceChatServerError,
    VoiceChatTimeoutError,
    VoiceChatUnavailableError,
    VoiceChatValidationError,
    chunk_pcm,
    pcm16_to_wav,
    resample_to_pcm16,
)

WS_URL = "wss://gpu.example/v1/realtime"


def event(etype: str, **fields) -> str:
    payload = {"type": etype, "event_id": "e-1"}
    payload.update(fields)
    return json.dumps(payload)


class ScriptedSession:
    """Fake session: sends fixed server events, records client messages."""

    def __init__(self, server_events: list[str], fail_connect: bool = False):
        self.server_events = list(server_events)
        self.sent: list[dict] = []
        self.closed = False
        self._fail_connect = fail_connect

    async def send(self, message: str) -> None:
        self.sent.append(json.loads(message))

    async def recv(self) -> str:
        if self._fail_connect:
            raise OSError("connection refused (fake)")
        if self.server_events:
            return self.server_events.pop(0)
        raise TimeoutError("no more scripted events (fake)")

    async def aclose(self) -> None:
        self.closed = True


def make_client(server_events: list[str]) -> tuple[VoiceChatClient, ScriptedSession]:
    session = ScriptedSession(server_events + [event("session.end")])
    return VoiceChatClient(
        WS_URL,
        connect_timeout_s=1.0,
        idle_timeout_s=1.0,
        transport_factory=lambda url, timeout: _connect_later(session),
    ), session


def _connect_later(session):
    async def connect():
        return session

    return connect


def happy_path_events() -> list[str]:
    return [
        event("session.created", session={"model": "nemotron-voicechat"}),
        event("session.updated"),
        event("input_audio_buffer.speech_started"),
        event("response.output_audio.delta", delta=base64.b64encode(b"AAAA").decode()),
        event("response.output_audio.delta", delta=base64.b64encode(b"BBBB").decode()),
        event("response.output_audio_transcript.done", transcript="Hello there."),
        event("conversation.item.input_audio_transcription.completed",
              transcript="Hi agent."),
        event("response.output_audio.done"),
        event("response.done"),
    ]


def tone(seconds: float = 1.0, rate: int = 16000) -> tuple[np.ndarray, int, int]:
    t = np.arange(int(seconds * rate)) / rate
    samples = (0.5 * np.sin(2 * np.pi * 440 * t))
    return samples.astype(np.float64), rate, 1


def test_resample_to_pcm16_shape_and_rate():
    samples, rate, ch = tone(1.0, rate=16000)
    pcm = resample_to_pcm16(samples, rate, ch)
    # 24 kHz * 2 bytes * 1 s
    assert len(pcm) == PCM_RATE * 2
    assert isinstance(pcm, (bytes, bytearray))


def test_resample_stereo_is_mixed_to_mono():
    t = np.arange(16000) / 16000
    left = (0.25 * np.sin(2 * np.pi * 440 * t))
    stereo = np.stack([left, left], axis=1)
    pcm = resample_to_pcm16(stereo, 16000, 2)
    assert len(pcm) == PCM_RATE * 2


def test_chunk_pcm_matches_documented_80ms_chunks():
    pcm = bytes(CHUNK_BYTES * 2 + 100)
    chunks = chunk_pcm(pcm)
    assert len(chunks) == 3
    assert all(len(c) == CHUNK_BYTES for c in chunks[:2])


def test_pcm16_to_wav_is_valid_riff():
    wav = pcm16_to_wav(b"ABCD")
    assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE"


def test_ws_url_must_use_tls_off_localhost():
    with pytest.raises(VoiceChatValidationError):
        VoiceChatClient("ws://gpu.example/v1/realtime")
    with pytest.raises(VoiceChatValidationError):
        VoiceChatClient("")
    # localhost ws:// is allowed (private deployment).
    VoiceChatClient("ws://localhost:9000/v1/realtime")


def test_converse_rejects_oversized_input():
    client, _ = make_client(happy_path_events())
    with pytest.raises(VoiceChatValidationError):
        client_max = VoiceChatClient(
            WS_URL, max_input_seconds=1.0, idle_timeout_s=1.0,
            transport_factory=lambda url, timeout: _connect_later(
                ScriptedSession([])),
        )
        import asyncio

        asyncio.run(client_max.converse(*tone(2.0, rate=16000)))


def run(coro):
    import asyncio

    return asyncio.run(coro)


def test_full_converse_lifecycle():
    client, session = make_client(happy_path_events())
    turn = run(client.converse(*tone(1.0, rate=16000)))
    assert turn.agent_audio_pcm == b"AAAABBBB"
    assert turn.agent_transcript == "Hello there."
    assert turn.user_transcript == "Hi agent."
    assert turn.provider == "voicechat"
    assert turn.is_synthetic is True
    assert turn.generator_id == "voicechat:nemotron-voicechat"
    assert pcm16_to_wav(turn.agent_audio_pcm)[:4] == b"RIFF"
    # Protocol: session.update first, then chunked base64 appends.
    assert session.sent[0]["type"] == "session.update"
    assert session.sent[0]["session"]["audio"]["input"]["format"] == {
        "type": "audio/pcm", "rate": PCM_RATE,
    }
    assert session.sent[0]["session"]["tools"] == []
    appends = [m for m in session.sent if m["type"] == "input_audio_buffer.append"]
    assert appends and all(
        base64.b64decode(a["audio"])[:CHUNK_BYTES].decode("latin-1", "ignore") != ""
        for a in appends
    )
    close = session.sent[-1]
    assert close["type"] == "session.close"
    assert session.closed is True


def test_converse_sends_instructions():
    client, session = make_client(happy_path_events())
    run(client.converse(*tone(0.5, rate=16000), instructions="Be terse."))
    assert session.sent[0]["session"]["instructions"] == "Be terse."


def test_converse_error_event_raises_server_error():
    events = happy_path_events()[:4] + [
        event("error", error={"code": "inference_error", "message": "boom"}),
    ]
    client, _ = make_client(events)
    with pytest.raises(VoiceChatServerError) as excinfo:
        run(client.converse(*tone(0.5, rate=16000)))
    assert excinfo.value.code == "inference_error"


def test_converse_inference_timeout_maps_to_timeout():
    events = happy_path_events()[:4] + [
        event("error", error={"code": "inference_timeout", "message": "too slow"}),
    ]
    client, _ = make_client(events)
    with pytest.raises(VoiceChatTimeoutError):
        run(client.converse(*tone(0.5, rate=16000)))


def test_converse_rejects_protocol_deviation():
    events = [event("session.created", session={"model": "m"}),
              event("response.output_audio.delta", delta="not-base64!!")]
    client, _ = make_client(events)
    with pytest.raises(VoiceChatProtocolError):
        run(client.converse(*tone(0.5, rate=16000)))


def test_converse_session_end_before_done_is_protocol_error():
    events = happy_path_events()[:1] + [event("session.created", session={}),
                                        event("session.end")]
    client, _ = make_client(events)
    with pytest.raises((VoiceChatProtocolError, VoiceChatTimeoutError)):
        run(client.converse(*tone(0.5, rate=16000)))


def test_converse_non_json_message_is_protocol_error():
    events = [event("session.created", session={}), "this is not json{"]
    client, _ = make_client(events)
    with pytest.raises(VoiceChatProtocolError):
        run(client.converse(*tone(0.5, rate=16000)))


def test_health_without_url_is_honest():
    client = VoiceChatClient(
        "ws://localhost:9000/v1/realtime",
        transport_factory=lambda url, timeout: _connect_later(ScriptedSession([])),
    )
    with pytest.raises(VoiceChatUnavailableError):
        run(client.check_health())


def test_connection_failure_is_typed():
    session = ScriptedSession([], fail_connect=True)
    client = VoiceChatClient(
        WS_URL,
        connect_timeout_s=1.0,
        transport_factory=lambda url, timeout: _connect_later(session),
    )
    with pytest.raises((VoiceChatConnectionError, VoiceChatTimeoutError)):
        run(client.converse(*tone(0.5, rate=16000)))
