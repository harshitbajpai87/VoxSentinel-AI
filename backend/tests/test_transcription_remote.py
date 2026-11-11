"""Cloud transcriber contract: honest statuses, no invented speech."""

from __future__ import annotations

import io
import wave

import numpy as np
import pytest

from app.protection.transcription_remote import GeminiTranscriber, _to_wav_bytes


def _speech_like(seconds: float = 1.0, sample_rate: int = 44100) -> np.ndarray:
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    return (0.3 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)


def _wav_frame_count(payload: bytes) -> tuple[int, int]:
    with wave.open(io.BytesIO(payload), "rb") as handle:
        return handle.getframerate(), handle.getnframes()


def test_wav_bytes_are_16k_mono_pcm():
    rate, frames = _wav_frame_count(_to_wav_bytes(_speech_like(1.0, 44100), 44100))
    assert rate == 16000
    assert frames == pytest.approx(16000, abs=2)


def test_wav_bytes_survive_non_finite_samples():
    audio = _speech_like(0.2, 16000)
    audio[0] = np.nan
    audio[1] = np.inf
    payload = _to_wav_bytes(audio, 16000)
    assert len(payload) > 44  # header plus data, no exception raised


def test_unavailable_without_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    transcriber = GeminiTranscriber(api_key="")
    assert transcriber.is_available() is False
    result = transcriber.transcribe(_speech_like(), 44100)
    assert result.status == "unavailable"
    assert result.text == ""
    assert transcriber.describe()["service"] == "gemini-audio"


def test_parses_provider_answer(monkeypatch):
    transcriber = GeminiTranscriber(api_key="test-key")
    answer = (
        '{"language": "en", "text": "Send 50,000 rupees now.", '
        '"segments": [{"start": 0, "end": 1.2, "text": "Send 50,000 rupees now."}]}'
    )

    class _Response:
        status_code = 200

        @staticmethod
        def json():
            return {"candidates": [{"content": {"parts": [{"text": answer}]}}]}

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        @staticmethod
        def post(*args, **kwargs):
            return _Response()

    import app.protection.transcription_remote as module

    monkeypatch.setattr(module, "httpx", type("m", (), {"Client": _Client}), raising=False)

    result = transcriber.transcribe(_speech_like(), 44100)
    assert result.status == "completed"
    assert result.text == "Send 50,000 rupees now."
    assert result.language == "en"
    assert result.segments[0].end == 1.2


def test_provider_error_is_a_failure_not_a_transcript(monkeypatch):
    transcriber = GeminiTranscriber(api_key="test-key")

    class _Response:
        status_code = 429

        @staticmethod
        def json():
            return {}

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        @staticmethod
        def post(*args, **kwargs):
            return _Response()

    import app.protection.transcription_remote as module

    monkeypatch.setattr(module, "httpx", type("m", (), {"Client": _Client}), raising=False)

    result = transcriber.transcribe(_speech_like(), 44100)
    assert result.status == "failed"
    assert result.text == ""
    assert "429" in (result.error or "")


def test_unreadable_answer_never_becomes_text(monkeypatch):
    transcriber = GeminiTranscriber(api_key="test-key")

    class _Response:
        status_code = 200

        @staticmethod
        def json():
            return {"candidates": [{"content": {"parts": [{"text": "I cannot do that."}]}}]}

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        @staticmethod
        def post(*args, **kwargs):
            return _Response()

    import app.protection.transcription_remote as module

    monkeypatch.setattr(module, "httpx", type("m", (), {"Client": _Client}), raising=False)

    result = transcriber.transcribe(_speech_like(), 44100)
    assert result.status == "failed"
    assert result.text == ""
