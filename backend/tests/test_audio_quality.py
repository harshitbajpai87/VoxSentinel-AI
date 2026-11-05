"""Quality confidence must stay separate from synthetic-speech confidence.

Fixtures are short real clips (TTS-generated, checked in as test assets) so the
quality assessment is exercised on encoded audio rather than idealised tones.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.protection.quality import assess_quality, qualify_voice

SR = 16000
FIX = Path(__file__).parent / "fixtures" / "audio"


def _load(name: str) -> tuple[np.ndarray, int]:
    path = FIX / name
    if not path.exists():
        pytest.skip(f"audio fixture missing: {name}")
    return sf.read(path, dtype="float32", always_2d=False)


def _burst(seconds: float = 3.0, amp: float = 0.3, mod: float = 0.8) -> np.ndarray:
    """Speech-like: bursts separated by silence, amplitude modulated."""
    t = np.arange(int(SR * seconds)) / SR
    carrier = sum(
        np.sin(2 * np.pi * f * t) / k
        for k, f in enumerate([150, 420, 900, 1800, 3200, 4800], 1)
    )
    carrier /= np.max(np.abs(carrier))
    envelope = np.where((t % 1.2) < 0.85, mod * (0.4 + 0.6 * np.sin(2 * np.pi * 3 * t)), 0.0)
    return (amp * envelope * carrier).astype(np.float32)


def test_clean_recording_is_good():
    samples, sr = _load("clean_speech.mp3")
    q = assess_quality(samples, sr, 1, transcript_words=12)
    assert q.verdict == "good"
    assert q.clipping is False


def test_narrowband_recording_is_degraded():
    samples, sr = _load("narrowband_noisy.mp3")
    q = assess_quality(samples, sr, 1, transcript_words=10)
    assert q.verdict == "degraded"
    assert q.reasons, "a degraded verdict must explain itself"


def test_clipped_recording_is_degraded():
    samples, sr = _load("clipped.mp3")
    q = assess_quality(samples, sr, 1, transcript_words=6)
    assert q.clipping is True
    assert q.verdict == "degraded"


def test_silence_is_insufficient():
    q = assess_quality(np.zeros(SR * 2, dtype=np.float32), SR, 1, transcript_words=0)
    assert q.verdict == "insufficient"
    assert q.insufficient_speech is True


def test_very_short_clip_is_insufficient():
    q = assess_quality(_burst(seconds=0.4), SR, 1, transcript_words=2)
    assert q.too_short is True
    assert q.verdict == "insufficient"


def test_insufficient_audio_yields_inconclusive_voice_verdict():
    voice = {"status": "completed", "score": 0.997, "suspected_synthetic": True}
    q = assess_quality(np.zeros(SR, dtype=np.float32), SR, 1, transcript_words=0)
    out = qualify_voice(voice, q)
    assert out["verdict"] == "INCONCLUSIVE"
    assert out["score_raw"] == 0.997  # raw detector output preserved for audit
    assert out["suspected_synthetic"] is None
    assert out["score_usable"] is False


def test_degraded_audio_qualifies_but_preserves_score():
    samples, sr = _load("narrowband_noisy.mp3")
    q = assess_quality(samples, sr, 1, transcript_words=10)
    out = qualify_voice({"status": "completed", "score": 0.93, "suspected_synthetic": True}, q)
    assert out["verdict"] == "QUALIFIED"
    assert out["score"] == 0.93
    assert out["score_raw"] == 0.93


def test_clean_audio_keeps_detector_semantics():
    samples, sr = _load("clean_speech.mp3")
    q = assess_quality(samples, sr, 1, transcript_words=12)
    out = qualify_voice({"status": "completed", "score": 0.91, "suspected_synthetic": True}, q)
    assert out["verdict"] == "SUSPECTED_SYNTHETIC"
    assert out["suspected_synthetic"] is True


def test_detector_threshold_is_never_modified():
    samples, sr = _load("clean_speech.mp3")
    q = assess_quality(samples, sr, 1, transcript_words=12)
    out = qualify_voice(
        {"status": "completed", "score": 0.5, "suspected_synthetic": False, "threshold": 0.65}, q
    )
    assert out["threshold"] == 0.65
    assert out["score"] == 0.5


def test_missing_score_is_reported_as_unavailable():
    out = qualify_voice({"status": "unavailable", "score": None}, assess_quality(_burst(), SR, 1, 12))
    assert out["verdict"] == "UNAVAILABLE"
    assert out["score_raw"] is None
