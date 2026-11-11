"""Real MuRIL/DeBERTa heads and the real whisper-small checkpoint.

These run the shipped artifacts when they are present and skip with a clear
reason when they are not, so the suite stays honest on a host that has only the
slim dependencies.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.protection.asr_engines import MultiEngineTranscriber
from app.protection.text_ensemble import BRANCHES, HEADS_DIR, TextEnsemble
from app.services.hf_models import BACKEND_DIR, resolve_model

MURIL = HEADS_DIR / "muril_scam_head.joblib"
DEBERTA = HEADS_DIR / "deberta_scam_head.joblib"
WHISPER = BACKEND_DIR / "models" / "hf" / "whisper-small"
MURIL_WEIGHTS = resolve_model(BRANCHES["muril"]["model_id"], BRANCHES["muril"]["slug"])
DEBERTA_WEIGHTS = resolve_model(BRANCHES["deberta"]["model_id"], BRANCHES["deberta"]["slug"])


@pytest.mark.skipif(
    not (MURIL.is_file() and DEBERTA.is_file() and MURIL_WEIGHTS.local and DEBERTA_WEIGHTS.local),
    reason="trained heads and local encoder weights are required; download via the documented model setup first",
)
def test_trained_heads_separate_scam_from_ordinary_speech():
    ensemble = TextEnsemble()
    scam = ensemble.classify(
        "Papa, it's me. I had an accident, send 1 lakh rupees within 10 minutes, don't tell anyone."
    )
    benign = ensemble.classify("Hi Papa, I reached college safely and will call you after class.")

    assert scam["status"] == "completed"
    assert benign["status"] == "completed"
    completed = {branch["branch"] for branch in scam["branches"] if branch["status"] == "completed"}
    assert {"muril", "deberta"} <= completed
    assert scam["score"] > 0.8
    assert benign["score"] < 0.2
    assert scam["score"] - benign["score"] > 0.6


@pytest.mark.skipif(not WHISPER.is_dir(), reason="whisper-small is not downloaded locally")
def test_whisper_small_engine_is_discovered_and_described():
    transcriber = MultiEngineTranscriber()
    status = {entry["engine"]: entry for entry in transcriber.describe()["engines"]}
    assert status["whisper_small"]["available"] is True
    assert status["whisper_small"]["model_id"] == "openai/whisper-small"
    assert transcriber.describe()["available"] is True


@pytest.mark.skipif(not WHISPER.is_dir(), reason="whisper-small is not downloaded locally")
def test_whisper_small_transcribes_real_speech():
    from app.core.config import Settings
    from app.protection.asr_engines import WhisperSmallEngine
    from app.services.audio_validation import validate_and_decode

    fixture = Path(__file__).parent / "fixtures" / "audio" / "clean_speech.mp3"
    decoded = validate_and_decode(fixture.read_bytes(), fixture.name, Settings())
    engine = WhisperSmallEngine()
    result = engine.transcribe(decoded.samples, decoded.sample_rate)
    assert result.status == "completed", result.detail
    assert result.model == "openai/whisper-small"
    assert "college" in result.text.lower()
