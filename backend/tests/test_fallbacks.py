"""Torch-free fallbacks: the acoustic forensics screen and MFCC speaker comparison.

These run on hosts where torch/librosa cannot be installed (the Vercel build).
They must produce a real, explainable measurement and never pretend to be the
trained detector or a calibrated speaker verdict.

The signals below are noise-shaped on purpose: they let each test assert on ONE
physical property (bandwidth, digital silence, dynamics) instead of hoping a
synthetic "voice" happens to resemble the property under test.
"""

from __future__ import annotations

import pathlib
import tempfile

import numpy as np

from app.core.config import Settings
from app.main import create_app
from app.services.acoustic_forensics import (
    AcousticForensicsDetector,
    ForensicsEvidence,
    bandwidth_cutoff,
    measure,
    score,
    silence_evidence,
)
from app.services.speaker_mfcc import MfccStatSpeakerVerifier

SR = 44100


def _noise(seconds: float = 4.0, seed: int = 1, limit_hz: float | None = None) -> np.ndarray:
    """White noise, optionally brick-walled at `limit_hz` (a 16 kHz render's edge)."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    sig = rng.standard_normal(n)
    if limit_hz is not None:
        spec = np.fft.rfft(sig)
        freqs = np.fft.rfftfreq(n, 1 / SR)
        spec[freqs > limit_hz] = 0.0
        sig = np.fft.irfft(spec, n=n)
    return (sig / np.abs(sig).max() * 0.6).astype(np.float32)


def _gate(sig: np.ndarray, gap_s: float = 0.05, at_s: float = 0.5) -> np.ndarray:
    """Insert an exact-zero run, the way a renderer emits phrase gaps."""
    out = sig.copy()
    start = int(at_s * SR)
    out[start : start + int(gap_s * SR)] = 0.0
    return out


# --- physical measurements ---------------------------------------------------
def test_band_limited_audio_has_no_high_band_ratio():
    limited_edge, limited_ratio = bandwidth_cutoff(_noise(limit_hz=7900), SR)
    wide_edge, wide_ratio = bandwidth_cutoff(_noise(), SR)
    assert limited_ratio < 0.01, "a 7.9 kHz brick wall must leave ~no energy above 8.5 kHz"
    assert wide_ratio > 0.02, "wideband noise must carry energy above 8.5 kHz"
    # The reported edge is only meaningful on speech (sibilance above a quiet
    # floor); for flat noise it is informational, so assert it stays plausible.
    assert 0 <= limited_edge < SR / 2


def test_exact_zero_runs_are_detected_and_distinguished_from_room_noise():
    gated = silence_evidence(_gate(_noise()), SR)
    clean = silence_evidence(_noise(), SR)
    assert gated[1] >= 0.04, "a 50 ms hard gap must be reported"
    assert clean[1] == 0.0, "continuous audio with room tone has no exact zeros"
    assert gated[0] > 0, clean[0] == 0


# --- scoring weights ---------------------------------------------------------
def _evidence(**kwargs) -> ForensicsEvidence:
    base = dict(
        duration_s=10.0,
        sample_rate=44100,
        bandwidth_cutoff_hz=16000.0,
        hf_energy_ratio=0.2,
        zero_sample_fraction=0.0,
        longest_zero_run_s=0.0,
        silence_fraction=0.15,
        head_silence_s=0.3,
        f0_mean_hz=150.0,
        f0_cv=0.2,
        f0_median_step_hz=6.0,
        level_std_db=8.0,
        spectral_flatness=0.02,
    )
    base.update(kwargs)
    return ForensicsEvidence(**base)


def test_rendered_artefacts_raise_the_score():
    rendered = _evidence(
        bandwidth_cutoff_hz=8100.0,
        hf_energy_ratio=0.0004,
        longest_zero_run_s=0.12,
        zero_sample_fraction=0.02,
        f0_cv=0.08,
        f0_median_step_hz=0.4,
        level_std_db=2.0,
        head_silence_s=0.0,
    )
    natural = _evidence()
    rendered_score, rendered_reasons = score(rendered)
    natural_score, _ = score(natural)
    assert rendered_score >= 0.75
    assert natural_score <= 0.35
    assert any("band" in r for r in rendered_reasons)
    assert any("silence" in r for r in rendered_reasons)


def test_a_band_limited_file_on_a_low_sample_rate_is_not_penalised():
    """16 kHz captures are band-limited by definition, so the ratio must be ignored."""
    ev = _evidence(sample_rate=16000, bandwidth_cutoff_hz=7900.0, hf_energy_ratio=0.0)
    value, reasons = score(ev)
    assert not any("band" in r for r in reasons)
    assert value <= 0.5


def test_band_limiting_plus_digital_silence_cannot_be_cancelled_by_dynamics():
    """Two independent renderer fingerprints outweigh a wide loudness range.

    This is the real case that motivated the rule: a rendered clip whose loudness
    still varies enough to look human. Structure evidence must win.
    """
    ev = _evidence(
        bandwidth_cutoff_hz=8800.0,
        hf_energy_ratio=0.0015,
        longest_zero_run_s=0.16,
        zero_sample_fraction=0.02,
        level_std_db=8.0,   # would otherwise subtract
        f0_cv=0.13,
    )
    value, reasons = score(ev)
    assert value >= 0.65
    assert any("together" in r for r in reasons)


def test_expressive_dynamics_pull_the_score_down():
    tight = score(_evidence(f0_cv=0.05, level_std_db=1.5, f0_median_step_hz=0.2))[0]
    loose = score(_evidence(f0_cv=0.25, level_std_db=9.0, f0_median_step_hz=8.0))[0]
    assert loose < tight


# --- detector wrapper --------------------------------------------------------
def test_detector_reports_a_heuristic_without_inventing_a_probability():
    detector = AcousticForensicsDetector()
    result = detector.analyze(_gate(_noise(limit_hz=7900)), SR, 1)
    assert detector.is_ready() is True
    assert result.calibrated_probability is None
    assert "not a trained classifier" in result.raw_score_definition.lower()
    assert result.explanation and "silence" in result.explanation.lower()
    assert detector.describe().mode == "baseline"
    assert detector.elevated_threshold == 0.50


def test_measure_populates_every_field_it_can():
    ev = measure(_noise(), SR)
    assert ev.bandwidth_cutoff_hz is not None
    assert ev.hf_energy_ratio is not None
    assert ev.level_std_db is not None
    assert ev.duration_s > 3.5
    assert ev.signals == [] or isinstance(ev.signals, list)


# --- speaker fallback --------------------------------------------------------
def test_speaker_fallback_compares_without_claiming_a_verdict():
    verifier = MfccStatSpeakerVerifier()
    voice_a = _noise(seed=3)
    voice_b = _noise(seed=11)
    same = verifier.compare(voice_a, SR, voice_a, SR)
    other = verifier.compare(voice_a, SR, voice_b, SR)
    assert same.status == "completed"
    assert same.similarity is not None and same.similarity > 0.99
    assert other.similarity is not None and other.similarity < same.similarity
    # No calibrated threshold exists for this method, so no decision is emitted.
    assert same.outcome is None and same.threshold is None
    assert any("not" in limit.lower() for limit in same.limitations)


def test_speaker_fallback_declines_on_silence():
    result = MfccStatSpeakerVerifier().compare(np.zeros(SR // 4), SR, _noise(), SR)
    assert result.status == "unavailable"
    assert result.similarity is None


# --- app wiring --------------------------------------------------------------
def test_missing_detector_stays_unavailable_with_optional_forensics(tmp_path):
    settings = Settings(
        db_url=f"sqlite:///{tmp_path}/f.db",
        detector_model_path=str(tmp_path / "missing.joblib"),
        speaker_model_path=str(tmp_path / "missing.joblib"),
        speaker_model_dir=str(tmp_path / "missing-dir"),
        forensics_enabled=True,
        mfcc_speaker_enabled=True,
    )
    app = create_app(settings=settings)
    detector = app.state.detector_service
    assert detector.is_ready() is False
    assert detector.describe().loaded is False
    assert isinstance(app.state.protection_service._forensics, AcousticForensicsDetector)
    assert app.state.speaker_service.is_ready() is True


def test_fallbacks_can_be_switched_off(tmp_path):
    settings = Settings(
        db_url=f"sqlite:///{tmp_path}/g.db",
        detector_model_path=str(tmp_path / "missing.joblib"),
        speaker_model_path=str(tmp_path / "missing.joblib"),
        speaker_model_dir=str(tmp_path / "missing-dir"),
        forensics_enabled=False,
        mfcc_speaker_enabled=False,
    )
    app = create_app(settings=settings)
    assert app.state.detector_service.is_ready() is False
    assert app.state.protection_service._forensics is None
    assert app.state.speaker_service.is_ready() is False


def test_temp_writer_falls_back_when_configured_dir_is_read_only(tmp_path, monkeypatch):
    """A read-only bundle filesystem must not turn an upload into a 500.

    Serverless hosts ship the code on a read-only mount; the configured temp
    directory then cannot be created. The decoder has to fall back to the system
    temp dir instead of failing the request.
    """
    from app.core.config import Settings
    from app.services import ffmpeg_decoder

    settings = Settings(temp_upload_dir=str(tmp_path / "ro" / "nested"))

    def deny(self, *args, **kwargs):
        raise OSError(30, "Read-only file system")

    monkeypatch.setattr(pathlib.Path, "mkdir", deny)
    written = ffmpeg_decoder._write_temp(b"RIFF0000", settings)
    try:
        assert pathlib.Path(written).exists()
        assert pathlib.Path(written).parent == pathlib.Path(tempfile.gettempdir())
    finally:
        pathlib.Path(written).unlink(missing_ok=True)
