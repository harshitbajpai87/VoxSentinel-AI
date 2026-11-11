"""Acoustic forensics: a torch-free authenticity screen for the serverless build.

Why this exists
---------------
The trained detector (models/detector.joblib) needs librosa and, in the container
build, the full scientific stack. Vercel's Python function cannot carry torch or
ffmpeg, so uploads there had no authenticity signal at all and the API honestly
reported "unavailable". This module fills that gap with signals that need only
numpy and are checkable by a human:

  * a hard bandwidth cutoff (also produced by telephony, denoising, codecs,
    resampling and other genuine processing)
  * digital silence (also produced by editing, noise gating and rendering)
  * pitch-contour smoothness (frame-to-frame F0 steps far smaller than laryngeal
    jitter produces)
  * loudness uniformity (flat, normalised level with no natural dynamics)

It is a HEURISTIC SCREEN, not a trained anti-spoof classifier, and it says so in
every result it returns. A learned detector always wins when one is installed;
this module is only the local fallback (VOXSENTINEL_FORENSICS_ENABLED=false
disables it).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from app.services.detector import BaseVoiceDetector, DetectorResult, ModelDescription

logger = logging.getLogger("voxsentinel.detector.forensics")

FORENSICS_VERSION = "0.1.0"
FRAME = 1024
HOP = 256
MIN_SPEECH_S = 1.0
# An MP3 re-encode of a 16 kHz source leaks a little energy above 8 kHz; a real
# wideband capture keeps content past 14 kHz. 9.2 kHz separates the two cleanly.
CUTOFF_HZ = 9200.0


@dataclass
class ForensicsEvidence:
    """Measured signals behind the score. Every field is directly checkable."""

    duration_s: float = 0.0
    sample_rate: int = 0
    bandwidth_cutoff_hz: float | None = None
    hf_energy_ratio: float | None = None
    zero_sample_fraction: float | None = None
    longest_zero_run_s: float | None = None
    silence_fraction: float | None = None
    head_silence_s: float | None = None
    f0_mean_hz: float | None = None
    f0_cv: float | None = None
    f0_median_step_hz: float | None = None
    level_std_db: float | None = None
    spectral_flatness: float | None = None
    spoken_s: float = 0.0
    signals: list[str] = field(default_factory=list)


def _mono(samples: np.ndarray) -> np.ndarray:
    y = np.asarray(samples, dtype=np.float64)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y


def _frames(y: np.ndarray) -> np.ndarray:
    if len(y) < FRAME:
        y = np.pad(y, (0, FRAME - len(y)))
    n = 1 + (len(y) - FRAME) // HOP
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(n)[:, None]
    return y[idx] * np.hamming(FRAME)[None, :]


def _power(frames: np.ndarray) -> np.ndarray:
    return np.abs(np.fft.rfft(frames, axis=1)) ** 2 / FRAME


def bandwidth_cutoff(y: np.ndarray, sr: int) -> tuple[float, float]:
    """Spectral edge in Hz, and the high-band ratio that really decides it.

    The ratio compares energy above 8.5 kHz with the 2-6 kHz band that every
    voice carries. Averaging the raw spectrum cannot do this job: it is
    dominated by the fundamental, so a cumulative-energy cutoff reads ~800 Hz
    even for wideband audio. Sibilance and room noise put real energy above
    8.5 kHz (ratio well above 0.01); a 16 kHz render or telephony path has
    almost none (ratio below 0.002).
    """
    y = _mono(y)
    spec = _power(_frames(y)).mean(axis=0)
    freqs = np.fft.rfftfreq(FRAME, 1 / sr)
    ref = spec[(freqs >= 2000) & (freqs < 6000)].sum()
    hf = spec[freqs >= 8500].sum()
    ratio = float(hf / ref) if ref > 0 else 0.0
    # Reportable edge: the last frequency still above the file's own noise
    # floor. Judging against the peak would just find the fundamental; judging
    # against the 14 kHz-and-up floor finds where real content really stops.
    db = 10 * np.log10(spec + 1e-20)
    floor_band = db[freqs >= 14000]
    floor = float(np.median(floor_band)) if len(floor_band) else float(db.min())
    above = np.where(db > floor + 8.0)[0]
    edge = float(freqs[above[-1]]) if len(above) else 0.0
    return edge, ratio


def silence_evidence(y: np.ndarray, sr: int) -> tuple[float, float, float, float]:
    """(zero fraction, longest exact-zero run s, silent-frame fraction, head silence s)."""
    exact = np.abs(y) < 1e-7
    zero_fraction = float(exact.mean())
    longest = 0
    run = 0
    for flag in exact:  # samples are ints here; loop cost is acceptable for <=60 s
        if flag:
            run += 1
            longest = max(longest, run)
        else:
            run = 0
    db = 20 * np.log10(np.abs(_frames(y)).max(axis=1) + 1e-12)
    loud = np.percentile(db, 95) if len(db) else 0.0
    silent = db < (loud - 35)
    silence_fraction = float(silent.mean())
    speech_idx = np.where(~silent)[0]
    head = float(speech_idx[0] * HOP / sr) if len(speech_idx) else 0.0
    return zero_fraction, longest / sr, silence_fraction, head


def pitch_evidence(y: np.ndarray, sr: int) -> tuple[float | None, float | None, float | None, int]:
    """Autocorrelation F0 track -> (mean Hz, coefficient of variation, median step, voiced frames).

    Implemented without librosa so the serverless build can run it. Frames are
    25 ms with a 10 ms hop; a frame is voiced when its normalised autocorrelation
    peak clears 0.30 inside 70-350 Hz.
    """
    frame = int(0.025 * sr)
    hop = int(0.010 * sr)
    if len(y) < frame * 4:
        return None, None, None, 0
    lo, hi = int(sr / 350), int(sr / 70)
    f0s: list[float] = []
    for start in range(0, len(y) - frame, hop):
        seg = y[start : start + frame]
        seg = seg - seg.mean()
        energy = float(np.sqrt((seg**2).mean()))
        if energy < 0.01:  # skip silence without consulting the whole file
            continue
        seg = seg * np.hanning(frame)
        corr = np.correlate(seg, seg, mode="full")[frame - 1 :]
        if corr[0] <= 0:
            continue
        corr = corr / corr[0]
        window = corr[lo:hi]
        if len(window) == 0:
            continue
        best = float(window.max())
        if best < 0.30:
            continue
        # Take the SMALLEST lag that is nearly as periodic as the best one.
        # The global maximum often lands on a multiple of the true period
        # (octave error), which would fake pitch instability in a steady voice.
        near = np.where(window >= 0.9 * best)[0]
        lag = lo + int(near[0])
        if 1 <= lag < len(corr) - 1:  # sub-sample refine for smoother steps
            a, b, c = corr[lag - 1], corr[lag], corr[lag + 1]
            denom = a - 2 * b + c
            offset = 0.5 * (a - c) / denom if denom != 0 else 0.0
            lag = lag + float(np.clip(offset, -0.5, 0.5))
        f0s.append(sr / float(lag))
    if len(f0s) < 25:
        return None, None, None, len(f0s)
    arr = np.asarray(f0s)
    mean = float(arr.mean())
    cv = float(arr.std() / mean) if mean > 0 else None
    step = float(np.median(np.abs(np.diff(arr)))) if len(arr) > 1 else None
    return mean, cv, step, len(arr)


def _level_and_flatness(y: np.ndarray) -> tuple[float, float]:
    frames = _frames(y)
    db = 20 * np.log10(np.abs(frames).max(axis=1) + 1e-12)
    loud = np.percentile(db, 95)
    speech = db[db > loud - 35]
    level_std = float(speech.std()) if len(speech) > 5 else None
    spec = _power(frames)
    spec = spec[spec.sum(axis=1) > 0]
    if len(spec) == 0:
        return 0.0, 0.0
    flat = np.exp(np.mean(np.log(spec + 1e-20), axis=1)) / (np.mean(spec, axis=1) + 1e-20)
    return level_std, float(np.mean(flat))


def measure(y: np.ndarray, sr: int) -> ForensicsEvidence:
    """Run every measurement once and return them with human-readable signals."""
    y = _mono(y)
    ev = ForensicsEvidence(duration_s=len(y) / sr if sr else 0.0, sample_rate=sr)
    if len(y) == 0 or sr <= 0:
        return ev
    ev.bandwidth_cutoff_hz, ev.hf_energy_ratio = bandwidth_cutoff(y, sr)
    zero_frac, zero_run, sil_frac, head = silence_evidence(y, sr)
    ev.zero_sample_fraction = zero_frac
    ev.longest_zero_run_s = zero_run
    ev.silence_fraction = sil_frac
    ev.head_silence_s = head
    ev.f0_mean_hz, ev.f0_cv, ev.f0_median_step_hz, voiced = pitch_evidence(y, sr)
    ev.level_std_db, ev.spectral_flatness = _level_and_flatness(y)
    ev.spoken_s = max(0.0, ev.duration_s * (1 - (ev.silence_fraction or 0.0)))

    signals: list[str] = []
    if (
        ev.bandwidth_cutoff_hz is not None
        and sr >= 32000
        and ev.bandwidth_cutoff_hz < CUTOFF_HZ
        and (ev.hf_energy_ratio or 0) < 0.01
    ):
        signals.append(
            f"Bandwidth stops at {ev.bandwidth_cutoff_hz / 1000:.1f} kHz in a "
            f"{sr / 1000:.1f} kHz file; this is consistent with a bandwidth-limited "
            "or processed source, including telephony or resampling, and does not establish synthesis."
        )
    if (ev.longest_zero_run_s or 0) >= 0.01:
        signals.append(
            f"Contains {ev.longest_zero_run_s * 1000:.0f} ms of exact digital silence "
            f"({(ev.zero_sample_fraction or 0) * 100:.2f}% of samples): "
            "this can result from editing, a noise gate, or rendering and is not specific to synthetic speech."
        )
    if ev.f0_cv is not None and ev.f0_cv < 0.11:
        signals.append(
            f"Pitch variation is only {ev.f0_cv:.3f} (coefficient of variation), "
            "flatter than conversational speech."
        )
    if ev.f0_median_step_hz is not None and ev.f0_median_step_hz < 2.5:
        signals.append(
            f"Pitch moves {ev.f0_median_step_hz:.1f} Hz per 10 ms frame on median, "
            "smoother than human laryngeal jitter."
        )
    if ev.level_std_db is not None and ev.level_std_db < 3.0:
        signals.append(
            f"Loudness is normalised to {ev.level_std_db:.1f} dB spread across the clip; "
            "natural speech varies more."
        )
    if ev.head_silence_s is not None and ev.head_silence_s < 0.05 and ev.duration_s > 3:
        signals.append("Audio starts at full level with no lead-in room tone.")
    ev.signals = signals
    return ev


def score(ev: ForensicsEvidence) -> tuple[float, list[str]]:
    """Weighted heuristic score in [0.05, 0.95] plus the reasons that moved it.

    Weights are hand-set from the physical meaning of each signal, not fitted on
    labelled spoof data. That is why the result is labelled a screen.
    """
    total = 0.05
    reasons: list[str] = []
    sr = ev.sample_rate or 0
    band_limited = (
        ev.bandwidth_cutoff_hz is not None
        and sr >= 32000
        and ev.bandwidth_cutoff_hz < CUTOFF_HZ
        and (ev.hf_energy_ratio or 0) < 0.01
    )
    hard_gated = (ev.longest_zero_run_s or 0) >= 0.01
    if band_limited:
        total += 0.25
        reasons.append("band-limited spectrum (also consistent with telephony or resampling)")
    if hard_gated:
        total += 0.20
        reasons.append("digital silence between phrases")
    if ev.f0_cv is not None and ev.f0_cv < 0.11:
        total += 0.20
        reasons.append("flat pitch contour")
    if ev.f0_median_step_hz is not None and ev.f0_median_step_hz < 1.0:
        total += 0.15
        reasons.append("smoothed pitch transitions")
    elif ev.f0_median_step_hz is not None and ev.f0_median_step_hz < 2.5:
        total += 0.05
        reasons.append("slightly smooth pitch transitions")
    if ev.level_std_db is not None and ev.level_std_db < 3.0:
        total += 0.10
        reasons.append("uniform loudness")
    if ev.head_silence_s is not None and ev.head_silence_s < 0.05 and ev.duration_s > 3:
        total += 0.05
        reasons.append("abrupt start with little lead-in room tone")
    # Honest counterweight: clearly expressive dynamics suggests a human talker.
    if ev.f0_cv is not None and ev.f0_cv > 0.18:
        total -= 0.15
        reasons.append("expressive pitch range (human-like)")
    if ev.level_std_db is not None and ev.level_std_db > 6.0:
        total -= 0.10
        reasons.append("wide dynamic range (human-like)")
    # Band limitation plus exact zeros is a stronger processing cue than either
    # alone, but neither establishes whether the source is synthetic.
    if band_limited and (ev.longest_zero_run_s or 0) >= 0.02 and total < 0.65:
        total = 0.65
        reasons.append("band limitation and exact silence together (non-exclusive processing cues)")
    return float(min(0.95, max(0.05, total))), reasons


class AcousticForensicsDetector(BaseVoiceDetector):
    """Torch-free detector used when the trained model cannot be loaded."""

    kind = "detection"

    def __init__(self, threshold: float = 0.50) -> None:
        # The hand-set weights put a clearly rendered clip near 0.7-0.9 and a
        # natural capture near 0.15-0.35, so this screen's elevated band sits at
        # 0.50 rather than the trained model's 0.35.
        self.elevated_threshold = threshold
        # Scores within this distance of the threshold are reported as
        # undetermined rather than forced into a yes/no answer.
        self.ambiguity_band = 0.15

    def is_ready(self) -> bool:
        return True

    @property
    def calibrated(self) -> bool:
        return False

    def analyze(self, samples: np.ndarray, sample_rate: int, channels: int) -> DetectorResult:
        ev = measure(samples, sample_rate)
        raw, reasons = score(ev)
        evidence = "; ".join(ev.signals) if ev.signals else "No spoofing-oriented artefact measured."
        return DetectorResult(
            raw_score=raw,
            raw_score_definition=(
                "Weighted heuristic from bandwidth, digital silence, pitch and loudness "
                "measurements. NOT a trained classifier output and not a calibrated "
                "probability."
            ),
            calibrated_probability=None,
            uncertainty=None,
            uncertainty_method=None,
            model_name="VoxSentinel Acoustic Forensics (heuristic)",
            model_version=FORENSICS_VERSION,
            model_mode="baseline",
            explanation=f"{evidence} Signals: {', '.join(reasons) or 'none'}.",
            window_scores=(),
        )

    def describe(self) -> ModelDescription:
        return ModelDescription(
            id="voice_detector",
            name="Acoustic forensics screen (torch-free)",
            version=FORENSICS_VERSION,
            mode="baseline",
            kind="detection",
            description=(
                "Measures bandwidth cutoff, digital silence, pitch smoothness and loudness "
                "uniformity, then scores them with hand-set weights. Useful as a screen and "
                "explainable to a user, but weaker than the trained detector that runs in "
                "the full container build."
            ),
            loaded=True,
        )
