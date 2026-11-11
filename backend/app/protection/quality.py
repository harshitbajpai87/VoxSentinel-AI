"""Audio quality assessment for the protection pipeline.

Purpose: keep *audio quality confidence* separate from *synthetic speech
confidence*. A degraded recording can drive a deepfake detector to an extreme
score (compression, clipping and noise all look "not human" to these models), so
the raw score is preserved and explicitly qualified instead of silently trusted.

This module never changes a detector threshold and never edits detector output.
It only describes the input signal.

Method notes (deliberately simple and explainable):
- Clipping: share of samples pinned at full scale.
- Level: overall RMS and peak in dBFS.
- Noise: minimum-statistics SNR (per-bin 10th percentile across frames is the
  noise estimate; speech energy is total minus noise). This separates broadband
  hiss from silence-padded, cleanly encoded speech, which a naive frame-energy
  floor cannot do.
- Band limiting: share of spectral energy above 3.5 kHz.
- Speech presence: frames above 20% of the 95th-percentile level, plus the
  transcript word count when available.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CLIP_LEVEL = 0.995
CLIP_RATIO_DEGRADED = 0.005
QUIET_DBFS = -30.0
LOUD_DBFS = -1.0
SNR_DEGRADED_DB = 8.0
BAND_LIMIT_HF_RATIO = 0.002
MIN_SPEECH_SECONDS = 1.2
MIN_DURATION_SECONDS = 1.0
SILENCE_AMPLITUDE = 1e-3


@dataclass
class QualityReport:
    duration_s: float
    speech_s: float
    speech_ratio: float
    rms_dbfs: float
    peak_dbfs: float
    clip_ratio: float
    snr_db: float
    noise_floor_dbfs: float = -120.0
    hf_ratio: float = 0.0
    clipping: bool = False
    noisy: bool = False
    low_volume: bool = False
    band_limited: bool = False
    too_short: bool = False
    insufficient_speech: bool = False
    verdict: str = "good"  # good | degraded | insufficient
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "duration_s": round(self.duration_s, 2),
            "speech_s": round(self.speech_s, 2),
            "speech_ratio": round(self.speech_ratio, 3),
            "rms_dbfs": round(self.rms_dbfs, 1),
            "peak_dbfs": round(self.peak_dbfs, 1),
            "clip_ratio": round(self.clip_ratio, 5),
            "snr_db": round(self.snr_db, 1),
            "noise_floor_dbfs": round(self.noise_floor_dbfs, 1),
            "hf_ratio": round(self.hf_ratio, 4),
            "clipping": self.clipping,
            "noisy": self.noisy,
            "low_volume": self.low_volume,
            "band_limited": self.band_limited,
            "too_short": self.too_short,
            "insufficient_speech": self.insufficient_speech,
            "verdict": self.verdict,
            "reasons": self.reasons,
        }


def _dbfs(value: float) -> float:
    return float(20.0 * np.log10(max(value, 1e-8)))


def _frames(x: np.ndarray, sample_rate: int, frame_s: float = 0.02) -> np.ndarray:
    frame = max(1, int(frame_s * sample_rate))
    usable = len(x) - (len(x) % frame)
    if usable < frame:
        return x.reshape(1, -1)
    return x[:usable].reshape(-1, frame)


def _stft_magnitudes(x: np.ndarray, sample_rate: int, n: int = 512) -> np.ndarray:
    if len(x) < n:
        n = max(64, 2 ** int(np.floor(np.log2(max(len(x), 64)))))
    hop = max(1, n // 2)
    window = np.hanning(n).astype(np.float32)
    cols = []
    for start in range(0, len(x) - n + 1, hop):
        cols.append(np.abs(np.fft.rfft(x[start : start + n] * window)))
    if not cols:
        return np.zeros((1, n // 2 + 1), dtype=np.float32)
    return np.array(cols, dtype=np.float32)


def _hf_ratio(x: np.ndarray, sample_rate: int) -> float:
    """Share of spectral energy above 3.5 kHz (narrowband/compressed detection)."""
    mags = _stft_magnitudes(x, sample_rate)
    power = mags**2
    if power.size == 0:
        return 0.0
    bins = power.shape[1]
    freqs = np.fft.rfftfreq((bins - 1) * 2, 1.0 / sample_rate)
    mask = freqs >= 3500.0
    if not np.any(mask):
        return 0.0
    return float(np.sum(power[:, mask]) / max(np.sum(power), 1e-12))


def _snr_from_frames(frame_rms: np.ndarray) -> tuple[float, float]:
    """Signal-to-noise estimate from frame levels.

    The quietest frames that still contain audio approximate the noise floor;
    the loudest approximate speech. Frames of exact digital silence are ignored
    so a codec that mutes gaps is not mistaken for a clean recording.
    """
    audible = frame_rms[frame_rms > 1e-3]
    if audible.size < 4:
        return 60.0, -120.0
    noise = float(np.percentile(audible, 10))
    speech = float(np.percentile(frame_rms, 90))
    snr = float(np.clip(20.0 * np.log10(max(speech, 1e-8) / max(noise, 1e-8)) - 3.0, -20.0, 60.0))
    return snr, _dbfs(noise)


def assess_quality(
    samples: np.ndarray,
    sample_rate: int,
    channels: int = 1,
    transcript_words: int | None = None,
) -> QualityReport:
    mono = np.asarray(samples, dtype=np.float32)
    if mono.ndim > 1:
        mono = mono.mean(axis=1)
    mono = np.nan_to_num(mono.flatten(), nan=0.0, posinf=0.0, neginf=0.0)
    if mono.size == 0 or sample_rate <= 0:
        return QualityReport(
            0, 0, 0, -120, -120, 0, 0, too_short=True, insufficient_speech=True,
            verdict="insufficient", reasons=["The recording contains no audio samples."],
        )

    duration = mono.size / float(sample_rate)
    peak = float(np.max(np.abs(mono)))
    rms = float(np.sqrt(np.mean(mono**2) + 1e-12))
    clip_ratio = float(np.mean(np.abs(mono) >= CLIP_LEVEL))

    frame_rms = np.sqrt(np.mean(_frames(mono, sample_rate) ** 2, axis=1) + 1e-12)
    p95 = float(np.percentile(frame_rms, 95))
    speech_threshold = max(0.2 * p95, SILENCE_AMPLITUDE)
    speech_frames = int(np.count_nonzero(frame_rms >= speech_threshold))
    speech_s = speech_frames * (len(mono) / max(len(frame_rms), 1)) / sample_rate
    speech_ratio = speech_s / duration if duration else 0.0

    snr_db, noise_floor_dbfs = _snr_from_frames(frame_rms)
    hf_ratio = _hf_ratio(mono, sample_rate)
    report = QualityReport(
        duration_s=duration,
        speech_s=speech_s,
        speech_ratio=speech_ratio,
        rms_dbfs=_dbfs(rms),
        peak_dbfs=_dbfs(peak),
        clip_ratio=clip_ratio,
        snr_db=snr_db,
        noise_floor_dbfs=noise_floor_dbfs,
        hf_ratio=hf_ratio,
    )

    if clip_ratio > CLIP_RATIO_DEGRADED or report.peak_dbfs >= LOUD_DBFS:
        report.clipping = True
        report.reasons.append(
            "The signal is clipped, which destroys the detail a detector needs."
        )
    if snr_db < SNR_DEGRADED_DB:
        report.noisy = True
        report.reasons.append(
            f"Background noise dominates the recording (estimated SNR {snr_db:.1f} dB)."
        )
    if report.rms_dbfs < QUIET_DBFS:
        report.low_volume = True
        report.reasons.append(f"Recording level is very low ({report.rms_dbfs:.1f} dBFS).")
    if hf_ratio < BAND_LIMIT_HF_RATIO:
        report.band_limited = True
        report.reasons.append(
            "The audio is band-limited (very little energy above 3.5 kHz), typical of "
            "heavy compression or a narrowband codec, which confuses deepfake detectors."
        )
    if duration < MIN_DURATION_SECONDS:
        report.too_short = True
        report.reasons.append(f"Recording is only {duration:.2f}s long.")
    words = transcript_words or 0
    if speech_s < MIN_SPEECH_SECONDS or words < 3:
        report.insufficient_speech = True
        report.reasons.append(
            "Not enough speech was detected to make a reliable audio-authenticity judgement."
        )

    if report.insufficient_speech or report.too_short:
        report.verdict = "insufficient"
    elif report.clipping or report.noisy or report.low_volume or report.band_limited:
        report.verdict = "degraded"
    return report


def qualify_voice(voice: dict, quality: QualityReport) -> dict:
    """Attach quality context to a voice block without altering raw detector output."""
    block = dict(voice)
    raw_score = voice.get("score")
    block["quality"] = quality.as_dict()
    block["score_raw"] = raw_score
    block["score_usable"] = quality.verdict == "good"
    if raw_score is None:
        block["verdict"] = "UNAVAILABLE"
        block["verdict_detail"] = "No detector score was produced for this recording."
        return block
    if quality.verdict == "insufficient":
        block["verdict"] = "INCONCLUSIVE"
        block["verdict_detail"] = (
            "Audio quality is insufficient for a synthetic-speech judgement; "
            "the raw detector score is kept for audit only."
        )
        block["suspected_synthetic"] = None
        return block
    if quality.verdict == "degraded":
        block["verdict"] = "QUALIFIED"
        block["verdict_detail"] = (
            "Detector ran on degraded audio ("
            + "; ".join(r.rstrip(".") for r in quality.reasons)
            + "); treat this score as indicative, not conclusive."
        )
        return block
    if voice.get("ambiguous"):
        block["verdict"] = "INCONCLUSIVE"
        block["verdict_detail"] = (
            "Score sits inside the screen's uncertainty band: the clip has the acoustics of a "
            "real recording and of a neural voice at the same time. No verdict is claimed."
        )
        return block
    if voice.get("suspected_synthetic"):
        block["verdict"] = "SUSPECTED_SYNTHETIC"
        block["verdict_detail"] = (
            "Renderer-style artefacts were measured in this clip (see evidence). That is "
            "positive evidence of generation or re-rendering, not proof of intent."
        )
    else:
        # Deliberately NOT "likely human". A neural voice and a genuine recording
        # can present the same acoustics, so the absence of artefacts is reported
        # as an absence of evidence, never as proof of authenticity.
        block["verdict"] = "NO_SYNTHETIC_ARTEFACTS"
        block["verdict_detail"] = (
            "No renderer-style artefact was measured. This does not establish that the "
            "voice is human: modern neural voices can leave no measurable trace, so treat "
            "this as unverified."
        )
    return block
