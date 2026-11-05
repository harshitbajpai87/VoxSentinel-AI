"""Enhanced feature extraction for candidate detectors.

Superset of the baseline MFCC statistics (app/services/features.py). The
baseline path stays untouched for the production model; this module adds
delta-MFCC dynamics and spectral-shape statistics, giving candidate models
more discriminative signal while keeping the same design rules:

- deterministic, fixed-length output independent of clip duration;
- same preprocessing for training and inference (shared code path);
- never returns NaN/Inf; unusable input raises FeatureExtractionError;
- configuration stored verbatim in the model artifact and re-validated on load.

Feature layout (default configuration, 91 dims):
  - MFCC mean + std          2 * n_mfcc = 40
  - delta-MFCC mean + std   2 * n_mfcc = 40
  - spectral shape stats    11 (zcr, rms, centroid, rolloff, flatness,
    bandwidth; mean and std except flatness, which is mean-only)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import librosa
import numpy as np

from app.services.features import (
    FeatureExtractionError,
    _fixed_segment,
    _resample,
    _to_mono_float32,
)


@dataclass(frozen=True)
class EnhancedFeatureConfig:
    """Deterministic enhanced-feature configuration, stored in the artifact."""

    # Shared audio/MFCC front-end (same defaults as the baseline FeatureConfig).
    target_sample_rate: int = 16000
    segment_seconds: float = 4.0
    n_mfcc: int = 20
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 128
    fmin: float = 0.0
    fmax: float = 8000.0
    # Enhanced additions.
    include_delta_mfcc: bool = True
    include_spectral_shape: bool = True

    @property
    def feature_dim(self) -> int:
        dim = self.n_mfcc * 2
        if self.include_delta_mfcc:
            dim += self.n_mfcc * 2
        if self.include_spectral_shape:
            dim += 11
        return dim

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict[str, object]) -> EnhancedFeatureConfig:
        known = {f for f in EnhancedFeatureConfig.__dataclass_fields__}
        unknown = set(data.keys()) - known
        if unknown:
            raise ValueError(f"Unknown enhanced feature configuration keys: {sorted(unknown)}")
        return EnhancedFeatureConfig(**{k: v for k, v in data.items() if k in known})


def enhanced_feature_names(config: EnhancedFeatureConfig) -> list[str]:
    """Ordered feature names, matching extract_enhanced_features output."""
    names = [f"mfcc{i}_mean" for i in range(config.n_mfcc)]
    names += [f"mfcc{i}_std" for i in range(config.n_mfcc)]
    if config.include_delta_mfcc:
        names += [f"dmfcc{i}_mean" for i in range(config.n_mfcc)]
        names += [f"dmfcc{i}_std" for i in range(config.n_mfcc)]
    if config.include_spectral_shape:
        names += [
            "zcr_mean", "zcr_std",
            "rms_mean", "rms_std",
            "centroid_mean", "centroid_std",
            "rolloff_mean", "rolloff_std",
            "flatness_mean",
            "bandwidth_mean", "bandwidth_std",
        ]
    return names


def _safe_stats(values: np.ndarray) -> np.ndarray:
    """Mean and std of a 1-D frame series, made finite or rejected."""
    mean = float(np.mean(values))
    std = float(np.std(values))
    if not (np.isfinite(mean) and np.isfinite(std)):
        raise FeatureExtractionError("Enhanced features produced non-finite statistics.")
    return np.array([mean, std], dtype=np.float32)


def extract_enhanced_features(
    samples: np.ndarray,
    sample_rate: int,
    config: EnhancedFeatureConfig | None = None,
) -> np.ndarray:
    """Mono audio in, fixed-length enhanced feature vector out.

    Raises FeatureExtractionError for empty, non-finite or otherwise unusable
    input. Never returns NaN/Inf.
    """
    config = config or EnhancedFeatureConfig()

    mono = _to_mono_float32(samples)
    if mono.size == 0:
        raise FeatureExtractionError("Cannot extract features from empty audio.")
    if not np.all(np.isfinite(mono)):
        raise FeatureExtractionError("Audio contains non-finite samples.")
    mono = _resample(mono, int(sample_rate), config.target_sample_rate)
    segment = _fixed_segment(mono, config.target_sample_rate, config.segment_seconds)
    if segment.size < 2:
        raise FeatureExtractionError("Audio segment is too short for analysis.")
    if float(np.max(np.abs(segment))) == 0.0:
        raise FeatureExtractionError("Audio segment is silent (all zeros).")

    mfcc = librosa.feature.mfcc(
        y=segment,
        sr=config.target_sample_rate,
        n_mfcc=config.n_mfcc,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
        fmin=config.fmin,
        fmax=config.fmax,
    )
    parts: list[np.ndarray] = [
        np.asarray(mfcc.mean(axis=1), dtype=np.float32),
        np.asarray(mfcc.std(axis=1), dtype=np.float32),
    ]

    if config.include_delta_mfcc:
        delta = librosa.feature.delta(mfcc)
        parts.append(np.asarray(delta.mean(axis=1), dtype=np.float32))
        parts.append(np.asarray(delta.std(axis=1), dtype=np.float32))

    if config.include_spectral_shape:
        zcr = librosa.feature.zero_crossing_rate(segment, frame_length=config.n_fft,
                                                 hop_length=config.hop_length)[0]
        rms = librosa.feature.rms(y=segment, frame_length=config.n_fft,
                                  hop_length=config.hop_length)[0]
        centroid = librosa.feature.spectral_centroid(
            y=segment, sr=config.target_sample_rate, n_fft=config.n_fft,
            hop_length=config.hop_length,
        )[0]
        rolloff = librosa.feature.spectral_rolloff(
            y=segment, sr=config.target_sample_rate, n_fft=config.n_fft,
            hop_length=config.hop_length, roll_percent=0.85,
        )[0]
        flatness = librosa.feature.spectral_flatness(
            y=segment, n_fft=config.n_fft, hop_length=config.hop_length,
        )[0]
        bandwidth = librosa.feature.spectral_bandwidth(
            y=segment, sr=config.target_sample_rate, n_fft=config.n_fft,
            hop_length=config.hop_length,
        )[0]
        parts.append(np.concatenate([
            _safe_stats(zcr), _safe_stats(rms), _safe_stats(centroid),
            _safe_stats(rolloff), np.array([float(np.mean(flatness))], dtype=np.float32),
            _safe_stats(bandwidth),
        ]))

    features = np.concatenate(parts).astype(np.float32)
    if features.size != config.feature_dim:
        raise FeatureExtractionError(
            f"Enhanced feature dimension {features.size} != expected {config.feature_dim}."
        )
    if not np.all(np.isfinite(features)):
        raise FeatureExtractionError("Enhanced features contain non-finite values.")
    return features
