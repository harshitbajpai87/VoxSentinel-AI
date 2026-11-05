"""MFCC feature extraction — the single shared path for training AND inference.

This module is imported by the training scripts and by the live detector, so
preprocessing can never drift between training and serving.

The representation is fixed and documented: librosa MFCC over a fixed-length
segment, summarized as per-coefficient mean and standard deviation across
frames. `FeatureConfig` is stored inside the model artifact and re-validated
at load time.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import librosa
import numpy as np


class FeatureExtractionError(RuntimeError):
    """Raised when audio cannot be turned into valid features."""


@dataclass(frozen=True)
class FeatureConfig:
    """Deterministic feature configuration. Stored verbatim in the artifact."""

    target_sample_rate: int = 16000
    segment_seconds: float = 4.0
    n_mfcc: int = 20
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 128
    fmin: float = 0.0
    fmax: float = 8000.0

    @property
    def feature_dim(self) -> int:
        """Two statistics (mean, std) per MFCC coefficient."""
        return self.n_mfcc * 2

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict[str, object]) -> FeatureConfig:
        known = {f for f in FeatureConfig.__dataclass_fields__}
        unknown = set(data.keys()) - known
        if unknown:
            raise ValueError(f"Unknown feature configuration keys: {sorted(unknown)}")
        return FeatureConfig(**{k: v for k, v in data.items() if k in known})


def feature_names(config: FeatureConfig) -> list[str]:
    return [f"mfcc{i}_mean" for i in range(config.n_mfcc)] + [
        f"mfcc{i}_std" for i in range(config.n_mfcc)
    ]


def _to_mono_float32(samples: np.ndarray) -> np.ndarray:
    array = np.asarray(samples, dtype=np.float32)
    if array.ndim == 2:
        array = array.mean(axis=1)
    return array.reshape(-1)


def _resample(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return samples
    if orig_sr <= 0:
        raise FeatureExtractionError(f"Invalid source sample rate: {orig_sr}")
    return librosa.resample(samples.astype(np.float32), orig_sr=orig_sr, target_sr=target_sr)


def _fixed_segment(samples: np.ndarray, target_sr: int, segment_seconds: float) -> np.ndarray:
    """Take the first `segment_seconds` of audio, zero-padding shorter clips.

    Fixed length keeps the frame count — and therefore the feature dimension —
    independent of input duration.
    """
    needed = int(round(target_sr * segment_seconds))
    if samples.size >= needed:
        return samples[:needed]
    return np.pad(samples, (0, needed - samples.size), mode="constant")


def extract_features(
    samples: np.ndarray,
    sample_rate: int,
    config: FeatureConfig | None = None,
) -> np.ndarray:
    """Mono audio in, fixed-length feature vector out.

    Raises FeatureExtractionError for empty, non-finite or otherwise unusable
    input, or when the resulting features are invalid. Never returns NaN/Inf.
    """
    config = config or FeatureConfig()

    mono = _to_mono_float32(samples)
    if mono.size == 0:
        raise FeatureExtractionError("Cannot extract features from empty audio.")
    if not bool(np.isfinite(mono).all()):
        raise FeatureExtractionError("Cannot extract features: audio contains non-finite samples.")

    mono = _resample(mono, sample_rate, config.target_sample_rate)
    segment = _fixed_segment(mono, config.target_sample_rate, config.segment_seconds)

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
    if mfcc.shape[1] < 2:
        raise FeatureExtractionError(
            "Audio segment is too short to compute MFCC statistics (need >= 2 frames)."
        )

    features = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)]).astype(np.float64)
    if features.shape[0] != config.feature_dim:
        raise FeatureExtractionError(
            f"Feature dimension mismatch: expected {config.feature_dim}, got {features.shape[0]}."
        )
    if not bool(np.isfinite(features).all()):
        raise FeatureExtractionError("Feature extraction produced non-finite values.")
    return features
