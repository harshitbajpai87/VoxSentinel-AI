"""Feature extraction: the shared training/inference preprocessing path."""
from __future__ import annotations

import numpy as np
import pytest

from app.services.features import (
    FeatureConfig,
    FeatureExtractionError,
    extract_features,
    feature_names,
)


def tone(duration_s: float = 2.0, sr: int = 8000, freq: float = 220.0) -> np.ndarray:
    t = np.arange(int(duration_s * sr)) / sr
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_features_from_valid_audio_have_fixed_dimension():
    config = FeatureConfig()
    features = extract_features(tone(), 8000, config)
    assert features.shape == (config.feature_dim,)
    assert np.isfinite(features).all()
    assert feature_names(config)[:3] == ["mfcc0_mean", "mfcc1_mean", "mfcc2_mean"]


def test_stereo_input_is_mixed_to_mono():
    mono = tone()
    stereo = np.stack([mono, mono * 0.5], axis=1)  # (frames, 2)
    features = extract_features(stereo, 8000)
    assert features.shape == (FeatureConfig().feature_dim,)
    assert np.isfinite(features).all()


def test_sample_rate_is_resampled_to_target():
    """The same signal at 8 kHz and 16 kHz must produce near-identical features."""
    config = FeatureConfig()
    low = extract_features(tone(sr=8000), 8000, config)
    high = extract_features(tone(sr=16000), 16000, config)
    assert low.shape == high.shape
    assert np.max(np.abs(low - high)) < 1.0  # resampling tolerance


def test_short_audio_is_padded_to_fixed_segment():
    features = extract_features(tone(duration_s=1.0), 8000)
    assert features.shape == (FeatureConfig().feature_dim,)
    assert np.isfinite(features).all()


def test_empty_audio_is_rejected():
    with pytest.raises(FeatureExtractionError):
        extract_features(np.array([], dtype=np.float32), 8000)


def test_non_finite_audio_is_rejected():
    bad = tone()
    bad[10] = np.nan
    with pytest.raises(FeatureExtractionError):
        extract_features(bad, 8000)
    inf = tone()
    inf[10] = np.inf
    with pytest.raises(FeatureExtractionError):
        extract_features(inf, 8000)


def test_extraction_is_deterministic():
    audio = tone(duration_s=3.0)
    first = extract_features(audio, 8000)
    second = extract_features(audio, 8000)
    assert np.array_equal(first, second)


def test_feature_config_round_trips():
    config = FeatureConfig(n_mfcc=13)
    restored = FeatureConfig.from_dict(config.to_dict())
    assert restored == config
    assert restored.feature_dim == 26


def test_feature_config_rejects_unknown_keys():
    with pytest.raises(ValueError):
        FeatureConfig.from_dict({"target_sample_rate": 16000, "mystery": 1})
