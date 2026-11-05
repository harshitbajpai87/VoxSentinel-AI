"""Enhanced feature extraction: determinism, dimension, honest failures."""
from __future__ import annotations

import numpy as np
import pytest

from app.services.features import FeatureExtractionError
from app.services.features_enhanced import (
    EnhancedFeatureConfig,
    enhanced_feature_names,
    extract_enhanced_features,
)

SR = 16000


def speech_like(seed: int, seconds: float = 4.0) -> np.ndarray:
    """Deterministic speech-like signal: modulated formants + noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(SR * seconds)) / SR
    envelope = 0.5 * (1 + np.sin(2 * np.pi * 3.0 * t + seed))
    signal = (
        0.4 * np.sin(2 * np.pi * 200 * t)
        + 0.25 * np.sin(2 * np.pi * 800 * t + rng.uniform(0, 3))
        + 0.1 * np.sin(2 * np.pi * 2400 * t)
        + 0.02 * rng.standard_normal(t.size)
    )
    return (signal * envelope).astype(np.float32)


def test_feature_dim_matches_names():
    config = EnhancedFeatureConfig()
    assert config.feature_dim == 91
    assert len(enhanced_feature_names(config)) == config.feature_dim


def test_deterministic_and_finite():
    config = EnhancedFeatureConfig()
    a = extract_enhanced_features(speech_like(1), SR, config)
    b = extract_enhanced_features(speech_like(1), SR, config)
    np.testing.assert_array_equal(a, b)
    assert np.all(np.isfinite(a))


def test_dimension_independent_of_duration():
    config = EnhancedFeatureConfig()
    short = extract_enhanced_features(speech_like(2, seconds=1.0), SR, config)
    long = extract_enhanced_features(speech_like(2, seconds=6.0), SR, config)
    assert short.shape == long.shape == (config.feature_dim,)


def test_resampling_path():
    config = EnhancedFeatureConfig()
    at_44k = speech_like(3)
    # Feed the same waveform labeled 44.1kHz: resampled then extracted.
    features = extract_enhanced_features(at_44k, 44100, config)
    assert features.shape == (config.feature_dim,)
    assert np.all(np.isfinite(features))


def test_rejects_empty_silent_and_nan():
    config = EnhancedFeatureConfig()
    with pytest.raises(FeatureExtractionError):
        extract_enhanced_features(np.zeros(0, dtype=np.float32), SR, config)
    with pytest.raises(FeatureExtractionError):
        extract_enhanced_features(np.zeros(SR * 4, dtype=np.float32), SR, config)
    bad = speech_like(4)
    bad[10] = np.nan
    with pytest.raises(FeatureExtractionError):
        extract_enhanced_features(bad, SR, config)


def test_config_roundtrip_rejects_unknown_keys():
    config = EnhancedFeatureConfig()
    restored = EnhancedFeatureConfig.from_dict(config.to_dict())
    assert restored == config
    with pytest.raises(ValueError, match="[Uu]nknown"):
        EnhancedFeatureConfig.from_dict({"n_mfcc": 20, "bogus": 1})


def test_flags_toggle_dimensions():
    lean = EnhancedFeatureConfig(include_delta_mfcc=False, include_spectral_shape=False)
    assert lean.feature_dim == 40  # baseline-compatible MFCC mean/std only
    assert extract_enhanced_features(speech_like(5), SR, lean).shape == (40,)
