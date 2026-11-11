"""Generated signals validate numerical parity, NOT speech detection accuracy."""
import numpy as np
import pytest

from app.services.features_enhanced import EnhancedFeatureConfig, extract_enhanced_features
from app.services.features_numpy import extract_enhanced_features_numpy, resample_numpy


@pytest.mark.parametrize("rate", [8000, 16000, 22050, 24000, 44100, 48000])
def test_runtime_feature_parity(rate):
    rng = np.random.default_rng(42)
    t = np.arange(rate * 4) / rate
    audio = (0.2 * np.sin(2*np.pi*223*t) + rng.normal(0, .03, t.size)).astype(np.float32)
    cfg = EnhancedFeatureConfig()
    expected = extract_enhanced_features(audio, rate, cfg)
    actual = extract_enhanced_features_numpy(audio, rate, cfg)
    np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=2e-4)

@pytest.mark.parametrize("rate", [8000, 22050, 24000, 44100, 48000])
def test_resampling_matches_training(rate):
    import librosa
    audio = np.random.default_rng(3).normal(0,.1,rate*2+1).astype(np.float32)
    expected = librosa.resample(audio, orig_sr=rate, target_sr=16000)
    np.testing.assert_array_equal(resample_numpy(audio,rate,16000), expected)
