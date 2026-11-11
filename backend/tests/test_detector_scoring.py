"""Detector mechanics with explicit doubles; no persisted artifacts are loaded."""
from types import SimpleNamespace

import numpy as np
import pytest

from app.services.detector import DetectorInferenceError, TrainedVoiceDetector
from app.services.features import FeatureConfig, FeatureExtractionError


class ClassifierDouble:
    def __init__(self, score):
        self.score = score

    def predict_proba(self, matrix):
        return np.array([[1 - self.score, self.score]])


class ScalerDouble:
    mean_ = np.array([1.0, 2.0])
    scale_ = np.array([2.0, 4.0])

    def transform(self, matrix):
        return (matrix - self.mean_) / self.scale_


def detector(monkeypatch, score=0.5, features=None, scaler=None, **kwargs):
    instance = TrainedVoiceDetector(
        classifier=ClassifierDouble(score), scaler=scaler or ScalerDouble(),
        feature_config=FeatureConfig(target_sample_rate=10, segment_seconds=4),
        model_name="explicit-test-double", model_version="test", synthetic_index=1,
        **kwargs,
    )
    vector = np.array([1.0, 2.0]) if features is None else np.asarray(features)
    monkeypatch.setattr(instance, "_extractor", lambda: lambda *_: vector)
    return instance


@pytest.mark.parametrize("score,expected", [(0, 0), (0.1, 0.2), (0.5, 1), (0.9, 0.2), (1, 0)])
@pytest.mark.parametrize("calibrated", [False, True])
def test_margin_uncertainty_is_not_an_error_probability(monkeypatch, score, expected, calibrated):
    result = detector(monkeypatch, score, calibrated=calibrated).analyze(np.ones(20), 10, 1)
    assert result.uncertainty == pytest.approx(expected)
    assert "NOT a validated error probability" in result.uncertainty_method
    assert result.calibrated_probability == (score if calibrated else None)
    if calibrated:
        assert "explicitly calibrated" in result.raw_score_definition
        assert "uncalibrated" not in result.raw_score_definition
        assert "validation distribution" in result.raw_score_definition
    else:
        assert "uncalibrated" in result.raw_score_definition


@pytest.mark.parametrize("score", [np.nan, np.inf, -0.1, 1.1])
def test_invalid_classifier_scores_still_fail(monkeypatch, score):
    with pytest.raises(DetectorInferenceError):
        detector(monkeypatch, score).analyze(np.ones(20), 10, 1)


@pytest.mark.parametrize("length,expected", [
    (0, []), (1, [(0, 0.1)]), (30, [(0, 3)]), (40, [(0, 4)]),
    (41, [(0, 4), (2, 4.1)]), (80, [(0, 4), (2, 6), (4, 8)]),
    (90, [(0, 4), (2, 6), (4, 8), (6, 9)]),
])
def test_windows_overlap_and_include_real_tail(monkeypatch, length, expected):
    instance = detector(monkeypatch)
    seen = []
    def extract(samples, rate, config):
        seen.append(samples.copy())
        return np.array([1.0, 2.0])
    monkeypatch.setattr(instance, "_extractor", lambda: extract)
    windows = instance._window_scores(np.arange(length), 10)
    assert [(w["start_s"], w["end_s"]) for w in windows] == expected
    assert all(0 < len(segment) <= 40 for segment in seen)
    assert all(w["score"] == 0.5 for w in windows)
    if length:
        assert seen[-1][-1] == length - 1


def test_bad_window_does_not_hide_following_tail(monkeypatch):
    instance = detector(monkeypatch)
    def extract(samples, *_):
        if samples[0] == 20:
            raise FeatureExtractionError("deliberate test failure")
        return np.array([1.0, 2.0])
    monkeypatch.setattr(instance, "_extractor", lambda: extract)
    windows = instance._window_scores(np.arange(90), 10)
    assert [w["start_s"] for w in windows] == [0, 4, 6]
    assert windows[-1]["end_s"] == 9


@pytest.mark.parametrize("resampled_length", [40, 41])
def test_resampling_does_not_report_padding_as_real_audio(monkeypatch, resampled_length):
    instance = detector(monkeypatch)
    monkeypatch.setattr("app.services.features._resample", lambda *_: np.ones(resampled_length))
    windows = instance._window_scores(np.ones((81, 2)), 20)
    assert windows[-1]["end_s"] == 4.05
    assert all(w["end_s"] <= 4.05 for w in windows)


def test_ood_warning_uses_training_statistics_not_confidence(monkeypatch):
    result = detector(monkeypatch, score=1, features=[15, 2]).analyze(np.ones(20), 10, 1)
    assert result.uncertainty == 0
    assert "Feature-space warning" in result.explanation
    assert "not a validated OOD test" in result.explanation
    assert detector(monkeypatch, features=[13, 2])._feature_space_warning(np.array([13, 2])) is None


@pytest.mark.parametrize("scaler", [
    SimpleNamespace(), SimpleNamespace(mean_=None, scale_=None),
    SimpleNamespace(mean_=[0], scale_=[1]),
    SimpleNamespace(mean_=[0, 0], scale_=[0, 1]),
    SimpleNamespace(mean_=[0, np.nan], scale_=[1, 1]),
    SimpleNamespace(mean_=[0, 0], scale_=[1, np.inf]),
])
def test_missing_or_invalid_reference_statistics_do_not_invent_ood(monkeypatch, scaler):
    assert detector(monkeypatch, scaler=scaler)._feature_space_warning(np.array([100, 100])) is None


def test_explanation_receives_actual_standardized_features(monkeypatch):
    observed = []
    def explain(raw, scaled):
        observed.append((raw.copy(), scaled.copy()))
        return "Measured linear contributions (test double)"
    result = detector(monkeypatch, features=[5, 10], explanation_fn=explain).analyze(np.ones(20), 10, 1)
    np.testing.assert_array_equal(observed[0][0], [5, 10])
    np.testing.assert_array_equal(observed[0][1], [[2, 2]])
    assert result.explanation == "Measured linear contributions (test double)"


def test_failed_explanation_does_not_fabricate_one(monkeypatch):
    def explain(*_):
        raise RuntimeError("deliberate test failure")
    result = detector(monkeypatch, explanation_fn=explain).analyze(np.ones(20), 10, 1)
    assert result.explanation is None


class RecordingNonlinearDouble:
    """Explicit nonlinear probability double, not a trained detector."""
    def __init__(self, constant=False):
        self.inputs = []
        self.constant = constant

    def predict_proba(self, matrix):
        self.inputs.append(matrix.copy())
        score = 0.5 if self.constant else float(1 / (1 + np.exp(-matrix[0, 0] ** 2)))
        return np.array([[1 - score, score]])


@pytest.mark.parametrize("constant", [False, True])
def test_nonlinear_occlusion_reports_real_changes_and_five_call_budget(monkeypatch, constant):
    from app.services.features_enhanced import EnhancedFeatureConfig
    config = EnhancedFeatureConfig(n_mfcc=1)
    size = config.feature_dim
    scaler = ScalerDouble()
    scaler.mean_ = np.arange(size, dtype=float)
    scaler.scale_ = np.full(size, 2.0)
    instance = detector(monkeypatch, scaler=scaler, feature_kind="mfcc_enhanced")
    instance._config = config
    classifier = RecordingNonlinearDouble(constant)
    instance._classifier = classifier
    vector = scaler.mean_ + 2.0
    original = instance.score_features(vector)
    classifier.inputs.clear()
    explanation = instance._occlusion_explanation(vector, original)
    assert len(classifier.inputs) == 5
    assert "not causal proof" in explanation
    assert "replacement score minus original score" in explanation
    for observed, (start, end) in zip(
        classifier.inputs, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 15)], strict=True
    ):
        expected = np.ones((1, size))
        expected[:, start:end] = 0
        np.testing.assert_array_equal(observed, expected)
    if constant:
        assert "MFCC means: +0" in explanation
        assert "MFCC standard deviations: +0" in explanation
        assert "delta-MFCC means: +0" in explanation
        assert "delta-MFCC standard deviations: +0" in explanation
        assert "spectral shape: +0" in explanation
    else:
        assert "MFCC means: -0.231059" in explanation
        assert "spectral shape: +0" in explanation


def test_baseline_occlusion_fallback_is_used_without_callback(monkeypatch):
    from dataclasses import replace
    instance = detector(monkeypatch)
    instance._config = replace(instance._config, n_mfcc=1)
    result = instance.analyze(np.ones(20), 10, 1)
    assert "Grouped feature-occlusion sensitivity" in result.explanation
    assert "MFCC means: +0" in result.explanation
    assert "delta-MFCC" not in result.explanation


@pytest.mark.parametrize("mean", [None, [0], [0, np.nan], [[0, 0]]])
def test_occlusion_missing_or_malformed_mean_produces_no_explanation(monkeypatch, mean):
    from dataclasses import replace
    instance = detector(monkeypatch)
    instance._config = replace(instance._config, n_mfcc=1)
    instance._scaler.mean_ = mean
    classifier = RecordingNonlinearDouble()
    instance._classifier = classifier
    assert instance._occlusion_explanation(np.array([1, 2]), 0.5) is None
    assert classifier.inputs == []


def test_occlusion_failure_does_not_fabricate_partial_explanation(monkeypatch):
    from dataclasses import replace
    instance = detector(monkeypatch)
    instance._config = replace(instance._config, n_mfcc=1)
    def fail(_):
        raise DetectorInferenceError("deliberate test failure")
    monkeypatch.setattr(instance, "score_features", fail)
    assert instance._occlusion_explanation(np.array([1, 2]), 0.5) is None
