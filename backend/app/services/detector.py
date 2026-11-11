"""Detection-model interface.

AI integrity rule for this project: the API never fabricates a prediction. If
no genuine trained detector is installed, `NullVoiceDetector.is_ready()` is
False, `/analyze` answers HTTP 503 `model_unavailable`, and nothing is stored
as a completed detection.

Phase 5 adds a real detector by implementing `BaseVoiceDetector` — no route,
schema or frontend change is required.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from app.core.config import Settings

if TYPE_CHECKING:  # annotation-only import; keeps the interface module light
    from app.services.features import FeatureConfig


class ModelNotReadyError(RuntimeError):
    """Raised when analysis is requested but no trained detector is available."""


class DetectorInferenceError(RuntimeError):
    """Raised when a loaded model fails or produces an invalid output.

    The API refuses to serve, clamp or invent values from a misbehaving model;
    the request fails honestly instead."""


@dataclass(frozen=True)
class DetectorResult:
    """Output of a genuine detector.

    `raw_score` is an unscaled model output and is NOT a probability. A
    calibrated probability is only present when calibration was performed on
    validation data; otherwise it stays None instead of being invented.
    """

    raw_score: float
    raw_score_definition: str
    calibrated_probability: float | None = None
    uncertainty: float | None = None
    uncertainty_method: str | None = None
    model_name: str = "unknown"
    model_version: str = "0.0.0"
    model_mode: str = "baseline"
    # Human-readable, model-supported explanation (e.g. top logistic-regression
    # contributions). None when the model supports no explanation method.
    explanation: str | None = None
    # Window-level evidence scores from the same model (Module 6). Empty when
    # extraction fails for all windows. Evidence, not the official score.
    window_scores: tuple[dict, ...] = ()


@dataclass(frozen=True)
class ModelDescription:
    id: str
    name: str
    version: str
    mode: str
    kind: str
    description: str
    loaded: bool


class BaseVoiceDetector(abc.ABC):
    """Contract every real detector must satisfy."""

    kind = "detection"

    @abc.abstractmethod
    def is_ready(self) -> bool:
        """True only when a genuine, loaded model can produce real scores."""

    @abc.abstractmethod
    def analyze(self, samples: np.ndarray, sample_rate: int, channels: int) -> DetectorResult:
        """Analyze mono float32 samples. Must raise ModelNotReadyError if unready."""

    @abc.abstractmethod
    def describe(self) -> ModelDescription:
        """Truthful metadata for GET /api/v1/models."""


class NullVoiceDetector(BaseVoiceDetector):
    """Honest placeholder used when no trained model artifact is installed.

    It never returns scores, never claims readiness, and never breaks health
    checks. It exists so the API can run and clearly report that real
    synthetic-voice detection is not available yet.
    """

    def is_ready(self) -> bool:
        return False

    def analyze(self, samples: np.ndarray, sample_rate: int, channels: int) -> DetectorResult:
        raise ModelNotReadyError(
            "No trained synthetic-speech detector is installed on this server."
        )

    def describe(self) -> ModelDescription:
        return ModelDescription(
            id="voice_detector",
            name="Detection model (not installed)",
            version="n/a",
            mode="baseline",
            kind="detection",
            description=(
                "No trained synthetic-speech detector is installed. The API validates and "
                "measures audio, but real detection stays unavailable until a trained, "
                "licensed model is integrated (Phase 5)."
            ),
            loaded=False,
        )


class VoiceDetectorService:
    """Thin service wrapper used by the routes (single seam for testing)."""

    def __init__(self, detector: BaseVoiceDetector) -> None:
        self._detector = detector

    def is_ready(self) -> bool:
        return self._detector.is_ready()

    def describe(self) -> ModelDescription:
        return self._detector.describe()

    def analyze(self, samples: np.ndarray, sample_rate: int, channels: int) -> DetectorResult:
        if not self._detector.is_ready():
            raise ModelNotReadyError(
                "The detection model is not ready. Real detection is unavailable until a "
                "trained model is integrated."
            )
        return self._detector.analyze(samples, sample_rate, channels)

    # --- evaluation accessors (Module 5): delegated so the evaluation job
    # uses EXACTLY the serving model, never a parallel reimplementation. ---
    def score_features(self, features: np.ndarray) -> float:
        if not self._detector.is_ready():
            raise ModelNotReadyError("The detection model is not ready.")
        return self._detector.score_features(features)

    def feature_config_asdict(self) -> dict:
        if not self._detector.is_ready():
            raise ModelNotReadyError("The detection model is not ready.")
        return self._detector.feature_config_asdict()

    def model_name(self) -> str:
        return self._detector.describe().name

    def model_version(self) -> str:
        return self._detector.describe().version


class TrainedVoiceDetector(BaseVoiceDetector):
    """A genuine detector backed by an artifact from this project's trainer.

    Wraps a fitted scikit-learn pipeline (StandardScaler + classifier) over the
    shared MFCC/enhanced feature extraction. Every output is validated: the
    positive probability must be finite and within [0, 1] or the inference fails
    loudly. `calibrated_probability` is reported only when the artifact carries
    an explicit calibration (sigmoid fit on validation data); otherwise it
    stays None and is never invented.
    """

    def __init__(
        self,
        classifier: object,
        scaler: object,
        feature_config: FeatureConfig,
        model_name: str,
        model_version: str,
        synthetic_index: int,
        model_mode: str = "baseline",
        explanation_fn: object | None = None,
        feature_kind: str = "mfcc_baseline",
        calibrated: bool = False,
    ) -> None:
        import numpy  # local import keeps the module importable without sklearn

        self._np = numpy
        self._classifier = classifier
        self._scaler = scaler
        self._config = feature_config
        self._model_name = model_name
        self._model_version = model_version
        self._model_mode = model_mode
        self._synthetic_index = synthetic_index
        self._explanation_fn = explanation_fn
        self._feature_kind = feature_kind
        self._calibrated = calibrated

    def is_ready(self) -> bool:
        return True  # only constructed from a validated artifact (see model_loader)

    @property
    def calibrated(self) -> bool:
        """Whether the artifact declares an explicit score calibration."""
        return self._calibrated

    def analyze(self, samples: np.ndarray, sample_rate: int, channels: int) -> DetectorResult:
        from app.services.features import FeatureExtractionError

        extract_fn = self._extractor()
        try:
            features = extract_fn(samples, sample_rate, self._config)
        except FeatureExtractionError as exc:
            raise DetectorInferenceError(f"Feature extraction failed: {exc}") from exc

        score = self.score_features(features)

        # Window-level evidence (Module 6): the SAME model applied to
        # overlapping, bounded segments. Honest attribution — the official
        # result remains the whole-clip score above.
        window_scores = self._window_scores(samples, sample_rate)

        explanation: str | None = None
        if self._explanation_fn is not None:
            try:
                scaled = self._scaler.transform(
                    self._np.asarray(features).reshape(1, -1)
                )
                explanation = str(self._explanation_fn(features, scaled))
            except Exception:  # noqa: BLE001 - explanations must never break inference
                explanation = None
        else:
            explanation = self._occlusion_explanation(features, score)

        ood_warning = self._feature_space_warning(features)
        if ood_warning is not None:
            explanation = f"{explanation}\n{ood_warning}" if explanation else ood_warning

        return DetectorResult(
            raw_score=score,
            raw_score_definition=(
                "Estimated P(synthetic) from the explicitly calibrated classifier; "
                "calibration is limited to the artifact's validation distribution."
                if self._calibrated else
                "Estimated P(synthetic) from the classifier (uncalibrated; "
                "treat as a ranking score, not a validated probability)."
            ),
            # A probability is reported ONLY when the artifact carries an
            # explicit calibration; otherwise it stays None (never invented).
            calibrated_probability=score if self._calibrated else None,
            uncertainty=1.0 - 2.0 * abs(score - 0.5),
            uncertainty_method=(
                "Classifier margin: 1 - 2*abs(p - 0.5). Higher means nearer the "
                "decision boundary; this is NOT a validated error probability and "
                "does not measure out-of-distribution uncertainty."
            ),
            model_name=self._model_name,
            model_version=self._model_version,
            model_mode=self._model_mode,
            explanation=explanation,
            window_scores=window_scores,
        )

    def _extractor(self):
        """Feature-extraction function matching the artifact's feature kind."""
        from app.services.features import extract_features
        from app.services.features_enhanced import extract_enhanced_features

        if self._feature_kind == "mfcc_enhanced":
            return extract_enhanced_features
        return extract_features

    def _occlusion_explanation(self, features: np.ndarray, original_score: float) -> str | None:
        """Actual grouped replacement sensitivity, requiring fitted means.

        At most five additional predictions, with no sampling or invented
        contributions. These perturbations need not represent plausible audio
        and are not additive attributions or causal evidence.
        """
        try:
            vector = self._np.asarray(features, dtype=float).reshape(-1)
            mean = self._np.asarray(self._scaler.mean_, dtype=float)
            if (mean.ndim != 1 or mean.shape != vector.shape
                    or not self._np.isfinite(mean).all()
                    or not self._np.isfinite(vector).all()):
                return None
            n = self._config.n_mfcc
            if not isinstance(n, int) or n <= 0:
                return None
            groups = [("MFCC means", 0, n), ("MFCC standard deviations", n, 2 * n)]
            expected = 2 * n
            if self._feature_kind == "mfcc_enhanced":
                if self._config.include_delta_mfcc:
                    groups.extend([
                        ("delta-MFCC means", expected, expected + n),
                        ("delta-MFCC standard deviations", expected + n, expected + 2 * n),
                    ])
                    expected += 2 * n
                if self._config.include_spectral_shape:
                    groups.append(("spectral shape", expected, expected + 11))
                    expected += 11
            elif self._feature_kind != "mfcc_baseline":
                return None
            if vector.size != expected:
                return None  # unknown feature layout: never guess group boundaries
            changes = []
            for label, start, end in groups:
                replaced = vector.copy()
                replaced[start:end] = mean[start:end]
                occluded_score = self.score_features(replaced)
                changes.append(f"{label}: {occluded_score - original_score:+.6g}")
            return (
                "Grouped feature-occlusion sensitivity (replacement score minus "
                "original score): " + "; ".join(changes)
                + ". Each group was replaced by fitted scaler training means; "
                "other features were held fixed. These are actual model score "
                "changes, not causal proof or additive contributions; replacement "
                "vectors may not correspond to realistic audio."
            )
        except Exception:  # optional XAI must not break valid inference
            return None

    def _feature_space_warning(self, features: np.ndarray) -> str | None:
        """Conservative marginal-distance heuristic, not a validated OOD test.

        Use only fitted training statistics, never the current upload as a
        reference distribution. Missing/malformed scaler statistics do not
        justify claiming either in-distribution or out-of-distribution input.
        Constant training dimensions use unit scale, as StandardScaler does.
        """
        try:
            mean = self._np.asarray(self._scaler.mean_, dtype=float).reshape(-1)
            scale = self._np.asarray(self._scaler.scale_, dtype=float).reshape(-1)
            vector = self._np.asarray(features, dtype=float).reshape(-1)
            if (mean.size != vector.size or scale.size != vector.size
                    or not mean.size or not self._np.isfinite(mean).all()
                    or not self._np.isfinite(scale).all() or (scale <= 0).any()
                    or not self._np.isfinite(vector).all()):
                return None
            with self._np.errstate(over="ignore", invalid="ignore"):
                distance = self._np.abs((vector - mean) / scale)
            # Six fitted marginal training-scale units is a deliberately
            # conservative heuristic, not a learned cutoff or coverage bound.
            if not (distance > 6.0).any():
                return None
        except (AttributeError, TypeError, ValueError):
            return None
        return (
            "Feature-space warning: at least one input feature is more than six "
            "fitted training-scale units from the scaler mean. This "
            "heuristic suggests possible out-of-distribution audio; it is not a "
            "validated OOD test or an error probability. A confident classifier "
            "score can still be unreliable."
        )

    def _window_scores(self, samples: np.ndarray, sample_rate: int) -> tuple[dict, ...]:
        """Score segment-bounded windows with 50% overlap and a final tail.

        Short and trailing windows are padded by the shared extractor only;
        their timestamps describe real samples, never synthetic padding.
        The official score still comes from the existing whole-clip path
        (whose extractor takes the first model segment), not these windows.
        """
        from app.services.features import (
            FeatureExtractionError,
            _resample,
            _to_mono_float32,
        )

        mono = _to_mono_float32(samples)
        mono = _resample(mono, sample_rate, self._config.target_sample_rate)
        rate = self._config.target_sample_rate
        segment = int(round(rate * self._config.segment_seconds))
        if segment <= 0 or rate <= 0:
            raise DetectorInferenceError("Invalid detector window configuration.")
        hop = max(1, segment // 2)
        windows: list[dict] = []
        extract_fn = self._extractor()
        start = 0
        # Preserve the source duration even if the resampler rounds up by one
        # sample. No reported endpoint may exceed the real recording length.
        duration = len(samples) / sample_rate
        while start < mono.size:
            end = min(start + segment, mono.size)
            try:
                feats = extract_fn(mono[start:end], rate, self._config)
                windows.append({
                    "start_s": start / rate,
                    "end_s": duration if end == mono.size else min(end / rate, duration),
                    "score": self.score_features(feats),
                })
            except (FeatureExtractionError, DetectorInferenceError):
                pass  # one bad window must not fail the whole analysis
            if end == mono.size:
                break
            start += hop
        return tuple(windows)

    def score_features(self, features: np.ndarray) -> float:
        """Scale and classify an already-extracted feature vector (shared path)."""
        matrix = self._np.asarray(features).reshape(1, -1)
        try:
            scaled = self._scaler.transform(matrix)
            probabilities = self._classifier.predict_proba(scaled)[0]
        except Exception as exc:  # noqa: BLE001 - any sklearn failure is an inference failure
            raise DetectorInferenceError(f"The detector failed during inference: {exc}") from exc
        probabilities = self._np.asarray(probabilities)
        if self._synthetic_index >= probabilities.size:
            raise DetectorInferenceError("Detector output does not contain the synthetic class.")
        score = float(probabilities[self._synthetic_index])
        if not self._np.isfinite(score) or not (0.0 <= score <= 1.0):
            raise DetectorInferenceError(
                f"The detector produced an invalid score ({score}); refusing to serve it."
            )
        return score

    def feature_config_asdict(self) -> dict:
        from dataclasses import asdict

        return asdict(self._config)

    def model_name(self) -> str:
        return self._model_name

    def model_version(self) -> str:
        return self._model_version

    def describe(self) -> ModelDescription:
        return ModelDescription(
            id="voice_detector",
            name=self._model_name,
            version=self._model_version,
            mode=self._model_mode,
            kind="detection",
            description=(
                "MFCC-statistics baseline detector trained by this project's pipeline. "
                "Uncalibrated score; evaluated only on the dataset and synthesis methods "
                "listed in its training report (see docs/MODEL_CARD.md)."
                if self._feature_kind == "mfcc_baseline" else
                "Enhanced-feature candidate detector (delta-MFCC + spectral shape) trained "
                "by the multi-dataset pipeline; see its training report for provenance "
                "and calibration status."
            ),
            loaded=True,
        )


def try_load_detector(settings: Settings) -> BaseVoiceDetector | None:
    """Phase 5 hook: load a trained artifact from `settings.detector_model_path`.

    Loads ONLY a server-configured artifact produced by this project's training
    script. Never used on user uploads (joblib artifacts are pickles: loading
    one from an untrusted source can execute arbitrary code).
    """
    from app.services.model_loader import load_detector_from_path  # lazy: avoids cycle

    return load_detector_from_path(settings.detector_model_path)
