"""Speaker-verification interface.

Kept strictly separate from synthetic-speech detection: similarity to a
reference speaker is NOT evidence that a recording is genuine or cloned.

Module 3: a REAL verifier (ECAPA-TDNN embeddings + cosine similarity, with
thresholds calibrated on labelled LibriSpeech speaker pairs) is loaded when
the model files exist. Otherwise NullSpeakerVerifier keeps the API honest
with an explicit `unavailable` status and `similarity: null`.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np

from app.services.detector import ModelDescription


@dataclass(frozen=True)
class SpeakerVerificationResult:
    status: str
    similarity: float | None
    threshold: float | None
    outcome: str | None
    embedding_model: str | None
    warnings: list[str]
    limitations: list[str]


class BaseSpeakerVerifier(abc.ABC):
    kind = "speaker"

    @abc.abstractmethod
    def is_ready(self) -> bool:
        """True only when a genuine speaker-embedding model is loaded."""

    @abc.abstractmethod
    def compare(
        self, reference: np.ndarray, reference_sr: int, test: np.ndarray, test_sr: int
    ) -> SpeakerVerificationResult:
        """Compare two recordings. Must not fabricate a similarity score."""

    @abc.abstractmethod
    def describe(self) -> ModelDescription:
        """Truthful metadata for GET /api/v1/models."""


def unavailable_result(detail: str | None = None) -> SpeakerVerificationResult:
    """The honest "no similarity produced" payload, used by every fallback."""
    limitations = [
        detail
        or "No speaker-embedding model is installed, so no similarity score is produced.",
        "Similarity is not authenticity: a cloned voice can resemble its target, "
        "and a genuine recording can score low because of noise or conditions.",
        "Identity is never claimed from voice alone; this feature requires "
        "explicit authorization for the reference recording.",
    ]
    return SpeakerVerificationResult(
        status="unavailable",
        similarity=None,
        threshold=None,
        outcome=None,
        embedding_model=None,
        warnings=[detail] if detail else [],
        limitations=limitations,
    )


class NullSpeakerVerifier(BaseSpeakerVerifier):
    """Used until a real speaker-embedding model is integrated (Phase 6)."""

    def is_ready(self) -> bool:
        return False

    def compare(
        self, reference: np.ndarray, reference_sr: int, test: np.ndarray, test_sr: int
    ) -> SpeakerVerificationResult:
        return unavailable_result()

    def describe(self) -> ModelDescription:
        return ModelDescription(
            id="speaker_verifier",
            name="Speaker embedding model (not installed)",
            version="n/a",
            mode="baseline",
            kind="speaker",
            description=(
                "No speaker-embedding model is installed. Comparisons report an explicit "
                "unavailable status instead of a fabricated similarity score."
            ),
            loaded=False,
        )


class EcapaSpeakerVerifier(BaseSpeakerVerifier):
    """Real verifier: ECAPA-TDNN embeddings + cosine similarity.

    The similarity is RAW cosine similarity in [-1, 1] — a geometric
    closeness measure between embeddings, explicitly NOT the probability
    that both recordings contain the same person. The decision threshold
    comes from a calibration artifact built on real labelled pairs
    (scripts/evaluate_speaker.py); it is never invented here.
    """

    def __init__(self, embedder, thresholds) -> None:
        self._embedder = embedder
        self._thresholds = thresholds

    def is_ready(self) -> bool:
        return self._embedder.is_ready()

    def decision_threshold(self) -> float | None:
        raw = self._thresholds.eer_threshold
        calibrated_on = (self._thresholds.calibrated_on or "").lower()
        test_markers = ("fixture", "not real", "test", "synthetic", "placeholder")
        if (
            any(marker in calibrated_on for marker in test_markers)
            or self._thresholds.calibration_pairs < 100
        ):
            return None
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None
        return value if np.isfinite(value) and -1.0 <= value <= 1.0 else None

    def compare(
        self, reference: np.ndarray, reference_sr: int, test: np.ndarray, test_sr: int
    ) -> SpeakerVerificationResult:
        ref_speech, ref_warn = _speech_region(reference)
        test_speech, test_warn = _speech_region(test)
        warnings = ref_warn + test_warn

        emb_a = self._embedder.embed(ref_speech, reference_sr)
        emb_b = self._embedder.embed(test_speech, test_sr)

        similarity = float(
            np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-9)
        )
        if not np.isfinite(similarity):
            raise RuntimeError("similarity computation produced a non-finite value")

        threshold = self.decision_threshold()
        outcome: str | None = None
        if threshold is None:
            warnings.append(
                "No valid, non-placeholder calibration threshold is installed; raw similarity "
                "is reported without a same-speaker/no-speaker decision."
            )
        else:
            # A small uncertainty band around the EER threshold keeps borderline
            # pairs out of confident decisions. The threshold value itself is
            # never recomputed here (artifacts are authoritative).
            band = 0.03
            if similarity >= threshold + band:
                outcome = "likely_same_speaker"
            elif similarity <= threshold - band:
                outcome = "likely_different_speaker"
            else:
                outcome = "inconclusive"

        return SpeakerVerificationResult(
            status="completed",
            similarity=similarity,
            threshold=threshold,
            outcome=outcome,
            embedding_model="ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb)",
            warnings=warnings,
            limitations=[
                "Similarity is raw cosine similarity between 192-dim embeddings — "
                "a closeness measure, NOT the probability that both recordings "
                "contain the same person.",
                "A same/different decision is returned only when a valid calibration "
                "artifact with real calibration pairs is installed.",
                "A cloned voice can score high similarity to its target, and a genuine "
                "clip can score low because of noise or recording conditions.",
                "Identity is never claimed from voice alone; this feature requires "
                "explicit authorization for the reference recording.",
            ],
        )

    def describe(self) -> ModelDescription:
        info = self._embedder.describe()
        return ModelDescription(
            id="speaker_verifier",
            name=info["name"],
            version=info["version"],
            mode="baseline",
            kind="speaker",
            description=(
                f"ECAPA-TDNN speaker embeddings ({info['dim']} dims, 16 kHz). "
                "Produces raw cosine similarity; a decision is exposed only when a valid "
                "calibration artifact with real labelled pairs is installed."
            ),
            loaded=info["loaded"],
        )


def _speech_region(samples: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Trim leading/trailing near-silence (simple energy VAD) before embedding.

    Returns (speech_region_samples, warnings). A tiny middle region keeps the
    clip analysable; total silence is rejected honestly upstream.
    """
    warnings: list[str] = []
    mono = samples if samples.ndim == 1 else samples.mean(axis=1)
    frame = max(1, int(mono.size / 100))
    levels = np.sqrt(
        np.array(
            [np.mean(mono[i * frame : (i + 1) * frame] ** 2) for i in range(100)]
            + [0.0]
        )
    )
    threshold = 0.01
    voiced = np.where(levels > threshold)[0]
    if voiced.size == 0:
        return mono, ["No voiced region detected; embedded the full signal as-is."]
    start = int(voiced[0] * mono.size / (levels.size - 1))
    end = int((voiced[-1] + 1) * mono.size / (levels.size - 1))
    speech = mono[max(0, start) : min(mono.size, end)]
    if speech.size < 16000:
        warnings.append("Less than 1 second of voiced speech was detected.")
    return speech, warnings


class SpeakerVerificationService:
    def __init__(self, verifier: BaseSpeakerVerifier) -> None:
        self._verifier = verifier

    def is_ready(self) -> bool:
        return self._verifier.is_ready()

    def describe(self) -> ModelDescription:
        return self._verifier.describe()

    def supports_same_different_decision(self) -> bool:
        return getattr(self._verifier, "decision_threshold", lambda: None)() is not None

    def compare(
        self, reference: np.ndarray, reference_sr: int, test: np.ndarray, test_sr: int
    ) -> SpeakerVerificationResult:
        """Compare two clips; an unavailable model is a documented result, not a 500.

        Before this guard, a host that had the ECAPA artifact on disk but could
        not import torch raised out of the route and returned HTTP 500
        ("internal_error"), which the UI showed as a generic failure.
        """
        from app.services.speaker_model import SpeakerModelNotReadyError

        try:
            ready = self._verifier.is_ready()
        except Exception as exc:  # noqa: BLE001 - a probe failure is unavailability
            return unavailable_result(
                f"The speaker-embedding model could not be initialised "
                f"({type(exc).__name__}), so no similarity score is produced."
            )
        if not ready:
            return unavailable_result(
                "No speaker-embedding model is available in this runtime, so no "
                "similarity score is produced."
            )
        try:
            return self._verifier.compare(reference, reference_sr, test, test_sr)
        except (SpeakerModelNotReadyError, ImportError, RuntimeError, ValueError) as exc:
            return unavailable_result(
                "The speaker-embedding model failed during this comparison "
                f"({type(exc).__name__}), so no similarity score is produced."
            )


def speaker_stack_available() -> bool:
    """True when torch AND speechbrain can be imported in this process.

    Checked BEFORE selecting the ECAPA verifier: a host with the artifact on
    disk but without the torch/speechbrain stack (the slim serverless bundle)
    must fall through to the torch-free fallback instead of failing at
    inference time with a 500.
    """
    import importlib.util

    return (
        importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("speechbrain") is not None
    )


def try_load_speaker_verifier(settings) -> BaseSpeakerVerifier | None:
    """Return a real verifier when the artifact AND its runtime exist, else None.

    The API falls back to the MFCC-statistics comparison (or the explicit
    NullSpeakerVerifier when that is disabled) when this returns None. Model
    files load lazily on first comparison, so a missing or broken model never
    crashes the API.
    """
    from pathlib import Path

    from app.services.speaker_model import EcapaSpeakerEmbedder, SimilarityThresholds

    model_dir = Path(settings.speaker_model_dir)
    if not (model_dir / "embedding_model.ckpt").is_file():
        return None
    if not speaker_stack_available():
        import logging

        logging.getLogger("voxsentinel.speaker").warning(
            "ECAPA artifact found at %s but torch/speechbrain are not importable in this "
            "runtime; falling back to the torch-free MFCC comparison.",
            model_dir,
        )
        return None
    embedder = EcapaSpeakerEmbedder(str(model_dir))
    thresholds = SimilarityThresholds(settings.speaker_thresholds_path)
    verifier = EcapaSpeakerVerifier(embedder, thresholds)
    return verifier


# Class-name -> backend label reported by /health and /models. The label is
# what actually ran, so a slim deployment never pretends ECAPA is active.
_BACKEND_LABELS = {
    "EcapaSpeakerVerifier": "ecapa",
    "MfccStatSpeakerVerifier": "mfcc_statistics",
    "NullSpeakerVerifier": "none",
}


def speaker_backend_name(verifier: object | None) -> str:
    """The speaker comparison backend in use: 'ecapa', 'mfcc_statistics' or 'none'.

    Accepts either the verifier itself or the SpeakerVerificationService
    wrapper around it, so callers do not need to know which they hold.
    """
    if verifier is None:
        return "none"
    inner = getattr(verifier, "_verifier", None)
    if inner is not None:
        verifier = inner
    return _BACKEND_LABELS.get(type(verifier).__name__, type(verifier).__name__)


ECAPA_INSTALL_HINT = (
    "Install the speaker stack where the API runs: "
    "`pip install -r backend/requirements-speaker.txt` (CPU torch, torchaudio, "
    "speechbrain) and fetch the model once with "
    "`python backend/scripts/download_speaker_model.py`. Vercel serverless "
    "functions cannot host it (250 MB bundle limit, no system libraries); use a "
    "container host for ECAPA."
)
