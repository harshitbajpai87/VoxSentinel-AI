"""Speaker embedding model: SpeechBrain ECAPA-TDNN (VoxCeleb).

Model: speechbrain/spkrec-ecapa-voxceleb (HuggingFace), 192-dim embeddings,
CPU inference, ~83 MB. Downloaded once into ``settings.speaker_model_dir``
(never per request) and lazy-loaded on first use so the API starts fast and
stays alive if the model files are missing or torch is not importable.

The model is initialized from a LOCAL directory only: we do not fetch from
the network at request time, and we never load user-supplied weights.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger("voxsentinel.speaker")

EMBEDDING_DIM = 192
TARGET_SAMPLE_RATE = 16000
MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"


class SpeakerModelNotReadyError(RuntimeError):
    """Raised when embeddings are requested without a usable model."""


class EcapaSpeakerEmbedder:
    """Thin, honest wrapper around the ECAPA-TDNN encoder."""

    def __init__(self, model_dir: str) -> None:
        self._model_dir = Path(model_dir)
        self._classifier = None
        self._load_error: str | None = None
        self._lock = threading.Lock()

    # -- readiness ---------------------------------------------------------
    def is_ready(self) -> bool:
        return self._classifier is not None or self._try_load()

    def describe(self) -> dict:
        return {
            "id": "speaker_ecapa",
            "name": "ECAPA-TDNN (VoxCeleb) speaker embedding",
            "version": "1.0",
            "source": MODEL_ID,
            "dim": EMBEDDING_DIM,
            "sample_rate": TARGET_SAMPLE_RATE,
            "loaded": self._classifier is not None,
            "load_error": self._load_error,
        }

    # -- embedding ---------------------------------------------------------
    def embed(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """Embed mono float32 speech at ``sample_rate`` into 192 dims.

        SpeechBrain's ECAPA pipeline expects 16 kHz; we resample here so
        preprocessing is identical for every caller.
        """
        classifier = self._classifier or (self.is_ready() and self._classifier)
        if classifier is None:
            raise SpeakerModelNotReadyError(
                "speaker model is not loaded: " + (self._load_error or "unknown error")
            )
        import torch

        mono = _to_mono(samples)
        if mono.dtype != np.float32:
            mono = mono.astype(np.float32)
        if sample_rate != TARGET_SAMPLE_RATE:
            mono = _resample_linear(mono, sample_rate, TARGET_SAMPLE_RATE)
        if mono.size == 0:
            raise ValueError("no audio samples to embed")
        with torch.no_grad():
            batch = torch.from_numpy(mono).unsqueeze(0)
            embedding = classifier.encode_batch(batch)
        vector = embedding.squeeze().cpu().numpy().astype(np.float32)
        if vector.shape[0] != EMBEDDING_DIM:
            raise RuntimeError(
                f"unexpected embedding dimension {vector.shape[0]} (want {EMBEDDING_DIM})"
            )
        return vector

    # -- internals ---------------------------------------------------------
    def _try_load(self) -> bool:
        with self._lock:
            if self._classifier is not None:
                return True
            if not self._model_dir.is_dir() or not (
                self._model_dir / "embedding_model.ckpt"
            ).exists():
                self._load_error = f"model files not found under {self._model_dir}"
                return False
            try:
                from speechbrain.inference.speaker import EncoderClassifier

                started = time.time()
                self._classifier = EncoderClassifier.from_hparams(
                    source=str(self._model_dir),
                    savedir=str(self._model_dir),
                    run_opts={"device": "cpu"},
                )
                logger.info(
                    "ECAPA speaker model loaded from %s in %.1fs",
                    self._model_dir,
                    time.time() - started,
                )
                self._load_error = None
                return True
            except Exception as exc:  # noqa: BLE001 - logged, never leaked
                self._load_error = f"model load failed: {type(exc).__name__}"
                logger.warning("Speaker model failed to load: %s", type(exc).__name__, exc_info=True)
                return False


class SimilarityThresholds:
    """Decision thresholds calibrated on real labelled speaker pairs.

    Loaded from a JSON artifact produced by ``scripts/evaluate_speaker.py``
    (dev-clean pairs); never invented at runtime.
    """

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._data: dict | None = None

    def is_ready(self) -> bool:
        if self._data is not None:
            return True
        if not self._path.is_file():
            return False
        try:
            loaded = json.loads(self._path.read_text())
            if loaded.get("format") != "voxsentinel-speaker-thresholds":
                logger.warning("threshold artifact has wrong format: %s", self._path)
                return False
            self._data = loaded
            return True
        except (OSError, json.JSONDecodeError):
            logger.warning("Could not read threshold artifact %s", self._path)
            return False

    @property
    def eer_threshold(self) -> float | None:
        if not self.is_ready():
            return None
        return self._data.get("eer_threshold")  # type: ignore[return-value]

    @property
    def calibration_pairs(self) -> int:
        if not self.is_ready():
            return 0
        try:
            return int(self._data.get("calibration_pairs") or 0)
        except (TypeError, ValueError):
            return 0

    @property
    def calibrated_on(self) -> str | None:
        if not self.is_ready():
            return None
        return self._data.get("calibrated_on")

    def as_dict(self) -> dict | None:
        return dict(self._data) if self.is_ready() else None


def _to_mono(samples: np.ndarray) -> np.ndarray:
    if samples.ndim == 1:
        return samples
    return samples.mean(axis=1)


def _resample_linear(mono: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Deterministic linear resample (mirrors the training-time convention)."""
    if orig_sr == target_sr or mono.size == 0:
        return mono
    duration = mono.size / orig_sr
    n_out = max(1, int(round(duration * target_sr)))
    positions = np.linspace(0.0, mono.size - 1, n_out, dtype=np.float64)
    low = np.floor(positions).astype(np.int64)
    high = np.minimum(low + 1, mono.size - 1)
    frac = (positions - low).astype(np.float32)
    return mono[low] * (1.0 - frac) + mono[high] * frac
