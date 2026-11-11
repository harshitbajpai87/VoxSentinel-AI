"""AASIST anti-spoofing detector, run through its exported ONNX graph.

The checkpoint and its exported graph come from the published AASIST release
(``SpeechAntiSpoofingBenchmarks/AASIST``). Running the exported graph rather than
re-implementing the network means the mel front-end is the one the weights were
trained with, so a score is a score and not an artefact of a guessed feature
extractor.

Input contract of the graph: ``wav`` float32 ``(batch, 64600)`` at 16 kHz
(4.04 s), output ``logits`` ``(batch, 2)`` with class 0 = bonafide, 1 = spoof,
the ASVspoof convention. Longer clips are scored in 4.04 s windows and averaged
in logit space; shorter clips are repeat-padded, which is the padding the model
was trained with.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger("voxsentinel.aasist")

BACKEND_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BACKEND_DIR / "models" / "aasist"
ONNX_PATH = MODEL_DIR / "aasist.onnx"
CHECKPOINT = MODEL_DIR / "AASIST.pth"

MODEL_ID = "clovaai/AASIST"
VARIANT = "AASIST (ASVspoof 2019 LA, ONNX export)"
SAMPLE_RATE = 16000
WINDOW = 64600  # 4.04 s at 16 kHz, the graph's fixed time axis


def _fit(samples: np.ndarray) -> np.ndarray:
    """Repeat-pad short clips to one window, exactly as training did."""
    if samples.size == 0:
        return np.zeros((1, WINDOW), dtype="float32")
    if samples.size < WINDOW:
        repeats = int(np.ceil(WINDOW / samples.size))
        samples = np.tile(samples, repeats)
    return samples[:WINDOW].astype("float32")[None, :]


class AASISTDetector:
    """Lazy ONNX session; reports unavailable with a reason instead of guessing."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or ONNX_PATH
        self._session = None
        self._lock = threading.Lock()
        self._error: str | None = None
        self._enabled = (__import__("os").environ.get("VOXSENTINEL_AASIST") or "true").lower() not in {
            "0",
            "false",
            "no",
        }

    def _load(self):
        if self._session is not None or self._error is not None:
            return self._session
        with self._lock:
            if self._session is not None or self._error is not None:
                return self._session
            if not self._path.is_file():
                self._error = (
                    f"No exported AASIST graph at {self._path}. Fetch it with "
                    "`python -m scripts.fetch_models`."
                )
                return None
            try:
                import onnxruntime as ort  # noqa: PLC0415 - optional dependency

                self._session = ort.InferenceSession(
                    str(self._path), providers=["CPUExecutionProvider"]
                )
            except Exception as exc:  # noqa: BLE001 - never fatal
                self._error = f"{type(exc).__name__}: {str(exc)[:160]}"
            return self._session

    def status(self) -> dict:
        reason = None
        if not self._enabled:
            reason = "Disabled by configuration (VOXSENTINEL_AASIST=false)."
        else:
            reason = self._error or (
                None if self._load() is not None else f"No usable graph at {self._path}."
            )
        return {
            "model": MODEL_ID,
            "variant": VARIANT,
            "status": "completed" if reason is None else "unavailable",
            "checkpoint": str(CHECKPOINT),
            "graph": str(self._path),
            "loaded": self._session is not None,
            "calibrated": False,
            "score_definition": "Softmax of AASIST bonafide/spoof logits; not calibrated on a deployment validation set.",
            "threshold": 0.5,
            "window_s": round(WINDOW / SAMPLE_RATE, 2),
            "detail": reason,
        }

    def is_available(self) -> bool:
        return self.status()["status"] == "completed"

    def score(self, samples: np.ndarray, sample_rate: int) -> dict:
        """Uncalibrated spoof-class softmax score over logit-averaged windows."""
        started = time.perf_counter()
        session = self._load()
        if session is None:
            return {
                "status": "unavailable",
                "probability": None,
                "calibrated": False,
                "score_definition": "AASIST softmax spoof-class score; not deployment-calibrated.",
                "detail": self._error,
                "windows": 0,
                "latency_ms": 0,
            }
        audio = np.asarray(samples, dtype="float32")
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"AASIST expects {SAMPLE_RATE} Hz audio, got {sample_rate} Hz.")
        if audio.size == 0:
            return {
                "status": "failed",
                "probability": None,
                "calibrated": False,
                "score_definition": "AASIST softmax spoof-class score; not deployment-calibrated.",
                "detail": "Empty audio.",
                "windows": 0,
                "latency_ms": 0,
            }
        windows = max(1, int(np.ceil(audio.size / WINDOW)))
        logits = []
        for index in range(windows):
            chunk = audio[index * WINDOW : (index + 1) * WINDOW]
            out = session.run(None, {"wav": _fit(chunk)})[0][0]
            logits.append(out)
        mean = np.mean(np.asarray(logits), axis=0)
        shifted = np.exp(mean - mean.max())
        probability = float((shifted / shifted.sum())[1])
        return {
            "status": "completed",
            "probability": round(probability, 4),
            "calibrated": False,
            "score_definition": "AASIST softmax spoof-class score; not deployment-calibrated.",
            "logits": [round(float(v), 4) for v in mean],
            "windows": windows,
            "latency_ms": int((time.perf_counter() - started) * 1000),
            "detail": None,
        }


_DETECTOR: AASISTDetector | None = None
_LOCK = threading.Lock()


def get_detector() -> AASISTDetector:
    global _DETECTOR
    if _DETECTOR is None:
        with _LOCK:
            if _DETECTOR is None:
                _DETECTOR = AASISTDetector()
    return _DETECTOR
