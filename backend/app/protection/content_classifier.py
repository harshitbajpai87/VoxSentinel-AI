"""Trained content classifier (supporting signal for scam analysis).

The rule engine is explainable but literal: it misses phrasings nobody wrote a
pattern for. This module serves a small, cheap model trained on the labelled
scam corpus in this repo, so the content layer keeps working when no LLM key
is configured, when the provider is rate-limited, or on a host that cannot run
one at all.

Design rules (the same ones the rest of the protection module follows):

  * it is a SUPPORTING signal. Rules and the contextual model stay the primary
    evidence; this layer can raise a weak verdict, never veto a strong one, and
    the report always says which layer produced what.
  * confidence is a model probability on the training distribution, not a
    probability that a real call is fraud. The API label says so.
  * no model file means status="unavailable" with a reason, never a guess.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger("voxsentinel.protection.content_classifier")

DEFAULT_MODEL_PATH = os.environ.get(
    "VOXSENTINEL_CONTENT_MODEL",
    str(Path(__file__).resolve().parent.parent.parent / "models" / "scam_text_classifier.joblib"),
)
# Minimum probability before the model is allowed to call a call suspicious.
# Deliberately high: a false accusation in a family emergency is expensive.
DEFAULT_THRESHOLD = float(os.environ.get("VOXSENTINEL_CONTENT_THRESHOLD", "0.65"))

_lock = threading.Lock()
_loaded: dict | None = None
_load_error: str | None = None


@dataclass
class Classification:
    status: str  # completed | unavailable | failed
    label: str | None = None  # scam | benign
    confidence: float | None = None
    model: str | None = None
    top_category: str | None = None
    detail: str | None = None

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "label": self.label,
            "confidence": self.confidence,
            "model": self.model,
            "top_category": self.top_category,
            "detail": self.detail,
            "confidence_definition": (
                "model probability on the labelled training corpus, not a real-call probability"
            ),
        }


def _load() -> dict | None:
    global _loaded, _load_error
    if _loaded is not None or _load_error is not None:
        return _loaded
    with _lock:
        if _loaded is not None or _load_error is not None:
            return _loaded
        path = Path(DEFAULT_MODEL_PATH)
        if not path.exists():
            _load_error = (
                f"No trained content model at {path.name}; rules and the LLM layer still run."
            )
            return None
        try:
            import joblib

            payload = joblib.load(path)
            if isinstance(payload, dict) and "pipeline" in payload:
                _loaded = payload
            else:  # legacy: a bare estimator
                _loaded = {"pipeline": payload, "version": "unknown", "metrics": {}}
            return _loaded
        except Exception as exc:  # noqa: BLE001 - a broken artifact must not break the API
            _load_error = f"Content model could not be loaded: {type(exc).__name__}"
            logger.warning("Content model load failed: %s", exc)
            return None


def is_available() -> bool:
    return _load() is not None


def describe() -> dict:
    payload = _load()
    if payload is None:
        return {"service": "trained-text-classifier", "available": False, "detail": _load_error}
    return {
        "service": "trained-text-classifier",
        "available": True,
        "version": payload.get("version"),
        "trained_on": payload.get("trained_on"),
        "threshold": payload.get("threshold", DEFAULT_THRESHOLD),
        "metrics": payload.get("metrics", {}),
    }


def classify_text(text: str) -> Classification:
    payload = _load()
    if payload is None:
        return Classification(status="unavailable", detail=_load_error)
    if not text or not text.strip():
        return Classification(
            status="unavailable",
            detail="No transcript text to classify.",
            model=payload.get("version"),
        )
    try:
        pipeline = payload["pipeline"]
        proba = pipeline.predict_proba([text])[0]
        classes = list(pipeline.classes_)
        scam_index = classes.index("scam") if "scam" in classes else int(np.argmax(proba))
        confidence = float(proba[scam_index])
        label = "scam" if confidence >= float(payload.get("threshold", DEFAULT_THRESHOLD)) else "benign"
        return Classification(
            status="completed",
            label=label,
            confidence=round(confidence, 4),
            model=payload.get("version"),
            top_category=None,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Content classification failed: %s", exc)
        return Classification(status="failed", detail=f"Classification failed: {type(exc).__name__}")
