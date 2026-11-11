"""Synthetic-speech branches on self-supervised speech embeddings.

Two branches, one per checkpoint: WavLM base-plus and XLS-R 300M. Each branch is
an embedding mean-pool followed by a small logistic head trained on labelled
synthetic/natural audio.

A branch with no trained head refuses to produce a number. A pretrained SSL
checkpoint knows nothing about spoofing on its own, so inventing a score from raw
embeddings would look authoritative and mean nothing. The status says what is
missing and how to build it.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger("voxsentinel.ssl_spoof")

BACKEND_DIR = Path(__file__).resolve().parents[2]
HF_DIR = BACKEND_DIR / "models" / "hf"
HEADS_DIR = BACKEND_DIR / "models" / "spoof_heads"

SPECS = {
    "wavlm": {
        "model": "microsoft/wavlm-base-plus",
        "dir": "wavlm-base-plus",
        "description": "WavLM base-plus mean-pooled embeddings + logistic spoof head",
    },
    "xlsr": {
        "model": "facebook/wav2vec2-xls-r-300m",
        "dir": "xlsr-900m",
        "description": "XLS-R 300M mean-pooled embeddings + logistic spoof head",
    },
}


@dataclass
class HeadPayload:
    coef: np.ndarray
    intercept: float
    mean: np.ndarray
    scale: np.ndarray
    threshold: float
    metrics: dict
    metadata: dict

    @classmethod
    def load(cls, path: Path) -> "HeadPayload":
        import joblib  # noqa: PLC0415 - optional dependency

        payload = joblib.load(path)
        return cls(
            coef=np.asarray(payload["coef"], dtype="float64"),
            intercept=float(payload["intercept"]),
            mean=np.asarray(payload["mean"], dtype="float64"),
            scale=np.asarray(payload["scale"], dtype="float64"),
            threshold=float(payload.get("threshold", 0.5)),
            metrics=dict(payload.get("metrics") or {}),
            metadata=dict(payload.get("metadata") or {}),
        )

    def probability(self, embedding: np.ndarray) -> float:
        z = (np.asarray(embedding, dtype="float64") - self.mean) / self.scale
        logit = float(z @ self.coef + self.intercept)
        return float(1.0 / (1.0 + np.exp(-logit)))


class SpoofBranch:
    """One SSL checkpoint plus its spoof head."""

    def __init__(self, key: str) -> None:
        self.key = key
        self._spec = SPECS[key]
        self.name = self._spec["model"]
        self._artifact = HEADS_DIR / f"{key}_spoof.joblib"
        self._model = None
        self._payload: HeadPayload | None = None
        self._error: str | None = None
        self._lock = threading.Lock()

    def _model_dir(self) -> Path:
        return HF_DIR / self._spec["dir"]

    def _reason(self) -> str | None:
        if not self._artifact.is_file():
            return (
                f"No trained head artifact {self._artifact.name}. Run "
                f"`python -m scripts.train_spoof_heads --branch {self.key}` to build it "
                "from labelled audio."
            )
        if not self._model_dir().is_dir():
            return f"SSL checkpoint missing at {self._model_dir()}."
        if self._error:
            return self._error
        return None

    def _load_head(self) -> HeadPayload | None:
        if self._payload is not None:
            return self._payload
        try:
            self._payload = HeadPayload.load(self._artifact)
        except Exception as exc:  # noqa: BLE001
            self._error = f"Head artifact unreadable: {type(exc).__name__}: {str(exc)[:120]}"
            return None
        return self._payload

    def _load_model(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                from transformers import AutoModel  # noqa: PLC0415

                self._model = AutoModel.from_pretrained(
                    str(self._model_dir()), local_files_only=True
                )
                self._model.eval()
            except Exception as exc:  # noqa: BLE001
                self._error = f"{type(exc).__name__}: {str(exc)[:160]}"
            return self._model

    def status(self) -> dict:
        reason = self._reason()
        if reason is None and self._load_head() is not None and self._payload is not None:
            ood = (self._payload.metadata.get("out_of_domain") or {})
            if not ood.get("validated"):
                reason = (
                    "Head trained on ASVspoof2019 LA and not validated outside that corpus: "
                    + (ood.get("reason") or "no out-of-domain probe has passed")
                    + (f" (probe scores {ood.get('scores')})" if ood.get("scores") else "")
                    + ". It is listed but does not vote."
                )
        metadata = {}
        if reason is None and self._load_head() is not None and self._payload is not None:
            metadata = self._payload.metadata
        return {
            "branch": self.key,
            "model": self.name,
            "status": "completed" if reason is None else ("unvalidated" if self._payload is not None else "unavailable"),
            "description": self._spec["description"],
            "artifact": str(self._artifact),
            "checkpoint_dir": str(self._model_dir()),
            "loaded": self._model is not None,
            "calibrated": bool(metadata.get("calibration") == "platt"),
            "threshold": metadata.get("threshold"),
            "metrics": metadata.get("metrics"),
            "detail": reason,
        }

    def is_available(self) -> bool:
        return self.status()["status"] == "completed"

    def embed(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        import torch  # noqa: PLC0415

        model = self._load_model()
        if model is None:
            raise RuntimeError(self._error or "SSL checkpoint unavailable")
        wav = torch.from_numpy(np.asarray(samples, dtype="float32")).unsqueeze(0)
        mask = torch.ones_like(wav, dtype=torch.long)
        with torch.no_grad():
            out = model(wav, attention_mask=mask)
        hidden = getattr(out, "last_hidden_state", None)
        if hidden is None:
            raise RuntimeError("Unexpected SSL output")
        return hidden.mean(dim=1).squeeze(0).cpu().numpy().astype("float64")

    def score(self, samples: np.ndarray, sample_rate: int) -> dict:
        started = time.perf_counter()
        status = self.status()
        if status["status"] != "completed":
            return {
                "status": "unavailable",
                "probability": None,
                "detail": status["detail"],
                "latency_ms": 0,
            }
        try:
            embedding = self.embed(samples, sample_rate)
            assert self._payload is not None
            probability = self._payload.probability(embedding)
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "failed",
                "probability": None,
                "detail": f"{type(exc).__name__}: {str(exc)[:160]}",
                "latency_ms": int((time.perf_counter() - started) * 1000),
            }
        return {
            "status": "completed",
            "probability": round(probability, 4),
            "threshold": self._payload.threshold if self._payload else None,
            "metrics": self._payload.metrics if self._payload else None,
            "latency_ms": int((time.perf_counter() - started) * 1000),
            "detail": None,
        }


_BRANCHES: dict[str, SpoofBranch] = {}
_LOCK = threading.Lock()


def build_branches() -> list[SpoofBranch]:
    with _LOCK:
        for key in SPECS:
            if key not in _BRANCHES:
                _BRANCHES[key] = SpoofBranch(key)
        return [_BRANCHES[key] for key in SPECS]


def describe() -> dict:
    branches = [branch.status() for branch in build_branches()]
    return {
        "branch_count": len(branches),
        "available": sum(1 for b in branches if b["status"] == "completed"),
        "branches": branches,
    }
