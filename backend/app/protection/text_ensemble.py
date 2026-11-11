"""Transformer text-ensemble for scam intent (MuRIL + DeBERTa).

The shipped ``scam_text_classifier.joblib`` is a TF-IDF + logistic-regression
model: fast, dependency-free and explainable through its own coefficients, but
blind to word order and to the sub-word morphology that Hinglish/Hindi fraud
messages rely on.

This module adds two frozen encoder branches:

  * ``google/muril-base-cased``     multilingual Indic BERT, Hindi + Hinglish
  * ``microsoft/deberta-v3-small``  English sub-word model, code-mixed English

Each branch is a real trained head over mean-pooled hidden states, produced by
``python -m scripts.train_transformer_text_heads`` on the labelled corpus in
``data/corpus`` and evaluated on the hand-authored set the corpus never contains.
No head artifact means the branch reports ``unavailable`` with the reason; the
TF-IDF classifier and the rule engine keep working, and the ensemble averages
only over the branches that actually ran.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.protection.asr_engines import looks_hinglish
from app.services.hf_models import BACKEND_DIR, resolve_model

logger = logging.getLogger("voxsentinel.protection.text_ensemble")

HEADS_DIR = Path(
    os.environ.get("VOXSENTINEL_TEXT_HEADS_DIR", str(BACKEND_DIR / "models" / "text_heads"))
)
DEFAULT_THRESHOLD = float(os.environ.get("VOXSENTINEL_TEXT_ENSEMBLE_THRESHOLD", "0.55"))
MAX_CHARS = 512

BRANCHES: dict[str, dict] = {
    "muril": {
        "model_id": "google/muril-base-cased",
        "slug": "muril-base-cased",
        "artifact": "muril_scam_head.joblib",
        "languages": ["en", "hi", "hi-en"],
    },
    "deberta": {
        "model_id": "microsoft/deberta-v3-small",
        "slug": "deberta-v3-small",
        "artifact": "deberta_scam_head.joblib",
        "languages": ["en"],
    },
    "xlmr": {
        "model_id": "FacebookAI/xlm-roberta-base",
        "slug": "xlmr-base",
        "artifact": "xlmr_scam_head.joblib",
        "languages": ["en", "hi", "hi-en"],
    },
}


#: Loaded encoders shared across TextEnsemble instances. Without this, every
#: app rebuild (tests, hot reload) re-read the same 1 GB of weights for nothing.
_SHARED: dict[str, tuple[object, object]] = {}
_SHARED_LOCK = threading.Lock()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class BranchResult:
    name: str
    model_id: str
    status: str
    score: float | None = None
    version: str | None = None
    detail: str | None = None

    def as_dict(self) -> dict:
        return {
            "branch": self.name,
            "model": self.model_id,
            "status": self.status,
            "score": self.score,
            "version": self.version,
            "detail": self.detail,
        }


class TransformerHead:
    """Frozen encoder + trained linear head, loaded lazily per process."""

    def __init__(self, name: str, spec: dict) -> None:
        self.name = name
        self._model_id = spec["model_id"]
        self._artifact = HEADS_DIR / spec["artifact"]
        self._source = resolve_model(self._model_id, spec["slug"])
        self._allow_download = _env_bool("VOXSENTINEL_TEXT_ENSEMBLE_DOWNLOAD", False)
        self._tokenizer = None
        self._model = None
        self._payload: dict | None = None
        self._load_error: str | None = None
        self._lock = threading.Lock()
        self._tried = False

    # -- availability -----------------------------------------------------
    def _reason(self) -> str | None:
        if not self._artifact.is_file():
            return (
                f"No trained head artifact {self._artifact.name}. Run "
                "`python -m scripts.train_transformer_text_heads` to build it from the "
                "labelled corpus."
            )
        if not self._source.available:
            return self._source.reason
        if not self._source.local and not self._allow_download:
            return (
                f"{self._model_id} is not installed locally. Run "
                "`python -m scripts.download_text_models` or set "
                "VOXSENTINEL_TEXT_ENSEMBLE_DOWNLOAD=true."
            )
        return None

    def status(self) -> dict:
        if self._load_error:
            return {
                "branch": self.name,
                "model": self._model_id,
                "status": "failed",
                "artifact": self._artifact.name,
                "local_model": self._source.local,
                "detail": self._load_error,
            }
        reason = self._reason()
        return {
            "branch": self.name,
            "model": self._model_id,
            "status": "completed" if reason is None else "unavailable",
            "artifact": self._artifact.name,
            "local_model": self._source.local,
            "detail": reason or self._load_error,
        }

    def is_available(self) -> bool:
        return self.status()["status"] == "completed"

    # -- inference --------------------------------------------------------
    def _load(self) -> bool:
        if self._payload is not None and self._model is not None:
            return True
        if self._tried:
            return False
        with self._lock:
            if self._payload is not None and self._model is not None:
                return True
            if self._tried:
                return False
            self._tried = True
            try:
                import joblib

                payload = joblib.load(self._artifact)
                self._payload = payload if isinstance(payload, dict) else {"pipeline": payload}
                source = self._source.source
                with _SHARED_LOCK:
                    cached = _SHARED.get(self.name)
                    if cached is not None and cached[2] == source:  # type: ignore[index]
                        self._tokenizer, self._model = cached[0], cached[1]
                    else:
                        import torch
                        from transformers import AutoModel, AutoTokenizer

                        self._tokenizer = AutoTokenizer.from_pretrained(source)
                        self._model = AutoModel.from_pretrained(source).eval()
                        torch.set_grad_enabled(False)
                        _SHARED[self.name] = (self._tokenizer, self._model, source)
                logger.info("%s head ready (%s + %s)", self.name, self._model_id, self._artifact.name)
                return True
            except Exception as exc:  # noqa: BLE001
                self._load_error = f"{self.name} branch failed to load: {type(exc).__name__}"
                logger.warning("%s branch failed to load: %s", self.name, exc)
                return False

    def _embed(self, text: str) -> np.ndarray:
        import torch

        assert self._tokenizer is not None and self._model is not None
        encoded = self._tokenizer(
            text[: MAX_CHARS * 4],
            return_tensors="pt",
            truncation=True,
            max_length=MAX_CHARS,
            padding=True,
        )
        with torch.no_grad():
            hidden = self._model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        return pooled.squeeze(0).numpy().astype(np.float32)

    def score(self, text: str) -> BranchResult:
        status = self.status()
        if status["status"] != "completed":
            return BranchResult(self.name, self._model_id, "unavailable", detail=status["detail"])
        if not self._load():
            return BranchResult(
                self.name, self._model_id, "failed", detail=self._load_error or "load failed"
            )
        try:
            pipeline = self._payload["pipeline"]  # type: ignore[index]
            embedding = self._embed(text)
            proba = pipeline.predict_proba([embedding])[0]
            classes = list(getattr(pipeline, "classes_", []))
            if "scam" not in classes or len(proba) != len(classes):
                return BranchResult(
                    self.name,
                    self._model_id,
                    "failed",
                    detail="The trained head artifact has no valid 'scam' class mapping.",
                )
            index = classes.index("scam")
            if not np.isfinite(proba[index]):
                return BranchResult(
                    self.name, self._model_id, "failed", detail="The head returned a non-finite score."
                )
            return BranchResult(
                self.name,
                self._model_id,
                "completed",
                score=round(float(proba[index]), 4),
                version=(self._payload or {}).get("version"),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s inference failed: %s", self.name, exc)
            return BranchResult(
                self.name, self._model_id, "failed", detail=f"inference failed: {type(exc).__name__}"
            )


class TextEnsemble:
    """Averages the available encoder branches into one calibrated-ish score."""

    def __init__(self, heads: dict[str, TransformerHead] | None = None,
                 threshold: float = DEFAULT_THRESHOLD) -> None:
        self._enabled = _env_bool("VOXSENTINEL_TEXT_ENSEMBLE", True)
        self._heads = heads or {name: TransformerHead(name, spec) for name, spec in BRANCHES.items()}
        self._threshold = threshold
        self._cache_key: tuple[str, str] | None = None
        self._cache_value: dict | None = None

    def describe(self) -> dict:
        branches = [head.status() for head in self._heads.values()]
        return {
            "service": "transformer-text-ensemble",
            "enabled": self._enabled,
            "threshold": self._threshold,
            "branches": branches,
            "available": self._enabled and any(item["status"] == "completed" for item in branches),
        }

    def classify(self, text: str, language: str | None = None) -> dict:
        if not self._enabled:
            return {
                "status": "unavailable",
                "detail": "Disabled by configuration (VOXSENTINEL_TEXT_ENSEMBLE=false).",
                "branches": [],
                "score": None,
                "label": None,
                "threshold": self._threshold,
            }
        clean = (text or "").strip()
        if not clean:
            return {
                "status": "unavailable",
                "detail": "No transcript text to classify.",
                "branches": [],
                "score": None,
                "label": None,
                "threshold": self._threshold,
            }
        language_key = _language_key(language, clean)
        cache_key = (clean, language_key)
        if cache_key == self._cache_key and self._cache_value is not None:
            return self._cache_value

        branch_results = []
        for head in self._heads.values():
            spec = BRANCHES.get(head.name)
            supported = set((spec or {}).get("languages", []))
            if supported and language_key not in supported:
                branch_results.append(
                    BranchResult(
                        head.name,
                        getattr(head, "_model_id", head.name),
                        "skipped",
                        detail=f"Branch is not validated for language '{language_key}'.",
                    )
                )
                continue
            branch_results.append(head.score(clean))
        scored = [b for b in branch_results if b.status == "completed" and b.score is not None]
        if not scored:
            payload = {
                "status": "unavailable",
                "detail": (
                    "No encoder branch is ready on this host; the TF-IDF classifier and the "
                    "rule engine still produced the content verdict."
                ),
                "branches": [b.as_dict() for b in branch_results],
                "score": None,
                "label": None,
                "threshold": self._threshold,
            }
            self._cache_key, self._cache_value = cache_key, payload
            return payload

        weights = {"xlmr": 0.40, "muril": 0.35, "deberta": 0.25}
        total = sum(weights.get(b.name, 0.5) for b in scored)
        score = round(sum((b.score or 0.0) * weights.get(b.name, 0.5) for b in scored) / total, 4)
        payload = {
            "status": "completed",
            "score": score,
            "label": "scam" if score >= self._threshold else "benign",
            "threshold": self._threshold,
            "branches": [b.as_dict() for b in branch_results],
            "definition": (
                "Weighted mean of the encoder branches that ran, on the labelled training "
                "distribution. A supporting signal, not a probability that a real call is fraud."
            ),
        }
        self._cache_key, self._cache_value = cache_key, payload
        return payload


def _language_key(language: str | None, text: str) -> str:
    """Normalize supported language labels; infer script only when absent."""
    value = (language or "").strip().lower().replace("_", "-")
    if looks_hinglish(text):
        return "hi-en"
    if value.startswith("hi") or value in {"hinglish", "hindi"}:
        return "hi-en" if "en" in value or value == "hinglish" else "hi"
    if value.startswith("en") or value in {"english", "eng"}:
        return "en"
    if value:
        return value
    letters = [char for char in text if char.isalpha()]
    devanagari = sum("\u0900" <= char <= "\u097f" for char in letters)
    return "hi" if letters and devanagari / len(letters) >= 0.2 else "en"
