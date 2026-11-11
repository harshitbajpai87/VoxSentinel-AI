"""Acoustic forensics: learned representations fused with a trained detector.

The unified pipeline asks three independent questions about a recording:

  1. ``detector``      does the shipped detector.joblib (enhanced MFCC features)
                       consider this synthetic?
  2. ``handcrafted``   do the measurable acoustic cues (bandwidth ceiling,
                       digital silence, pitch smoothness, level uniformity) look
                       like a render rather than a microphone capture?
  3. ``wav2vec2_head`` do facebook/wav2vec2-large-xlsr-53 representations, fed
                       to a head trained on this project's labels, separate this
                       clip from genuine speech?

Components 1 and 2 always work (pure numpy + scikit-learn). Component 3 needs the
XLSR checkpoint (~1.2 GB) plus a head artifact trained with
``scripts/train_forensics_head.py``; when either is missing it reports
``unavailable`` with the reason and the fusion renormalises over what is real.

The fused score is a weighted mean over the components that actually ran, and it
always lists which ones were used. It is a screening score on labelled data, not
a probability that a specific call is fraudulent.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.services.hf_models import BACKEND_DIR, downloads_allowed, resolve_model

logger = logging.getLogger("voxsentinel.forensics.unified")

MODEL_ID = "facebook/wav2vec2-large-xlsr-53"
MODEL_SLUG = "wav2vec2-xlsr-53"
HEAD_PATH = Path(
    os.environ.get(
        "VOXSENTINEL_FORENSICS_HEAD",
        str(BACKEND_DIR / "models" / "forensics_wav2vec2_head.joblib"),
    )
)
TARGET_SR = 16000
#: Weights for the fusion step. Declared here so the API can report them.
FUSION_WEIGHTS = {"detector": 0.5, "wav2vec2_head": 0.3, "handcrafted": 0.2}

#: Encoders are shared across instances so a reload never re-reads the weights.
_SHARED: dict = {}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class Wav2Vec2ForensicsEmbedder:
    """Mean-pooled XLSR-53 hidden states, cached per process."""

    def __init__(self) -> None:
        self._source = resolve_model(MODEL_ID, MODEL_SLUG)
        self._model = None
        self._processor = None
        self._lock = threading.Lock()
        self._error: str | None = None
        self._enabled = _env_bool("VOXSENTINEL_WAV2VEC2_FORENSICS", True)
        # A 1.2 GB fetch inside a user request is never a good trade; require an
        # explicit opt-in (or a pre-downloaded local copy).
        self._allow_download = _env_bool("VOXSENTINEL_WAV2VEC2_FORENSICS_DOWNLOAD", False)

    # -- availability -----------------------------------------------------
    def status(self) -> dict:
        reason = self._error
        if self._error:
            reason = self._error
        elif not self._enabled:
            reason = (
                "Disabled by configuration (VOXSENTINEL_WAV2VEC2_FORENSICS=false); "
                "forensics uses the trained detector and acoustic cues."
            )
        elif not self._source.available:
            reason = self._source.reason
        elif not self._source.local and not self._allow_download:
            reason = (
                f"{MODEL_ID} is not installed locally; install its Hugging Face "
                "snapshot under backend/models/hf/xlsr-53 or set "
                "VOXSENTINEL_WAV2VEC2_FORENSICS_DOWNLOAD=true to enable the "
                "learned forensics branch."
            )
        return {
            "model": MODEL_ID,
            "status": "failed" if self._error else "completed" if reason is None else "unavailable",
            "loaded": self._model is not None,
            "local": self._source.local,
            "detail": reason,
        }

    def is_available(self) -> bool:
        return self.status()["status"] == "completed"

    # -- inference --------------------------------------------------------
    def embed(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        status = self.status()
        if status["status"] != "completed":
            raise RuntimeError(status["detail"] or "wav2vec2 forensics branch unavailable")
        try:
            import torch
            from transformers import AutoFeatureExtractor, Wav2Vec2Model

            if self._model is None:
                with self._lock:
                    if self._model is None:
                        source = self._source.source
                        if _SHARED.get("source") == source and _SHARED.get("model") is not None:
                            self._processor = _SHARED["processor"]
                            self._model = _SHARED["model"]
                        else:
                            self._processor = AutoFeatureExtractor.from_pretrained(source)
                            self._model = Wav2Vec2Model.from_pretrained(source).eval()
                            _SHARED.update(
                                {"source": source, "processor": self._processor, "model": self._model}
                            )
                        logger.info("wav2vec2 forensics encoder ready (%s)", source)
        except Exception as exc:  # noqa: BLE001 - report a stable failed state
            self._error = f"XLSR encoder load failed: {type(exc).__name__}"
            logger.warning("wav2vec2 forensics encoder failed: %s", exc)
            raise RuntimeError(self._error) from exc
        audio = np.asarray(samples, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sample_rate != TARGET_SR:
            audio = _resample(audio, sample_rate, TARGET_SR)
        inputs = self._processor(audio, sampling_rate=TARGET_SR, return_tensors="pt")
        with torch.no_grad():
            hidden = self._model(**inputs).last_hidden_state
        return hidden.mean(dim=1).squeeze(0).numpy().astype(np.float32)


@dataclass
class HeadResult:
    status: str
    score: float | None = None
    version: str | None = None
    detail: str | None = None


class ForensicsHead:
    """Trained head over XLSR embeddings (artifact produced by a script)."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = Path(path or HEAD_PATH)
        self._payload: dict | None = None
        self._error: str | None = None
        self._loaded = False

    def _load(self) -> dict | None:
        if self._loaded:
            return self._payload
        self._loaded = True
        if not self._path.is_file():
            self._error = (
                f"No trained forensics head at {self._path.name}. This checkout does not include "
                "a supported training script; install a compatible externally trained artifact "
                "only after validation on separate real and synthetic audio."
            )
            return None
        try:
            import joblib

            payload = joblib.load(self._path)
            if isinstance(payload, dict) and "pipeline" in payload:
                self._payload = payload
            else:
                self._payload = {"pipeline": payload, "version": "unknown"}
        except Exception as exc:  # noqa: BLE001 - a broken artifact must not break the API
            self._error = f"Forensics head could not be loaded: {type(exc).__name__}"
            logger.warning("forensics head load failed: %s", exc)
            self._payload = None
        return self._payload

    def status(self) -> dict:
        payload = self._load()
        return {
            "artifact": self._path.name,
            "status": "completed" if payload is not None else "unavailable",
            "version": (payload or {}).get("version"),
            "detail": self._error,
        }

    def score(self, embedding: np.ndarray) -> HeadResult:
        payload = self._load()
        if payload is None:
            return HeadResult(status="unavailable", detail=self._error)
        try:
            pipeline = payload["pipeline"]
            proba = pipeline.predict_proba([embedding])[0]
            classes = list(pipeline.classes_)
            index = classes.index("synthetic") if "synthetic" in classes else int(np.argmax(proba))
            return HeadResult(
                status="completed",
                score=round(float(proba[index]), 4),
                version=payload.get("version"),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("forensics head inference failed: %s", exc)
            return HeadResult(status="failed", detail=f"Inference failed: {type(exc).__name__}")


class UnifiedForensics:
    """Merges detector, handcrafted cues and the learned forensics branch."""

    def __init__(self, handcrafted=None, embedder: Wav2Vec2ForensicsEmbedder | None = None,
                 head: ForensicsHead | None = None, threshold: float = 0.5) -> None:
        self._handcrafted = handcrafted
        self._embedder = embedder or Wav2Vec2ForensicsEmbedder()
        self._head = head or ForensicsHead()
        self._threshold = float(threshold)

    # -- helpers ----------------------------------------------------------
    def _handcrafted_block(self, samples: np.ndarray, sample_rate: int, channels: int) -> dict:
        if self._handcrafted is None:
            return {
                "status": "unavailable",
                "detail": "Acoustic forensics is disabled on this server.",
                "signals": [],
                "evidence": {},
            }
        try:
            result = self._handcrafted.analyze(samples, sample_rate, channels)
            evidence = {}
            measure = getattr(result, "evidence", None)
            if measure is not None:
                evidence = {
                    key: value
                    for key, value in vars(measure).items()
                    if isinstance(value, (int, float, str, list)) or value is None
                }
            return {
                "status": "completed",
                "score": float(result.raw_score),
                "verdict": "elevated" if result.raw_score >= self._threshold else "not_elevated",
                "signals": evidence.pop("signals", []) or [],
                "evidence": evidence,
                "explanation": result.explanation,
                "kind": "heuristic_evidence_only",
            }
        except Exception as exc:  # noqa: BLE001
            return {"status": "failed", "detail": type(exc).__name__, "signals": [], "evidence": {}}

    def _embedding_block(self, samples: np.ndarray, sample_rate: int) -> dict:
        embedder_status = self._embedder.status()
        if embedder_status["status"] != "completed":
            return {"status": "unavailable", "detail": embedder_status["detail"],
                    "model": MODEL_ID, "score": None}
        head_status = self._head.status()
        if head_status["status"] != "completed":
            return {"status": "unavailable", "detail": head_status["detail"],
                    "model": MODEL_ID, "dim": None, "score": None}
        try:
            embedding = self._embedder.embed(samples, sample_rate)
        except Exception as exc:  # noqa: BLE001
            return {"status": "failed", "detail": f"Embedding failed: {type(exc).__name__}",
                    "model": MODEL_ID, "score": None}
        scored = self._head.score(embedding)
        if scored.status != "completed":
            return {"status": scored.status, "detail": scored.detail,
                    "model": MODEL_ID, "dim": int(embedding.size), "score": None}
        return {
            "status": "completed",
            "model": MODEL_ID,
            "dim": int(embedding.size),
            "head_version": scored.version,
            "score": scored.score,
            "verdict": "elevated" if (scored.score or 0) >= self._threshold else "not_elevated",
        }

    # -- public -----------------------------------------------------------
    def analyze(
        self,
        samples: np.ndarray,
        sample_rate: int,
        channels: int,
        detector_block: dict | None = None,
    ) -> dict:
        detector = detector_block or {}
        detector_score = _detector_score(detector)
        handcrafted = self._handcrafted_block(samples, sample_rate, channels)
        embedding = self._embedding_block(samples, sample_rate)

        components: dict[str, float] = {}
        if detector_score is not None:
            components["detector"] = float(detector_score)
        if handcrafted.get("status") == "completed" and handcrafted.get("score") is not None:
            components["handcrafted"] = float(handcrafted["score"])
        if embedding.get("status") == "completed" and embedding.get("score") is not None:
            components["wav2vec2_head"] = float(embedding["score"])

        fused, used, weights = _fuse(components)
        verdict = _verdict(fused, self._threshold)
        return {
            "status": "completed" if components else "unavailable",
            "detector": {
                "status": detector.get("status", "unavailable"),
                "model": detector.get("model"),
                "score": detector.get("score"),
                "calibrated_probability": detector.get("calibrated_probability"),
                "threshold": detector.get("threshold"),
                "verdict": detector.get("verdict") or detector.get("suspected_synthetic"),
            },
            "handcrafted": handcrafted,
            "wav2vec2_head": embedding,
            "fusion": {
                "score": fused,
                "verdict": verdict,
                "threshold": self._threshold,
                "weights": weights,
                "components_used": used,
                "components_missing": [c for c in FUSION_WEIGHTS if c not in used],
                "definition": (
                    "Weighted mean of the components that ran, on a 0-1 scale where higher "
                    "means more synthetic-sounding. A screening score on labelled data, "
                    "not the probability that this specific call is fraudulent."
                ),
            },
            "explanation": _explain(fused, verdict, used, handcrafted, embedding, detector),
        }

    def describe(self) -> dict:
        return {
            "fusion_weights": FUSION_WEIGHTS,
            "embedding_model": self._embedder.status(),
            "head": self._head.status(),
            "handcrafted_available": self._handcrafted is not None,
            "downloads_allowed": downloads_allowed(),
        }


def _detector_score(detector: dict) -> float | None:
    for key in ("calibrated_probability", "score"):
        value = detector.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _fuse(components: dict[str, float]) -> tuple[float | None, list[str], dict]:
    if not components:
        return None, [], {}
    total_weight = sum(FUSION_WEIGHTS[name] for name in components)
    weights = {
        name: round(FUSION_WEIGHTS[name] / total_weight, 4) for name in components
    }
    fused = sum(value * weights[name] for name, value in components.items())
    return round(min(max(fused, 0.0), 1.0), 4), sorted(components), weights


def _verdict(fused: float | None, threshold: float) -> str:
    if fused is None:
        return "unavailable"
    band = 0.08
    if fused >= threshold + band:
        return "elevated"
    if fused <= threshold - band:
        return "not_elevated"
    return "inconclusive"


def _explain(
    fused: float | None,
    verdict: str,
    used: list[str],
    handcrafted: dict,
    embedding: dict,
    detector: dict,
) -> str:
    if fused is None:
        return (
            "No acoustic component produced a score on this host, so authenticity could not "
            "be screened. The text-level evidence still stands on its own."
        )
    parts = [f"Fused authenticity screen {fused:.2f} ({verdict.replace('_', ' ')}) from {', '.join(used)}."]
    if "detector" in used:
        parts.append("The trained detector scored the recording's MFCC profile.")
    if "handcrafted" in used and handcrafted.get("signals"):
        parts.append("Measurable cues: " + ", ".join(str(s) for s in handcrafted["signals"][:3]) + ".")
    if "wav2vec2_head" in used:
        parts.append("The learned XLSR branch contributed to the fused screen; its contribution alone does not establish agreement.")
    elif embedding.get("status") == "unavailable" and embedding.get("detail"):
        parts.append("The learned XLSR branch did not run: " + str(embedding["detail"]))
    if verdict == "inconclusive":
        parts.append(
            "The score sits in the ambiguity band: modern neural cloning and a clean real "
            "recording can land in the same range, so this is undetermined, not innocent."
        )
    return " ".join(parts)


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr or audio.size == 0:
        return audio
    try:
        import librosa

        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    except Exception:  # pragma: no cover
        duration = audio.size / float(orig_sr)
        n_out = max(1, int(round(duration * target_sr)))
        positions = np.linspace(0.0, max(audio.size - 1, 0), n_out)
        low = np.floor(positions).astype(np.int64)
        high = np.minimum(low + 1, max(audio.size - 1, 0))
        frac = (positions - low).astype(np.float32)
        return (audio[low] * (1.0 - frac) + audio[high] * frac).astype(np.float32)


@dataclass
class ForensicsReport:
    """Small container used by scripts and tests."""

    block: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return self.block
