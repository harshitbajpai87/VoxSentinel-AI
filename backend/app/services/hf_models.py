"""Local-first resolution of Hugging Face model artifacts.

Every heavy model in the unified pipeline (Whisper ASR, IndicWav2Vec2 Hindi ASR,
wav2vec2 forensics, MuRIL / DeBERTa text heads) follows the same rule:

    a local directory wins, otherwise the Hub id is used only when downloads
    are explicitly allowed, otherwise the component reports `unavailable`
    together with the reason.

Nothing here downloads anything by itself: it resolves a source string and
answers "is this model usable on this host". Model loading stays in the
component that owns it, so a broken optional model can never break the API.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("voxsentinel.hf")

BACKEND_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BACKEND_DIR / "models"
HF_CACHE_SUBDIR = MODELS_DIR / "hf"

#: A model directory is only trusted when it carries a real config file.
_CONFIG_MARKERS = ("config.json",)


def _truthy(raw: str | None, default: bool) -> bool:
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def downloads_allowed() -> bool:
    """Whether the Hub may be contacted for a missing model.

    Default is "yes" for a normal install (the first request pays the download,
    then the Hub cache serves it) and "no" for the serverless shim, which sets
    VOXSENTINEL_HF_DOWNLOAD=false because its bundle cannot hold these models.
    """
    if os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in {"1", "true", "yes"}:
        return False
    if os.environ.get("TRANSFORMERS_OFFLINE", "").strip().lower() in {"1", "true", "yes"}:
        return False
    return _truthy(os.environ.get("VOXSENTINEL_HF_DOWNLOAD"), True)


def transformers_available() -> bool:
    try:
        import transformers  # noqa: F401
    except Exception:  # pragma: no cover - environment dependent
        return False
    return True


def local_model_dir(slug: str) -> Path:
    return HF_CACHE_SUBDIR / slug


@dataclass(frozen=True)
class ModelSource:
    """Where a model should be loaded from, or why it cannot be."""

    model_id: str
    source: str | None
    local: bool
    reason: str | None = None

    @property
    def available(self) -> bool:
        return self.source is not None


def _looks_like_model_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any((path / marker).exists() for marker in _CONFIG_MARKERS)


GATED_MODELS = {
    "ai4bharat/indicwav2vec-hindi",
}


def has_hf_token() -> bool:
    return bool(
        (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    )


def resolve_model(model_id: str, slug: str) -> ModelSource:
    """Resolve ``model_id`` to a locally usable source, never downloading here."""
    if not transformers_available():
        return ModelSource(
            model_id=model_id,
            source=None,
            local=False,
            reason=(
                "`transformers` is not installed on this host, so this model cannot run. "
                "Install the unified requirements (`pip install -r backend/requirements.txt`)."
            ),
        )
    local = local_model_dir(slug)
    if _looks_like_model_dir(local):
        return ModelSource(model_id=model_id, source=str(local), local=True)
    if model_id in GATED_MODELS and not has_hf_token():
        # A gated repo returns HTTP 401 at load time; saying "available" here and
        # failing on the first request would be a silent lie, so refuse upfront.
        return ModelSource(
            model_id=model_id,
            source=None,
            local=False,
            reason=(
                f"{model_id} is a gated Hugging Face repository: accept the model terms "
                "and set HF_TOKEN, or place a copy under "
                f"models/hf/{slug}. Until then this engine stays unavailable."
            ),
        )
    if downloads_allowed():
        return ModelSource(model_id=model_id, source=model_id, local=False)
    return ModelSource(
        model_id=model_id,
        source=None,
        local=False,
        reason=(
            f"No local copy under models/hf/{slug} and Hub downloads are disabled "
            "(VOXSENTINEL_HF_DOWNLOAD=false). Place the model there or enable downloads."
        ),
    )


def describe_source(source: ModelSource) -> dict:
    return {
        "model_id": source.model_id,
        "source": source.source,
        "local": source.local,
        "available": source.available,
        "reason": source.reason,
    }
