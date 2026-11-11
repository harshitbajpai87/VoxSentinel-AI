"""Vercel serverless entrypoint for the VoxSentinel API.

Why this file exists
--------------------
The real backend needs torch + speechbrain (speaker verification), faster-whisper
(ASR) and ffmpeg/libsndfile (audio decoding). Vercel's Python functions have a
250 MB unzipped bundle limit and no system packages, so that stack cannot be
installed here. This entrypoint therefore boots the SAME FastAPI application in a
deliberately reduced mode:

  * the rule engine, risk scoring, protection metadata and history endpoints all
    work (pure Python + scikit-learn)
  * speaker verification and audio transcription report "unavailable" instead of
    pretending: the heavy libraries are replaced by stubs that raise ImportError
    when called, which the application already handles as an honest
    unavailability (see NullVoiceDetector / NullSpeakerVerifier / status dicts)

The trained lightweight detector can process libsndfile-supported formats using
the same model artifact and soxr HQ resampling. ECAPA and local Whisper are not
available in this slim bundle. For the complete pipeline, run a container
backend (see docs/DEPLOY_VERCEL.md).
"""

from __future__ import annotations

import importlib.abc
import os
import sys
import types
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# --- serverless-safe defaults (set before Settings is built) ------------------
os.environ.setdefault("VOXSENTINEL_ENV", "production")
os.environ.setdefault("VOXSENTINEL_DB_URL", "sqlite:////tmp/voxsentinel.db")
os.environ.setdefault("VOXSENTINEL_ALLOW_DEV_ORIGINS", "false")
os.environ.setdefault("VOXSENTINEL_RETENTION_ENABLED", "false")
# /var/task is read-only; anything the pipeline writes must land in /tmp.
os.environ.setdefault("VOXSENTINEL_TEMP_UPLOAD_DIR", "/tmp/voxsentinel_tmp")
os.environ.setdefault("VOXSENTINEL_REPORT_DIR", "/tmp/reports")
os.environ.setdefault("VOXSENTINEL_DATASET_DIR", "/tmp/data")
# Models live next to the code; point at them explicitly so the working
# directory Vercel picks does not matter.
# The trained detector runs here: its features have a parity-tested numpy
# implementation (app/services/features_numpy.py), so the serverless build scores
# with the SAME artifact as the validated local pipeline instead of a weaker
# heuristic. Speaker verification needs torch/speechbrain and stays unavailable.
_env_root = Path(__file__).resolve().parents[1]
os.environ.setdefault(
    "VOXSENTINEL_DETECTOR_MODEL_PATH",
    os.environ.get("VOXSENTINEL_DETECTOR_MODEL_PATH")
    or str(_env_root / "backend" / "models" / "detector.joblib"),
)
os.environ.setdefault("VOXSENTINEL_SPEAKER_MODEL_DIR", "/tmp/no-artifact/speaker_ecapa")
os.environ.setdefault("VOXSENTINEL_SPEAKER_THRESHOLDS_PATH", "/tmp/no-artifact/speaker_thresholds.json")
os.environ.setdefault("VOXSENTINEL_FORENSICS_ENABLED", "true")
os.environ.setdefault("VOXSENTINEL_MFCC_SPEAKER_ENABLED", "true")


class _Unavailable(types.ModuleType):
    """Module stand-in that fails loudly on use, never on import.

    Needed only for libraries the app imports at module scope (librosa, via
    app.services.features). Using them raises ImportError, which the app already
    treats as "feature unavailable".
    """

    def __getattr__(self, item: str):
        raise ImportError(
            f"{self.__name__} is not installed in the Vercel deployment "
            "(audio stack requires a container host)"
        )


class _Blocked(importlib.abc.MetaPathFinder):
    """Makes `import x` raise, so capability probes report the truth.

    The pipeline decides "is ASR / speaker verification available?" by trying the
    import. A silent stub would make it claim availability it does not have, so
    these names must fail the import instead.
    """

    blocked = frozenset(
        {"torch", "torchaudio", "speechbrain", "faster_whisper", "ctranslate2"}
    )

    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".")[0]
        if root in self.blocked:
            raise ModuleNotFoundError(
                f"{root} is not installed in the Vercel deployment "
                "(audio/speaker stack requires a container host)"
            )
        return None


sys.meta_path.insert(0, _Blocked())

for _name in ("librosa", "numba", "llvmlite"):
    sys.modules.setdefault(_name, _Unavailable(_name))

from fastapi.responses import JSONResponse, Response  # noqa: E402
from app.main import create_app  # noqa: E402  (import after the stubs)

app = create_app()

# --- honest error for audio uploads on this host -----------------------------
# libsndfile decodes common formats; ffmpeg-only containers need an explicit
# capability error. This does not imply every invalid upload is a host problem.
_AUDIO_UNAVAILABLE_MESSAGE = (
    "This clip's container cannot be decoded on this host (no ffmpeg here). "
    "The web UI converts M4A/AAC/WebM/Opus automatically, so upload through the app "
    "or send WAV, MP3, FLAC or OGG directly."
)


@app.middleware("http")
async def _explain_missing_audio_stack(request, call_next):
    response = await call_next(request)
    if request.method != "POST" or not request.url.path.endswith(("/protection/analyze", "/analyze")):
        return response
    if response.status_code != 400:
        return response

    body = b""
    async for chunk in response.body_iterator:
        body += chunk
    if b"invalid_audio" not in body and b"unsupported_media_type" not in body:
        return Response(
            content=body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.headers.get("content-type"),
        )

    import json

    return JSONResponse(
        status_code=415,
        content={"error": {"code": "unsupported_media_type", "message": _AUDIO_UNAVAILABLE_MESSAGE}},
    )

@app.get("/api/_diag")
def _diag() -> dict:
    """Deployment diagnostics: which audio pieces actually work here."""
    import io
    import wave

    import numpy as np

    out: dict = {}
    try:
        import soundfile as sf

        out["soundfile"] = sf.__version__
        out["libsndfile"] = sf.__libsndfile_version__
        out["formats"] = sorted(sf.available_formats())
    except Exception as exc:  # noqa: BLE001
        out["soundfile_error"] = f"{type(exc).__name__}: {exc}"

    try:
        from app.core.config import get_settings
        from app.services.audio_validation import decode_audio

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes((np.zeros(16000 * 3) * 32767).astype("<i2").tobytes())
        samples, rate, channels = decode_audio(buffer.getvalue(), "wav", get_settings())
        out["decode_ok"] = {"samples": int(samples.size), "rate": rate, "channels": channels}
    except Exception as exc:  # noqa: BLE001
        out["decode_error"] = f"{type(exc).__name__}: {exc}"
    return out
