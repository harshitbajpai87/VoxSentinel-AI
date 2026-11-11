"""POST /api/analyze - the unified protection pipeline endpoint.

One endpoint, audio *or* text, every model family in one response:

    transcription        whisper-small / indicwav2vec-hindi / faster-whisper / remote
    speaker_verification ECAPA-TDNN against a saved family voice profile
    acoustic_forensics   detector.joblib + handcrafted cues + XLSR head
    scam_risk            rules + TF-IDF classifier + MuRIL/DeBERTa ensemble
    llm_reasoning        voxisentinel-mistral on Ollama, mistral-small-latest fallback
    risk_badges          the live states a dashboard renders
    explanation          structured JSON threat explanation

Accepted bodies
---------------
multipart/form-data   ``file`` (audio) and/or ``text``, plus ``consent``,
                      ``language``, ``profile_id``
application/json      ``{"text": "...", "language": "hi", "consent": true}``

Everything is additive: ``/api/v1/protection/analyze`` keeps working unchanged,
and this endpoint returns the legacy keys alongside the new sections so an older
screen keeps rendering.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from sqlalchemy.orm import Session

from app.db.session import Database
from app.models.protection import FamilyVoiceProfileORM, ProtectionAnalysisORM
from app.protection.deep_audio import augment
from app.protection.unified import build_unified
from app.services.audio_validation import AudioValidationError
from app.services.ingestion import read_upload_limited

logger = logging.getLogger("voxsentinel.api.unified")

router = APIRouter(prefix="/api", tags=["unified"])


def _session(request: Request):
    db: Database = request.app.state.db
    session: Session = db.session()
    try:
        yield session
    finally:
        session.close()


DbSession = Annotated[Session, Depends(_session)]


def _require_consent(consent: bool | None) -> None:
    if consent is False:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "missing_consent",
                "message": "Consent to analyze this recording is required.",
            },
        )


def _profile_and_reference(db: Session, profile_id: str | None) -> tuple[dict | None, bytes | None]:
    if not profile_id:
        return None, None
    row = db.get(FamilyVoiceProfileORM, profile_id)
    if row is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "profile_not_found", "message": "Family voice profile not found."},
        )
    profile = {
        "id": row.id,
        "created_at": row.created_at,
        "name": row.name,
        "relationship": row.relationship,
        "consent": row.consent,
        "duration_s": row.duration_s,
    }
    path = Path(row.reference_path)
    return profile, (path.read_bytes() if path.exists() else None)


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, AudioValidationError):
        return HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "message": exc.message},
        )
    code = getattr(exc, "code", None)
    if code:
        status = getattr(exc, "status_code", None) or 400
        return HTTPException(status_code=status, detail={"code": code, "message": str(exc)})
    if isinstance(exc, ValueError) and str(exc) == "text is empty":
        return HTTPException(
            status_code=400,
            detail={"code": "empty_text", "message": "The supplied text is empty."},
        )
    logger.warning("unified analyze failed: %s", type(exc).__name__, exc_info=True)
    return HTTPException(
        status_code=500,
        detail={
            "code": "analyze_failed",
            "message": "The protection analysis could not be completed.",
        },
    )


def _persist(db: Session, payload: dict) -> None:
    row = ProtectionAnalysisORM(
        id=payload["id"],
        created_at=payload["created_at"],
        created_at_epoch=payload.get("created_at_epoch")
        or __import__("datetime").datetime.now(__import__("datetime").UTC).timestamp(),
        display_name=payload.get("display_name") or "recording",
        size_bytes=payload.get("size_bytes") or 0,
        duration_s=payload.get("duration_s") or 0.0,
        status=payload.get("status") or "completed",
        risk_category=(payload.get("risk") or {}).get("category") or "INCONCLUSIVE",
        profile_id=payload.get("profile_id"),
        payload_json=payload,
    )
    db.add(row)
    db.commit()


@router.post("/analyze")
async def analyze(
    request: Request,
    db: DbSession,
    file: Annotated[UploadFile | None, File(description="Call recording to analyze")] = None,
    text: Annotated[str | None, Form(description="Transcript to analyze without audio")] = None,
    consent: Annotated[bool | None, Form()] = None,
    language: Annotated[str | None, Form()] = None,
    profile_id: Annotated[str | None, Form()] = None,
) -> dict:
    settings = request.app.state.settings
    service = request.app.state.protection_service
    content_type = (request.headers.get("content-type") or "").lower()

    if "application/json" in content_type:
        try:
            body = await request.json()
        except Exception as exc:  # noqa: BLE001 - bad JSON is a client error
            raise HTTPException(
                status_code=400,
                detail={"code": "invalid_json", "message": "The request body is not valid JSON."},
            ) from exc
        if not isinstance(body, dict):
            raise HTTPException(
                status_code=400,
                detail={"code": "invalid_body", "message": "Expected a JSON object."},
            )
        text = body.get("text") or text
        consent = body.get("consent", consent)
        language = body.get("language") or language
        profile_id = body.get("profile_id") or profile_id
        _require_consent(consent)
        if not (text or "").strip():
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "no_input",
                    "message": "Send an audio file (multipart) or a 'text' field to analyze.",
                },
            )
        try:
            result = service.analyze_text(text, language, display_name="pasted transcript")
        except Exception as exc:  # noqa: BLE001
            raise _http_error(exc) from exc
        payload = build_unified(result)
        payload = augment(payload, None)
        _persist(db, payload)
        return payload

    _require_consent(consent)
    has_file = file is not None and (file.filename or "").strip() != ""
    if not has_file and not (text or "").strip():
        raise HTTPException(
            status_code=400,
            detail={
                "code": "no_input",
                "message": "Attach an audio file or send a 'text' field to analyze.",
            },
        )

    data = None
    if has_file:
        data = await read_upload_limited(file, settings, "audio")
        profile, reference_audio = _profile_and_reference(db, profile_id)
        try:
            result = service.analyze(
                data,
                file.filename or "recording",
                profile=profile,
                reference_audio=reference_audio,
            )
        except Exception as exc:  # noqa: BLE001
            raise _http_error(exc) from exc
    else:
        try:
            result = service.analyze_text(text, language, display_name="pasted transcript")
        except Exception as exc:  # noqa: BLE001
            raise _http_error(exc) from exc

    payload = build_unified(result)
    payload = augment(payload, data, (file.filename if has_file and file else None) or "recording")
    _persist(db, payload)
    return payload


def _reasoning_capability(described: dict) -> dict:
    """Report the provider selected by the live ProtectionService instance."""
    return described.get("contextual_ai") or described.get("gemini") or {}


def _synthetic_capability() -> dict:
    """Per-model status for the synthetic-audio family."""
    from app.services.aasist import get_detector

    models: list[dict] = []
    status = get_detector().status()
    models.append(
        {
            "model": status.get("model"),
            "branch": "aasist",
            "variant": status.get("variant"),
            "status": status.get("status"),
            "threshold": status.get("threshold"),
            "calibrated": status.get("calibrated"),
            "detail": status.get("detail"),
        }
    )
    try:
        from app.services import ssl_spoof

        for branch in ssl_spoof.describe()["branches"]:
            models.append(
                {
                    "model": branch.get("model"),
                    "branch": branch.get("branch"),
                    "status": branch.get("status"),
                    "threshold": branch.get("threshold"),
                    "calibrated": branch.get("calibrated"),
                    "detail": branch.get("detail"),
                }
            )
    except Exception as exc:  # noqa: BLE001 - a probe must never 500
        models.append(
            {
                "model": "wavlm/xlsr spoof heads",
                "branch": "ssl_spoof",
                "status": "failed",
                "detail": f"{type(exc).__name__}: {str(exc)[:160]}",
            }
        )
    return {
        "models": models,
        "available": sum(1 for m in models if m.get("status") == "completed"),
        "total": len(models),
    }


def _fusion_capability() -> dict:
    from app.protection import fusion as fusion_mod

    return {
        "version": "1.0",
        "families": list(fusion_mod.FAMILIES),
        "levels": list(fusion_mod.LEVELS),
        "policy": fusion_mod.POLICY,
        "levels_order": fusion_mod.LEVEL_ORDER,
    }


def _diarization_capability() -> dict:
    try:
        from app.services.diarization import get_diarizer

        return get_diarizer().status()
    except Exception as exc:  # noqa: BLE001 - a probe must never 500
        return {"status": "failed", "detail": f"{type(exc).__name__}: {str(exc)[:160]}"}


@router.get("/capabilities")
def capabilities(request: Request) -> dict:
    """Which model families are actually live on this host.

    The dashboard reads this to label every card honestly instead of implying a
    model ran when it did not.
    """
    service = request.app.state.protection_service
    described = service.describe()
    return {
        "schema_version": "1.0",
        "endpoint": "POST /api/analyze",
        "models": {
            "transcription": described.get("transcription"),
            "speaker_verification": described.get("speaker_verifier") or {
                "ready": described.get("speaker_verifier_ready"),
                "backend": "unknown",
                "model": None,
                "embedding_dim": None,
            },
            "acoustic_forensics": described.get("acoustic_forensics"),
            "scam_text": {
                "trained_classifier": (described.get("text_ensemble") or {})
                or {"detail": "TF-IDF classifier ships with the API"},
                "ensemble": described.get("text_ensemble"),
            },
            "synthetic_detection": _synthetic_capability(),
            "speaker_diarization": _diarization_capability(),
            "llm_reasoning": _reasoning_capability(described),
        },
        "detector_ready": described.get("detector_ready"),
        "evidence_fusion": _fusion_capability(),
        "input_kinds": ["audio", "text"],
    }
