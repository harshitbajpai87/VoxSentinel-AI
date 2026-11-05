"""Family Emergency & Scam Protection API (additive module).

All routes live under /api/v1/protection and are isolated from the existing
analysis endpoints. Every route requires explicit consent acknowledgment.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.session import Database
from app.models.protection import FamilyVoiceProfileORM, ProtectionAnalysisORM

router = APIRouter(prefix="/protection", tags=["protection"])

_PROFILE_DIR = Path("data/family_voices")


def _session(request: Request):
    db: Database = request.app.state.db
    session: Session = db.session()
    try:
        yield session
    finally:
        session.close()


DbSession = Annotated[Session, Depends(_session)]


def _now() -> tuple[str, float]:
    now = datetime.now(UTC)
    return now.isoformat(), now.timestamp()


def _require_consent(consent: bool | None) -> None:
    if not consent:
        raise HTTPException(
            status_code=400,
            detail={"code": "missing_consent", "message": "Consent is required to analyze this recording."},
        )


def _profile_dict(row: FamilyVoiceProfileORM) -> dict:
    return {
        "id": row.id,
        "created_at": row.created_at,
        "name": row.name,
        "relationship": row.relationship,
        "consent": row.consent,
        "duration_s": row.duration_s,
    }


# ---------------------------------------------------------------- analyses


@router.post("/analyze")
async def analyze(
    request: Request,
    db: DbSession,
    file: Annotated[UploadFile, File(description="Suspicious call recording")],
    consent: Annotated[bool, Form()],
    profile_id: Annotated[str | None, Form()] = None,
) -> dict:
    _require_consent(consent)
    data = await file.read()
    service = request.app.state.protection_service

    profile = None
    reference_audio = None
    if profile_id:
        row = db.get(FamilyVoiceProfileORM, profile_id)
        if row is None:
            raise HTTPException(
                status_code=404,
                detail={"code": "profile_not_found", "message": "Family voice profile not found."},
            )
        profile = _profile_dict(row)
        path = Path(row.reference_path)
        if path.exists():
            reference_audio = path.read_bytes()

    try:
        result = service.analyze(
            data, file.filename or "recording", profile=profile, reference_audio=reference_audio
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - validation errors map to 400
        code = getattr(exc, "code", None)
        if code:
            raise HTTPException(status_code=400, detail={"code": code, "message": str(exc)}) from exc
        raise HTTPException(
            status_code=500,
            detail={
                "code": "protection_failed",
                "message": "The protection analysis could not be completed.",
            },
        ) from exc

    created_at, created_epoch = _now()
    row = ProtectionAnalysisORM(
        id=result["id"],
        created_at=created_at,
        created_at_epoch=created_epoch,
        display_name=result["display_name"],
        size_bytes=result["size_bytes"],
        duration_s=result["duration_s"],
        status=result["status"],
        risk_category=result["risk"]["category"],
        profile_id=result.get("profile_id"),
        payload_json=result,
    )
    db.add(row)
    db.commit()
    return result


@router.get("/analyses")
def list_analyses(db: DbSession, limit: int = 50) -> dict:
    rows = (
        db.execute(
            select(ProtectionAnalysisORM)
            .order_by(ProtectionAnalysisORM.created_at_epoch.desc())
            .limit(max(1, min(limit, 200)))
        )
        .scalars()
        .all()
    )
    return {
        "analyses": [
            {
                "id": r.id,
                "created_at": r.created_at,
                "display_name": r.display_name,
                "duration_s": r.duration_s,
                "status": r.status,
                "risk_category": r.risk_category,
                "profile_id": r.profile_id,
            }
            for r in rows
        ],
        "count": len(rows),
    }


@router.get("/analyses/{analysis_id}")
def get_analysis(analysis_id: str, db: DbSession) -> dict:
    row = db.get(ProtectionAnalysisORM, analysis_id)
    if row is None:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "Analysis not found."})
    return row.payload_json


@router.delete("/analyses/{analysis_id}")
def delete_analysis(analysis_id: str, db: DbSession) -> dict:
    row = db.get(ProtectionAnalysisORM, analysis_id)
    if row is None:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "Analysis not found."})
    db.delete(row)
    db.commit()
    return {"deleted": analysis_id}


@router.get("/analyses/{analysis_id}/report")
def export_report(analysis_id: str, db: DbSession) -> Response:
    """JSON report download (transcript, evidence, risk reasoning, versions)."""
    row = db.get(ProtectionAnalysisORM, analysis_id)
    if row is None:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "Analysis not found."})
    body = json.dumps(row.payload_json, ensure_ascii=False, indent=2)
    return Response(
        content=body,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="voxsentinel-protection-{row.id}.json"'},
    )


# ------------------------------------------------------------- family voices


@router.post("/family-voices")
async def create_profile(
    request: Request,
    db: DbSession,
    name: Annotated[str, Form()],
    relationship: Annotated[str, Form()],
    consent: Annotated[bool, Form()],
    reference: Annotated[UploadFile, File(description="Reference recording of the family member")],
) -> dict:
    if not consent:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "missing_authorization",
                "message": "The family member must have consented to this reference recording.",
            },
        )
    data = await reference.read()
    settings = request.app.state.settings
    from app.services.audio_validation import validate_and_decode

    try:
        decoded = validate_and_decode(data, reference.filename or "reference.wav", settings)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=400, detail={"code": "invalid_reference", "message": str(exc)}
        ) from exc

    profile_id = uuid.uuid4().hex[:16]
    _PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    path = _PROFILE_DIR / f"{profile_id}.wav"
    import soundfile as sf

    sf.write(str(path), decoded.samples, decoded.sample_rate, format="WAV")

    created_at, created_epoch = _now()
    row = FamilyVoiceProfileORM(
        id=profile_id,
        created_at=created_at,
        created_at_epoch=created_epoch,
        name=name.strip()[:120] or "Family member",
        relationship=(relationship.strip()[:64] or "family"),
        consent=True,
        reference_path=str(path),
        duration_s=decoded.duration_s,
    )
    db.add(row)
    db.commit()
    return _profile_dict(row)


@router.get("/family-voices")
def list_profiles(db: DbSession) -> dict:
    rows = (
        db.execute(select(FamilyVoiceProfileORM).order_by(FamilyVoiceProfileORM.created_at_epoch.desc()))
        .scalars()
        .all()
    )
    return {"profiles": [_profile_dict(r) for r in rows], "count": len(rows)}


@router.delete("/family-voices/{profile_id}")
def delete_profile(profile_id: str, db: DbSession) -> dict:
    row = db.get(FamilyVoiceProfileORM, profile_id)
    if row is None:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "Profile not found."})
    Path(row.reference_path).unlink(missing_ok=True)
    db.delete(row)
    db.commit()
    return {"deleted": profile_id}


@router.get("/capabilities")
def capabilities(request: Request) -> dict:
    return request.app.state.protection_service.describe()
