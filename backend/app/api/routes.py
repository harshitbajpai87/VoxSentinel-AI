"""HTTP routes implementing docs/API_CONTRACT.md (v1).

Everything here is deliberately thin: validation lives in services, decisions
in the risk engine, persistence in the repository.
"""
from __future__ import annotations

import base64
import csv
import io
import json
import logging
import uuid
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import Response
from sqlalchemy.orm import Session

from app.schemas.analysis import (
    AnalysisDecision,
    AnalysisEvidence,
    AnalysisInput,
    AnalysisListResponse,
    AnalysisModelInfo,
    AnalysisRecordOut,
    AnalysisScores,
    AuditEventOut,
    AuditListResponse,
    EvaluationJobOut,
    EvaluationMetricsResponse,
    HealthResponse,
    MeasuredFeature,
    ModelInfoOut,
    ModelsResponse,
    RetentionPolicyOut,
    SpeakerVerifyResponse,
    WindowScore,
)
from app.schemas.synthesis import (
    ProviderCapabilitiesOut,
    ProvidersResponse,
    SynthesizeRequest,
    VoiceChatTurnOut,
)
from app.services.audio_validation import (
    AudioValidationError,
    DecodedAudio,
)
from app.services.audit import list_events, log_event
from app.services.detector import DetectorInferenceError, DetectorResult, ModelNotReadyError
from app.services.evaluation import EvaluationError, EvaluationJobManager
from app.services.gnani_client import (
    GnaniAuthError,
    GnaniConnectionError,
    GnaniMalformedResponseError,
    GnaniProviderError,
    GnaniRateLimitError,
    GnaniTimeoutError,
    GnaniUnavailableError,
    GnaniValidationError,
    TTSRequest,
)
from app.services.history import AnalysisRepository, utc_now_iso
from app.services.ingestion import (
    audio_capabilities,
    describe_environment,
    effective_upload_limit,
    ingest_audio,
    probe_decoders,
)
from app.services.model_loader import artifact_status
from app.services.retention import last_cleanup_info, run_cleanup
from app.services.risk_engine import STANDARD_LIMITATIONS
from app.services.speaker import speaker_backend_name
from app.services.voicechat_client import (
    VoiceChatConnectionError,
    VoiceChatProtocolError,
    VoiceChatServerError,
    VoiceChatTimeoutError,
    VoiceChatUnavailableError,
    VoiceChatValidationError,
)

logger = logging.getLogger("voxsentinel.routes")

health_router = APIRouter()
api_router = APIRouter(prefix="/api/v1")


def get_db(request: Request) -> Iterator[Session]:
    """One session per request; always closed."""
    session = request.app.state.db.session()
    try:
        yield session
    finally:
        session.close()


def size_limit_message(settings) -> str:
    """Kept for callers/tests; now states the deployment's real ceiling."""
    from app.services.ingestion import limit_message

    return limit_message(settings)


def _raise_validation_error(exc: AudioValidationError) -> None:
    raise HTTPException(status_code=exc.status_code, detail={"code": exc.code, "message": exc.message})


async def _validated_audio(
    file: UploadFile, request: Request, label: str = "audio"
) -> tuple[DecodedAudio, str]:
    """Delegate to the ONE shared ingestion layer every audio route uses."""
    decoded, display_name, _size = await ingest_audio(file, request.app.state.settings, label)
    return decoded, display_name


def _evidence(decoded: DecodedAudio, result: DetectorResult | None = None) -> AnalysisEvidence:
    """Measured features are facts about the input, not model inference."""
    return AnalysisEvidence(
        measured_features=[
            MeasuredFeature(name="duration", value=round(decoded.duration_s, 3), unit="s"),
            MeasuredFeature(name="sample_rate", value=float(decoded.sample_rate), unit="Hz"),
            MeasuredFeature(name="channels", value=float(decoded.channels), unit="count"),
            MeasuredFeature(name="peak_level", value=round(decoded.peak, 6), unit="ratio"),
            MeasuredFeature(name="rms_level", value=round(decoded.rms, 6), unit="ratio"),
        ],
        # Present only when the loaded model supports an explanation method.
        model_explanation=result.explanation if result is not None else None,
        # Same-model scores over consecutive segments (evidence, not the verdict).
        time_window_scores=[
            WindowScore(**w) for w in (result.window_scores if result is not None else ())
        ],
        limitations=list(STANDARD_LIMITATIONS),
    )


def _contextual_ready(contextual: dict) -> bool:
    """True only when the reasoning chain has a member that can actually answer.

    A configured router with a stopped Ollama daemon and no hosted fallback key
    used to report ready; it could not answer a single request.
    """
    if not contextual.get("configured"):
        return False
    chain = contextual.get("resolved_chain") or []
    if not chain:
        return True
    return any(member.get("reachable", member.get("configured")) for member in chain)


@health_router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Readiness of the API, database and each model, reported separately.

    A running API never claims a model is ready when none is loaded, and this
    endpoint must not crash when an optional model is missing.
    """
    settings = request.app.state.settings
    db_ok = request.app.state.db.ping()
    decoders = probe_decoders(settings)
    protection = request.app.state.protection_service.describe()
    transcription = protection.get("transcription") or {}
    contextual = protection.get("contextual_ai") or {}
    # Optional external services are REPORTED, never required: an unavailable
    # provider must not make the API look offline.
    services = {
        "detector": request.app.state.detector_service.is_ready(),
        "speaker_verifier": request.app.state.speaker_service.is_ready(),
        "transcription": bool(transcription.get("available")),
        "contextual_ai": _contextual_ready(contextual),
        "speech_generation": bool(getattr(request.app.state.gnani_service, "enabled", False)),
        "voicechat": bool(getattr(request.app.state.voicechat_service, "enabled", False)),
        # Which speaker backend actually runs here: 'ecapa', 'mfcc_statistics'
        # or 'none'. `speaker_model_ready` alone cannot tell you that.
        "speaker_verifier_backend": speaker_backend_name(
            request.app.state.speaker_service
        ),
        # Why the detector artifact did or did not load, verbatim, including a
        # missing module name when the pickle needs a dependency this host lacks.
        "detector_artifact": artifact_status(),
    }
    return HealthResponse(
        status="ok" if db_ok else "degraded",
        backend_ready=db_ok,
        model_ready=request.app.state.detector_service.is_ready(),
        speaker_model_ready=request.app.state.speaker_service.is_ready(),
        version=settings.api_version,
        mode=settings.app_mode,
        audio_decoder_ready=bool(decoders.ffmpeg and decoders.ffprobe),
        max_upload_bytes=settings.max_upload_bytes,
        effective_max_upload_bytes=effective_upload_limit(settings),
        audio=audio_capabilities(settings),
        services=services,
        environment=describe_environment(),
    )


@api_router.get("/models", response_model=ModelsResponse)
def list_models(request: Request) -> ModelsResponse:
    """Truthful metadata for the installed detection and speaker pipelines."""
    return ModelsResponse(
        models=[
            ModelInfoOut(**asdict(request.app.state.detector_service.describe())),
            ModelInfoOut(**asdict(request.app.state.speaker_service.describe())),
        ]
    )


@api_router.post("/analyze", response_model=AnalysisRecordOut)
async def analyze_audio(
    request: Request,
    file: Annotated[UploadFile, File(description="Audio file to analyze")],
    consent: Annotated[bool, Form(description="Must be true: the user is authorized to analyze this audio")],
    db: Annotated[Session, Depends(get_db)],
    persist: Annotated[str | None, Form(description=(
        "'false' = compute but do NOT store in history (used for provisional "
        "live-recording windows; the UI makes this explicit). Default: stored."
    ))] = None,
) -> AnalysisRecordOut:
    """Validate, measure and (when a real model is installed) classify audio.

    Without a trained detector this returns 503 `model_unavailable`. It never
    returns a fabricated score, verdict or confidence.
    """
    if not consent:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "missing_consent",
                "message": "Consent is required before analyzing audio.",
            },
        )

    try:
        decoded, display_name = await _validated_audio(file, request, "audio")
    except HTTPException as exc:
        code = str(((exc.detail or {}).get("code")) if isinstance(exc.detail, dict) else "invalid")
        log_event(db, "analysis", "request", "failed",
                  request_id=getattr(request.state, "request_id", None),
                  metadata={"error_code": code, "http_status": exc.status_code})
        raise
    detector = request.app.state.detector_service

    started = perf_counter()
    try:
        result: DetectorResult = detector.analyze(
            decoded.samples, decoded.sample_rate, decoded.channels
        )
    except ModelNotReadyError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "model_unavailable",
                "message": (
                    "No trained synthetic-speech detection model is installed on this server, "
                    "so no analysis was performed. Real detection becomes available once a "
                    "trained model is integrated (Phase 5)."
                ),
            },
        ) from exc
    except DetectorInferenceError as exc:
        log_event(db, "analysis", "inference", "failed",
                  request_id=getattr(request.state, "request_id", None),
                  metadata={"error_code": "model_inference_error"})
        raise HTTPException(
            status_code=503,
            detail={
                "code": "model_unavailable",
                "message": (
                    "The detection model could not produce a valid result for this audio. "
                    "No analysis was recorded."
                ),
            },
        ) from exc
    latency_ms = int((perf_counter() - started) * 1000)

    decision = request.app.state.risk_engine.decide(result, decoded)
    decision_model = AnalysisDecision(
        category=decision.category,
        label=decision.label,
        reasons=decision.reasons,
        recommendations=decision.recommendations,
    )
    record = AnalysisRecordOut(
        id=uuid.uuid4().hex,
        created_at=utc_now_iso(),
        input=AnalysisInput(
            display_name=display_name,
            duration_s=round(decoded.duration_s, 3),
            sample_rate_hz=decoded.sample_rate,
            channels=decoded.channels,
            quality_warnings=list(decoded.warnings),
        ),
        model=AnalysisModelInfo(
            name=result.model_name,
            version=result.model_version,
            mode=result.model_mode,
        ),
        scores=AnalysisScores(
            raw_score=result.raw_score,
            raw_score_definition=result.raw_score_definition,
            calibrated_probability=result.calibrated_probability,
            uncertainty=result.uncertainty,
            uncertainty_method=result.uncertainty_method,
        ),
        decision=decision_model,
        evidence=_evidence(decoded, result),
        latency_ms=latency_ms,
    )
    request_id = getattr(request.state, "request_id", None)
    should_persist = (persist or "true").strip().lower() not in ("false", "0", "no")
    if not should_persist:
        # Provisional (e.g. live-recording window): computed, never stored.
        log_event(db, "analysis", "provisional_window", "skipped_persist",
                  resource_id=record.id, request_id=request_id,
                  metadata={"model_version": record.model.version,
                            "category": record.decision.category})
        return record
    stored = AnalysisRepository(db).create(record)
    log_event(db, "analysis", "create", "completed",
              resource_id=stored.id, request_id=request_id,
              metadata={"model_version": stored.model.version,
                        "category": stored.decision.category,
                        "latency_ms": stored.latency_ms})
    # The ORIGINAL record carries window evidence, which is response-only:
    # returning the stored copy would silently drop it.
    return record


@api_router.post("/speaker-verify", response_model=SpeakerVerifyResponse)
async def speaker_verify(
    request: Request,
    reference: Annotated[UploadFile, File(description="Authorized reference recording")],
    test: Annotated[UploadFile, File(description="Recording to compare")],
    consent: Annotated[bool, Form()],
    reference_authorized: Annotated[bool, Form()],
    db: Annotated[Session, Depends(get_db)],
) -> SpeakerVerifyResponse:
    """Compare two recordings when a speaker-embedding model exists.

    Without the model the response is an explicit `unavailable` status with a
    null similarity — never a fabricated score.
    """
    if not consent:
        raise HTTPException(
            status_code=400,
            detail={"code": "missing_consent", "message": "Consent is required for this comparison."},
        )
    if not reference_authorized:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "missing_authorization",
                "message": "The reference recording must be explicitly authorized before comparison.",
            },
        )

    # Each file is validated and size-checked INDEPENDENTLY, so an oversized
    # reference is named as such instead of a generic "request too large".
    reference_audio, reference_name = await _validated_audio(reference, request, "reference")
    test_audio, test_name = await _validated_audio(test, request, "test")

    service = request.app.state.speaker_service
    # A missing/broken embedding model yields the documented explicit
    # `unavailable` result (HTTP 200, similarity null) instead of an opaque 500;
    # the service owns that decision so no route can accidentally fabricate one.
    result = service.compare(
        reference_audio.samples,
        reference_audio.sample_rate,
        test_audio.samples,
        test_audio.sample_rate,
    )
    log_event(db, "speaker_comparison", "compare", result.status,
              request_id=getattr(request.state, "request_id", None),
              metadata={"similarity": result.similarity, "outcome": result.outcome,
                        "embedding_model": result.embedding_model})
    # Include the verifier's own warnings: they carry "no voiced region detected"
    # and the uncalibrated-threshold notice, which were previously dropped.
    warnings = (
        list(reference_audio.warnings)
        + list(test_audio.warnings)
        + list(result.warnings)
    )
    return SpeakerVerifyResponse(
        status=result.status,
        similarity=result.similarity,
        threshold=result.threshold,
        outcome=result.outcome,
        embedding_model=result.embedding_model,
        warnings=warnings,
        limitations=list(result.limitations),
    )


@api_router.get("/analyses", response_model=AnalysisListResponse)
def list_analyses(
    db: Annotated[Session, Depends(get_db)],
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    category: str | None = Query(default=None, description="Server-side decision-category filter"),
) -> AnalysisListResponse:
    """Completed analyses only, newest first; filter executes in the database."""
    records, total = AnalysisRepository(db).list(limit=limit, offset=offset, category=category)
    return AnalysisListResponse(analyses=records, total=total)


@api_router.get("/analyses/export.csv")
def export_analyses_csv(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    category: str | None = Query(default=None, description="Filter by decision category"),
) -> object:
    """Stream the persisted history as UTF-8 CSV (with BOM for spreadsheets).

    Rows come from the DATABASE, not a browser array. Cells are escaped with
    the csv module (injection-safe). Raw audio and audio bytes are never
    included. Logged in the audit trail after a successful export.
    """
    from fastapi.responses import StreamingResponse

    records, total = AnalysisRepository(db).list(limit=10000, offset=0, category=category)

    def _clean(value: object) -> str:
        return "" if value is None else str(value)

    def generate() -> Iterator[str]:
        buffer = io.StringIO()
        writer = csv.writer(buffer, quoting=csv.QUOTE_MINIMAL, lineterminator="\r\n")
        writer.writerow([
            "analysis_id", "created_at_utc", "file_name", "duration_s", "sample_rate_hz",
            "channels", "quality_warnings", "model_name", "model_version",
            "raw_score", "raw_score_definition", "decision_category", "decision_label",
            "model_explanation", "latency_ms",
        ])
        yield "\ufeff"  # UTF-8 BOM so Excel opens UTF-8 correctly
        yield buffer.getvalue()
        for record in records:
            buffer = io.StringIO()
            writer = csv.writer(buffer, quoting=csv.QUOTE_MINIMAL, lineterminator="\r\n")
            writer.writerow([
                record.id, record.created_at,
                record.input.display_name, record.input.duration_s,
                record.input.sample_rate_hz, record.input.channels,
                ";".join(record.input.quality_warnings),
                record.model.name, record.model.version,
                record.scores.raw_score, record.scores.raw_score_definition,
                record.decision.category, record.decision.label,
                (record.evidence.model_explanation if record.evidence else ""),
                record.latency_ms,
            ])
            yield buffer.getvalue()

    log_event(db, "data_export", "export_csv", "success",
              request_id=getattr(request.state, "request_id", None),
              metadata={"rows": total, "category": category or "all"})
    filename = f"voxsentinel-history-{utc_now_iso().replace(':', '')}.csv"
    return StreamingResponse(
        generate(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@api_router.get("/analyses/{analysis_id}", response_model=AnalysisRecordOut)
def get_analysis(analysis_id: str, db: Annotated[Session, Depends(get_db)]) -> AnalysisRecordOut:
    record = AnalysisRepository(db).get(analysis_id)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "not_found", "message": "No analysis exists with that id."},
        )
    return record


@api_router.get("/evaluation")
def evaluation_summary(request: Request) -> dict[str, object]:
    """Summary of the latest REAL evaluation run; honest when none has run."""
    settings = request.app.state.settings
    report_path = Path(settings.report_dir) / "evaluation_latest.json"
    if not report_path.is_file():
        return {
            "evaluated": False,
            "message": (
                "No evaluation has been run yet. POST /api/v1/evaluation/run starts a real "
                "inference job over the configured eval partition."
            ),
        }
    report = json.loads(report_path.read_text())
    return {
        "evaluated": True,
        "model": report.get("model"),
        "split": report.get("split"),
        "evaluated_at": report.get("evaluated_at"),
        "metrics": report.get("metrics"),
    }


@api_router.get("/evaluation/metrics", response_model=EvaluationMetricsResponse)
def evaluation_metrics(request: Request) -> EvaluationMetricsResponse:
    """Measured metrics only. Honest: evaluated=false until a real run is recorded."""
    settings = request.app.state.settings
    report_path = Path(settings.report_dir) / "evaluation_latest.json"
    if not report_path.is_file():
        return EvaluationMetricsResponse(evaluated=False)
    report = json.loads(report_path.read_text())
    return EvaluationMetricsResponse(
        evaluated=True,
        metrics=report.get("metrics"),
        evaluated_at=report.get("evaluated_at"),
    )


# ---------------------------------------------------------------------------
# Module 4 — CSV export of the authorized history (DB-backed, escaped)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Module 7 — Audit trail (append-only from the API's perspective)
# ---------------------------------------------------------------------------
@api_router.get("/audit", response_model=AuditListResponse)
def list_audit_events(
    db: Annotated[Session, Depends(get_db)],
    event_type: str | None = Query(default=None),
    outcome: str | None = Query(default=None),
    since: str | None = Query(default=None, description="ISO-8601 UTC lower bound"),
    until: str | None = Query(default=None, description="ISO-8601 UTC upper bound"),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> AuditListResponse:
    """Audit events are read-only through this endpoint; never editable."""
    from datetime import datetime

    def _epoch(value: str | None) -> float | None:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed.timestamp()
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "invalid_date",
                    "message": "Dates must be ISO-8601 (e.g. 2026-10-23T00:00:00Z).",
                },
            ) from exc

    events, total = list_events(
        db, event_type=event_type, outcome=outcome,
        since_epoch=_epoch(since), until_epoch=_epoch(until),
        limit=limit, offset=offset,
    )
    return AuditListResponse(
        events=[AuditEventOut(
            id=e["id"], created_at=e["created_at"], event_type=e["event_type"],
            action=e["action"], outcome=e["outcome"], actor=e["actor"],
            resource_id=e["resource_id"], request_id=e["request_id"], metadata=e["metadata"],
        ) for e in events],
        total=total,
    )


# ---------------------------------------------------------------------------
# Module 5 — Evaluation run (real background inference job)
# ---------------------------------------------------------------------------
def _evaluation_manager(request: Request) -> EvaluationJobManager:
    manager = getattr(request.app.state, "evaluation_manager", None)
    if manager is None:
        manager = EvaluationJobManager(
            request.app.state.settings,
            request.app.state.detector_service,
        )
        request.app.state.evaluation_manager = manager
    return manager


@api_router.post("/evaluation/run", response_model=EvaluationJobOut, status_code=202)
def start_evaluation_run(request: Request, partition: str | None = Query(default=None)) -> EvaluationJobOut:
    """Start a REAL evaluation: manifest -> features -> model -> metrics.

    The job runs real inference in a background thread; poll
    GET /api/v1/evaluation/run/{job_id} for progress and results.
    """
    manager = _evaluation_manager(request)
    try:
        state = manager.start(partition)
    except EvaluationError as exc:
        raise HTTPException(status_code=409, detail={"code": "evaluation_busy", "message": str(exc)}) from exc
    return EvaluationJobOut(**{k: v for k, v in state.as_dict().items() if k != "report"})


@api_router.get("/evaluation/run/{job_id}", response_model=EvaluationJobOut)
def evaluation_run_status(job_id: str, request: Request) -> EvaluationJobOut:
    state = _evaluation_manager(request).get(job_id)
    if state is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "job_not_found", "message": "Unknown evaluation job id."},
        )
    return EvaluationJobOut(**{k: v for k, v in state.as_dict().items() if k != "report"})


# ---------------------------------------------------------------------------
# Module 7 — Retention policy status and execution (dry-run supported)
# ---------------------------------------------------------------------------
@api_router.get("/admin/retention", response_model=RetentionPolicyOut)
def retention_status(
    db: Annotated[Session, Depends(get_db)], request: Request,
) -> RetentionPolicyOut:
    settings = request.app.state.settings
    info = last_cleanup_info(db).get("last_cleanup")
    return RetentionPolicyOut(
        enabled=settings.retention_enabled,
        analyses_days=settings.retention_analyses_days,
        audit_days=settings.retention_audit_days,
        temp_file_max_age_minutes=settings.temp_max_age_seconds / 60.0,
        interval_hours=settings.retention_interval_hours,
        last_run=(AuditEventOut(
            id="", created_at=info["created_at"], event_type="data_retention",
            action=info["action"], outcome=info["action"], actor="system",
            resource_id=None, request_id=None, metadata=info["metadata"],
        ) if info else None),
    )


@api_router.post("/admin/retention/run")
def run_retention_now(
    request: Request, db: Annotated[Session, Depends(get_db)],
    dry_run: bool = Query(default=True, description=(
        "Preview what would be deleted without deleting it (default)")),
) -> dict[str, object]:
    """Execute the retention policy now. Defaults to DRY-RUN (preview only)."""
    from app.db.session import Database

    settings = request.app.state.settings
    database = Database(settings.db_url)
    try:
        session = database.session()
        try:
            outcome = run_cleanup(settings, session, dry_run=dry_run)
        finally:
            session.close()
        return {
            "dry_run": dry_run,
            "analyses_deleted": outcome.analyses_deleted,
            "audit_events_deleted": outcome.audit_events_deleted,
            "removed_temp_files": 0,
            "cutoff_analyses_iso": outcome.cutoff_analyses_iso,
            "cutoff_audit_iso": outcome.cutoff_audit_iso,
            "message": (
                "Preview complete — nothing was deleted (run with dry_run=false to enforce)."
                if dry_run else "Retention executed; eligible records were permanently deleted."
            ),
        }
    finally:
        database.dispose()


# ---------------------------------------------------------------------------
# Phase 6 — External speech generation (Gnani.ai / Vach) and provider
# capability reporting. Generation is a SEPARATE capability from detection:
# the provider's documented API offers TTS only (verified 2026-10-25), so the
# local classifier remains the sole detection source. Generated audio is
# always labelled synthetic and is never mixed into evaluation sets.
# ---------------------------------------------------------------------------
@api_router.get("/providers", response_model=ProvidersResponse)
def providers(request: Request) -> ProvidersResponse:
    detector = request.app.state.detector_service
    capabilities: list[dict[str, object]] = []
    for service_attr in ("gnani_service", "voicechat_service"):
        service = getattr(request.app.state, service_attr, None)
        if service is not None:
            capabilities.append(service.capabilities())
    return ProvidersResponse(
        detection=asdict(detector.describe()),
        detection_note=(
            "Detection is performed by VoxSentinel's local classifier. "
            "No external provider is used for detection."
        ),
        generations=[ProviderCapabilitiesOut(**cap) for cap in capabilities],
    )


def _rate_limit_message(reason: str, settings, per_minute: int | None = None) -> tuple[int, dict[str, str]]:
    if reason == "daily":
        return 429, {
            "code": "rate_limited",
            "message": (
                "The daily speech-generation quota has been reached. "
                "Try again tomorrow or ask an administrator to raise it."
            ),
        }
    limit = per_minute if per_minute is not None else settings.gnani_rate_limit_per_minute
    return 429, {
        "code": "rate_limited",
        "message": (
            f"Too many speech-generation requests. Limit: {limit} per minute."
        ),
    }


def _store_generated_sample(audio, settings, request: Request) -> dict[str, object] | None:
    """Optionally persist a generated sample WITH its provenance manifest entry.

    Off by default; when enabled, every stored file gets a JSONL manifest row
    that labels it synthetic and records the generator, model, voice, and
    language. Never stored without that provenance.
    """
    if not settings.gnani_store_generated:
        return None
    import hashlib
    import json as _json
    from pathlib import Path

    out_dir = Path(settings.gnani_generated_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(audio.audio).hexdigest()[:16]
    file_name = f"gnani_{audio.model}_{audio.voice}_{digest}.{audio.container}".replace(" ", "_")
    file_path = out_dir / file_name
    file_path.write_bytes(audio.audio)
    manifest_row = {
        "path": file_name,
        "label": "synthetic",
        "partition": "generated",
        "source": "gnani_timbre",
        "generator_id": audio.generator_id,
        "model": audio.model,
        "voice": audio.voice,
        "language": audio.language,
        "container": audio.container,
        "sha256_prefix": digest,
        "created": utc_now_iso(),
    }
    with (out_dir / "provenance.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(_json.dumps(manifest_row) + "\n")
    return manifest_row


@api_router.post("/synthesize")
async def synthesize_speech(
    payload: SynthesizeRequest,
    request: Request,
    db: Annotated[Session, Depends(get_db)],
):
    """Authorized speech generation via the external provider.

    Returns raw audio bytes with explicit provenance headers. The response is
    provider output, labelled synthetic; it is NOT a VoxSentinel detection
    result and never feeds the evaluation corpus.
    """
    settings = request.app.state.settings
    service = getattr(request.app.state, "gnani_service", None)

    if service is None or not service.enabled:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "provider_disabled",
                "message": (
                    "External speech generation is not enabled on this deployment. "
                    "It must be explicitly enabled and configured with a provider "
                    "API key by an administrator."
                ),
            },
        )

    if len(payload.text) > settings.gnani_max_text_chars:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "validation_error",
                "message": f"Text exceeds the {settings.gnani_max_text_chars}-character limit.",
            },
        )

    limiter = request.app.state.gnani_limiter
    client_key = request.client.host if request.client else "unknown"
    decision = limiter.check(client_key)
    if not decision.allowed:
        status, detail = _rate_limit_message(decision.reason or "per_minute", settings)
        raise HTTPException(status_code=status, detail=detail)

    try:
        tts_request = TTSRequest(
            text=payload.text,
            model=payload.model,
            voice=payload.voice,
            language=payload.language,
            speed=payload.speed,
            sample_rate=payload.sample_rate,
            container=payload.container,
        )
        audio = await service.synthesize(tts_request)
    except GnaniValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "validation_error", "message": str(exc)},
        ) from exc
    except GnaniAuthError as exc:
        # Credentials are a deployment configuration problem; never exposed.
        logger.warning("provider auth failure during synthesize")
        raise HTTPException(status_code=503, detail={
            "code": "provider_auth_failed",
            "message": "The speech provider rejected the configured credentials.",
        }) from exc
    except GnaniRateLimitError as exc:
        raise HTTPException(status_code=429, detail={
            "code": "rate_limited",
            "message": "The speech provider is rate limiting requests. Try again later.",
        }) from exc
    except (GnaniUnavailableError, GnaniTimeoutError, GnaniConnectionError) as exc:
        raise HTTPException(status_code=503, detail={
            "code": "provider_unavailable",
            "message": "The speech provider is currently unavailable. Try again later.",
        }) from exc
    except (GnaniProviderError, GnaniMalformedResponseError) as exc:
        logger.warning("provider failure during synthesize: %s", type(exc).__name__)
        raise HTTPException(status_code=502, detail={
            "code": "provider_error",
            "message": "The speech provider failed to generate the requested audio.",
        }) from exc

    limiter.record(client_key)

    # Audit metadata: sizes and attribution only — never the text or audio.
    try:
        log_event(
            db, "provider", "synthesize", "success",
            request_id=getattr(request.state, "request_id", None),
            metadata={
                "provider": audio.provider,
                "model": audio.model,
                "voice": audio.voice,
                "language": audio.language,
                "text_chars": len(payload.text),
                "audio_bytes": len(audio.audio),
                "container": audio.container,
                "synthetic": True,
            },
        )
    except Exception:  # noqa: BLE001 — audit must never break a request
        logger.warning("synthesize audit write failed", exc_info=True)

    _store_generated_sample(audio, settings, request)

    return Response(
        content=audio.audio,
        media_type=audio.content_type,
        headers={
            "X-Synthetic-Content": "true",
            "X-Content-Provenance": "synthetic",
            "X-Provider": audio.provider,
            "X-Provider-Model": audio.model,
            "X-Provider-Voice": audio.voice,
            "X-Provider-Language": audio.language,
        },
    )


# ---------------------------------------------------------------------------
# Phase 6C — Self-hosted NVIDIA NemotronLabs VoiceChat 11B: one
# speech-to-speech turn against the official realtime container. The agent's
# audio is SYNTHETIC by construction; detection is never claimed or performed
# by this provider (verified 2026-10-25, see docs/VOICECHAT_PROVIDER.md).
# ---------------------------------------------------------------------------
def _store_voicechat_turn(turn, settings) -> dict[str, object] | None:
    """Optionally persist a generated turn WITH a provenance manifest row.

    Off by default; when enabled, every stored file is labelled synthetic
    with its generator and transcripts recorded as metadata only.
    """
    if not settings.voicechat_store_generated:
        return None
    import hashlib
    import json as _json
    from pathlib import Path

    wav = turn.as_wav()
    out_dir = Path(settings.voicechat_generated_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(wav).hexdigest()[:16]
    file_name = f"voicechat_{turn.model}_{digest}.wav".replace(" ", "_")
    (out_dir / file_name).write_bytes(wav)
    manifest_row = {
        "path": file_name,
        "label": "synthetic",
        "partition": "generated",
        "source": "voicechat_s2s",
        "generator_id": turn.generator_id,
        "model": turn.model,
        "voice": None,
        "language": None,
        "sample_rate": turn.sample_rate,
        "sha256_prefix": digest,
        "created": utc_now_iso(),
    }
    with (out_dir / "provenance.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(_json.dumps(manifest_row) + "\n")
    return manifest_row


@api_router.post("/voicechat", response_model=VoiceChatTurnOut)
async def voicechat_turn(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    file: Annotated[UploadFile, File(description="User speech (WAV/MP3/FLAC/OGG/M4A).")],
    instructions: Annotated[str | None, Form(description=(
        "Optional system prompt for the agent (server-validated length)."
    ))] = None,
) -> VoiceChatTurnOut:
    """Run one conversational speech-to-speech turn via VoiceChat 11B.

    The response audio is the model's GENERATED agent speech: synthetic by
    construction, explicitly labelled, and never a detection result.
    """
    settings = request.app.state.settings
    service = getattr(request.app.state, "voicechat_service", None)

    if service is None or not service.enabled:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "provider_disabled",
                "message": (
                    "The self-hosted voice-chat provider is not enabled on this "
                    "deployment. It requires an NVIDIA GPU inference server "
                    "(NemotronLabs VoiceChat 11B container) and must be enabled "
                    "by an administrator."
                ),
            },
        )

    if instructions is not None and len(instructions) > settings.gnani_max_text_chars:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "validation_error",
                "message": (
                    f"Instructions exceed the {settings.gnani_max_text_chars}-character limit."
                ),
            },
        )

    decoded, _mime = await _validated_audio(file, request)

    limiter = request.app.state.voicechat_limiter
    client_key = request.client.host if request.client else "unknown"
    decision = limiter.check(client_key)
    if not decision.allowed:
        status, detail = _rate_limit_message(
            decision.reason or "per_minute", settings,
            per_minute=settings.voicechat_rate_limit_per_minute,
        )
        raise HTTPException(status_code=status, detail=detail)

    try:
        turn = await service.converse(
            decoded.samples,
            decoded.sample_rate,
            decoded.channels,
            instructions=instructions,
        )
    except VoiceChatValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "validation_error", "message": str(exc)},
        ) from exc
    except VoiceChatUnavailableError as exc:
        raise HTTPException(status_code=503, detail={
            "code": "provider_unavailable",
            "message": "The voice provider is currently unavailable. Try again later.",
        }) from exc
    except (VoiceChatConnectionError, VoiceChatTimeoutError) as exc:
        raise HTTPException(status_code=504, detail={
            "code": "provider_timeout",
            "message": "The voice provider did not complete the turn in time. Try again.",
        }) from exc
    except VoiceChatServerError as exc:
        logger.warning("voicechat provider error event: %s", exc.code)
        raise HTTPException(status_code=502, detail={
            "code": "provider_error",
            "message": "The voice provider failed to complete the requested turn.",
        }) from exc
    except VoiceChatProtocolError as exc:
        logger.warning("voicechat protocol violation: %s", exc)
        raise HTTPException(status_code=502, detail={
            "code": "provider_error",
            "message": "The voice provider returned an unexpected response.",
        }) from exc

    if not turn.agent_audio_pcm:
        raise HTTPException(status_code=502, detail={
            "code": "provider_error",
            "message": "The voice provider returned no agent audio for this turn.",
        })

    limiter.record(client_key)

    # Audit metadata: durations and attribution only — never transcripts.
    try:
        log_event(
            db, "provider", "voicechat_turn", "success",
            request_id=getattr(request.state, "request_id", None),
            metadata={
                "provider": turn.provider,
                "model": turn.model,
                "input_seconds": decoded.duration_s,
                "agent_audio_bytes": len(turn.agent_audio_pcm),
                "agent_transcript_chars": len(turn.agent_transcript),
                "user_transcript_chars": len(turn.user_transcript),
                "synthetic": True,
            },
        )
    except Exception:  # noqa: BLE001 — audit must never break a request
        logger.warning("voicechat audit write failed", exc_info=True)

    _store_voicechat_turn(turn, settings)

    return VoiceChatTurnOut(
        provider=turn.provider,
        model=turn.model,
        generator_id=turn.generator_id,
        audio_b64=base64.b64encode(turn.as_wav()).decode("ascii"),
        sample_rate=turn.sample_rate,
        agent_transcript=turn.agent_transcript,
        user_transcript=turn.user_transcript,
    )
