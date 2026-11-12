"""VoxSentinel AI API — application factory.

Run locally with:
    uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

Design notes:
  - `create_app()` accepts overrides (settings, detector, speaker) so tests can
    inject a temporary database or a clearly-labelled test double.
  - Debug mode is never enabled; internal errors return a safe 500 payload.
  - Real synthetic-speech detection is unavailable until a trained detector is
    integrated, and the API says so instead of inventing predictions.
"""
from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import api_router, health_router
from app.api.assistant_routes import router as assistant_router
from app.api.unified_routes import router as unified_router
from app.core.config import Settings, get_settings
from app.core.errors import install_error_handlers
from app.db.session import Database
from app.protection.router import router as protection_router
from app.protection.service import ProtectionService
from app.protection.text_ensemble import TextEnsemble
from app.services.acoustic_forensics import AcousticForensicsDetector
from app.services.detector import (
    BaseVoiceDetector,
    NullVoiceDetector,
    VoiceDetectorService,
    try_load_detector,
)
from app.services.forensics_embeddings import ForensicsHead, UnifiedForensics, Wav2Vec2ForensicsEmbedder
from app.services.gnani_client import GnaniClient, GnaniSpeechService
from app.services.ingestion import describe_environment, human_mb, probe_decoders
from app.services.rate_limit import SlidingWindowLimiter
from app.services.risk_engine import RiskEngine, RiskThresholds
from app.services.speaker import (
    ECAPA_INSTALL_HINT,
    BaseSpeakerVerifier,
    NullSpeakerVerifier,
    SpeakerVerificationService,
    speaker_backend_name,
    try_load_speaker_verifier,
)
from app.services.speaker_mfcc import MfccStatSpeakerVerifier
from app.services.voicechat_client import VoiceChatClient, VoiceChatService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("voxsentinel")


#: Coarse failure classes so one message is never used for every problem
#: (docs: "Analysis failed" hid upload, decode, model and network faults alike).
_ERROR_CATEGORIES = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    413: "upload_too_large",
    415: "unsupported_media_type",
    422: "invalid_or_undecodable_audio",
    429: "rate_limited",
    500: "internal_error",
    502: "upstream_unavailable",
    503: "service_unavailable",
    504: "upstream_timeout",
}


def _error_category(status_code: int) -> str | None:
    if status_code < 400:
        return None
    return _ERROR_CATEGORIES.get(status_code, f"http_{status_code}")


def create_app(
    settings: Settings | None = None,
    detector: BaseVoiceDetector | None = None,
    speaker: BaseSpeakerVerifier | None = None,
    gnani_client: GnaniClient | None = None,
    voicechat_client: VoiceChatClient | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    # Missing trained detectors remain explicitly unavailable. Acoustic heuristics
    # are optional evidence only, never a detector/readiness substitute.
    resolved_detector = detector if detector is not None else try_load_detector(settings)
    resolved_detector = resolved_detector or NullVoiceDetector()
    forensics = (
        AcousticForensicsDetector(threshold=settings.forensics_threshold)
        if settings.forensics_enabled else None
    )

    # Speaker resolution mirrors the detector: ECAPA embeddings when torch is
    # present, otherwise the MFCC-statistics comparison (which reports no
    # same/different-speaker verdict because no threshold is calibrated for it).
    resolved_speaker = speaker or try_load_speaker_verifier(settings)
    if resolved_speaker is None and settings.mfcc_speaker_enabled:
        resolved_speaker = MfccStatSpeakerVerifier()
    resolved_speaker = resolved_speaker or NullSpeakerVerifier()
    # Strict mode: when ECAPA is required, refuse to start on a host that cannot
    # run it, instead of silently serving the torch-free comparison.
    if settings.require_ecapa_speaker and speaker_backend_name(resolved_speaker) != "ecapa":
        raise RuntimeError(
            "VOXSENTINEL_REQUIRE_ECAPA_SPEAKER is on but the ECAPA backend is not "
            f"active (resolved: {speaker_backend_name(resolved_speaker)}). "
            + ECAPA_INSTALL_HINT
        )
    # External speech generation (Phase 6). Built ONLY when the integration is
    # explicitly enabled AND a backend-only key is configured; otherwise the
    # API honestly reports the provider as disabled. Tests inject a client
    # with a mocked transport, so no real key or network is ever used in CI.
    resolved_gnani = gnani_client
    if resolved_gnani is None and settings.gnani_enabled and settings.gnani_api_key.strip():
        resolved_gnani = GnaniClient(
            settings.gnani_api_key,
            base_url=settings.gnani_base_url,
            connect_timeout_s=settings.gnani_connect_timeout_s,
            read_timeout_s=settings.gnani_read_timeout_s,
            max_text_chars=settings.gnani_max_text_chars,
        )
    resolved_gnani_service = GnaniSpeechService(resolved_gnani)
    # Self-hosted NemotronLabs VoiceChat 11B (Phase 6C). Enabled only when
    # explicitly configured with a WebSocket URL; talks to a private NVIDIA
    # GPU deployment of the official container. Tests inject a client with a
    # scripted fake session, so no test performs a network connection.
    resolved_voicechat = voicechat_client
    if (
        resolved_voicechat is None
        and settings.voicechat_enabled
        and settings.voicechat_ws_url.strip()
    ):
        resolved_voicechat = VoiceChatClient(
            settings.voicechat_ws_url,
            health_url=settings.voicechat_health_url,
            connect_timeout_s=settings.voicechat_connect_timeout_s,
            idle_timeout_s=settings.voicechat_idle_timeout_s,
            max_input_seconds=settings.voicechat_max_input_seconds,
        )
    resolved_voicechat_service = VoiceChatService(resolved_voicechat)

    database = Database(settings.db_url)
    # Create tables eagerly, not only in lifespan. Serverless hosts (Vercel)
    # hand a cold instance a fresh /tmp, and some runtimes invoke the ASGI app
    # without running lifespan startup first. Without this a completed analysis
    # crashed on INSERT with "no such table", so the user lost the result.
    # create_all() is idempotent and does nothing when the schema is present.
    try:
        database.create_all()
    except Exception:  # noqa: BLE001 - a read-only host must still boot
        logger.warning("schema creation failed; history may be unavailable", exc_info=True)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        database.create_all()
        logger.info(
            "VoxSentinel API ready | env=%s | detector_ready=%s | speaker_ready=%s | db=%s",
            settings.app_env,
            resolved_detector.is_ready(),
            resolved_speaker.is_ready(),
            settings.db_url.split("://", 1)[0],
        )
        # --- startup diagnostics for the audio pipeline -------------------
        # A missing decoder is the single most common cause of "415 / cannot
        # decode" reports, so it is stated loudly at boot instead of surfacing
        # only when a user uploads an M4A.
        try:
            decoders = probe_decoders(settings, use_cache=False)
            logger.info(
                "audio decoders | ffmpeg=%s (%s) | ffprobe=%s | libsndfile=%s | formats=%s",
                decoders.ffmpeg,
                decoders.ffmpeg_version or "missing",
                decoders.ffprobe,
                decoders.libsndfile or "n/a",
                ",".join(decoders.libsndfile_formats) or "n/a",
            )
            for note in decoders.notes:
                logger.warning("audio decoder: %s", note)
            if not decoders.ffmpeg:
                logger.warning(
                    "ffmpeg is NOT available: M4A/AAC/WebM/Opus uploads will be refused with "
                    "503 decoder_unavailable. Install ffmpeg in this runtime "
                    "(deploy/docker/Dockerfile installs it) or convert clips to WAV/MP3/FLAC/OGG."
                )
            logger.info(
                "upload policy | configured=%s | effective=%s | hosting_body_limit=%s | runtime=%s",
                human_mb(settings.max_upload_bytes),
                human_mb(settings.effective_upload_limit_bytes()),
                human_mb(settings.hosting_body_limit_bytes)
                if settings.hosting_body_limit_bytes
                else "none",
                describe_environment().get("platform"),
            )
            if (
                settings.hosting_body_limit_bytes
                and settings.hosting_body_limit_bytes < settings.max_upload_bytes
            ):
                logger.warning(
                    "this hosting platform rejects request bodies above %s, below the "
                    "configured %s upload limit; large uploads need the container backend",
                    human_mb(settings.hosting_body_limit_bytes),
                    human_mb(settings.max_upload_bytes),
                )
        except Exception:  # noqa: BLE001 - diagnostics must never block startup
            logger.warning("startup audio diagnostics failed", exc_info=True)
        # Audit startup: which models are actually serving.
        try:
            session = database.session()
            try:
                from app.services.audit import log_event

                log_event(session, "system", "startup", "success", metadata={
                    "detector_ready": resolved_detector.is_ready(),
                    "speaker_ready": resolved_speaker.is_ready(),
                    "api_version": settings.api_version,
                })
            finally:
                session.close()
        except Exception:  # noqa: BLE001 - audit must never block startup
            logger.warning("startup audit write failed", exc_info=True)

        retention_task = None
        if settings.retention_enabled:
            import asyncio

            async def _retention_loop() -> None:

                from app.services.retention import run_cleanup

                logger.info("retention loop enabled (every %sh)", settings.retention_interval_hours)
                while True:
                    await asyncio.sleep(settings.retention_interval_hours * 3600)
                    try:
                        session = database.session()
                        try:
                            outcome = run_cleanup(settings, session, dry_run=False)
                            logger.info("scheduled retention: %s", outcome.as_dict())
                        finally:
                            session.close()
                    except Exception:  # noqa: BLE001
                        logger.warning("scheduled retention failed", exc_info=True)

            retention_task = asyncio.create_task(_retention_loop())
        try:
            yield
        finally:
            if retention_task is not None:
                retention_task.cancel()
            await resolved_gnani_service.aclose()
            await resolved_voicechat_service.aclose()
            database.dispose()

    app = FastAPI(
        title="VoxSentinel AI API",
        description=(
            "Voice-cloning impersonation detection and risk-aware prevention. "
            "Results are decision support, never proof of authenticity."
        ),
        version=settings.api_version,
        lifespan=lifespan,
    )

    # Explicit origins from configuration; credentials are not used, and no
    # wildcard origin is configured by default.
    # Development additionally accepts loopback/LAN origins on any port via a
    # regex (see Settings.cors_origin_regex). Production keeps the strict
    # explicit allowlist, so this cannot widen a deployed environment.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.allowed_origins),
        allow_origin_regex=settings.cors_origin_regex(),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS", "HEAD", "PUT", "PATCH", "DELETE"],
        allow_headers=["*"],
        expose_headers=["Content-Disposition"],
    )

    app.state.settings = settings
    app.state.db = database
    app.state.detector_service = VoiceDetectorService(resolved_detector)
    app.state.speaker_service = SpeakerVerificationService(resolved_speaker)
    app.state.risk_engine = RiskEngine(RiskThresholds())
    # Unified pipeline extras: MuRIL/DeBERTa text ensemble and the fused
    # acoustic-forensics layer (detector + handcrafted cues + XLSR head).
    text_ensemble = TextEnsemble()
    forensics_engine = (
        UnifiedForensics(
            handcrafted=forensics,
            embedder=Wav2Vec2ForensicsEmbedder(),
            head=ForensicsHead(),
            threshold=settings.forensics_threshold,
        )
        if settings.forensics_enabled
        else None
    )
    app.state.protection_service = ProtectionService(
        settings, app.state.detector_service, app.state.speaker_service,
        forensics_detector=forensics,
        text_ensemble=text_ensemble,
        forensics_engine=forensics_engine,
    )
    app.state.gnani_service = resolved_gnani_service
    app.state.gnani_limiter = SlidingWindowLimiter(
        per_minute=settings.gnani_rate_limit_per_minute,
        daily_quota=settings.gnani_daily_quota,
    )
    app.state.voicechat_service = resolved_voicechat_service
    app.state.voicechat_limiter = SlidingWindowLimiter(
        per_minute=settings.voicechat_rate_limit_per_minute,
        daily_quota=settings.voicechat_daily_quota,
    )

    # Request correlation: every response carries an X-Request-ID, and audit
    # events can reference it (traceability, Module 7).
    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        import time

        from app.services.ingestion import json_log

        request.state.request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception as exc:  # noqa: BLE001 - log, then let the handler respond
            json_log(
                "request_failed",
                request_id=request.state.request_id,
                endpoint=request.url.path,
                method=request.method,
                stage="unhandled",
                error_category=type(exc).__name__,
                duration_ms=int((time.perf_counter() - started) * 1000),
                http_status=500,
            )
            raise
        duration_ms = int((time.perf_counter() - started) * 1000)
        response.headers["X-Request-ID"] = request.state.request_id
        # Never log audio bytes or credentials: size, timing and status only.
        if request.url.path.startswith("/api/") or request.url.path == "/health":
            json_log(
                "request",
                request_id=request.state.request_id,
                endpoint=request.url.path,
                method=request.method,
                http_status=response.status_code,
                content_length=request.headers.get("content-length"),
                error_category=_error_category(response.status_code),
                duration_ms=duration_ms,
            )
        return response

    install_error_handlers(app)
    app.include_router(health_router)
    app.include_router(api_router)
    # Additive module: Family Emergency & Scam Protection (isolated routes).
    app.include_router(protection_router, prefix="/api/v1")
    # Unified multi-model pipeline: POST /api/analyze (audio or text).
    app.include_router(unified_router)
    # Additive module: in-app help assistant (Gemini, server-side key only).
    app.include_router(assistant_router, prefix="/api/v1")
    return app


app = create_app()
