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
from app.core.config import Settings, get_settings
from app.core.errors import install_error_handlers
from app.db.session import Database
from app.protection.router import router as protection_router
from app.protection.service import ProtectionService
from app.services.detector import (
    BaseVoiceDetector,
    NullVoiceDetector,
    VoiceDetectorService,
    try_load_detector,
)
from app.services.gnani_client import GnaniClient, GnaniSpeechService
from app.services.rate_limit import SlidingWindowLimiter
from app.services.risk_engine import RiskEngine, RiskThresholds
from app.services.speaker import (
    BaseSpeakerVerifier,
    NullSpeakerVerifier,
    SpeakerVerificationService,
    try_load_speaker_verifier,
)
from app.services.voicechat_client import VoiceChatClient, VoiceChatService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("voxsentinel")


def create_app(
    settings: Settings | None = None,
    detector: BaseVoiceDetector | None = None,
    speaker: BaseSpeakerVerifier | None = None,
    gnani_client: GnaniClient | None = None,
    voicechat_client: VoiceChatClient | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    resolved_detector = detector or try_load_detector(settings) or NullVoiceDetector()
    resolved_speaker = speaker or try_load_speaker_verifier(settings) or NullSpeakerVerifier()
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
    app.state.protection_service = ProtectionService(
        settings, app.state.detector_service, app.state.speaker_service
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
        request.state.request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    install_error_handlers(app)
    app.include_router(health_router)
    app.include_router(api_router)
    # Additive module: Family Emergency & Scam Protection (isolated routes).
    app.include_router(protection_router, prefix="/api/v1")
    return app


app = create_app()
