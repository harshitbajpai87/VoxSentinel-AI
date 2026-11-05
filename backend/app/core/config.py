"""Central configuration.

Every value can be overridden with an environment variable (see .env.example).
Nothing machine-specific is hard-coded, and no secrets live in code.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_ALLOWED_ORIGINS = ("http://localhost:5173", "http://127.0.0.1:5173")


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    return raw if raw is not None and raw.strip() != "" else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    items = tuple(part.strip() for part in raw.split(",") if part.strip())
    return items or default


@dataclass(frozen=True)
class Settings:
    """Runtime settings. Frozen so it can be shared safely across requests."""

    app_env: str = "development"
    # API contract version exposed by /health (the frontend types mirror v1).
    api_version: str = "0.1.0"
    # Descriptive pipeline mode ("baseline" / "demo" / "production").
    # IMPORTANT: this is NOT a readiness claim. Readiness is reported only by
    # the `model_ready` field, which is derived from the loaded detector.
    app_mode: str = "baseline"

    db_url: str = "sqlite:///./voxsentinel.db"

    allowed_origins: tuple[str, ...] = DEFAULT_ALLOWED_ORIGINS
    # In development only, also accept any loopback/LAN origin (any port, plus
    # 127.0.0.1/localhost). Without this, opening the Vite dev server on a LAN
    # IP or a second port makes every browser request fail CORS and the UI shows
    # "Could not reach the backend (network error or CORS)". Production keeps
    # the strict explicit allowlist below.
    allow_dev_origins: bool = True

    max_upload_bytes: int = 25 * 1024 * 1024
    min_duration_s: float = 2.0
    max_duration_s: float = 600.0

    silence_rms: float = 0.005
    quiet_rms: float = 0.02
    clipping_peak: float = 0.995

    # Phase 5 will load a trained artifact from here. Phase 4 ships the
    # interface only: if the file is absent, the API reports the model as
    # not ready instead of inventing predictions.
    detector_model_path: str = "models/detector.joblib"
    speaker_model_path: str = "models/speaker.joblib"
    speaker_model_dir: str = "models/speaker_ecapa"
    speaker_thresholds_path: str = "models/speaker_thresholds.json"
    # Data retention (Module 7). Disabled by default: enabling it makes the
    # API delete analyses older than retention_analyses_days and audit events
    # older than retention_audit_days, permanently, on a periodic background sweep.
    retention_enabled: bool = False
    retention_analyses_days: float = 90.0
    retention_audit_days: float = 365.0
    retention_interval_hours: float = 24.0
    # Evaluation job defaults (Module 5)
    eval_manifest_path: str = "data/librispeech_piper_manifest.jsonl"
    eval_dataset_dir: str = "data"
    eval_partition: str = "eval"
    # FFmpeg fallback decoder (M4A/AAC/WebM/Opus). Binaries are fixed argv,
    # never user-supplied; see app/services/ffmpeg_decoder.py.
    ffmpeg_binary: str = "ffmpeg"
    ffprobe_binary: str = "ffprobe"
    ffmpeg_timeout_seconds: int = 30
    # Gnani.ai / Vach speech provider (Phase 6). The API key is BACKEND-ONLY:
    # read from the environment where the FastAPI process runs. Never place it
    # in Vite/browser variables (VITE_*) or any client-visible file. Disabled
    # by default; generation is refused until explicitly enabled AND a key is
    # present. See docs/GNANI_PROVIDER.md for the verified contract.
    gnani_enabled: bool = False
    gnani_api_key: str = ""
    gnani_base_url: str = "https://api.vachana.ai"
    gnani_connect_timeout_s: float = 5.0
    gnani_read_timeout_s: float = 30.0
    gnani_max_text_chars: int = 1000
    gnani_rate_limit_per_minute: int = 10
    gnani_daily_quota: int = 200
    # Store generated audio only when explicitly necessary (training data).
    # Generated files always carry a synthetic provenance manifest entry.
    gnani_store_generated: bool = False
    gnani_generated_dir: str = "data/generated/gnani"
    # NVIDIA NemotronLabs VoiceChat 11B (Phase 6C) — a SELF-HOSTED,
    # full-duplex speech-to-speech model served by the official NVIDIA
    # inference container (OpenAI-Realtime-compatible WebSocket API).
    # Requires an NVIDIA GPU deployment elsewhere (e.g. an AWS GPU
    # instance); this backend only talks to it over the network. No key:
    # the deployment is private infrastructure. Disabled by default and
    # additionally requires an explicit WebSocket URL.
    voicechat_enabled: bool = False
    voicechat_ws_url: str = ""
    voicechat_health_url: str = ""
    voicechat_connect_timeout_s: float = 5.0
    voicechat_idle_timeout_s: float = 60.0
    voicechat_max_input_seconds: float = 60.0
    voicechat_rate_limit_per_minute: int = 5
    voicechat_daily_quota: int = 100
    # Store agent audio only when explicitly necessary (training data),
    # always with a synthetic provenance manifest entry.
    voicechat_store_generated: bool = False
    voicechat_generated_dir: str = "data/generated/voicechat"
    # Server-named temp files for the ffmpeg layer (always cleaned up; the
    # retention sweeper removes crash leftovers older than this).
    temp_upload_dir: str = ".voxsentinel_tmp"
    temp_max_age_seconds: float = 3600.0
    dataset_dir: str = "data"
    report_dir: str = "reports"

    def cors_origin_regex(self) -> str | None:
        """Regex for extra dev origins, or None when disabled/non-dev."""
        if not self.allow_dev_origins or self.app_env.lower() not in {"development", "dev", "local"}:
            return None
        # http(s)://<loopback or private LAN>:<any port>
        host = (
            r"(?:localhost|127\.0\.0\.1|0\.0\.0\.0|"
            r"10\.\d{1,3}\.\d{1,3}\.\d{1,3}|"
            r"192\.168\.\d{1,3}\.\d{1,3}|"
            r"172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})"
        )
        return rf"^https?://{host}(?::\d{{1,5}})?$"



def get_settings() -> Settings:
    """Build Settings from environment variables (defaults are safe)."""
    return Settings(
        app_env=_env_str("VOXSENTINEL_ENV", "development"),
        api_version=_env_str("VOXSENTINEL_API_VERSION", "0.1.0"),
        app_mode=_env_str("VOXSENTINEL_APP_MODE", "baseline"),
        db_url=_env_str("VOXSENTINEL_DB_URL", "sqlite:///./voxsentinel.db"),
        allowed_origins=_env_list("VOXSENTINEL_ALLOWED_ORIGINS", DEFAULT_ALLOWED_ORIGINS),
        allow_dev_origins=_env_bool("VOXSENTINEL_ALLOW_DEV_ORIGINS", True),
        max_upload_bytes=_env_int("VOXSENTINEL_MAX_UPLOAD_BYTES", 25 * 1024 * 1024),
        min_duration_s=_env_float("VOXSENTINEL_MIN_DURATION_S", 2.0),
        max_duration_s=_env_float("VOXSENTINEL_MAX_DURATION_S", 600.0),
        silence_rms=_env_float("VOXSENTINEL_SILENCE_RMS", 0.005),
        quiet_rms=_env_float("VOXSENTINEL_QUIET_RMS", 0.02),
        clipping_peak=_env_float("VOXSENTINEL_CLIPPING_PEAK", 0.995),
        detector_model_path=_env_str("VOXSENTINEL_DETECTOR_MODEL_PATH", "models/detector.joblib"),
        speaker_model_path=_env_str("VOXSENTINEL_SPEAKER_MODEL_PATH", "models/speaker.joblib"),
        dataset_dir=_env_str("VOXSENTINEL_DATASET_DIR", "data"),
        report_dir=_env_str("VOXSENTINEL_REPORT_DIR", "reports"),
        gnani_enabled=_env_str("VOXSENTINEL_GNANI_ENABLED", "false").strip().lower()
        in ("1", "true", "yes", "on"),
        # Backend-only secret. VOXSENTINEL_GNANI_API_KEY wins; GNANI_API_KEY
        # (the provider SDK's documented variable) is accepted as a fallback.
        gnani_api_key=_env_str("VOXSENTINEL_GNANI_API_KEY", _env_str("GNANI_API_KEY", "")),
        gnani_base_url=_env_str("VOXSENTINEL_GNANI_BASE_URL", "https://api.vachana.ai"),
        gnani_connect_timeout_s=_env_float("VOXSENTINEL_GNANI_CONNECT_TIMEOUT_S", 5.0),
        gnani_read_timeout_s=_env_float("VOXSENTINEL_GNANI_READ_TIMEOUT_S", 30.0),
        gnani_max_text_chars=_env_int("VOXSENTINEL_GNANI_MAX_TEXT_CHARS", 1000),
        gnani_rate_limit_per_minute=_env_int("VOXSENTINEL_GNANI_RATE_LIMIT_PER_MINUTE", 10),
        gnani_daily_quota=_env_int("VOXSENTINEL_GNANI_DAILY_QUOTA", 200),
        gnani_store_generated=_env_str("VOXSENTINEL_GNANI_STORE_GENERATED", "false").strip().lower()
        in ("1", "true", "yes", "on"),
        gnani_generated_dir=_env_str("VOXSENTINEL_GNANI_GENERATED_DIR", "data/generated/gnani"),
        voicechat_enabled=_env_str("VOXSENTINEL_VOICECHAT_ENABLED", "false").strip().lower()
        in ("1", "true", "yes", "on"),
        voicechat_ws_url=_env_str("VOXSENTINEL_VOICECHAT_WS_URL", ""),
        voicechat_health_url=_env_str("VOXSENTINEL_VOICECHAT_HEALTH_URL", ""),
        voicechat_connect_timeout_s=_env_float("VOXSENTINEL_VOICECHAT_CONNECT_TIMEOUT_S", 5.0),
        voicechat_idle_timeout_s=_env_float("VOXSENTINEL_VOICECHAT_IDLE_TIMEOUT_S", 60.0),
        voicechat_max_input_seconds=_env_float("VOXSENTINEL_VOICECHAT_MAX_INPUT_SECONDS", 60.0),
        voicechat_rate_limit_per_minute=_env_int("VOXSENTINEL_VOICECHAT_RATE_LIMIT_PER_MINUTE", 5),
        voicechat_daily_quota=_env_int("VOXSENTINEL_VOICECHAT_DAILY_QUOTA", 100),
        voicechat_store_generated=_env_str("VOXSENTINEL_VOICECHAT_STORE_GENERATED", "false")
        .strip().lower() in ("1", "true", "yes", "on"),
        voicechat_generated_dir=_env_str("VOXSENTINEL_VOICECHAT_GENERATED_DIR", "data/generated/voicechat"),
    )
