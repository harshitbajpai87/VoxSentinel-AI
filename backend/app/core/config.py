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


def _env_optional_int(name: str, default: int | None) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else None


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
    # Extra origin regex honoured in ANY environment. Deployments whose hostname
    # changes per build (Vercel preview URLs) need this; an exact allowlist
    # cannot cover them, and a missing origin here blocks every browser request
    # with a 400 preflight while curl keeps working.
    allowed_origin_regex: str | None = None

    max_upload_bytes: int = 25 * 1024 * 1024
    # Infrastructure ceiling for a request body, in bytes. Serverless platforms
    # cap the body BEFORE the application runs (Vercel: 4.5 MB), so the API must
    # report the smaller value it can really honour instead of promising 25 MB
    # and then failing with an opaque platform error.
    hosting_body_limit_bytes: int | None = None
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
    # Torch-free fallbacks. On a host without librosa/torch (Vercel) the trained
    # detector and ECAPA embedder cannot load; these keep a real, honestly
    # labelled signal instead of returning "unavailable" for every upload.
    forensics_enabled: bool = True
    forensics_threshold: float = 0.50
    mfcc_speaker_enabled: bool = True
    # Strict mode for the speaker stack. Off by default: a host without
    # torch/speechbrain keeps serving the honestly-labelled MFCC comparison.
    # Turn it on when ECAPA is a hard requirement, so a misconfigured
    # deployment fails at startup instead of silently degrading.
    require_ecapa_speaker: bool = False
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
    # ------------------------------------------------------------------
    # One provider credential, resolved once, used by every backend model
    # that talks to Vach (Gnani): speech-to-text (/stt/v3) and
    # text-to-speech (/api/v1/tts/inference). The key is BACKEND-ONLY and is
    # never placed in a VITE_* variable, a response body, or the frontend.
    # Resolution order: VOXSENTINEL_VACH_API_KEY, then the deployment-wide
    # API_KEY, then the legacy GNANI_API_KEY.
    # ------------------------------------------------------------------
    vach_api_key: str = ""
    vach_base_url: str = "https://api.vachana.ai"
    vach_stt_timeout_s: float = 120.0
    #: Provider hard cap is 30 s per request; stay under it and chunk longer calls.
    vach_stt_max_chunk_seconds: float = 28.0
    #: indic-auto = provider-side language detection (Hindi/English + 10 Indic).
    vach_stt_language_code: str = "indic-auto"

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

    def effective_upload_limit_bytes(self) -> int:
        """Largest upload this deployment can actually accept."""
        if self.hosting_body_limit_bytes and self.hosting_body_limit_bytes > 0:
            return min(self.max_upload_bytes, self.hosting_body_limit_bytes)
        return self.max_upload_bytes

    def cors_origin_regex(self) -> str | None:
        """Regex for extra allowed origins: configured pattern, plus dev origins."""
        patterns: list[str] = []
        if self.allowed_origin_regex:
            patterns.append(self.allowed_origin_regex)
        if self.allow_dev_origins and self.app_env.lower() in {"development", "dev", "local"}:
            patterns.append(self._dev_origin_regex())
        return "|".join(f"(?:{p})" for p in patterns) if patterns else None

    def _dev_origin_regex(self) -> str:
        # http(s)://<loopback or private LAN>:<any port>
        host = (
            r"(?:localhost|127\.0\.0\.1|0\.0\.0\.0|"
            r"10\.\d{1,3}\.\d{1,3}\.\d{1,3}|"
            r"192\.168\.\d{1,3}\.\d{1,3}|"
            r"172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})"
        )
        return rf"^https?://{host}(?::\d{{1,5}})?$"



def _resolve_vach_api_key() -> str:
    """The single backend credential for Vach (Gnani) STT + TTS.

    Checked in this order so an explicit per-service variable can override the
    deployment-wide secret while a plain ``API_KEY`` still works on its own:

      1. ``VOXSENTINEL_VACH_API_KEY``
      2. ``VOXSENTINEL_GNANI_API_KEY`` (legacy explicit name)
      3. ``API_KEY`` (the shared deployment secret)
      4. ``GNANI_API_KEY`` (the provider SDK's documented variable)
    """
    for name in (
        "VOXSENTINEL_VACH_API_KEY",
        "VOXSENTINEL_GNANI_API_KEY",
        "API_KEY",
        "GNANI_API_KEY",
    ):
        value = (os.environ.get(name) or "").strip()
        if value:
            # A key pasted with wrapping whitespace is the usual cause of a 401.
            return "".join(value.split()) if any(c.isspace() for c in value) else value
    return ""


def get_settings() -> Settings:
    """Build Settings from environment variables (defaults are safe)."""
    return Settings(
        app_env=_env_str("VOXSENTINEL_ENV", "development"),
        api_version=_env_str("VOXSENTINEL_API_VERSION", "0.1.0"),
        app_mode=_env_str("VOXSENTINEL_APP_MODE", "baseline"),
        db_url=_env_str("VOXSENTINEL_DB_URL", "sqlite:///./voxsentinel.db"),
        allowed_origins=_env_list("VOXSENTINEL_ALLOWED_ORIGINS", DEFAULT_ALLOWED_ORIGINS),
        allowed_origin_regex=(os.environ.get("VOXSENTINEL_ALLOWED_ORIGIN_REGEX") or "").strip() or None,
        allow_dev_origins=_env_bool("VOXSENTINEL_ALLOW_DEV_ORIGINS", True),
        max_upload_bytes=_env_int("VOXSENTINEL_MAX_UPLOAD_BYTES", 25 * 1024 * 1024),
        # Default is platform-derived: Vercel rejects bodies over 4.5 MB, so on
        # Vercel the honest ceiling is 4.5 MB unless the operator overrides it.
        hosting_body_limit_bytes=_env_optional_int(
            "VOXSENTINEL_HOSTING_BODY_LIMIT_BYTES",
            4_500_000 if os.environ.get("VERCEL") else None,
        ),
        ffmpeg_binary=_env_str("VOXSENTINEL_FFMPEG_BINARY", "ffmpeg"),
        ffprobe_binary=_env_str("VOXSENTINEL_FFPROBE_BINARY", "ffprobe"),
        ffmpeg_timeout_seconds=_env_int("VOXSENTINEL_FFMPEG_TIMEOUT_SECONDS", 30),
        temp_max_age_seconds=_env_float("VOXSENTINEL_TEMP_MAX_AGE_SECONDS", 3600.0),
        speaker_model_dir=_env_str("VOXSENTINEL_SPEAKER_MODEL_DIR", "models/speaker_ecapa"),
        speaker_thresholds_path=_env_str(
            "VOXSENTINEL_SPEAKER_THRESHOLDS_PATH", "models/speaker_thresholds.json"
        ),
        min_duration_s=_env_float("VOXSENTINEL_MIN_DURATION_S", 2.0),
        max_duration_s=_env_float("VOXSENTINEL_MAX_DURATION_S", 600.0),
        silence_rms=_env_float("VOXSENTINEL_SILENCE_RMS", 0.005),
        quiet_rms=_env_float("VOXSENTINEL_QUIET_RMS", 0.02),
        clipping_peak=_env_float("VOXSENTINEL_CLIPPING_PEAK", 0.995),
        detector_model_path=_env_str("VOXSENTINEL_DETECTOR_MODEL_PATH", "models/detector.joblib"),
        forensics_enabled=_env_bool("VOXSENTINEL_FORENSICS_ENABLED", True),
        forensics_threshold=_env_float("VOXSENTINEL_FORENSICS_THRESHOLD", 0.50),
        mfcc_speaker_enabled=_env_bool("VOXSENTINEL_MFCC_SPEAKER_ENABLED", True),
        require_ecapa_speaker=_env_bool("VOXSENTINEL_REQUIRE_ECAPA_SPEAKER", False),
        speaker_model_path=_env_str("VOXSENTINEL_SPEAKER_MODEL_PATH", "models/speaker.joblib"),
        temp_upload_dir=_env_str("VOXSENTINEL_TEMP_UPLOAD_DIR", ".voxsentinel_tmp"),
        dataset_dir=_env_str("VOXSENTINEL_DATASET_DIR", "data"),
        report_dir=_env_str("VOXSENTINEL_REPORT_DIR", "reports"),
        # One credential for the whole product. A deployment that sets only
        # API_KEY (the platform secret form) still gets STT and TTS working.
        vach_api_key=_resolve_vach_api_key(),
        vach_base_url=_env_str("VOXSENTINEL_VACH_BASE_URL", _env_str("VOXSENTINEL_GNANI_BASE_URL", "https://api.vachana.ai")),
        vach_stt_timeout_s=_env_float("VOXSENTINEL_VACH_STT_TIMEOUT_S", 120.0),
        vach_stt_max_chunk_seconds=_env_float("VOXSENTINEL_VACH_STT_MAX_CHUNK_SECONDS", 28.0),
        vach_stt_language_code=_env_str("VOXSENTINEL_VACH_STT_LANGUAGE_CODE", "indic-auto"),
        # TTS is enabled automatically when a key exists, unless explicitly
        # disabled, so dropping in the credential is enough to light it up.
        gnani_enabled=_env_bool("VOXSENTINEL_GNANI_ENABLED", bool(_resolve_vach_api_key())),
        # Backend-only secret, sharing the same resolution as STT.
        gnani_api_key=_resolve_vach_api_key(),
        gnani_base_url=_env_str("VOXSENTINEL_GNANI_BASE_URL", _env_str("VOXSENTINEL_VACH_BASE_URL", "https://api.vachana.ai")),
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
