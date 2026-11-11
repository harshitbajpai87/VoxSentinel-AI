"""Pydantic schemas mirroring docs/API_CONTRACT.md and frontend/src/lib/types.ts."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

RiskCategory = Literal[
    "low_observed_risk",
    "elevated_risk",
    "high_risk",
    "inconclusive",
    "unable_to_analyze",
]


class HealthResponse(BaseModel):
    """Liveness + capability report.

    The original six fields are unchanged (the frontend types rely on them).
    Everything below is additive and optional, so older clients keep working:
    a missing optional external service must not make the backend look offline.
    """

    status: Literal["ok", "degraded", "down"]
    backend_ready: bool
    model_ready: bool
    speaker_model_ready: bool
    version: str
    mode: str
    # --- additive capability block ---
    audio_decoder_ready: bool = False
    max_upload_bytes: int = 0
    effective_max_upload_bytes: int = 0
    audio: dict = Field(default_factory=dict)
    services: dict = Field(default_factory=dict)
    environment: dict = Field(default_factory=dict)


class ModelInfoOut(BaseModel):
    id: str
    name: str
    version: str
    mode: str
    kind: Literal["detection", "speaker"]
    description: str
    loaded: bool


class ModelsResponse(BaseModel):
    models: list[ModelInfoOut]


class AnalysisInput(BaseModel):
    display_name: str
    duration_s: float
    sample_rate_hz: int
    channels: int
    quality_warnings: list[str] = Field(default_factory=list)


class AnalysisModelInfo(BaseModel):
    name: str
    version: str
    mode: str


class AnalysisScores(BaseModel):
    raw_score: float | None
    raw_score_definition: str
    calibrated_probability: float | None = None
    uncertainty: float | None = None
    uncertainty_method: str | None = None


class AnalysisDecision(BaseModel):
    category: RiskCategory
    label: str
    reasons: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


class MeasuredFeature(BaseModel):
    name: str
    value: float
    unit: str


class WindowScore(BaseModel):
    """Same-model score over one fixed segment of the clip (evidence only)."""
    start_s: float
    end_s: float
    score: float


class AnalysisEvidence(BaseModel):
    measured_features: list[MeasuredFeature] = Field(default_factory=list)
    model_explanation: str | None = None
    time_window_scores: list[WindowScore] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class AnalysisRecordOut(BaseModel):
    id: str
    created_at: str
    input: AnalysisInput
    model: AnalysisModelInfo
    scores: AnalysisScores
    decision: AnalysisDecision
    evidence: AnalysisEvidence | None = None
    latency_ms: int


class AnalysisListResponse(BaseModel):
    analyses: list[AnalysisRecordOut]
    total: int


class SpeakerVerifyResponse(BaseModel):
    status: Literal["completed", "unavailable"]
    similarity: float | None = None
    threshold: float | None = None
    outcome: str | None = None
    embedding_model: str | None = None
    warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class EvaluationMetricsResponse(BaseModel):
    evaluated: bool
    metrics: dict[str, Any] | None = None
    evaluated_at: str | None = None


class AuditEventOut(BaseModel):
    id: str
    created_at: str
    event_type: str
    action: str
    outcome: str
    actor: str
    resource_id: str | None = None
    request_id: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class AuditListResponse(BaseModel):
    events: list[AuditEventOut]
    total: int


class EvaluationJobOut(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    progress: dict[str, int]
    error: str | None = None
    started_at: str
    finished_at: str | None = None
    metrics: dict[str, object] | None = None


class RetentionPolicyOut(BaseModel):
    enabled: bool
    analyses_days: int
    audit_days: int
    temp_file_max_age_minutes: int
    interval_hours: int
    last_run: AuditEventOut | None = None
