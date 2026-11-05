"""Persisted analysis metadata.

Only analyses that were actually completed by a detector are stored here.
Rejected uploads and model-unavailable responses are NOT recorded as results,
so the history can never contain a fabricated verdict. Raw audio is never
stored.
"""
from __future__ import annotations

from sqlalchemy import JSON, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class AnalysisRecordORM(Base):
    __tablename__ = "analysis_records"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[str] = mapped_column(String(40), nullable=False, index=True)

    # Input metadata (no audio data, no personal identifiers beyond a display name).
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    duration_s: Mapped[float] = mapped_column(Float, nullable=False)
    sample_rate_hz: Mapped[int] = mapped_column(Integer, nullable=False)
    channels: Mapped[int] = mapped_column(Integer, nullable=False)
    quality_warnings: Mapped[list] = mapped_column(JSON, nullable=False, default=list)

    # Model provenance.
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    model_mode: Mapped[str] = mapped_column(String(32), nullable=False)

    # Scores are kept separate and never merged into one "probability".
    raw_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    raw_score_definition: Mapped[str] = mapped_column(String(512), nullable=False)
    calibrated_probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    uncertainty: Mapped[float | None] = mapped_column(Float, nullable=True)
    uncertainty_method: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Policy decision.
    decision_category: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    decision_label: Mapped[str] = mapped_column(String(255), nullable=False)
    decision_reasons: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    decision_recommendations: Mapped[list] = mapped_column(JSON, nullable=False, default=list)

    # Evidence (optional; measured features are facts about the audio).
    evidence_measured_features: Mapped[list | None] = mapped_column(JSON, nullable=True)
    evidence_model_explanation: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    evidence_limitations: Mapped[list | None] = mapped_column(JSON, nullable=True)

    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
