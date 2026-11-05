"""Persistence for completed analyses.

Only records produced by a genuine detector are stored. Rejected uploads and
model-unavailable responses are never written as results, so the history
cannot contain a fabricated verdict.
"""
from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.analysis_record import AnalysisRecordORM
from app.schemas.analysis import (
    AnalysisDecision,
    AnalysisEvidence,
    AnalysisInput,
    AnalysisModelInfo,
    AnalysisRecordOut,
    AnalysisScores,
)


def utc_now_iso() -> str:
    """Consistent UTC timestamps: 2026-10-22T13:45:10.123Z."""
    return (
        datetime.now(UTC)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _to_schema(row: AnalysisRecordORM) -> AnalysisRecordOut:
    evidence = None
    if row.evidence_measured_features is not None or row.evidence_limitations is not None:
        evidence = AnalysisEvidence(
            measured_features=row.evidence_measured_features or [],
            model_explanation=row.evidence_model_explanation,
            limitations=row.evidence_limitations or [],
        )
    return AnalysisRecordOut(
        id=row.id,
        created_at=row.created_at,
        input=AnalysisInput(
            display_name=row.display_name,
            duration_s=row.duration_s,
            sample_rate_hz=row.sample_rate_hz,
            channels=row.channels,
            quality_warnings=list(row.quality_warnings or []),
        ),
        model=AnalysisModelInfo(
            name=row.model_name,
            version=row.model_version,
            mode=row.model_mode,
        ),
        scores=AnalysisScores(
            raw_score=row.raw_score,
            raw_score_definition=row.raw_score_definition,
            calibrated_probability=row.calibrated_probability,
            uncertainty=row.uncertainty,
            uncertainty_method=row.uncertainty_method,
        ),
        decision=AnalysisDecision(
            category=row.decision_category,
            label=row.decision_label,
            reasons=list(row.decision_reasons or []),
            recommendations=list(row.decision_recommendations or []),
        ),
        evidence=evidence,
        latency_ms=row.latency_ms,
    )


class AnalysisRepository:
    """Data access for analysis records (one session per request)."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def create(self, record: AnalysisRecordOut) -> AnalysisRecordOut:
        row = AnalysisRecordORM(
            id=record.id,
            created_at=record.created_at,
            display_name=record.input.display_name,
            duration_s=record.input.duration_s,
            sample_rate_hz=record.input.sample_rate_hz,
            channels=record.input.channels,
            quality_warnings=list(record.input.quality_warnings),
            model_name=record.model.name,
            model_version=record.model.version,
            model_mode=record.model.mode,
            raw_score=record.scores.raw_score,
            raw_score_definition=record.scores.raw_score_definition,
            calibrated_probability=record.scores.calibrated_probability,
            uncertainty=record.scores.uncertainty,
            uncertainty_method=record.scores.uncertainty_method,
            decision_category=record.decision.category,
            decision_label=record.decision.label,
            decision_reasons=list(record.decision.reasons),
            decision_recommendations=list(record.decision.recommendations),
            evidence_measured_features=(
                [feature.model_dump() for feature in record.evidence.measured_features]
                if record.evidence
                else None
            ),
            evidence_model_explanation=(
                record.evidence.model_explanation if record.evidence else None
            ),
            evidence_limitations=(list(record.evidence.limitations) if record.evidence else None),
            latency_ms=record.latency_ms,
        )
        self._session.add(row)
        self._session.commit()
        return _to_schema(row)

    def get(self, analysis_id: str) -> AnalysisRecordOut | None:
        row = self._session.get(AnalysisRecordORM, analysis_id)
        return _to_schema(row) if row is not None else None

    def list(
        self, limit: int, offset: int, category: str | None = None
    ) -> tuple[list[AnalysisRecordOut], int]:
        """Completed analyses, newest first; optional server-side category filter."""
        where = [AnalysisRecordORM.decision_category == category] if category else []
        total = int(
            self._session.scalar(
                select(func.count()).select_from(AnalysisRecordORM).where(*where)
            ) or 0
        )
        rows = (
            self._session.execute(
                select(AnalysisRecordORM)
                .where(*where)
                .order_by(AnalysisRecordORM.created_at.desc(), AnalysisRecordORM.id.desc())
                .limit(limit)
                .offset(offset)
            )
            .scalars()
            .all()
        )
        return [_to_schema(row) for row in rows], total
