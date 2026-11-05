"""Data-retention enforcement (Module 7C).

A real, idempotent cleanup that deletes records older than the configured
policy — not a settings toggle that does nothing. Conservative by default:
retention is DISABLED unless VOXSENTINEL_RETENTION_ENABLED=true, so a fresh
deployment never silently destroys history.

Deletion is PERMANENT for database rows (documented in the UI): we do not
claim erasure of any external backups the operator may keep.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import delete, func, select

from app.core.config import Settings
from app.models.analysis_record import AnalysisRecordORM
from app.models.audit_event import AuditEventORM
from app.services.audit import log_event

logger = logging.getLogger("voxsentinel.retention")


@dataclass(frozen=True)
class CleanupResult:
    analyses_deleted: int
    audit_events_deleted: int
    dry_run: bool
    cutoff_analyses_iso: str | None
    cutoff_audit_iso: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "analyses_deleted": self.analyses_deleted,
            "audit_events_deleted": self.audit_events_deleted,
            "dry_run": self.dry_run,
            "cutoff_analyses_iso": self.cutoff_analyses_iso,
            "cutoff_audit_iso": self.cutoff_audit_iso,
        }


def run_cleanup(settings: Settings, session, *, dry_run: bool = False,
                now: float | None = None) -> CleanupResult:
    """Delete (or preview) records older than the configured retention.

    Idempotent and safe to retry: each run recomputes cutoffs from now().
    Audit events about the cleanup itself are always appended AFTER the
    sweep, are excluded from their own deletion window, and record what was
    removed so evidence survives the cleanup.
    """
    now = now if now is not None else time.time()
    # days <= 0 means "keep forever": that class is never touched.
    analyses_enabled = settings.retention_analyses_days > 0
    audit_enabled = settings.retention_audit_days > 0
    cutoff_analyses = now - settings.retention_analyses_days * 86400.0 if analyses_enabled else 0.0
    cutoff_audit = now - settings.retention_audit_days * 86400.0 if audit_enabled else 0.0

    def iso(epoch: float) -> str:
        return (
            datetime.fromtimestamp(epoch, tz=UTC)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )

    # created_at is stored as a uniform ISO-8601 UTC string, so a plain
    # lexicographic comparison is exact and needs NO schema migration.
    iso_cutoff_analyses = iso(cutoff_analyses) if analyses_enabled else None

    analyses_expired = session.execute(
        select(func.count()).select_from(AnalysisRecordORM).where(
            analyses_enabled
            and AnalysisRecordORM.created_at < iso_cutoff_analyses
        )
    ).scalar() or 0
    audit_expired = session.execute(
        select(func.count()).select_from(AuditEventORM).where(
            AuditEventORM.created_at_epoch < cutoff_audit
        )
    ).scalar() or 0

    if not dry_run:
        if analyses_enabled:
            session.execute(
                delete(AnalysisRecordORM).where(AnalysisRecordORM.created_at < iso_cutoff_analyses)
            )
        if audit_enabled:
            session.execute(
                delete(AuditEventORM).where(AuditEventORM.created_at_epoch < cutoff_audit)
            )
        session.commit()
    else:
        session.rollback()

    result = CleanupResult(
        analyses_deleted=int(analyses_expired),
        audit_events_deleted=int(audit_expired),
        dry_run=dry_run,
        cutoff_analyses_iso=iso_cutoff_analyses if settings.retention_analyses_days else None,
        cutoff_audit_iso=iso(cutoff_audit) if settings.retention_audit_days else None,
    )
    log_event(
        session,
        event_type="data_retention",
        action="cleanup" if not dry_run else "cleanup_preview",
        outcome="success",
        metadata={
            **result.as_dict(),
            "policy": {
                "retention_enabled": settings.retention_enabled,
                "analyses_days": settings.retention_analyses_days,
                "audit_days": settings.retention_audit_days,
                "note": "Permanent deletion of database rows; operator backups are outside this system.",
            },
        },
    )
    logger.info("retention cleanup dry_run=%s analyses=%d audit=%d",
                dry_run, result.analyses_deleted, result.audit_events_deleted)
    return result


def last_cleanup_info(session) -> dict[str, Any]:
    """The most recent cleanup event, for the security UI."""
    from sqlalchemy import select as _select

    stmt = (
        _select(AuditEventORM)
        .where(AuditEventORM.event_type == "data_retention")
        .order_by(AuditEventORM.created_at_epoch.desc())
        .limit(1)
    )
    row = session.execute(stmt).scalars().first()
    if row is None:
        return {"last_cleanup": None}
    return {
        "last_cleanup": {
            "created_at": row.created_at,
            "action": row.action,
            "metadata": row.metadata_json or {},
        }
    }
