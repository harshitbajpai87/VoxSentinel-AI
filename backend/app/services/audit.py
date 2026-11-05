"""Audit-log service: structured, append-only security events (Module 7B)."""
from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.audit_event import AuditEventORM

logger = logging.getLogger("voxsentinel.audit")

REDACTED_KEYS = {"password", "token", "api_key", "authorization", "secret", "consent_token"}


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _sanitize(metadata: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in (metadata or {}).items():
        if key.lower() in REDACTED_KEYS:
            safe[key] = "[redacted]"
        else:
            safe[key] = value
    return safe


def log_event(
    session: Session,
    event_type: str,
    action: str,
    outcome: str,
    *,
    actor: str = "local-user",
    resource_id: str | None = None,
    request_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Append one audit event. Returns the event id. Never raises upward:
    audit failures are logged server-side and must not break the request."""
    event_id = uuid.uuid4().hex
    now = datetime.now(UTC)
    try:
        event = AuditEventORM(
            id=event_id,
            created_at=now.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            created_at_epoch=now.timestamp(),
            event_type=event_type,
            action=action,
            outcome=outcome,
            actor=actor,
            resource_id=resource_id,
            request_id=request_id,
            metadata_json=_sanitize(metadata or {}),
        )
        session.add(event)
        session.commit()
    except Exception:  # noqa: BLE001 - availability beats audit completeness
        session.rollback()
        logger.warning("audit event write failed", exc_info=True)
    return event_id


def list_events(
    session: Session,
    *,
    event_type: str | None = None,
    outcome: str | None = None,
    since_epoch: float | None = None,
    until_epoch: float | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[dict], int]:
    stmt = select(AuditEventORM)
    if event_type:
        stmt = stmt.where(AuditEventORM.event_type == event_type)
    if outcome:
        stmt = stmt.where(AuditEventORM.outcome == outcome)
    if since_epoch is not None:
        stmt = stmt.where(AuditEventORM.created_at_epoch >= since_epoch)
    if until_epoch is not None:
        stmt = stmt.where(AuditEventORM.created_at_epoch <= until_epoch)

    from sqlalchemy import func

    count_stmt = stmt.order_by(None)
    total = session.execute(select(func.count()).select_from(count_stmt.subquery())).scalar() or 0
    rows = session.execute(
        stmt.order_by(AuditEventORM.created_at_epoch.desc()).limit(limit).offset(offset)
    ).scalars().all()
    events = [
        {
            "id": row.id,
            "created_at": row.created_at,
            "event_type": row.event_type,
            "action": row.action,
            "outcome": row.outcome,
            "actor": row.actor,
            "resource_id": row.resource_id,
            "request_id": row.request_id,
            "metadata": row.metadata_json or {},
        }
        for row in rows
    ]
    return events, int(total)
