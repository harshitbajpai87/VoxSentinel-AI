"""Persistent audit events (Module 7).

Security events are APPEND-ONLY from the API's perspective. No passwords,
tokens, API keys, raw audio or audio bytes are ever stored — only event
metadata safe for investigation. Ordinary API endpoints can write events but
the application offers no update/delete route for them.
"""
from __future__ import annotations

from sqlalchemy import JSON, Float, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class AuditEventORM(Base):
    __tablename__ = "audit_events"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    created_at_epoch: Mapped[float] = mapped_column(Float, nullable=False, index=True)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    outcome: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    actor: Mapped[str] = mapped_column(String(64), nullable=False, default="local-user")
    resource_id: Mapped[str] = mapped_column(String(128), nullable=True)
    request_id: Mapped[str] = mapped_column(String(64), nullable=True, index=True)
    metadata_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
