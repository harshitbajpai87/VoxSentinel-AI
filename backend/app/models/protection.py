"""Additive tables for the Family Emergency & Scam Protection module.

Isolated from the existing `analysis_records` / `audit_events` tables: nothing
here changes existing schemas, and dropping these tables would not affect the
core VoxSentinel workflow.
"""
from __future__ import annotations

from sqlalchemy import JSON, Boolean, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class ProtectionAnalysisORM(Base):
    """One completed (or failed) protection analysis.

    The full evidence document is stored as JSON so the result can be reviewed
    later exactly as it was produced. Raw audio is never stored.
    """

    __tablename__ = "protection_analyses"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    created_at_epoch: Mapped[float] = mapped_column(Float, nullable=False, index=True)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    duration_s: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    status: Mapped[str] = mapped_column(String(24), nullable=False, index=True)
    risk_category: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    profile_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    payload_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)


class FamilyVoiceProfileORM(Base):
    """Authorized family voice reference used for speaker comparison.

    Stores a consented reference clip path on local disk plus consent metadata.
    Deleting the profile deletes the stored clip (see router).
    """

    __tablename__ = "family_voice_profiles"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    created_at_epoch: Mapped[float] = mapped_column(Float, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    relationship: Mapped[str] = mapped_column(String(64), nullable=False)
    consent: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    reference_path: Mapped[str] = mapped_column(String(512), nullable=False)
    duration_s: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
