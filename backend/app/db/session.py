"""Database engine, session lifecycle and connectivity check.

SQLite is the documented development store. The URL comes from configuration
(VOXSENTINEL_DB_URL), so swapping to PostgreSQL later means changing one value.
"""
from __future__ import annotations

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.db.base import Base


class Database:
    def __init__(self, url: str) -> None:
        connect_args: dict[str, object] = {}
        if url.startswith("sqlite"):
            # FastAPI runs sync endpoints in a threadpool; sessions may cross threads.
            connect_args["check_same_thread"] = False
        self._engine: Engine = create_engine(url, connect_args=connect_args, future=True)
        self._session_factory = sessionmaker(bind=self._engine, expire_on_commit=False, future=True)

    @property
    def engine(self) -> Engine:
        return self._engine

    def create_all(self) -> None:
        """Import models so they register on Base.metadata, then create tables."""
        from app.models import analysis_record, audit_event, protection  # noqa: F401  (registers the tables)

        Base.metadata.create_all(self._engine)

    def session(self) -> Session:
        return self._session_factory()

    def ping(self) -> bool:
        """True when the database answers a trivial query."""
        try:
            with self._engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            return True
        except Exception:  # noqa: BLE001 - health must never crash
            return False

    def dispose(self) -> None:
        self._engine.dispose()
