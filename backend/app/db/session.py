"""Async SQLAlchemy engine/session management.

The relational layer is SQLite-compatible end to end: a local file in
development, Turso (hosted libSQL) in production. Because both speak SQLite,
the schema and queries are identical in either environment -- there is no
dialect drift between what you test against and what you deploy on.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger("ragx.db")

_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def _require_libsql_driver() -> None:
    """Fail with an actionable message when the Turso driver is absent.

    ``sqlalchemy-libsql < 0.2`` ships only a *sync* dialect, and the async one
    depends on ``libsql-experimental``, which builds from Rust source on
    platforms without a prebuilt wheel. Without this check SQLAlchemy raises a
    bare ``NoSuchModuleError`` that says nothing about how to fix it.
    """
    try:
        import sqlalchemy_libsql.aiolibsql  # noqa: F401, PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "Turso is configured (TURSO_DATABASE_URL) but the async libSQL driver "
            "is unavailable.\n"
            "  Install it with:  pip install \"sqlalchemy-libsql>=0.2\"\n"
            "  That pulls in libsql-experimental, which compiles from Rust source "
            "when no wheel exists for your platform — it needs the Rust toolchain "
            "and cmake (`pip install cmake`).\n"
            "  If it will not build, unset TURSO_DATABASE_URL to fall back to a "
            "local SQLite file."
        ) from exc


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        settings = get_settings()
        url = settings.sqlalchemy_url
        kwargs: dict = {"echo": settings.db_echo, "future": True}

        if settings.uses_turso:
            _require_libsql_driver()
            # Turso is a remote HTTP database, so connections are worth reusing
            # and worth checking for staleness — unlike a local SQLite file.
            kwargs.update(pool_pre_ping=True, pool_recycle=300)

        _engine = create_async_engine(url, **kwargs)
        log.info(
            "db.engine_created",
            flavour=settings.database_flavour,
            dialect=url.split("://", 1)[0],
        )
    return _engine


def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    global _sessionmaker
    if _sessionmaker is None:
        _sessionmaker = async_sessionmaker(
            bind=get_engine(), expire_on_commit=False, class_=AsyncSession
        )
    return _sessionmaker


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency yielding a transactional session."""
    async with get_sessionmaker()() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Context manager for background tasks and scripts."""
    async with get_sessionmaker()() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def dispose_engine() -> None:
    global _engine, _sessionmaker
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _sessionmaker = None


__all__ = ["get_engine", "get_sessionmaker", "get_session", "session_scope", "dispose_engine"]
