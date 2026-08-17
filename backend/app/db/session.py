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


LIBSQL_DRIVER_HELP = (
    "The async libSQL driver is unavailable, so Turso cannot be used.\n"
    '  Install it with:  pip install "sqlalchemy-libsql>=0.2"\n'
    "  It depends on libsql-experimental, which ships prebuilt wheels for Linux "
    "and macOS only. On Windows pip compiles it from Rust source, so it needs "
    "the Rust toolchain (https://rustup.rs) plus `pip install cmake`.\n"
    "  Alternatively, clear TURSO_DATABASE_URL to use a local SQLite file."
)


def libsql_driver_available() -> bool:
    try:
        import sqlalchemy_libsql.aiolibsql  # noqa: F401, PLC0415

        return True
    except ImportError:
        return False


def _resolve_database_url(settings) -> tuple[str, str | None]:
    """Return ``(url, degradation_warning)``.

    When Turso is configured but its driver cannot be imported, development
    environments fall back to a local SQLite file with a loud warning -- the
    same explicit-degradation contract used for Qdrant, Neo4j and embeddings.

    Production is different: quietly writing to a *different database* than the
    operator configured is a data-integrity problem, not an inconvenience. There
    the missing driver is a hard failure.
    """
    url = settings.sqlalchemy_url
    if not settings.uses_turso or libsql_driver_available():
        return url, None

    if settings.environment == "production":
        raise RuntimeError(
            f"Turso is configured (TURSO_DATABASE_URL) but its driver is missing, and "
            f"ENVIRONMENT=production so RAGX will not silently use a different "
            f"database.\n{LIBSQL_DRIVER_HELP}"
        )

    warning = (
        "TURSO_DATABASE_URL is set but the async libSQL driver is not installed — "
        "falling back to a local SQLite file. Data will NOT go to Turso. "
        + LIBSQL_DRIVER_HELP
    )
    log.warning("db.turso_driver_missing_using_sqlite", detail=warning.replace("\n", " "))

    fallback = settings.model_copy(update={"turso_database_url": "", "database_url": ""})
    return fallback.sqlalchemy_url, warning


#: Set when Turso was requested but unavailable; surfaced by /health and /settings.
DEGRADATION_WARNING: str | None = None


def _configure_sqlite(engine: AsyncEngine, busy_timeout_ms: int) -> None:
    """Make a local SQLite file safe for RAGX's concurrency.

    RAGX writes from several places at once: request handlers, background
    ingestion tasks and evaluation runs. SQLite's defaults (rollback journal, no
    busy timeout) fail such writes immediately with "database is locked".

    * WAL lets readers proceed while a writer holds the file, which is the usual
      pattern here (a query reading while ingestion writes).
    * busy_timeout makes a competing writer *wait* for the lock instead of
      erroring straight away.
    * synchronous=NORMAL is the standard, safe pairing with WAL.
    """
    from sqlalchemy import event  # noqa: PLC0415

    @event.listens_for(engine.sync_engine, "connect")
    def _set_sqlite_pragmas(dbapi_connection, _record):  # pragma: no cover - driver hook
        cursor = dbapi_connection.cursor()
        try:
            # Applied independently: switching to WAL needs brief exclusive
            # access, so it can fail when another process already holds the
            # file. busy_timeout is per-connection and always applies -- and it
            # is the one that actually prevents instant "database is locked",
            # so a WAL failure must not stop it being set.
            for pragma in (
                f"PRAGMA busy_timeout={busy_timeout_ms}",
                "PRAGMA journal_mode=WAL",
                "PRAGMA synchronous=NORMAL",
                "PRAGMA foreign_keys=ON",
            ):
                try:
                    cursor.execute(pragma)
                except Exception as exc:
                    log.warning("db.sqlite_pragma_failed", pragma=pragma, error=str(exc)[:120])
        finally:
            cursor.close()


def get_engine() -> AsyncEngine:
    global _engine, DEGRADATION_WARNING
    if _engine is None:
        settings = get_settings()
        url, DEGRADATION_WARNING = _resolve_database_url(settings)
        kwargs: dict = {"echo": settings.db_echo, "future": True}

        is_turso = "aiolibsql" in url or "libsql" in url
        if is_turso:
            # Turso is a remote database, so connections are worth reusing and
            # worth checking for staleness — unlike a local SQLite file.
            kwargs.update(pool_pre_ping=True, pool_recycle=300)

        _engine = create_async_engine(url, **kwargs)

        if not is_turso and url.startswith("sqlite"):
            _configure_sqlite(_engine, settings.sqlite_busy_timeout_ms)
        log.info(
            "db.engine_created",
            flavour="turso" if "aiolibsql" in url else "sqlite",
            dialect=url.split("://", 1)[0],
            degraded=bool(DEGRADATION_WARNING),
        )
    return _engine


def active_database_flavour() -> str:
    """What the engine is *actually* connected to, after any degradation."""
    settings = get_settings()
    if DEGRADATION_WARNING:
        return "sqlite"
    return settings.database_flavour


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
    global _engine, _sessionmaker, DEGRADATION_WARNING
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _sessionmaker = None
    DEGRADATION_WARNING = None


__all__ = [
    "get_engine",
    "get_sessionmaker",
    "get_session",
    "session_scope",
    "dispose_engine",
    "active_database_flavour",
    "libsql_driver_available",
    "LIBSQL_DRIVER_HELP",
]
