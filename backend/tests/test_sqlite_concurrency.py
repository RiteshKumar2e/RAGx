"""SQLite concurrency configuration.

RAGX writes from several places at once -- request handlers, background
ingestion tasks and evaluation runs (one row per strategy). With SQLite's
defaults those writes fail instantly with "database is locked", which showed up
as a 500 from `POST /api/v1/evaluation/run`.
"""

from __future__ import annotations

import asyncio

import pytest
from sqlalchemy import text

from app.db.session import get_engine


@pytest.mark.anyio
async def test_sqlite_pragmas_are_applied() -> None:
    engine = get_engine()
    async with engine.connect() as conn:
        busy = (await conn.execute(text("PRAGMA busy_timeout"))).scalar()
        journal = (await conn.execute(text("PRAGMA journal_mode"))).scalar()
        foreign_keys = (await conn.execute(text("PRAGMA foreign_keys"))).scalar()

    # busy_timeout is the one that prevents an instant lock failure.
    assert busy and int(busy) >= 5000, "a competing writer must wait, not fail immediately"
    assert str(journal).lower() == "wal", "WAL lets readers proceed while a writer holds the file"
    assert int(foreign_keys) == 1


@pytest.mark.anyio
async def test_concurrent_writes_do_not_deadlock() -> None:
    """Several concurrent writers must all succeed rather than raising."""
    from app.db.session import session_scope
    from app.models.evaluation import EvaluationRun

    async def write(index: int) -> str:
        async with session_scope() as session:
            run = EvaluationRun(
                name=f"concurrency-probe-{index}",
                dataset="test",
                strategy="naive",
                status="completed",
                question_count=0,
            )
            session.add(run)
            await session.flush()
            return run.id

    ids = await asyncio.gather(*(write(i) for i in range(8)))
    assert len(set(ids)) == 8

    # Clean up so the probe rows do not leak into other tests.
    from sqlalchemy import delete

    async with session_scope() as session:
        await session.execute(
            delete(EvaluationRun).where(EvaluationRun.name.like("concurrency-probe-%"))
        )
