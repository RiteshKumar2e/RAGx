"""Schema creation and seeding.

``create_all`` is used for first-run bootstrap; Alembic (``backend/alembic``)
owns schema evolution once a deployment is live.
"""

from __future__ import annotations

from sqlalchemy import select

from app.core.logging import get_logger
from app.db.session import get_engine, session_scope
from app.models import Base, User
from app.models.user import LOCAL_USER_ID

log = get_logger("ragx.db.init")


async def create_schema() -> None:
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    log.info("db.schema_ready", tables=len(Base.metadata.tables))


async def seed_defaults() -> None:
    async with session_scope() as session:
        existing = await session.scalar(select(User).where(User.id == LOCAL_USER_ID))
        if existing is None:
            session.add(
                User(
                    id=LOCAL_USER_ID,
                    email="local@ragx.local",
                    display_name="Local User",
                    preferences={},
                )
            )
            log.info("db.seeded_local_user")


async def init_db() -> None:
    await create_schema()
    await seed_defaults()


__all__ = ["init_db", "create_schema", "seed_defaults"]
