"""Object-store selection."""

from __future__ import annotations

from functools import lru_cache

from app.core.config import get_settings
from app.core.logging import get_logger
from app.storage.base import ObjectStore
from app.storage.local import LocalObjectStore

log = get_logger("ragx.storage")


@lru_cache(maxsize=1)
def get_object_store() -> ObjectStore:
    settings = get_settings()
    if settings.storage_backend == "s3":
        from app.storage.s3 import S3ObjectStore  # noqa: PLC0415

        try:
            store = S3ObjectStore()
            log.info("storage.backend_selected", backend="s3", bucket=settings.s3_bucket)
            return store
        except Exception as exc:
            log.error("storage.s3_unavailable_falling_back", error=str(exc))
    log.info("storage.backend_selected", backend="local", path=settings.storage_local_path)
    return LocalObjectStore()


__all__ = ["get_object_store"]
