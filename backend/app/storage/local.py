"""Filesystem-backed object store (default for local development)."""

from __future__ import annotations

import asyncio
from pathlib import Path

from app.core.config import get_settings
from app.core.errors import NotFoundError, StorageError
from app.storage.base import ObjectStore, StoredObject


class LocalObjectStore(ObjectStore):
    backend_name = "local"

    def __init__(self, root: str | Path | None = None):
        self.root = Path(root or get_settings().storage_local_path).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        # Resolve and confirm containment -- blocks ``../`` traversal in keys.
        candidate = (self.root / key).resolve()
        if not str(candidate).startswith(str(self.root)):
            raise StorageError("Invalid object key.")
        return candidate

    async def put(self, key: str, data: bytes, content_type: str | None = None) -> StoredObject:
        path = self._path(key)

        def _write() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)

        await asyncio.to_thread(_write)
        return StoredObject(key=key, size_bytes=len(data), backend=self.backend_name, content_type=content_type)

    async def get(self, key: str) -> bytes:
        path = self._path(key)
        if not path.exists():
            raise NotFoundError(f"Object '{key}' was not found in local storage.")
        return await asyncio.to_thread(path.read_bytes)

    async def delete(self, key: str) -> None:
        path = self._path(key)

        def _delete() -> None:
            if path.exists():
                path.unlink()
            parent = path.parent
            try:
                if parent != self.root and not any(parent.iterdir()):
                    parent.rmdir()
            except OSError:
                pass

        await asyncio.to_thread(_delete)

    async def exists(self, key: str) -> bool:
        return await asyncio.to_thread(self._path(key).exists)

    async def health(self) -> dict:
        try:
            probe = self.root / ".healthcheck"
            await asyncio.to_thread(probe.write_text, "ok", "utf-8")
            await asyncio.to_thread(probe.unlink)
            return {"backend": self.backend_name, "status": "healthy", "location": str(self.root)}
        except Exception as exc:  # pragma: no cover - filesystem failure
            return {"backend": self.backend_name, "status": "unhealthy", "error": str(exc)}


__all__ = ["LocalObjectStore"]
