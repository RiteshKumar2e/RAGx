"""Object-storage interface.

Original uploads and derived artefacts (figure crops, table renders) are stored
here, never in the database. Two backends implement the same contract: a local
filesystem store and an S3-compatible store (AWS S3, MinIO, R2, ...).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(slots=True)
class StoredObject:
    key: str
    size_bytes: int
    backend: str
    content_type: str | None = None


class ObjectStore(ABC):
    backend_name: str = "base"

    @abstractmethod
    async def put(self, key: str, data: bytes, content_type: str | None = None) -> StoredObject: ...

    @abstractmethod
    async def get(self, key: str) -> bytes: ...

    @abstractmethod
    async def delete(self, key: str) -> None: ...

    @abstractmethod
    async def exists(self, key: str) -> bool: ...

    @abstractmethod
    async def health(self) -> dict: ...

    @staticmethod
    def build_key(document_id: str, filename: str, prefix: str = "documents") -> str:
        return f"{prefix}/{document_id}/{filename}"


__all__ = ["ObjectStore", "StoredObject"]
