"""S3-compatible object store (AWS S3, MinIO, Cloudflare R2, ...).

boto3 is synchronous, so every call is offloaded to a worker thread to keep the
event loop free.
"""

from __future__ import annotations

import asyncio
from typing import Any

from app.core.config import get_settings
from app.core.errors import NotFoundError, StorageError
from app.storage.base import ObjectStore, StoredObject


class S3ObjectStore(ObjectStore):
    backend_name = "s3"

    def __init__(self) -> None:
        try:
            import boto3  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise StorageError("boto3 is required for the S3 storage backend.") from exc

        settings = get_settings()
        if not settings.s3_bucket:
            raise StorageError("S3_BUCKET must be set when STORAGE_BACKEND=s3.")

        self.bucket = settings.s3_bucket
        client_kwargs: dict[str, Any] = {"region_name": settings.s3_region}
        if settings.s3_endpoint_url:
            client_kwargs["endpoint_url"] = settings.s3_endpoint_url
        if settings.aws_access_key_id and settings.aws_secret_access_key:
            client_kwargs["aws_access_key_id"] = settings.aws_access_key_id
            client_kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
        self._client = boto3.client("s3", **client_kwargs)

    async def put(self, key: str, data: bytes, content_type: str | None = None) -> StoredObject:
        extra: dict[str, Any] = {"ContentType": content_type} if content_type else {}

        def _put() -> None:
            self._client.put_object(Bucket=self.bucket, Key=key, Body=data, **extra)

        try:
            await asyncio.to_thread(_put)
        except Exception as exc:
            raise StorageError(f"Failed to upload '{key}' to S3.", detail=str(exc)) from exc
        return StoredObject(key=key, size_bytes=len(data), backend=self.backend_name, content_type=content_type)

    async def get(self, key: str) -> bytes:
        def _get() -> bytes:
            response = self._client.get_object(Bucket=self.bucket, Key=key)
            return response["Body"].read()

        try:
            return await asyncio.to_thread(_get)
        except self._client.exceptions.NoSuchKey as exc:
            raise NotFoundError(f"Object '{key}' was not found in S3.") from exc
        except Exception as exc:
            raise StorageError(f"Failed to read '{key}' from S3.", detail=str(exc)) from exc

    async def delete(self, key: str) -> None:
        def _delete() -> None:
            self._client.delete_object(Bucket=self.bucket, Key=key)

        try:
            await asyncio.to_thread(_delete)
        except Exception as exc:
            raise StorageError(f"Failed to delete '{key}' from S3.", detail=str(exc)) from exc

    async def exists(self, key: str) -> bool:
        def _head() -> bool:
            try:
                self._client.head_object(Bucket=self.bucket, Key=key)
                return True
            except Exception:
                return False

        return await asyncio.to_thread(_head)

    async def health(self) -> dict:
        def _head_bucket() -> dict:
            self._client.head_bucket(Bucket=self.bucket)
            return {"backend": self.backend_name, "status": "healthy", "bucket": self.bucket}

        try:
            return await asyncio.to_thread(_head_bucket)
        except Exception as exc:
            return {"backend": self.backend_name, "status": "unhealthy", "error": str(exc)}


__all__ = ["S3ObjectStore"]
