"""Qdrant vector store.

Runs against a Qdrant server when ``QDRANT_URL`` is set, and in Qdrant's
embedded local mode (persisting to disk) otherwise. Both paths use the same
client API, so retrieval code is identical either way.

Payloads carry the citation provenance (document, page, section, figure/table
labels) so retrieval results are citable without a second database round-trip,
and so modality filters can be pushed down into the index.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any

from app.core.config import get_settings
from app.core.errors import StorageError
from app.core.logging import get_logger

log = get_logger("ragx.vector")

# Embedded Qdrant takes an exclusive lock on its storage directory, so only one
# process may hold it. Running a second backend (a forgotten `uvicorn`, a test
# run, a reload loop) otherwise fails deep inside retrieval with an opaque
# message and an empty result set.
_LOCK_HINT = (
    "The embedded Qdrant store at '{path}' is locked by another process. "
    "Only one RAGX backend can use embedded mode at a time — stop the other "
    "instance, or run a Qdrant server and set QDRANT_URL to share the index."
)


def _translate_storage_error(exc: Exception, path: str) -> StorageError:
    text = str(exc)
    if "already accessed by another instance" in text or "is locked" in text.lower():
        return StorageError(_LOCK_HINT.format(path=path), detail=text[:300])
    return StorageError("Qdrant is unavailable.", detail=text[:300])

# Qdrant point ids must be UUIDs or unsigned ints; chunk ids are 32-char hex,
# which maps to a UUID losslessly and reversibly.
def chunk_id_to_point_id(chunk_id: str) -> str:
    try:
        return str(uuid.UUID(hex=chunk_id))
    except ValueError:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


@dataclass(slots=True)
class VectorRecord:
    chunk_id: str
    vector: list[float]
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class VectorHit:
    chunk_id: str
    score: float
    payload: dict[str, Any] = field(default_factory=dict)


class QdrantVectorStore:
    def __init__(self) -> None:
        settings = get_settings()
        self.collection = settings.qdrant_collection
        self.dimension = settings.embedding_dimension
        self.mode = "server" if settings.qdrant_url else "embedded"
        self._settings = settings
        self._client: Any = None
        self._models: Any = None
        self._ready = False
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------- lifecycle
    def _build_client(self) -> Any:
        try:
            from qdrant_client import QdrantClient, models  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise StorageError("qdrant-client is not installed.") from exc

        self._models = models
        settings = self._settings
        if settings.qdrant_url:
            client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key or None,
                timeout=30,
            )
        else:
            # Embedded mode: real Qdrant, persisted to a local directory.
            client = QdrantClient(path=settings.qdrant_path)
        return client

    async def ensure_ready(self, dimension: int | None = None) -> None:
        """Create the collection on first use (idempotent)."""
        async with self._lock:
            if self._ready and dimension in (None, self.dimension):
                return
            if dimension:
                self.dimension = dimension
            if self._client is None:
                self._client = await asyncio.to_thread(self._build_client)
            models = self._models

            def _ensure() -> None:
                existing = {c.name for c in self._client.get_collections().collections}
                if self.collection not in existing:
                    self._client.create_collection(
                        collection_name=self.collection,
                        vectors_config=models.VectorParams(
                            size=self.dimension, distance=models.Distance.COSINE
                        ),
                    )
                    # Indexes that make modality/document filters cheap. They are
                    # a no-op in embedded mode (which scans anyway), so they are
                    # only created against a real server.
                    if self.mode == "server":
                        for field_name, schema in (
                            ("document_id", models.PayloadSchemaType.KEYWORD),
                            ("modality", models.PayloadSchemaType.KEYWORD),
                            ("page_number", models.PayloadSchemaType.INTEGER),
                        ):
                            try:
                                self._client.create_payload_index(
                                    collection_name=self.collection,
                                    field_name=field_name,
                                    field_schema=schema,
                                )
                            except Exception:  # pragma: no cover - index may exist
                                pass
                    log.info("vector.collection_created", collection=self.collection, dim=self.dimension)

            try:
                await asyncio.to_thread(_ensure)
            except Exception as exc:
                raise _translate_storage_error(exc, self._settings.qdrant_path) from exc
            self._ready = True

    # ---------------------------------------------------------------- writes
    async def upsert(self, records: list[VectorRecord]) -> int:
        if not records:
            return 0
        await self.ensure_ready(dimension=len(records[0].vector))
        models = self._models
        points = [
            models.PointStruct(
                id=chunk_id_to_point_id(r.chunk_id),
                vector=r.vector,
                payload={**r.payload, "chunk_id": r.chunk_id},
            )
            for r in records
        ]

        def _upsert() -> None:
            self._client.upsert(collection_name=self.collection, points=points, wait=True)

        try:
            await asyncio.to_thread(_upsert)
        except Exception as exc:
            raise StorageError("Failed to write vectors to Qdrant.", detail=str(exc)) from exc
        return len(points)

    async def delete_document(self, document_id: str) -> None:
        await self.ensure_ready()
        models = self._models

        def _delete() -> None:
            self._client.delete(
                collection_name=self.collection,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="document_id", match=models.MatchValue(value=document_id)
                            )
                        ]
                    )
                ),
                wait=True,
            )

        try:
            await asyncio.to_thread(_delete)
        except Exception as exc:
            log.error("vector.delete_failed", document_id=document_id, error=str(exc))

    async def clear(self) -> None:
        await self.ensure_ready()

        def _drop() -> None:
            self._client.delete_collection(collection_name=self.collection)

        await asyncio.to_thread(_drop)
        self._ready = False

    # ---------------------------------------------------------------- search
    def _build_filter(
        self,
        document_ids: list[str] | None,
        modalities: list[str] | None,
        exclude_chunk_ids: list[str] | None,
    ) -> Any:
        models = self._models
        must: list[Any] = []
        must_not: list[Any] = []
        if document_ids:
            must.append(
                models.FieldCondition(key="document_id", match=models.MatchAny(any=list(document_ids)))
            )
        if modalities:
            must.append(models.FieldCondition(key="modality", match=models.MatchAny(any=list(modalities))))
        if exclude_chunk_ids:
            must_not.append(
                models.FieldCondition(key="chunk_id", match=models.MatchAny(any=list(exclude_chunk_ids)))
            )
        if not must and not must_not:
            return None
        return models.Filter(must=must or None, must_not=must_not or None)

    async def search(
        self,
        vector: list[float],
        limit: int = 10,
        document_ids: list[str] | None = None,
        modalities: list[str] | None = None,
        exclude_chunk_ids: list[str] | None = None,
        score_threshold: float | None = None,
    ) -> list[VectorHit]:
        await self.ensure_ready()
        query_filter = self._build_filter(document_ids, modalities, exclude_chunk_ids)

        def _search() -> list[Any]:
            # qdrant-client >= 1.14 replaced `search()` with `query_points()`.
            # Support both so the same code runs against either version.
            if hasattr(self._client, "query_points"):
                return self._client.query_points(
                    collection_name=self.collection,
                    query=vector,
                    limit=limit,
                    query_filter=query_filter,
                    score_threshold=score_threshold,
                    with_payload=True,
                ).points
            return self._client.search(
                collection_name=self.collection,
                query_vector=vector,
                limit=limit,
                query_filter=query_filter,
                score_threshold=score_threshold,
                with_payload=True,
            )

        try:
            results = await asyncio.to_thread(_search)
        except Exception as exc:
            raise _translate_storage_error(exc, self._settings.qdrant_path) from exc

        hits: list[VectorHit] = []
        for point in results:
            payload = dict(point.payload or {})
            chunk_id = payload.get("chunk_id")
            if not chunk_id:
                continue
            hits.append(VectorHit(chunk_id=chunk_id, score=float(point.score), payload=payload))
        return hits

    # ---------------------------------------------------------------- status
    async def count(self) -> int:
        try:
            await self.ensure_ready()

            def _count() -> int:
                return self._client.count(collection_name=self.collection, exact=True).count

            return await asyncio.to_thread(_count)
        except Exception:
            return 0

    async def health(self) -> dict[str, Any]:
        status: dict[str, Any] = {
            "store": "qdrant",
            "mode": self.mode,
            "collection": self.collection,
            "dimension": self.dimension,
        }
        try:
            await self.ensure_ready()
            status.update(healthy=True, status_text="healthy", vectors=await self.count())
        except Exception as exc:
            status.update(healthy=False, status_text="unhealthy", error=str(exc)[:200])
        return status

    async def close(self) -> None:
        if self._client is not None:
            try:
                await asyncio.to_thread(self._client.close)
            except Exception:  # pragma: no cover
                pass
            self._client = None
            self._ready = False


_store: QdrantVectorStore | None = None


def get_vector_store() -> QdrantVectorStore:
    global _store
    if _store is None:
        _store = QdrantVectorStore()
    return _store


async def close_vector_store() -> None:
    global _store
    if _store is not None:
        await _store.close()
    _store = None


__all__ = [
    "QdrantVectorStore",
    "VectorRecord",
    "VectorHit",
    "get_vector_store",
    "close_vector_store",
    "chunk_id_to_point_id",
]
