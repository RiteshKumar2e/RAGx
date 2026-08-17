"""Knowledge-base management: upload, status, inspection, deletion, reindex."""

from __future__ import annotations

import asyncio
from typing import Any

from sqlalchemy import delete, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.config import get_settings
from app.core.errors import NotFoundError, ValidationError
from app.core.logging import get_logger
from app.core.security import file_extension, validate_upload
from app.indexing.bm25_index import get_bm25_index
from app.indexing.graph_store import get_graph_store
from app.indexing.vector_store import get_vector_store
from app.ingestion.pipeline import compute_checksum, get_pipeline
from app.models.document import (
    Chunk,
    Document,
    DocumentEntity,
    DocumentStatus,
    EntityRelation,
)
from app.storage import get_object_store

log = get_logger("ragx.documents")

ACTIVE_STATUSES = {
    DocumentStatus.UPLOADED,
    DocumentStatus.PARSING,
    DocumentStatus.CHUNKING,
    DocumentStatus.EMBEDDING,
    DocumentStatus.GRAPH_INDEXING,
}


class DocumentService:
    def __init__(self) -> None:
        self.object_store = get_object_store()
        self.pipeline = get_pipeline()

    # ------------------------------------------------------------------ upload
    async def upload(
        self,
        session: AsyncSession,
        filename: str,
        content_type: str | None,
        data: bytes,
        source_url: str | None = None,
    ) -> dict[str, Any]:
        settings = get_settings()
        safe_name = validate_upload(filename, content_type, data[:512], len(data))
        checksum = compute_checksum(data)

        existing = await session.scalar(
            select(Document).where(Document.checksum == checksum).limit(1)
        )
        if existing is not None:
            return {
                "document_id": existing.id,
                "filename": existing.filename,
                "status": existing.status.value,
                "message": f"'{safe_name}' is already indexed as '{existing.filename}'.",
                "duplicate_of": existing.id,
            }

        document = Document(
            filename=safe_name,
            title=None,
            file_type=file_extension(safe_name),
            mime_type=content_type,
            size_bytes=len(data),
            checksum=checksum,
            status=DocumentStatus.UPLOADED,
            processing_steps=self.pipeline.initial_steps(),
            storage_backend=self.object_store.backend_name,
            source_url=source_url,
        )
        session.add(document)
        await session.flush()

        object_key = self.object_store.build_key(document.id, safe_name)
        await self.object_store.put(object_key, data, content_type)
        document.object_key = object_key
        await session.flush()

        log.info(
            "document.uploaded",
            document_id=document.id,
            filename=safe_name,
            size_bytes=len(data),
            backend=self.object_store.backend_name,
        )
        return {
            "document_id": document.id,
            "filename": safe_name,
            "status": DocumentStatus.UPLOADED.value,
            "message": f"'{safe_name}' was uploaded and queued for processing.",
            "duplicate_of": None,
        }

    # ------------------------------------------------------------------- list
    async def list_documents(
        self,
        session: AsyncSession,
        page: int = 1,
        page_size: int = 20,
        status: str | None = None,
        file_type: str | None = None,
        search: str | None = None,
    ) -> dict[str, Any]:
        stmt = select(Document).order_by(desc(Document.created_at))
        count_stmt = select(func.count(Document.id))

        if status:
            try:
                status_enum = DocumentStatus(status)
            except ValueError as exc:
                raise ValidationError(f"'{status}' is not a valid document status.") from exc
            stmt = stmt.where(Document.status == status_enum)
            count_stmt = count_stmt.where(Document.status == status_enum)
        if file_type:
            normalized = file_type if file_type.startswith(".") else f".{file_type}"
            stmt = stmt.where(Document.file_type == normalized.lower())
            count_stmt = count_stmt.where(Document.file_type == normalized.lower())
        if search:
            pattern = f"%{search.strip()}%"
            condition = Document.filename.ilike(pattern) | Document.title.ilike(pattern)
            stmt = stmt.where(condition)
            count_stmt = count_stmt.where(condition)

        total = await session.scalar(count_stmt) or 0
        rows = await session.scalars(stmt.offset((page - 1) * page_size).limit(page_size))

        status_rows = await session.execute(
            select(Document.status, func.count(Document.id)).group_by(Document.status)
        )
        status_counts = {
            (s.value if hasattr(s, "value") else str(s)): n for s, n in status_rows.all()
        }

        return {
            "items": list(rows),
            "total": total,
            "page": page,
            "page_size": page_size,
            "status_counts": status_counts,
        }

    # ----------------------------------------------------------------- detail
    async def get_document(self, session: AsyncSession, document_id: str, chunk_limit: int = 50) -> dict[str, Any]:
        document = await session.scalar(
            select(Document)
            .options(selectinload(Document.entities))
            .where(Document.id == document_id)
        )
        if document is None:
            raise NotFoundError(f"Document '{document_id}' was not found.")

        chunks = await session.scalars(
            select(Chunk).where(Chunk.document_id == document_id).order_by(Chunk.ordinal).limit(chunk_limit)
        )
        modality_rows = await session.execute(
            select(Chunk.modality, func.count(Chunk.id))
            .where(Chunk.document_id == document_id)
            .group_by(Chunk.modality)
        )
        modality_breakdown = {
            (m.value if hasattr(m, "value") else str(m)): n for m, n in modality_rows.all()
        }

        entities = sorted(document.entities, key=lambda e: -e.salience)[:60]
        return {
            "document": document,
            "chunks": list(chunks),
            "entities": entities,
            "modality_breakdown": modality_breakdown,
        }

    async def get_chunk(self, session: AsyncSession, chunk_id: str) -> dict[str, Any]:
        from sqlalchemy.orm import joinedload  # noqa: PLC0415

        chunk = await session.scalar(
            select(Chunk).options(joinedload(Chunk.document)).where(Chunk.id == chunk_id)
        )
        if chunk is None:
            raise NotFoundError(f"Evidence chunk '{chunk_id}' was not found.")

        neighbors = await session.scalars(
            select(Chunk)
            .where(
                Chunk.document_id == chunk.document_id,
                Chunk.ordinal.between(max(0, chunk.ordinal - 1), chunk.ordinal + 1),
                Chunk.id != chunk.id,
            )
            .order_by(Chunk.ordinal)
        )
        return {"chunk": chunk, "neighbors": list(neighbors)}

    # ----------------------------------------------------------------- delete
    async def delete_document(self, session: AsyncSession, document_id: str) -> dict[str, Any]:
        document = await session.get(Document, document_id)
        if document is None:
            raise NotFoundError(f"Document '{document_id}' was not found.")
        filename = document.filename

        # Remove from every index *before* the SQL rows disappear, since the
        # object keys are read from those rows.
        await self.pipeline.remove_document(document_id)

        # Bulk deletes in FK order, then a bulk delete of the document itself.
        # ``session.delete(document)`` would re-issue per-row deletes for the
        # already-removed chunks via the ORM cascade.
        await session.execute(delete(EntityRelation).where(EntityRelation.document_id == document_id))
        await session.execute(delete(DocumentEntity).where(DocumentEntity.document_id == document_id))
        await session.execute(delete(Chunk).where(Chunk.document_id == document_id))
        session.expunge(document)
        await session.execute(delete(Document).where(Document.id == document_id))
        await session.flush()

        log.info("document.deleted", document_id=document_id, filename=filename)
        return {"ok": True, "message": f"'{filename}' and all of its indexed data were deleted."}

    # ---------------------------------------------------------------- reindex
    async def reindex(self, session: AsyncSession, document_id: str) -> dict[str, Any]:
        document = await session.get(Document, document_id)
        if document is None:
            raise NotFoundError(f"Document '{document_id}' was not found.")
        if not document.object_key:
            raise ValidationError("The original file is no longer available; re-upload it instead.")

        await self.pipeline.remove_document(document_id)
        await session.execute(delete(EntityRelation).where(EntityRelation.document_id == document_id))
        await session.execute(delete(DocumentEntity).where(DocumentEntity.document_id == document_id))
        await session.execute(delete(Chunk).where(Chunk.document_id == document_id))

        document.status = DocumentStatus.UPLOADED
        document.processing_steps = self.pipeline.initial_steps()
        document.error_message = None
        document.chunk_count = 0
        document.entity_count = 0
        await session.flush()

        return {
            "document_id": document_id,
            "status": DocumentStatus.UPLOADED.value,
            "message": f"'{document.filename}' was queued for reprocessing.",
        }

    # ------------------------------------------------------------------ stats
    async def stats(self, session: AsyncSession) -> dict[str, Any]:
        totals = (
            await session.execute(
                select(
                    func.count(Document.id),
                    func.coalesce(func.sum(Document.chunk_count), 0),
                    func.coalesce(func.sum(Document.token_count), 0),
                    func.coalesce(func.sum(Document.page_count), 0),
                    func.coalesce(func.sum(Document.table_count), 0),
                    func.coalesce(func.sum(Document.figure_count), 0),
                    func.coalesce(func.sum(Document.size_bytes), 0),
                )
            )
        ).one()

        status_rows = await session.execute(
            select(Document.status, func.count(Document.id)).group_by(Document.status)
        )
        status_counts = {
            (s.value if hasattr(s, "value") else str(s)): n for s, n in status_rows.all()
        }

        type_rows = await session.execute(
            select(Document.file_type, func.count(Document.id)).group_by(Document.file_type)
        )
        modality_rows = await session.execute(
            select(Chunk.modality, func.count(Chunk.id)).group_by(Chunk.modality)
        )

        entity_count = await session.scalar(select(func.count(DocumentEntity.id))) or 0
        relation_count = await session.scalar(select(func.count(EntityRelation.id))) or 0

        vector_count, bm25_size = await asyncio.gather(
            get_vector_store().count(), _bm25_size()
        )

        return {
            "total_documents": totals[0] or 0,
            "indexed_documents": status_counts.get("ready", 0),
            "processing_documents": sum(
                status_counts.get(s.value, 0) for s in ACTIVE_STATUSES
            ),
            "failed_documents": status_counts.get("failed", 0),
            "total_chunks": int(totals[1] or 0),
            "total_tokens": int(totals[2] or 0),
            "total_pages": int(totals[3] or 0),
            "total_tables": int(totals[4] or 0),
            "total_figures": int(totals[5] or 0),
            "total_entities": entity_count,
            "total_relations": relation_count,
            "vectors_indexed": vector_count,
            "bm25_documents": bm25_size,
            "storage_bytes": int(totals[6] or 0),
            "by_type": {t: n for t, n in type_rows.all()},
            "by_modality": {
                (m.value if hasattr(m, "value") else str(m)): n for m, n in modality_rows.all()
            },
        }

    # ------------------------------------------------------------- background
    async def process_in_background(self, document_id: str) -> None:
        """Entry point for FastAPI ``BackgroundTasks``. Never raises."""
        try:
            await self.pipeline.process(document_id)
        except Exception as exc:  # pragma: no cover - defensive
            log.error("document.background_processing_failed", document_id=document_id, error=str(exc))

    async def rebuild_indexes(self, session: AsyncSession) -> dict[str, Any]:
        """Rebuild the BM25 index from the relational store.

        Useful after restoring a database, or if the on-disk BM25 snapshot is
        lost. Vector and graph data are rebuilt by reindexing documents.
        """
        from app.indexing.bm25_index import BM25Document  # noqa: PLC0415
        from sqlalchemy.orm import joinedload  # noqa: PLC0415

        rows = await session.scalars(
            select(Chunk).options(joinedload(Chunk.document)).order_by(Chunk.document_id, Chunk.ordinal)
        )
        documents = [
            BM25Document(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                text=chunk.content,
                modality=chunk.modality.value if hasattr(chunk.modality, "value") else str(chunk.modality),
            )
            for chunk in rows.unique()
        ]
        count = await get_bm25_index().replace_all(documents)
        graph_stats = await get_graph_store().stats()
        return {
            "ok": True,
            "message": f"The BM25 index was rebuilt from {count} chunks.",
            "detail": {"bm25_documents": count, "graph_entities": graph_stats.get("entities", 0)},
        }


async def _bm25_size() -> int:
    index = get_bm25_index()
    await index.ensure_loaded()
    return index.size


_service: DocumentService | None = None


def get_document_service() -> DocumentService:
    global _service
    if _service is None:
        _service = DocumentService()
    return _service


__all__ = ["DocumentService", "get_document_service"]
