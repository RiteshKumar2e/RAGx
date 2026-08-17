"""Document ingestion pipeline.

    upload -> parse -> extract (OCR / tables / figures) -> chunk
           -> persist metadata -> embed -> index (vector + BM25)
           -> entity & relation extraction -> graph index -> ready

Each stage updates ``Document.processing_steps`` so the Knowledge Base UI can
render live progress, and every stage records its own duration. A failure in an
optional stage (graph indexing, figure description) degrades the document to
"ready with warnings" rather than losing the whole ingest.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.errors import IngestionError
from app.core.logging import get_logger
from app.core.security import file_extension
from app.db.session import session_scope
from app.indexing.bm25_index import BM25Document, get_bm25_index
from app.indexing.graph_store import get_graph_store
from app.indexing.vector_store import VectorRecord, get_vector_store
from app.ingestion.chunking import ChunkCandidate, StructuralChunker
from app.ingestion.entities import EntityExtractor
from app.ingestion.parsers import get_parser
from app.llm.embeddings import get_embedding_provider
from app.models.document import Chunk, ChunkModality, Document, DocumentEntity, DocumentStatus, EntityRelation
from app.storage import get_object_store

log = get_logger("ragx.ingest")

STEPS = ("upload", "parse", "chunk", "embed", "graph", "ready")


@dataclass
class IngestionOutcome:
    document_id: str
    status: str
    chunks: int = 0
    entities: int = 0
    relations: int = 0
    tables: int = 0
    figures: int = 0
    pages: int = 0
    tokens: int = 0
    duration_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)
    error: str | None = None


class IngestionPipeline:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.chunker = StructuralChunker()
        self.embedder = get_embedding_provider()
        self.vector_store = get_vector_store()
        self.bm25 = get_bm25_index()
        self.graph = get_graph_store()
        self.object_store = get_object_store()
        self.entity_extractor = EntityExtractor()

    # ------------------------------------------------------------- step book
    @staticmethod
    def initial_steps() -> dict[str, Any]:
        return {step: {"status": "pending", "duration_ms": 0.0, "detail": None} for step in STEPS}

    async def _set_step(
        self,
        session: AsyncSession,
        document: Document,
        step: str,
        status: str,
        duration_ms: float = 0.0,
        detail: str | None = None,
        doc_status: DocumentStatus | None = None,
    ) -> None:
        steps = dict(document.processing_steps or self.initial_steps())
        steps[step] = {"status": status, "duration_ms": round(duration_ms, 1), "detail": detail}
        document.processing_steps = steps
        if doc_status is not None:
            document.status = doc_status
        if detail:
            document.status_detail = detail
        await session.flush()
        await session.commit()

    # ------------------------------------------------------------------ main
    async def process(self, document_id: str) -> IngestionOutcome:
        """Run the full pipeline for an already-uploaded document."""
        started = time.perf_counter()
        outcome = IngestionOutcome(document_id=document_id, status="failed")

        async with session_scope() as session:
            document = await session.get(Document, document_id)
            if document is None:
                raise IngestionError(f"Document '{document_id}' does not exist.")
            filename = document.filename
            object_key = document.object_key
            extension = document.file_type

        try:
            data = await self.object_store.get(object_key) if object_key else b""
            if not data:
                raise IngestionError("The stored file is empty or missing from object storage.")

            # -- parse ------------------------------------------------------
            parsed = await self._run_parse(document_id, data, filename, extension)

            # -- chunk ------------------------------------------------------
            candidates = await self._run_chunk(document_id, parsed)

            # -- persist chunks + assets -----------------------------------
            chunk_rows = await self._persist_chunks(document_id, candidates, parsed)

            # -- embed + index ---------------------------------------------
            await self._run_embedding(document_id, chunk_rows)

            # -- graph ------------------------------------------------------
            graph_counts = await self._run_graph(document_id, filename, chunk_rows)

            # -- finalise ---------------------------------------------------
            duration_ms = (time.perf_counter() - started) * 1000
            async with session_scope() as session:
                document = await session.get(Document, document_id)
                document.page_count = parsed.page_count
                document.chunk_count = len(chunk_rows)
                document.table_count = parsed.table_count
                document.figure_count = parsed.figure_count
                document.entity_count = graph_counts["entities"]
                document.token_count = sum(c["token_count"] for c in chunk_rows)
                document.title = document.title or parsed.title or filename
                document.outline = parsed.outline[:200]
                document.doc_metadata = {
                    **(document.doc_metadata or {}),
                    **parsed.metadata,
                    "warnings": parsed.warnings,
                    "embedding_provider": self.embedder.name,
                    "embedding_dimension": self.embedder.dimension,
                    "graph_backend": self.graph.backend,
                    "relations": graph_counts["relations"],
                }
                document.processing_ms = duration_ms
                document.indexed_at = __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
                await self._set_step(
                    session, document, "ready", "completed", duration_ms,
                    detail=f"{len(chunk_rows)} chunks indexed", doc_status=DocumentStatus.READY,
                )

            outcome = IngestionOutcome(
                document_id=document_id,
                status="ready",
                chunks=len(chunk_rows),
                entities=graph_counts["entities"],
                relations=graph_counts["relations"],
                tables=parsed.table_count,
                figures=parsed.figure_count,
                pages=parsed.page_count,
                tokens=sum(c["token_count"] for c in chunk_rows),
                duration_ms=duration_ms,
                warnings=parsed.warnings,
            )
            log.info(
                "ingest.completed",
                document_id=document_id,
                chunks=outcome.chunks,
                entities=outcome.entities,
                relations=outcome.relations,
                duration_ms=round(duration_ms, 1),
            )
            return outcome

        except Exception as exc:
            message = str(exc)[:500]
            log.error("ingest.failed", document_id=document_id, error=message, exc_info=True)
            async with session_scope() as session:
                document = await session.get(Document, document_id)
                if document is not None:
                    document.status = DocumentStatus.FAILED
                    document.error_message = message
                    steps = dict(document.processing_steps or self.initial_steps())
                    for step in STEPS:
                        if steps.get(step, {}).get("status") == "running":
                            steps[step] = {"status": "failed", "duration_ms": 0.0, "detail": message}
                    document.processing_steps = steps
            outcome.error = message
            outcome.duration_ms = (time.perf_counter() - started) * 1000
            return outcome

    # ------------------------------------------------------------- stages
    async def _run_parse(self, document_id: str, data: bytes, filename: str, extension: str) -> Any:
        started = time.perf_counter()
        async with session_scope() as session:
            document = await session.get(Document, document_id)
            await self._set_step(session, document, "upload", "completed", detail=f"{len(data):,} bytes")
            await self._set_step(session, document, "parse", "running", doc_status=DocumentStatus.PARSING)

        parser = get_parser(extension or file_extension(filename))
        parsed = await parser.parse(data, filename)
        if not parsed.blocks:
            raise IngestionError(
                "No readable content was extracted from this file. "
                "If it is a scanned document, enable OCR or configure GEMINI_API_KEY."
            )
        duration = (time.perf_counter() - started) * 1000

        async with session_scope() as session:
            document = await session.get(Document, document_id)
            await self._set_step(
                session, document, "parse", "completed", duration,
                detail=f"{len(parsed.blocks)} blocks · {parsed.page_count} pages · "
                       f"{parsed.table_count} tables · {parsed.figure_count} figures",
            )
        return parsed

    async def _run_chunk(self, document_id: str, parsed: Any) -> list[ChunkCandidate]:
        started = time.perf_counter()
        async with session_scope() as session:
            document = await session.get(Document, document_id)
            await self._set_step(session, document, "chunk", "running", doc_status=DocumentStatus.CHUNKING)

        candidates = await asyncio.to_thread(self.chunker.chunk, parsed)
        if not candidates:
            raise IngestionError("Chunking produced no content for this document.")
        duration = (time.perf_counter() - started) * 1000

        async with session_scope() as session:
            document = await session.get(Document, document_id)
            await self._set_step(
                session, document, "chunk", "completed", duration,
                detail=f"{len(candidates)} chunks",
            )
        return candidates

    async def _persist_chunks(
        self, document_id: str, candidates: list[ChunkCandidate], parsed: Any
    ) -> list[dict[str, Any]]:
        """Write chunk rows and push figure/table artefacts to object storage."""
        rows: list[dict[str, Any]] = []
        async with session_scope() as session:
            for ordinal, candidate in enumerate(candidates):
                asset_key = None
                if candidate.asset_bytes:
                    suffix = (candidate.asset_mime or "image/png").split("/")[-1]
                    asset_key = f"assets/{document_id}/chunk_{ordinal:04d}.{suffix}"
                    try:
                        await self.object_store.put(asset_key, candidate.asset_bytes, candidate.asset_mime)
                    except Exception as exc:
                        log.warning("ingest.asset_store_failed", key=asset_key, error=str(exc)[:160])
                        asset_key = None

                chunk = Chunk(
                    document_id=document_id,
                    ordinal=ordinal,
                    content=candidate.text,
                    content_hash=candidate.content_hash,
                    token_count=candidate.token_count,
                    modality=ChunkModality(candidate.modality)
                    if candidate.modality in {m.value for m in ChunkModality}
                    else ChunkModality.TEXT,
                    page_number=candidate.page_number,
                    page_end=candidate.page_end,
                    section=candidate.section,
                    section_path=candidate.section_path,
                    figure_label=candidate.figure_label,
                    table_label=candidate.table_label,
                    bbox=candidate.bbox,
                    asset_key=asset_key,
                    embedding_model=self.embedder.name,
                    chunk_metadata=candidate.metadata,
                )
                session.add(chunk)
                await session.flush()
                rows.append(
                    {
                        "id": chunk.id,
                        "content": candidate.text,
                        "embedding_text": candidate.embedding_text,
                        "modality": candidate.modality,
                        "token_count": candidate.token_count,
                        "page_number": candidate.page_number,
                        "section": candidate.section,
                        "section_path": candidate.section_path,
                        "figure_label": candidate.figure_label,
                        "table_label": candidate.table_label,
                        "asset_key": asset_key,
                        "ordinal": ordinal,
                    }
                )
        return rows

    async def _run_embedding(self, document_id: str, chunk_rows: list[dict[str, Any]]) -> None:
        started = time.perf_counter()
        async with session_scope() as session:
            document = await session.get(Document, document_id)
            filename = document.filename
            title = document.title or filename
            await self._set_step(session, document, "embed", "running", doc_status=DocumentStatus.EMBEDDING)

        texts = [row["embedding_text"] for row in chunk_rows]
        vectors = await self.embedder.embed_documents(texts)
        if len(vectors) != len(chunk_rows):
            raise IngestionError(
                f"The embedder returned {len(vectors)} vectors for {len(chunk_rows)} chunks."
            )

        records = [
            VectorRecord(
                chunk_id=row["id"],
                vector=vector,
                payload={
                    "document_id": document_id,
                    "document_name": filename,
                    "document_title": title,
                    "modality": row["modality"],
                    "page_number": row["page_number"],
                    "section": row["section"],
                    "section_path": row["section_path"],
                    "figure_label": row["figure_label"],
                    "table_label": row["table_label"],
                    "asset_key": row["asset_key"],
                    "ordinal": row["ordinal"],
                    "token_count": row["token_count"],
                    # A short preview keeps single-hop retrieval readable
                    # without a database round-trip; the full text lives in SQL.
                    "preview": row["content"][:400],
                },
            )
            for row, vector in zip(chunk_rows, vectors)
        ]
        await self.vector_store.upsert(records)
        await self.bm25.add(
            [
                BM25Document(
                    chunk_id=row["id"],
                    document_id=document_id,
                    text=row["embedding_text"],
                    modality=row["modality"],
                )
                for row in chunk_rows
            ]
        )

        async with session_scope() as session:
            for row in chunk_rows:
                chunk = await session.get(Chunk, row["id"])
                if chunk is not None:
                    chunk.indexed_in_vector_store = True
            document = await session.get(Document, document_id)
            await self._set_step(
                session, document, "embed", "completed", (time.perf_counter() - started) * 1000,
                detail=f"{len(records)} vectors · {self.embedder.name} · dim {self.embedder.dimension}",
            )

    async def _run_graph(
        self, document_id: str, filename: str, chunk_rows: list[dict[str, Any]]
    ) -> dict[str, int]:
        counts = {"entities": 0, "relations": 0}
        if not self.settings.extract_entities:
            async with session_scope() as session:
                document = await session.get(Document, document_id)
                await self._set_step(
                    session, document, "graph", "skipped", detail="Entity extraction is disabled."
                )
            return counts

        started = time.perf_counter()
        async with session_scope() as session:
            document = await session.get(Document, document_id)
            await self._set_step(
                session, document, "graph", "running", doc_status=DocumentStatus.GRAPH_INDEXING
            )

        try:
            chunk_objects = [
                type("ChunkView", (), {
                    "id": row["id"],
                    "content": row["content"],
                    "section_path": row["section_path"],
                })()
                for row in chunk_rows
            ]
            extraction = await self.entity_extractor.extract_from_chunks(
                chunk_objects, document_id, filename
            )
            if extraction.entities:
                result = await self.graph.upsert(extraction.entities, extraction.relations)
                counts["entities"] = len(extraction.entities)
                counts["relations"] = result.get("relations", 0)

                async with session_scope() as session:
                    for entity in extraction.entities:
                        session.add(
                            DocumentEntity(
                                document_id=document_id,
                                name=entity.name,
                                normalized_name=entity.normalized,
                                entity_type=entity.entity_type,
                                description=entity.description,
                                salience=entity.salience,
                                mention_count=len(entity.chunk_ids),
                                chunk_ids=entity.chunk_ids[:50],
                            )
                        )
                    for relation in extraction.relations:
                        session.add(
                            EntityRelation(
                                document_id=document_id,
                                source_name=relation.source,
                                target_name=relation.target,
                                relation_type=relation.relation_type,
                                evidence_chunk_id=relation.chunk_id,
                                confidence=relation.confidence,
                                context=relation.context,
                            )
                        )

            async with session_scope() as session:
                document = await session.get(Document, document_id)
                await self._set_step(
                    session, document, "graph", "completed", (time.perf_counter() - started) * 1000,
                    detail=f"{counts['entities']} entities · {counts['relations']} relations "
                           f"· {self.graph.backend}",
                )
        except Exception as exc:
            # Graph indexing is enrichment: a failure must not sink the document.
            log.warning("ingest.graph_failed", document_id=document_id, error=str(exc)[:200])
            async with session_scope() as session:
                document = await session.get(Document, document_id)
                await self._set_step(
                    session, document, "graph", "failed", (time.perf_counter() - started) * 1000,
                    detail=f"Graph indexing failed: {str(exc)[:160]}",
                )
        return counts

    # ------------------------------------------------------------- teardown
    async def remove_document(self, document_id: str) -> None:
        """Delete a document from every index. Called before the SQL delete."""
        await asyncio.gather(
            self.vector_store.delete_document(document_id),
            self.bm25.remove_document(document_id),
            self.graph.delete_document(document_id),
            return_exceptions=True,
        )
        async with session_scope() as session:
            document = await session.get(Document, document_id)
            keys = [document.object_key] if document and document.object_key else []
            chunk_keys = await session.scalars(
                select(Chunk.asset_key).where(
                    Chunk.document_id == document_id, Chunk.asset_key.is_not(None)
                )
            )
            keys.extend(k for k in chunk_keys if k)
        for key in keys:
            try:
                await self.object_store.delete(key)
            except Exception as exc:  # pragma: no cover
                log.warning("ingest.object_delete_failed", key=key, error=str(exc)[:160])


def compute_checksum(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


_pipeline: IngestionPipeline | None = None


def get_pipeline() -> IngestionPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = IngestionPipeline()
    return _pipeline


def reset_pipeline() -> None:
    global _pipeline
    _pipeline = None


__all__ = ["IngestionPipeline", "IngestionOutcome", "get_pipeline", "reset_pipeline", "compute_checksum", "STEPS"]
