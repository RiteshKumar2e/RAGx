"""Chunk hydration.

Vector and BM25 hits carry only ids and payload previews. This module turns
them into fully-populated :class:`RetrievedChunk` objects in one batched SQL
round-trip, so no strategy ever issues per-hit queries.
"""

from __future__ import annotations

from typing import Any, Iterable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.models.document import Chunk
from app.retrieval.base import RetrievedChunk


async def load_chunks(session: AsyncSession, chunk_ids: Iterable[str]) -> dict[str, Chunk]:
    ids = [cid for cid in dict.fromkeys(chunk_ids) if cid]
    if not ids:
        return {}
    rows = await session.scalars(
        select(Chunk).options(joinedload(Chunk.document)).where(Chunk.id.in_(ids))
    )
    return {chunk.id: chunk for chunk in rows.unique()}


def to_retrieved(chunk: Chunk, score: float, strategy: str, **extra: Any) -> RetrievedChunk:
    document = chunk.document
    return RetrievedChunk(
        chunk_id=chunk.id,
        document_id=chunk.document_id,
        document_name=(document.filename if document else chunk.document_id),
        content=chunk.content,
        score=score,
        modality=chunk.modality.value if hasattr(chunk.modality, "value") else str(chunk.modality),
        page_number=chunk.page_number,
        page_end=chunk.page_end,
        section=chunk.section,
        section_path=list(chunk.section_path or []),
        figure_label=chunk.figure_label,
        table_label=chunk.table_label,
        asset_key=chunk.asset_key,
        ordinal=chunk.ordinal,
        token_count=chunk.token_count,
        sources=[strategy],
        strategy_scores={strategy: round(score, 6)},
        metadata={
            "document_title": document.title if document else None,
            **{k: v for k, v in extra.items() if v is not None},
        },
    )


async def hydrate(
    session: AsyncSession,
    scored_ids: list[tuple[str, float]],
    strategy: str,
    **extra: Any,
) -> list[RetrievedChunk]:
    """Turn ``[(chunk_id, score), ...]`` into ordered ``RetrievedChunk``s."""
    lookup = await load_chunks(session, (cid for cid, _ in scored_ids))
    out: list[RetrievedChunk] = []
    for chunk_id, score in scored_ids:
        chunk = lookup.get(chunk_id)
        if chunk is None:
            continue  # Index and database disagree; skip rather than fabricate.
        out.append(to_retrieved(chunk, score, strategy, **extra))
    return out


async def expand_neighbors(
    session: AsyncSession, chunks: list[RetrievedChunk], window: int = 1
) -> list[RetrievedChunk]:
    """Attach adjacent chunks from the same document.

    Useful when a hit lands mid-explanation: the neighbouring chunk usually
    carries the definition or the number the answer needs. Neighbours are added
    at a discounted score so they never outrank a genuine hit.
    """
    if not chunks or window <= 0:
        return chunks

    wanted: set[tuple[str, int]] = set()
    for chunk in chunks:
        for offset in range(-window, window + 1):
            if offset:
                wanted.add((chunk.document_id, chunk.ordinal + offset))
    if not wanted:
        return chunks

    existing = {c.chunk_id for c in chunks}
    conditions = [
        (Chunk.document_id == document_id) & (Chunk.ordinal == ordinal)
        for document_id, ordinal in wanted
        if ordinal >= 0
    ]
    if not conditions:
        return chunks

    from sqlalchemy import or_  # noqa: PLC0415

    rows = await session.scalars(
        select(Chunk).options(joinedload(Chunk.document)).where(or_(*conditions))
    )
    best_score = max((c.score for c in chunks), default=0.5)
    out = list(chunks)
    for chunk in rows.unique():
        if chunk.id in existing:
            continue
        neighbor = to_retrieved(chunk, round(best_score * 0.35, 6), "neighbor_expansion")
        neighbor.metadata["expanded"] = True
        out.append(neighbor)
    return out


__all__ = ["load_chunks", "to_retrieved", "hydrate", "expand_neighbors"]
