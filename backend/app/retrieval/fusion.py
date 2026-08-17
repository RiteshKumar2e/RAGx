"""Result fusion.

Two fusion modes, both used by RAGX:

* **Reciprocal Rank Fusion (RRF)** -- the default for combining ranked lists
  that come from different scoring universes (cosine similarity vs BM25 vs graph
  confidence). RRF only reads *positions*, so it needs no score normalisation
  and is robust to one retriever having a compressed score range.
* **Weighted score fusion** -- used when both lists are already on a comparable
  0..1 scale and the router wants to bias explicitly toward dense or sparse.

Both preserve provenance: the merged chunk keeps every contributing strategy in
``sources`` and each strategy's own score in ``strategy_scores``.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from app.core.config import get_settings
from app.retrieval.base import RetrievedChunk


def _merge_into(
    target: dict[str, RetrievedChunk], chunk: RetrievedChunk, strategy: str, score: float
) -> None:
    existing = target.get(chunk.chunk_id)
    if existing is None:
        clone = chunk
        clone.add_source(strategy, score)
        target[chunk.chunk_id] = clone
        return
    existing.add_source(strategy, score)
    for source in chunk.sources:
        if source not in existing.sources:
            existing.sources.append(source)
    for key, value in chunk.strategy_scores.items():
        existing.strategy_scores[key] = round(max(value, existing.strategy_scores.get(key, 0.0)), 6)
    if chunk.graph_path and not existing.graph_path:
        existing.graph_path = chunk.graph_path
    if len(chunk.content) > len(existing.content):
        existing.content = chunk.content


def reciprocal_rank_fusion(
    ranked_lists: Sequence[tuple[str, Sequence[RetrievedChunk]]],
    k: int | None = None,
    weights: dict[str, float] | None = None,
) -> list[RetrievedChunk]:
    """Fuse ranked lists with RRF: ``score = Σ w_s / (k + rank_s)``.

    ``ranked_lists`` is ``[(strategy_name, chunks_in_rank_order), ...]``.
    """
    k = k or get_settings().rrf_k
    weights = weights or {}
    merged: dict[str, RetrievedChunk] = {}
    fused_scores: dict[str, float] = {}

    for strategy, chunks in ranked_lists:
        weight = weights.get(strategy, 1.0)
        for rank, chunk in enumerate(chunks, start=1):
            _merge_into(merged, chunk, strategy, chunk.strategy_scores.get(strategy, chunk.score))
            fused_scores[chunk.chunk_id] = fused_scores.get(chunk.chunk_id, 0.0) + weight / (k + rank)

    if not fused_scores:
        return []

    # Normalise so downstream thresholds stay meaningful across query types.
    best = max(fused_scores.values()) or 1.0
    out: list[RetrievedChunk] = []
    for chunk_id, raw in fused_scores.items():
        chunk = merged[chunk_id]
        chunk.score = round(raw / best, 6)
        chunk.metadata["fusion"] = "rrf"
        chunk.metadata["rrf_raw"] = round(raw, 6)
        out.append(chunk)

    out.sort(key=lambda c: c.score, reverse=True)
    for index, chunk in enumerate(out, start=1):
        chunk.rank = index
    return out


def weighted_fusion(
    ranked_lists: Sequence[tuple[str, Sequence[RetrievedChunk]]],
    weights: dict[str, float],
) -> list[RetrievedChunk]:
    """Fuse by weighted sum of per-strategy normalised scores."""
    merged: dict[str, RetrievedChunk] = {}
    totals: dict[str, float] = {}

    for strategy, chunks in ranked_lists:
        weight = weights.get(strategy, 1.0)
        if not chunks:
            continue
        top = max((c.strategy_scores.get(strategy, c.score) for c in chunks), default=1.0) or 1.0
        for chunk in chunks:
            raw = chunk.strategy_scores.get(strategy, chunk.score)
            normalized = raw / top
            _merge_into(merged, chunk, strategy, raw)
            totals[chunk.chunk_id] = totals.get(chunk.chunk_id, 0.0) + weight * normalized

    if not totals:
        return []
    best = max(totals.values()) or 1.0
    out: list[RetrievedChunk] = []
    for chunk_id, raw in totals.items():
        chunk = merged[chunk_id]
        chunk.score = round(raw / best, 6)
        chunk.metadata["fusion"] = "weighted"
        out.append(chunk)
    out.sort(key=lambda c: c.score, reverse=True)
    for index, chunk in enumerate(out, start=1):
        chunk.rank = index
    return out


def deduplicate(chunks: Iterable[RetrievedChunk], similarity_threshold: float = 0.92) -> list[RetrievedChunk]:
    """Drop near-duplicate evidence.

    Overlapping chunks and repeated boilerplate otherwise waste context budget
    and inflate apparent agreement between sources.
    """
    from app.core.text import lexical_similarity  # noqa: PLC0415

    kept: list[RetrievedChunk] = []
    for chunk in sorted(chunks, key=lambda c: c.score, reverse=True):
        duplicate_of = None
        for existing in kept:
            if existing.document_id != chunk.document_id:
                continue
            if lexical_similarity(existing.content, chunk.content) >= similarity_threshold:
                duplicate_of = existing
                break
        if duplicate_of is None:
            kept.append(chunk)
        else:
            for source in chunk.sources:
                if source not in duplicate_of.sources:
                    duplicate_of.sources.append(source)
            duplicate_of.metadata.setdefault("duplicates_merged", 0)
            duplicate_of.metadata["duplicates_merged"] += 1
    for index, chunk in enumerate(kept, start=1):
        chunk.rank = index
    return kept


def enforce_document_diversity(
    chunks: list[RetrievedChunk], max_per_document: int, minimum_documents: int = 2
) -> list[RetrievedChunk]:
    """Cap how many chunks a single document may contribute.

    Cross-document questions fail when one verbose document floods the context;
    the cap is only applied when enough distinct documents are actually present.
    """
    distinct = {c.document_id for c in chunks}
    if len(distinct) < minimum_documents or max_per_document <= 0:
        return chunks

    counts: dict[str, int] = {}
    kept: list[RetrievedChunk] = []
    overflow: list[RetrievedChunk] = []
    for chunk in chunks:
        count = counts.get(chunk.document_id, 0)
        if count < max_per_document:
            counts[chunk.document_id] = count + 1
            kept.append(chunk)
        else:
            overflow.append(chunk)
    # Overflow is appended, not discarded, so a truncate() later still has
    # material if diversity leaves us short.
    return kept + overflow


__all__ = [
    "reciprocal_rank_fusion",
    "weighted_fusion",
    "deduplicate",
    "enforce_document_diversity",
]
