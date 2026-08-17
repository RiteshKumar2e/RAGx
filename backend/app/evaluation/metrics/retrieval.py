"""Retrieval metrics.

All four are standard IR measures computed against a set of *relevant chunk ids*
for a question. RAGX obtains those labels one of two ways, both recorded on the
run so results are interpretable:

``manual``
    Labels supplied in the benchmark file.

``pooled``
    Pooled relevance judgements (the TREC pooling method): every strategy under
    comparison contributes its top-N results to a shared pool, an LLM judge
    labels each pooled passage once for relevance to the question, and all
    strategies are then scored against those same labels. Pooling is the
    standard way to build judgements when exhaustive labelling is infeasible;
    its known limitation is that a passage no strategy retrieved is never
    judged, so recall is measured relative to the pool.

When no labels exist, these functions return ``None`` rather than a number, and
the dashboard shows "requires relevance labels" instead of a fabricated score.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float | None:
    relevant_set = {r for r in relevant if r}
    if not relevant_set:
        return None
    top_k = set(retrieved[:k])
    return round(len(top_k & relevant_set) / len(relevant_set), 4)


def precision_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float | None:
    relevant_set = {r for r in relevant if r}
    if not relevant_set:
        return None
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    return round(sum(1 for chunk_id in top_k if chunk_id in relevant_set) / len(top_k), 4)


def reciprocal_rank(retrieved: Sequence[str], relevant: Iterable[str]) -> float | None:
    relevant_set = {r for r in relevant if r}
    if not relevant_set:
        return None
    for index, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant_set:
            return round(1.0 / index, 4)
    return 0.0


def ndcg_at_k(
    retrieved: Sequence[str], relevant: Iterable[str], k: int, gains: dict[str, float] | None = None
) -> float | None:
    """Normalised DCG. Binary relevance unless graded ``gains`` are supplied."""
    relevant_set = {r for r in relevant if r}
    if not relevant_set:
        return None
    gains = gains or {}

    def gain(chunk_id: str) -> float:
        if chunk_id not in relevant_set:
            return 0.0
        return gains.get(chunk_id, 1.0)

    dcg = sum(gain(cid) / math.log2(index + 1) for index, cid in enumerate(retrieved[:k], start=1))
    ideal = sorted((gains.get(cid, 1.0) for cid in relevant_set), reverse=True)[:k]
    idcg = sum(value / math.log2(index + 1) for index, value in enumerate(ideal, start=1))
    return round(dcg / idcg, 4) if idcg else 0.0


def hit_rate(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float | None:
    relevant_set = {r for r in relevant if r}
    if not relevant_set:
        return None
    return 1.0 if set(retrieved[:k]) & relevant_set else 0.0


def compute_retrieval_metrics(
    retrieved: Sequence[str], relevant: Iterable[str], k: int, gains: dict[str, float] | None = None
) -> dict[str, float | None]:
    relevant_list = list(relevant)
    return {
        "recall_at_k": recall_at_k(retrieved, relevant_list, k),
        "precision_at_k": precision_at_k(retrieved, relevant_list, k),
        "reciprocal_rank": reciprocal_rank(retrieved, relevant_list),
        "ndcg_at_k": ndcg_at_k(retrieved, relevant_list, k, gains),
        "hit_rate": hit_rate(retrieved, relevant_list, k),
    }


def mean_ignoring_none(values: Iterable[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return round(sum(present) / len(present), 4) if present else None


__all__ = [
    "recall_at_k",
    "precision_at_k",
    "reciprocal_rank",
    "ndcg_at_k",
    "hit_rate",
    "compute_retrieval_metrics",
    "mean_ignoring_none",
]
