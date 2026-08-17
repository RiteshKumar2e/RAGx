"""Dashboard analytics.

Every figure is aggregated from rows the system actually wrote -- documents,
queries, retrieval logs and evaluation runs. When no queries have run, the
counters are zero and the frontend shows an empty state; nothing here
synthesises plausible-looking data.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document, DocumentEntity, DocumentStatus, EntityRelation
from app.models.evaluation import EvaluationRun
from app.models.query import QueryRecord

CONFIDENCE_BUCKETS = [
    ("0.0-0.2", 0.0, 0.2),
    ("0.2-0.4", 0.2, 0.4),
    ("0.4-0.6", 0.4, 0.6),
    ("0.6-0.8", 0.6, 0.8),
    ("0.8-1.0", 0.8, 1.01),
]


def _distribution(counter: Counter, limit: int = 12) -> list[dict[str, Any]]:
    total = sum(counter.values()) or 1
    return [
        {"label": label, "value": count, "percentage": round(count / total * 100, 2)}
        for label, count in counter.most_common(limit)
    ]


class AnalyticsService:
    async def dashboard(self, session: AsyncSession, days: int = 30) -> dict[str, Any]:
        since = datetime.now(timezone.utc) - timedelta(days=days)
        week_ago = datetime.now(timezone.utc) - timedelta(days=7)

        # -- documents ------------------------------------------------------
        doc_totals = (
            await session.execute(
                select(
                    func.count(Document.id),
                    func.coalesce(func.sum(Document.chunk_count), 0),
                )
            )
        ).one()
        status_rows = await session.execute(
            select(Document.status, func.count(Document.id)).group_by(Document.status)
        )
        status_counts = {
            (s.value if hasattr(s, "value") else str(s)): n for s, n in status_rows.all()
        }
        entity_count = await session.scalar(select(func.count(DocumentEntity.id))) or 0
        relation_count = await session.scalar(select(func.count(EntityRelation.id))) or 0

        # -- queries --------------------------------------------------------
        query_totals = (
            await session.execute(
                select(
                    func.count(QueryRecord.id),
                    func.coalesce(func.avg(QueryRecord.retrieval_latency_ms), 0.0),
                    func.coalesce(func.avg(QueryRecord.total_latency_ms), 0.0),
                    func.coalesce(func.avg(QueryRecord.confidence), 0.0),
                    func.coalesce(func.sum(QueryRecord.total_tokens), 0),
                    func.coalesce(func.sum(QueryRecord.estimated_cost_usd), 0.0),
                )
            )
        ).one()
        total_queries = query_totals[0] or 0

        recent_count = (
            await session.scalar(
                select(func.count(QueryRecord.id)).where(QueryRecord.created_at >= week_ago)
            )
            or 0
        )
        corrective_count = (
            await session.scalar(
                select(func.count(QueryRecord.id)).where(QueryRecord.corrective_triggered.is_(True))
            )
            or 0
        )
        abstention_count = (
            await session.scalar(
                select(func.count(QueryRecord.id)).where(QueryRecord.insufficient_evidence.is_(True))
            )
            or 0
        )
        evaluation_runs = await session.scalar(select(func.count(EvaluationRun.id))) or 0

        # -- per-query detail for distributions -----------------------------
        rows = await session.scalars(
            select(QueryRecord).where(QueryRecord.created_at >= since).order_by(desc(QueryRecord.created_at))
        )
        records = list(rows)

        strategy_counter: Counter = Counter()
        complexity_counter: Counter = Counter()
        intent_counter: Counter = Counter()
        provider_counter: Counter = Counter()
        confidence_counter: Counter = Counter()
        per_day: dict[str, list[float]] = {}

        for record in records:
            for strategy in record.strategies or []:
                strategy_counter[strategy] += 1
            if record.complexity:
                complexity_counter[record.complexity] += 1
            if record.intent:
                intent_counter[record.intent] += 1
            if record.llm_provider:
                provider_counter[record.llm_provider] += 1
            for label, low, high in CONFIDENCE_BUCKETS:
                if low <= (record.confidence or 0.0) < high:
                    confidence_counter[label] += 1
                    break
            day = record.created_at.date().isoformat()
            per_day.setdefault(day, []).append(record.total_latency_ms or 0.0)

        queries_over_time = [
            {"date": day, "value": len(values), "count": len(values)}
            for day, values in sorted(per_day.items())
        ]
        latency_over_time = [
            {"date": day, "value": round(sum(values) / len(values), 2), "count": len(values)}
            for day, values in sorted(per_day.items())
            if values
        ]

        recent_queries = [
            {
                "id": r.id,
                "question": r.question[:160],
                "strategies": r.strategies or [],
                "confidence": round(r.confidence or 0.0, 3),
                "confidence_label": r.confidence_label,
                "chunks_retrieved": r.chunks_retrieved,
                "corrective_triggered": r.corrective_triggered,
                "insufficient_evidence": r.insufficient_evidence,
                "llm_provider": r.llm_provider,
                "total_latency_ms": round(r.total_latency_ms or 0.0, 1),
                "created_at": r.created_at.isoformat(),
            }
            for r in records[:10]
        ]

        stats = {
            "total_documents": doc_totals[0] or 0,
            "indexed_documents": status_counts.get(DocumentStatus.READY.value, 0),
            "processing_documents": sum(
                status_counts.get(s, 0)
                for s in ("uploaded", "parsing", "chunking", "embedding", "graph_indexing")
            ),
            "failed_documents": status_counts.get(DocumentStatus.FAILED.value, 0),
            "total_chunks": int(doc_totals[1] or 0),
            "total_entities": entity_count,
            "total_relations": relation_count,
            "total_queries": total_queries,
            "queries_last_7_days": recent_count,
            "avg_retrieval_latency_ms": round(float(query_totals[1] or 0.0), 2),
            "avg_total_latency_ms": round(float(query_totals[2] or 0.0), 2),
            "avg_confidence": round(float(query_totals[3] or 0.0), 4),
            "most_used_strategy": strategy_counter.most_common(1)[0][0] if strategy_counter else None,
            "corrective_rate": round(corrective_count / total_queries, 4) if total_queries else 0.0,
            "abstention_rate": round(abstention_count / total_queries, 4) if total_queries else 0.0,
            "total_tokens": int(query_totals[4] or 0),
            "estimated_cost_usd": round(float(query_totals[5] or 0.0), 6),
            "evaluation_runs": evaluation_runs,
        }

        return {
            "stats": stats,
            "queries_over_time": queries_over_time,
            "latency_over_time": latency_over_time,
            "strategy_usage": _distribution(strategy_counter),
            "complexity_distribution": _distribution(complexity_counter),
            "intent_distribution": _distribution(intent_counter),
            "confidence_distribution": [
                {
                    "label": label,
                    "value": confidence_counter.get(label, 0),
                    "percentage": round(
                        confidence_counter.get(label, 0) / max(1, sum(confidence_counter.values())) * 100, 2
                    ),
                }
                for label, _, _ in CONFIDENCE_BUCKETS
            ],
            "provider_usage": _distribution(provider_counter),
            "recent_queries": recent_queries,
        }


_service: AnalyticsService | None = None


def get_analytics_service() -> AnalyticsService:
    global _service
    if _service is None:
        _service = AnalyticsService()
    return _service


__all__ = ["AnalyticsService", "get_analytics_service"]
