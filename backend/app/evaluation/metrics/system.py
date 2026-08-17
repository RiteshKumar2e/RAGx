"""System metrics: latency, tokens, cost and retrieval-call accounting.

These are the numbers that make the adaptive-routing hypothesis testable. The
claim is that adaptive selection improves grounding *while reducing* cost, so
cost has to be measured as carefully as quality.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class SystemMetrics:
    count: int = 0
    avg_latency_ms: float = 0.0
    median_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    total_tokens: int = 0
    avg_total_tokens: float = 0.0
    avg_prompt_tokens: float = 0.0
    avg_completion_tokens: float = 0.0
    estimated_cost_usd: float = 0.0
    avg_cost_usd: float = 0.0
    total_retrieval_calls: int = 0
    avg_retrieval_calls: float = 0.0
    total_llm_calls: int = 0
    avg_llm_calls: float = 0.0
    corrective_retrievals: int = 0
    abstentions: int = 0
    abstention_rate: float = 0.0
    failures: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "median_latency_ms": round(self.median_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "min_latency_ms": round(self.min_latency_ms, 2),
            "max_latency_ms": round(self.max_latency_ms, 2),
            "total_tokens": self.total_tokens,
            "avg_total_tokens": round(self.avg_total_tokens, 1),
            "avg_prompt_tokens": round(self.avg_prompt_tokens, 1),
            "avg_completion_tokens": round(self.avg_completion_tokens, 1),
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "avg_cost_usd": round(self.avg_cost_usd, 6),
            "total_retrieval_calls": self.total_retrieval_calls,
            "avg_retrieval_calls": round(self.avg_retrieval_calls, 2),
            "total_llm_calls": self.total_llm_calls,
            "avg_llm_calls": round(self.avg_llm_calls, 2),
            "corrective_retrievals": self.corrective_retrievals,
            "abstentions": self.abstentions,
            "abstention_rate": round(self.abstention_rate, 4),
            "failures": self.failures,
            **self.extra,
        }


def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolation percentile (matches numpy's default)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * p
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def aggregate_system_metrics(records: list[dict[str, Any]]) -> SystemMetrics:
    """``records`` are per-question dicts produced by the evaluation runner."""
    metrics = SystemMetrics(count=len(records))
    if not records:
        return metrics

    latencies = [float(r.get("latency_ms", 0.0)) for r in records]
    metrics.avg_latency_ms = sum(latencies) / len(latencies)
    metrics.median_latency_ms = statistics.median(latencies)
    metrics.p95_latency_ms = percentile(latencies, 0.95)
    metrics.min_latency_ms = min(latencies)
    metrics.max_latency_ms = max(latencies)

    prompt_tokens = sum(int(r.get("prompt_tokens", 0)) for r in records)
    completion_tokens = sum(int(r.get("completion_tokens", 0)) for r in records)
    metrics.total_tokens = sum(int(r.get("total_tokens", 0)) for r in records)
    metrics.avg_total_tokens = metrics.total_tokens / len(records)
    metrics.avg_prompt_tokens = prompt_tokens / len(records)
    metrics.avg_completion_tokens = completion_tokens / len(records)

    metrics.estimated_cost_usd = sum(float(r.get("estimated_cost_usd", 0.0)) for r in records)
    metrics.avg_cost_usd = metrics.estimated_cost_usd / len(records)

    metrics.total_retrieval_calls = sum(int(r.get("retrieval_calls", 0)) for r in records)
    metrics.avg_retrieval_calls = metrics.total_retrieval_calls / len(records)
    metrics.total_llm_calls = sum(int(r.get("llm_calls", 0)) for r in records)
    metrics.avg_llm_calls = metrics.total_llm_calls / len(records)

    metrics.corrective_retrievals = sum(int(r.get("corrective_rounds", 0)) for r in records)
    metrics.abstentions = sum(1 for r in records if r.get("abstained"))
    metrics.abstention_rate = metrics.abstentions / len(records)
    metrics.failures = sum(1 for r in records if r.get("error"))
    return metrics


__all__ = ["SystemMetrics", "aggregate_system_metrics", "percentile"]
