"""Structured logging and the retrieval/observability trace helper.

Two things live here:

1. ``configure_logging`` -- structlog wiring (console renderer for dev, JSON
   for production).
2. ``TraceRecorder`` -- an in-request accumulator for the observability signals
   RAGX must capture: which strategies ran, their latency, which chunk ids came
   back with which scores, corrective-retrieval events, LLM provider/model,
   token usage and the final confidence.

Redaction rule: we log *identifiers, scores and counts*, never API keys and
never full document bodies. Text snippets are truncated hard.
"""

from __future__ import annotations

import logging
import sys
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator

import structlog

from app.core.config import get_settings

_SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "gemini_api_key",
    "groq_api_key",
    "authorization",
    "password",
    "turso_auth_token",
    "neo4j_password",
    "secret",
    "token",
    "aws_secret_access_key",
}

_MAX_SNIPPET = 240

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


def _redact(_logger, _method, event_dict: dict) -> dict:
    for key in list(event_dict.keys()):
        if key.lower() in _SENSITIVE_KEYS:
            event_dict[key] = "***redacted***"
    rid = request_id_var.get()
    if rid and rid != "-":
        event_dict.setdefault("request_id", rid)
    return event_dict


def configure_logging() -> None:
    settings = get_settings()
    level = getattr(logging, settings.log_level, logging.INFO)

    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level)
    for noisy in ("httpx", "httpcore", "urllib3", "neo4j", "qdrant_client", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    renderer = (
        structlog.processors.JSONRenderer()
        if settings.log_json
        else structlog.dev.ConsoleRenderer(colors=False)
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            _redact,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = "ragx") -> Any:
    return structlog.get_logger(name)


def truncate(text: str | None, limit: int = _MAX_SNIPPET) -> str:
    if not text:
        return ""
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


# --------------------------------------------------------------------------
# Trace recorder
# --------------------------------------------------------------------------
@dataclass
class TraceEvent:
    name: str
    category: str
    started_at: float
    duration_ms: float = 0.0
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "duration_ms": round(self.duration_ms, 2),
            "detail": self.detail,
        }


@dataclass
class TraceRecorder:
    """Collects the observability record for a single query execution."""

    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    events: list[TraceEvent] = field(default_factory=list)
    llm_calls: list[dict[str, Any]] = field(default_factory=list)
    retrieval_calls: int = 0
    corrective_rounds: int = 0
    corrective_events: list[dict[str, Any]] = field(default_factory=list)
    started_at: float = field(default_factory=time.perf_counter)
    _log: Any = field(default_factory=lambda: get_logger("ragx.trace"))

    # -- timing ------------------------------------------------------------
    @contextmanager
    def span(self, name: str, category: str = "generic", **detail: Any) -> Iterator[TraceEvent]:
        event = TraceEvent(name=name, category=category, started_at=time.perf_counter(), detail=dict(detail))
        try:
            yield event
        finally:
            event.duration_ms = (time.perf_counter() - event.started_at) * 1000
            self.events.append(event)
            self._log.debug("trace.span", trace_id=self.trace_id, **event.as_dict())

    def event(self, name: str, category: str = "generic", **detail: Any) -> None:
        self.events.append(
            TraceEvent(name=name, category=category, started_at=time.perf_counter(), detail=dict(detail))
        )

    # -- domain-specific records -------------------------------------------
    def record_retrieval(
        self,
        strategy: str,
        duration_ms: float,
        chunk_ids: list[str],
        scores: list[float],
        **detail: Any,
    ) -> None:
        self.retrieval_calls += 1
        self.events.append(
            TraceEvent(
                name=f"retrieval:{strategy}",
                category="retrieval",
                started_at=time.perf_counter(),
                duration_ms=duration_ms,
                detail={
                    "strategy": strategy,
                    "returned": len(chunk_ids),
                    "chunk_ids": chunk_ids[:25],
                    "scores": [round(s, 4) for s in scores[:25]],
                    **detail,
                },
            )
        )
        self._log.info(
            "retrieval.completed",
            trace_id=self.trace_id,
            strategy=strategy,
            duration_ms=round(duration_ms, 2),
            returned=len(chunk_ids),
            top_score=round(max(scores), 4) if scores else 0.0,
        )

    def record_corrective(self, round_index: int, reason: str, action: str, **detail: Any) -> None:
        self.corrective_rounds = max(self.corrective_rounds, round_index)
        payload = {"round": round_index, "reason": reason, "action": action, **detail}
        self.corrective_events.append(payload)
        self.events.append(
            TraceEvent(
                name="corrective_retrieval",
                category="corrective",
                started_at=time.perf_counter(),
                detail=payload,
            )
        )
        self._log.info("retrieval.corrective", trace_id=self.trace_id, **payload)

    def record_llm(
        self,
        provider: str,
        model: str,
        purpose: str,
        latency_ms: float,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        fallback_used: bool = False,
        error: str | None = None,
    ) -> None:
        record = {
            "provider": provider,
            "model": model,
            "purpose": purpose,
            "latency_ms": round(latency_ms, 2),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cost_usd": round(cost_usd, 6),
            "fallback_used": fallback_used,
            "error": error,
        }
        self.llm_calls.append(record)
        self._log.info("llm.call", trace_id=self.trace_id, **record)

    # -- aggregates ---------------------------------------------------------
    @property
    def total_tokens(self) -> int:
        return sum(c["total_tokens"] for c in self.llm_calls)

    @property
    def prompt_tokens(self) -> int:
        return sum(c["prompt_tokens"] for c in self.llm_calls)

    @property
    def completion_tokens(self) -> int:
        return sum(c["completion_tokens"] for c in self.llm_calls)

    @property
    def total_cost_usd(self) -> float:
        return round(sum(c["cost_usd"] for c in self.llm_calls), 6)

    @property
    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.started_at) * 1000

    def stage_latencies(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for e in self.events:
            if e.duration_ms:
                out[e.name] = round(out.get(e.name, 0.0) + e.duration_ms, 2)
        return out

    def as_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "total_latency_ms": round(self.elapsed_ms, 2),
            "retrieval_calls": self.retrieval_calls,
            "corrective_rounds": self.corrective_rounds,
            "corrective_events": self.corrective_events,
            "llm_calls": self.llm_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": self.total_cost_usd,
            "stage_latency_ms": self.stage_latencies(),
            "events": [e.as_dict() for e in self.events],
        }


__all__ = [
    "configure_logging",
    "get_logger",
    "truncate",
    "TraceRecorder",
    "TraceEvent",
    "request_id_var",
]
