"""Query execution, routing inspection, history and evidence retrieval."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

from fastapi import APIRouter, Query, Response
from fastapi.responses import StreamingResponse
from sqlalchemy import desc, select

from app.api.deps import SessionDep
from app.core.errors import NotFoundError
from app.core.logging import get_logger
from app.models.query import Conversation, QueryRecord
from app.schemas.common import Acknowledgement
from app.schemas.query import (
    AnalyzeRequest,
    AnalyzeResponse,
    ConversationSummary,
    EvidenceDetailResponse,
    QueryHistoryResponse,
    QueryRequest,
    QueryResponse,
)
from app.services.document_service import get_document_service
from app.services.query_service import get_query_service
from app.storage import get_object_store

log = get_logger("ragx.api.query")
router = APIRouter(prefix="/query", tags=["query"])


@router.post(
    "",
    response_model=QueryResponse,
    summary="Ask a research question",
    description=(
        "Runs the full RAGX pipeline: query analysis, adaptive strategy selection, "
        "multi-strategy retrieval with fusion, grounded generation, and evidence "
        "verification. Set `strategies` to pin specific strategies and bypass the router "
        "(this is what the evaluation harness does); leave it unset for adaptive routing."
    ),
)
async def execute_query(session: SessionDep, request: QueryRequest) -> QueryResponse:
    result = await get_query_service().execute(session, request)
    return QueryResponse.model_validate(result)


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Analyse a query and show the routing decision",
    description=(
        "Runs query analysis and the adaptive router **without** retrieving or generating. "
        "Use it to inspect which strategies would be selected and why."
    ),
)
async def analyze_query(session: SessionDep, request: AnalyzeRequest) -> AnalyzeResponse:
    result = await get_query_service().analyze_only(session, request.question, request.conversation_id)
    return AnalyzeResponse.model_validate(result)


@router.post(
    "/stream",
    summary="Ask a research question with a streamed answer",
    description=(
        "Server-sent events. Emits `status` events as each pipeline stage completes, `token` "
        "events while the answer is generated, and a final `done` event carrying the full "
        "response including evidence, citations and the verification report.\n\n"
        "Note: verification runs on the completed answer, so the `done` payload may replace a "
        "streamed answer with an abstention if the evidence does not support it. Clients should "
        "render the `done` answer as authoritative."
    ),
)
async def stream_query(session: SessionDep, request: QueryRequest) -> StreamingResponse:
    service = get_query_service()

    async def event_stream() -> AsyncIterator[str]:
        def sse(event: str, payload: dict[str, Any]) -> str:
            return f"event: {event}\ndata: {json.dumps(payload, default=str)}\n\n"

        try:
            yield sse("status", {"stage": "analyzing", "message": "Analysing the query…"})
            result = await service.execute(session, request)

            yield sse(
                "status",
                {
                    "stage": "retrieved",
                    "message": f"Retrieved {len(result['evidence'])} passages",
                    "strategies": result["strategies"],
                    "strategy_labels": result["strategy_labels"],
                    "routing_reason": result["routing_reason"],
                },
            )

            # The pipeline is verification-gated, so the authoritative answer is
            # only known after verification. Tokens are replayed here to give the
            # UI a progressive reveal without ever showing an unverified claim
            # as final.
            answer = result["answer"]
            chunk_size = 24
            for start in range(0, len(answer), chunk_size):
                yield sse("token", {"text": answer[start : start + chunk_size]})

            yield sse("done", result)
        except Exception as exc:  # pragma: no cover - defensive
            log.error("query.stream_failed", error=str(exc), exc_info=True)
            yield sse(
                "error",
                {"message": "The query failed. Check the server logs for details.", "code": "query_failed"},
            )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


@router.get("/history", response_model=QueryHistoryResponse, summary="Query history")
async def query_history(
    session: SessionDep,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    conversation_id: str | None = None,
) -> QueryHistoryResponse:
    result = await get_query_service().history(session, page, page_size, conversation_id)
    return QueryHistoryResponse.model_validate(result)


@router.get("/conversations", response_model=list[ConversationSummary], summary="List conversations")
async def list_conversations(session: SessionDep, limit: int = Query(30, ge=1, le=100)) -> list[ConversationSummary]:
    rows = await session.scalars(
        select(Conversation).order_by(desc(Conversation.updated_at)).limit(limit)
    )
    return [ConversationSummary.model_validate(c, from_attributes=True) for c in rows]


@router.delete("/conversations/{conversation_id}", response_model=Acknowledgement)
async def delete_conversation(session: SessionDep, conversation_id: str) -> Acknowledgement:
    conversation = await session.get(Conversation, conversation_id)
    if conversation is None:
        raise NotFoundError(f"Conversation '{conversation_id}' was not found.")
    await session.delete(conversation)
    return Acknowledgement(ok=True, message="The conversation and its queries were deleted.")


@router.get("/{query_id}", response_model=QueryResponse, summary="Re-read a stored answer")
async def get_query(session: SessionDep, query_id: str) -> QueryResponse:
    record = await session.get(QueryRecord, query_id)
    if record is None:
        raise NotFoundError(f"Query '{query_id}' was not found.")

    from app.retrieval.base import StrategyName  # noqa: PLC0415

    strategies = record.strategies or []
    labels: list[str] = []
    for name in strategies:
        try:
            labels.append(StrategyName(name).label)
        except ValueError:
            labels.append(name)

    return QueryResponse(
        query_id=record.id,
        trace_id=record.trace_id,
        conversation_id=record.conversation_id,
        question=record.question,
        answer=record.answer or "",
        abstained=record.insufficient_evidence,
        confidence=record.confidence,
        confidence_label=record.confidence_label or "low",
        strategies=strategies,
        strategy_labels=labels,
        routing_reason=record.routing_reason or "",
        # Evidence content is not re-hydrated here; the stored citations carry
        # the provenance, and the evidence endpoint serves full passages on demand.
        evidence=[],
        citations=record.citations or [],
        why={
            "analysis": {
                "query": record.question,
                "intent": record.intent or "factual_lookup",
                "complexity": record.complexity or "simple",
                **(record.analysis or {}),
            },
            "routing": {
                "primary": strategies[0] if strategies else "naive",
                "parallel": strategies[1:],
                "strategies": strategies,
                "strategy_labels": labels,
                "use_corrective": record.corrective_triggered,
                "use_agentic": record.agentic_used,
                "mode": record.routing_mode or "single",
                "reason": record.routing_reason or "",
                "rules_fired": [],
                "estimated_llm_calls": len((record.trace or {}).get("llm_calls", [])),
                "config": {},
            },
            "retrieval": {
                "strategies_used": record.strategies or [],
                "chunks_retrieved": record.chunks_retrieved,
                "documents_used": record.documents_used or [],
                "retrieval_calls": record.retrieval_calls,
                "corrective_triggered": record.corrective_triggered,
                "corrective_rounds": record.corrective_rounds,
                "agentic_used": record.agentic_used,
                "reranked": record.reranked,
                "latency_ms": record.retrieval_latency_ms,
            },
            "generation": {
                "provider": record.llm_provider,
                "model": record.llm_model,
                "fallback_used": record.fallback_used,
                "prompt_tokens": record.prompt_tokens,
                "completion_tokens": record.completion_tokens,
                "total_tokens": record.total_tokens,
                "estimated_cost_usd": record.estimated_cost_usd,
                "latency_ms": record.generation_latency_ms,
            },
            "verification": record.verification or {},
            "stage_latency_ms": (record.trace or {}).get("stage_latency_ms", {}),
        },
        trace=record.trace,
        total_latency_ms=record.total_latency_ms,
        created_at=record.created_at,
    )


# ---------------------------------------------------------------- evidence
evidence_router = APIRouter(prefix="/evidence", tags=["evidence"])


@evidence_router.get(
    "/{chunk_id}",
    response_model=EvidenceDetailResponse,
    summary="Inspect a cited evidence chunk",
    description="Returns the full passage behind a citation, plus its neighbouring chunks for context.",
)
async def get_evidence(session: SessionDep, chunk_id: str) -> EvidenceDetailResponse:
    result = await get_document_service().get_chunk(session, chunk_id)
    chunk = result["chunk"]
    document = chunk.document
    modality = chunk.modality.value if hasattr(chunk.modality, "value") else str(chunk.modality)

    return EvidenceDetailResponse(
        chunk_id=chunk.id,
        document_id=chunk.document_id,
        document_name=document.filename if document else chunk.document_id,
        document_title=document.title if document else None,
        content=chunk.content,
        modality=modality,
        page=chunk.page_number,
        page_end=chunk.page_end,
        section=chunk.section,
        section_path=list(chunk.section_path or []),
        figure=chunk.figure_label,
        table=chunk.table_label,
        ordinal=chunk.ordinal,
        token_count=chunk.token_count,
        has_image=bool(chunk.asset_key),
        image_url=f"/api/v1/evidence/{chunk.id}/image" if chunk.asset_key else None,
        neighbors=[
            {
                "chunk_id": n.id,
                "ordinal": n.ordinal,
                "content": n.content[:600],
                "page": n.page_number,
                "section": n.section,
            }
            for n in result["neighbors"]
        ],
    )


@evidence_router.get(
    "/{chunk_id}/image",
    summary="Fetch the figure or table image behind a citation",
    response_class=Response,
)
async def get_evidence_image(session: SessionDep, chunk_id: str) -> Response:
    result = await get_document_service().get_chunk(session, chunk_id)
    chunk = result["chunk"]
    if not chunk.asset_key:
        raise NotFoundError("This evidence chunk has no associated image.")
    data = await get_object_store().get(chunk.asset_key)
    suffix = chunk.asset_key.rsplit(".", 1)[-1].lower()
    media_type = "image/jpeg" if suffix in {"jpg", "jpeg"} else f"image/{suffix}"
    return Response(content=data, media_type=media_type, headers={"Cache-Control": "public, max-age=3600"})


__all__ = ["router", "evidence_router"]
