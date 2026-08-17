"""Query orchestration -- the RAGX pipeline end to end.

    question
      -> conversation context
      -> query analysis          (app.retrieval.adaptive.analyzer)
      -> adaptive routing        (app.retrieval.adaptive.router)
      -> multi-strategy retrieval + fusion
      -> generation              (app.services.generation_service)
      -> evidence verification   (app.verification.pipeline)
      -> persistence + observability

Everything the "Why this answer?" panel shows is produced here and stored on the
``QueryRecord`` row, so an answer can be re-inspected long after the request.
"""

from __future__ import annotations

import time
from typing import Any

from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import TraceRecorder, get_logger
from app.models.document import Document, DocumentStatus
from app.models.query import Conversation, QueryRecord, RetrievalLog
from app.models.user import LOCAL_USER_ID
from app.retrieval.adaptive.strategy import AdaptiveRAG
from app.retrieval.base import RetrievalContext, RetrievalResult, RetrievedChunk
from app.retrieval.query_rewrite import condense_history
from app.schemas.query import QueryRequest
from app.services.generation_service import GenerationResult, get_generation_service
from app.verification.pipeline import VerificationReport, get_verification_pipeline

log = get_logger("ragx.query")

HISTORY_TURNS = 6


class QueryService:
    def __init__(self) -> None:
        self.adaptive = AdaptiveRAG()
        self.generator = get_generation_service()
        self.verifier = get_verification_pipeline()

    # ------------------------------------------------------------- context
    async def _conversation_history(
        self, session: AsyncSession, conversation_id: str | None
    ) -> list[dict[str, str]]:
        if not conversation_id:
            return []
        rows = await session.scalars(
            select(QueryRecord)
            .where(QueryRecord.conversation_id == conversation_id)
            .order_by(desc(QueryRecord.created_at))
            .limit(HISTORY_TURNS)
        )
        history: list[dict[str, str]] = []
        for record in reversed(list(rows)):
            history.append({"role": "user", "content": record.question})
            if record.answer:
                history.append({"role": "assistant", "content": record.answer[:900]})
        return history

    async def _document_titles(self, session: AsyncSession, limit: int = 40) -> list[str]:
        rows = await session.execute(
            select(Document.title, Document.filename)
            .where(Document.status == DocumentStatus.READY)
            .limit(limit)
        )
        return [(title or filename) for title, filename in rows.all()]

    async def _ensure_conversation(
        self, session: AsyncSession, conversation_id: str | None, question: str
    ) -> Conversation:
        if conversation_id:
            conversation = await session.get(Conversation, conversation_id)
            if conversation is not None:
                return conversation
        conversation = Conversation(
            title=question[:120] + ("…" if len(question) > 120 else ""),
            user_id=LOCAL_USER_ID,
        )
        session.add(conversation)
        await session.flush()
        return conversation

    # ------------------------------------------------------------------ main
    async def execute(self, session: AsyncSession, request: QueryRequest) -> dict[str, Any]:
        settings = get_settings()
        trace = TraceRecorder()
        started = time.perf_counter()

        conversation = await self._ensure_conversation(session, request.conversation_id, request.question)
        history = await self._conversation_history(session, conversation.id)
        titles = await self._document_titles(session)
        document_count = len(titles)

        context = RetrievalContext(
            session=session, trace=trace, history=history, document_titles=titles
        )

        # Follow-up questions are unretrievable in isolation; carry forward the
        # salient terms from recent turns before any embedding is computed.
        retrieval_query = condense_history(request.question, history)

        # -- 1. analyse + route ------------------------------------------------
        overrides: dict[str, Any] = {}
        if request.top_k:
            overrides["top_k"] = request.top_k
        if request.document_ids:
            overrides["document_ids"] = request.document_ids
        if request.rerank is not None:
            overrides["rerank"] = request.rerank

        analysis, decision = await self.adaptive.plan(
            request.question,
            context,
            document_count=document_count,
            forced_strategies=[s for s in (request.strategies or [])] or None,
            overrides=overrides or None,
        )

        # -- 2. retrieve --------------------------------------------------------
        retrieval_started = time.perf_counter()
        with trace.span("retrieval", category="retrieval"):
            retrieval: RetrievalResult = await self.adaptive.execute(retrieval_query, context, decision)
        retrieval_latency = (time.perf_counter() - retrieval_started) * 1000

        # -- 3. generate --------------------------------------------------------
        with trace.span("generation", category="generation"):
            generation: GenerationResult = await self.generator.generate(
                request.question,
                retrieval.chunks,
                history=history,
                trace=trace,
                provider_override=request.provider,
            )

        # Verification and citations must be computed against exactly the
        # evidence the model saw, not the full retrieval result.
        evidence_used: list[RetrievedChunk] = generation.evidence_used or retrieval.chunks

        # -- 4. verify ----------------------------------------------------------
        corrective_info = retrieval.diagnostics.get("corrective") or {}
        if not corrective_info:
            for detail in (retrieval.diagnostics.get("per_strategy") or {}).values():
                if isinstance(detail, dict) and detail.get("corrective"):
                    corrective_info = detail["corrective"]
                    break

        # A generation failure produces an operator-facing message, not a claim
        # about the corpus. Running it through claim extraction would surface
        # internal diagnostics as "unsupported claims", so it bypasses
        # verification and is reported as a degraded response instead.
        if generation.error:
            from app.verification.citations import analyze_citations  # noqa: PLC0415

            verification = VerificationReport(
                answer=generation.answer,
                original_answer=generation.answer,
                enabled=settings.verification_enabled,
                notes=[
                    "Verification was skipped because answer generation did not complete. "
                    "The retrieved evidence below is still valid."
                ],
            )
            verification.citations = analyze_citations("", evidence_used)
        elif request.verify and settings.verification_enabled:
            with trace.span("verification", category="verification"):
                verification: VerificationReport = await self.verifier.verify(
                    request.question,
                    generation.answer,
                    evidence_used,
                    trace=trace,
                    corrective_rounds=retrieval.corrective_rounds,
                    corrective_resolved=not corrective_info.get("insufficient_evidence", False),
                    retrieval_quality=retrieval.top_score,
                )
        else:
            verification = VerificationReport(
                answer=generation.answer,
                original_answer=generation.answer,
                enabled=False,
                notes=["Verification was disabled for this request."],
            )
            from app.verification.citations import analyze_citations  # noqa: PLC0415

            verification.citations = analyze_citations(generation.answer, evidence_used)

        total_latency = (time.perf_counter() - started) * 1000

        # -- 5. persist ---------------------------------------------------------
        record = await self._persist(
            session,
            conversation=conversation,
            request=request,
            trace=trace,
            analysis=analysis,
            decision=decision,
            retrieval=retrieval,
            retrieval_latency=retrieval_latency,
            generation=generation,
            verification=verification,
            evidence_used=evidence_used,
            total_latency=total_latency,
        )

        log.info(
            "query.completed",
            query_id=record.id,
            trace_id=trace.trace_id,
            strategies=retrieval.strategies_used,
            chunks=len(retrieval.chunks),
            confidence=verification.confidence.score,
            abstained=verification.abstained,
            corrective_rounds=retrieval.corrective_rounds,
            provider=generation.provider,
            tokens=trace.total_tokens,
            cost_usd=trace.total_cost_usd,
            latency_ms=round(total_latency, 1),
        )

        return self._build_response(
            record=record,
            conversation=conversation,
            request=request,
            trace=trace,
            analysis=analysis,
            decision=decision,
            retrieval=retrieval,
            retrieval_latency=retrieval_latency,
            generation=generation,
            verification=verification,
            evidence_used=evidence_used,
            total_latency=total_latency,
        )

    # -------------------------------------------------------------- persist
    async def _persist(
        self,
        session: AsyncSession,
        *,
        conversation: Conversation,
        request: QueryRequest,
        trace: TraceRecorder,
        analysis: Any,
        decision: Any,
        retrieval: RetrievalResult,
        retrieval_latency: float,
        generation: GenerationResult,
        verification: VerificationReport,
        evidence_used: list[RetrievedChunk],
        total_latency: float,
    ) -> QueryRecord:
        record = QueryRecord(
            conversation_id=conversation.id,
            user_id=LOCAL_USER_ID,
            trace_id=trace.trace_id,
            question=request.question,
            answer=verification.answer,
            analysis=analysis.as_dict(),
            intent=analysis.intent.value,
            complexity=analysis.complexity.value,
            strategies=retrieval.strategies_used,
            routing_reason=decision.reason,
            routing_mode=decision.mode,
            retrieved_chunk_ids=retrieval.chunk_ids,
            retrieval_scores={c.chunk_id: round(c.score, 4) for c in retrieval.chunks},
            chunks_retrieved=len(retrieval.chunks),
            documents_used=retrieval.document_ids,
            retrieval_calls=retrieval.retrieval_calls,
            corrective_triggered=retrieval.corrective_rounds > 0,
            corrective_rounds=retrieval.corrective_rounds,
            agentic_used=decision.use_agentic,
            reranked=retrieval.reranked,
            llm_provider=generation.provider,
            llm_model=generation.model,
            fallback_used=generation.fallback_used,
            prompt_tokens=trace.prompt_tokens,
            completion_tokens=trace.completion_tokens,
            total_tokens=trace.total_tokens,
            estimated_cost_usd=trace.total_cost_usd,
            confidence=verification.confidence.score,
            confidence_label=verification.confidence.label,
            claims_total=len(verification.claims),
            claims_supported=verification.claims_supported,
            citation_coverage=verification.citations.coverage,
            insufficient_evidence=verification.abstained,
            total_latency_ms=total_latency,
            retrieval_latency_ms=retrieval_latency,
            generation_latency_ms=generation.latency_ms,
            trace=trace.as_dict(),
            citations=[c.as_dict() for c in verification.citations.citations],
            evidence=[c.to_dict(include_content=False) for c in evidence_used],
            verification=verification.as_dict(),
            status="completed" if not generation.error else "degraded",
            error_message=generation.error,
        )
        session.add(record)
        await session.flush()

        # One retrieval log row per strategy invocation, for analytics.
        per_strategy = retrieval.diagnostics.get("per_strategy") or {}
        if per_strategy:
            for strategy, detail in per_strategy.items():
                session.add(
                    RetrievalLog(
                        query_id=record.id,
                        strategy=strategy,
                        effective_query=retrieval.effective_query[:1000],
                        latency_ms=float(detail.get("latency_ms", 0.0)),
                        results_returned=int(detail.get("chunks", 0)),
                        top_score=float(detail.get("top_score", 0.0)),
                        mean_score=retrieval.mean_score,
                        chunk_ids=retrieval.chunk_ids[:30],
                        scores=[round(s, 4) for s in retrieval.scores[:30]],
                        was_corrective=retrieval.corrective_rounds > 0,
                        details={k: v for k, v in detail.items() if k not in {"chunks", "top_score", "latency_ms"}},
                    )
                )
        else:
            session.add(
                RetrievalLog(
                    query_id=record.id,
                    strategy=retrieval.strategy,
                    effective_query=retrieval.effective_query[:1000],
                    latency_ms=retrieval.latency_ms,
                    results_returned=len(retrieval.chunks),
                    top_score=retrieval.top_score,
                    mean_score=retrieval.mean_score,
                    chunk_ids=retrieval.chunk_ids[:30],
                    scores=[round(s, 4) for s in retrieval.scores[:30]],
                    was_corrective=retrieval.corrective_rounds > 0,
                    details=retrieval.diagnostics,
                )
            )

        conversation.message_count = (conversation.message_count or 0) + 1
        await session.flush()
        return record

    # ------------------------------------------------------------- response
    @staticmethod
    def _build_response(
        *,
        record: QueryRecord,
        conversation: Conversation,
        request: QueryRequest,
        trace: TraceRecorder,
        analysis: Any,
        decision: Any,
        retrieval: RetrievalResult,
        retrieval_latency: float,
        generation: GenerationResult,
        verification: VerificationReport,
        evidence_used: list[RetrievedChunk],
        total_latency: float,
    ) -> dict[str, Any]:
        citation_by_chunk = {c.chunk_id: c for c in verification.citations.citations}
        evidence_payload: list[dict[str, Any]] = []
        for index, chunk in enumerate(evidence_used, start=1):
            citation = citation_by_chunk.get(chunk.chunk_id)
            evidence_payload.append(
                {
                    "marker": citation.marker if citation else index,
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "document_name": chunk.document_name,
                    "page": chunk.page_number,
                    "page_end": chunk.page_end,
                    "section": chunk.section,
                    "section_path": chunk.section_path,
                    "figure": chunk.figure_label,
                    "table": chunk.table_label,
                    "modality": chunk.modality,
                    "asset_key": chunk.asset_key,
                    "relevance": round(chunk.score, 4),
                    "content": chunk.content if request.include_evidence else "",
                    "excerpt": citation.excerpt if citation else chunk.content[:320],
                    "location": chunk.location,
                    "sources": chunk.sources,
                    "strategy_scores": chunk.strategy_scores,
                    "graph_path": chunk.graph_path,
                    "used_in_answer": citation.used_in_answer if citation else False,
                }
            )

        document_names = list(dict.fromkeys(c.document_name for c in evidence_used))

        why = {
            "analysis": analysis.as_dict(),
            "routing": decision.as_dict(),
            "retrieval": {
                "strategies_used": retrieval.strategies_used,
                "chunks_retrieved": len(retrieval.chunks),
                "documents_used": retrieval.document_ids,
                "document_names": document_names,
                "retrieval_calls": retrieval.retrieval_calls,
                "corrective_triggered": retrieval.corrective_rounds > 0,
                "corrective_rounds": retrieval.corrective_rounds,
                "agentic_used": decision.use_agentic,
                "reranked": retrieval.reranked,
                "top_score": round(retrieval.top_score, 4),
                "mean_score": round(retrieval.mean_score, 4),
                "latency_ms": round(retrieval_latency, 2),
                "notes": retrieval.notes,
                "diagnostics": retrieval.diagnostics,
            },
            "generation": generation.as_dict(),
            "verification": verification.as_dict(),
            "stage_latency_ms": trace.stage_latencies(),
        }

        return {
            "query_id": record.id,
            "trace_id": trace.trace_id,
            "conversation_id": conversation.id,
            "question": request.question,
            "answer": verification.answer,
            "abstained": verification.abstained,
            "confidence": verification.confidence.score,
            "confidence_label": verification.confidence.label,
            "strategies": retrieval.strategies_used,
            "strategy_labels": decision.strategy_labels,
            "routing_reason": decision.reason,
            "evidence": evidence_payload,
            "citations": [c.as_dict() for c in verification.citations.citations],
            "why": why if request.include_trace else None,
            "trace": trace.as_dict() if request.include_trace else None,
            "total_latency_ms": round(total_latency, 2),
            "created_at": record.created_at,
        }

    # -------------------------------------------------------------- analyse
    async def analyze_only(self, session: AsyncSession, question: str, conversation_id: str | None) -> dict[str, Any]:
        """Show what the router *would* do, without retrieving or generating."""
        started = time.perf_counter()
        trace = TraceRecorder()
        history = await self._conversation_history(session, conversation_id)
        titles = await self._document_titles(session)
        context = RetrievalContext(session=session, trace=trace, history=history, document_titles=titles)
        analysis, decision = await self.adaptive.plan(question, context, document_count=len(titles))
        return {
            "analysis": analysis.as_dict(),
            "routing": decision.as_dict(),
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    # --------------------------------------------------------------- history
    async def history(
        self, session: AsyncSession, page: int = 1, page_size: int = 20, conversation_id: str | None = None
    ) -> dict[str, Any]:
        stmt = select(QueryRecord).order_by(desc(QueryRecord.created_at))
        count_stmt = select(func.count(QueryRecord.id))
        if conversation_id:
            stmt = stmt.where(QueryRecord.conversation_id == conversation_id)
            count_stmt = count_stmt.where(QueryRecord.conversation_id == conversation_id)

        total = await session.scalar(count_stmt) or 0
        rows = await session.scalars(stmt.offset((page - 1) * page_size).limit(page_size))

        items = [
            {
                "id": r.id,
                "question": r.question,
                "answer_preview": (r.answer or "")[:220],
                "intent": r.intent,
                "complexity": r.complexity,
                "strategies": r.strategies or [],
                "confidence": r.confidence,
                "confidence_label": r.confidence_label,
                "chunks_retrieved": r.chunks_retrieved,
                "corrective_triggered": r.corrective_triggered,
                "agentic_used": r.agentic_used,
                "insufficient_evidence": r.insufficient_evidence,
                "llm_provider": r.llm_provider,
                "total_tokens": r.total_tokens,
                "estimated_cost_usd": r.estimated_cost_usd,
                "total_latency_ms": r.total_latency_ms,
                "status": r.status,
                "created_at": r.created_at,
            }
            for r in rows
        ]
        return {"items": items, "total": total, "page": page, "page_size": page_size}


_service: QueryService | None = None


def get_query_service() -> QueryService:
    global _service
    if _service is None:
        _service = QueryService()
    return _service


__all__ = ["QueryService", "get_query_service"]
