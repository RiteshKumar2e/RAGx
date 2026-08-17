"""Agentic RAG -- plan, act, observe, reflect.

For questions that cannot be answered by any single retrieval pass, the agent:

1. **Plans.** The LLM decomposes the question into ordered sub-questions, each
   assigned the retrieval tool best suited to it.
2. **Acts.** Independent steps run concurrently; dependent steps run after the
   findings they need, with the earlier findings injected into the query.
3. **Observes.** Each step's evidence is graded and recorded against its
   sub-question.
4. **Reflects.** The LLM decides whether the collected evidence answers the
   question. If a specific gap remains it issues one more targeted step.

Bounded by ``agentic_max_steps`` and ``agentic_max_subqueries`` so the loop
always terminates. The tools the agent can call are exactly the other RAG
strategies -- the agent selects strategies rather than reimplementing retrieval.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.text import truncate_words
from app.llm.base import Message
from app.llm.gateway import Purpose, get_gateway
from app.llm.prompts import (
    AGENT_PLANNING_SYSTEM,
    AGENT_PLANNING_USER,
    AGENT_REFLECTION_SYSTEM,
    AGENT_REFLECTION_USER,
)
from app.retrieval.base import (
    RetrievalConfig,
    RetrievalContext,
    RetrievalResult,
    RetrievalStrategy,
    RetrievedChunk,
    StrategyName,
)
from app.retrieval.corrective.grading import grade_retrieval
from app.retrieval.fusion import deduplicate, reciprocal_rank_fusion
from app.retrieval.registry import get_strategy
from app.retrieval.rerank import rerank

log = get_logger("ragx.retrieval.agentic")

# Tool name -> strategy. This is the agent's action space.
TOOLS: dict[str, StrategyName] = {
    "dense_search": StrategyName.NAIVE,
    "hybrid_search": StrategyName.HYBRID,
    "graph_search": StrategyName.GRAPH,
    "multimodal_search": StrategyName.MULTIMODAL,
    "hyde_search": StrategyName.HYDE,
}


@dataclass
class AgentStep:
    step: int
    sub_question: str
    tool: str
    reason: str = ""
    depends_on: list[int] = field(default_factory=list)
    status: str = "pending"
    chunks_found: int = 0
    top_score: float = 0.0
    evidence_quality: float = 0.0
    latency_ms: float = 0.0
    finding: str = ""
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "sub_question": self.sub_question,
            "tool": self.tool,
            "strategy": TOOLS.get(self.tool, StrategyName.NAIVE).value,
            "reason": self.reason,
            "depends_on": self.depends_on,
            "status": self.status,
            "chunks_found": self.chunks_found,
            "top_score": round(self.top_score, 4),
            "evidence_quality": round(self.evidence_quality, 4),
            "latency_ms": round(self.latency_ms, 2),
            "finding": self.finding,
            "error": self.error,
        }


class AgenticRAG(RetrievalStrategy):
    name = StrategyName.AGENTIC
    description = (
        "Plans a multi-step retrieval strategy, calls the other RAG strategies as tools, "
        "cross-checks findings and reflects on whether the evidence is sufficient."
    )
    uses_llm = True

    def __init__(self) -> None:
        self.gateway = get_gateway()

    # ------------------------------------------------------------------ plan
    async def build_plan(
        self, query: str, context: RetrievalContext, max_steps: int
    ) -> tuple[list[AgentStep], str]:
        analysis = context.analysis
        analysis_text = "(not available)"
        if analysis is not None:
            analysis_text = (
                f"intent={analysis.intent.value}, complexity={analysis.complexity.value}, "
                f"multi_hop={analysis.multi_hop}, entities={analysis.entities[:6]}, "
                f"sub_questions={analysis.sub_questions[:4]}"
            )

        if not self.gateway.any_configured:
            return self._fallback_plan(query, context), "No LLM available; used a heuristic plan."

        documents = "\n".join(f"- {t}" for t in context.document_titles[:20]) or "(none indexed)"
        try:
            payload, _ = await self.gateway.complete_json(
                [
                    Message.system(AGENT_PLANNING_SYSTEM.format(max_steps=max_steps)),
                    Message.user(
                        AGENT_PLANNING_USER.format(
                            question=query, analysis=analysis_text, documents=documents
                        )
                    ),
                ],
                Purpose.AGENT_PLANNING,
                default={},
                temperature=0.1,
                max_output_tokens=900,
                trace=context.trace,
            )
        except Exception as exc:
            log.warning("agentic.planning_failed", error=str(exc)[:160])
            return self._fallback_plan(query, context), f"Planning failed ({str(exc)[:80]}); used a heuristic plan."

        raw_steps = payload.get("plan") if isinstance(payload, dict) else None
        if not isinstance(raw_steps, list) or not raw_steps:
            return self._fallback_plan(query, context), "The planner returned no steps; used a heuristic plan."

        steps: list[AgentStep] = []
        for index, raw in enumerate(raw_steps[:max_steps], start=1):
            if not isinstance(raw, dict):
                continue
            sub_question = str(raw.get("sub_question", "")).strip()
            if not sub_question:
                continue
            tool = str(raw.get("tool", "hybrid_search")).strip().lower()
            if tool not in TOOLS:
                tool = "hybrid_search"
            depends = [
                int(d) for d in (raw.get("depends_on") or []) if isinstance(d, (int, float)) and int(d) < index
            ]
            steps.append(
                AgentStep(
                    step=index,
                    sub_question=sub_question,
                    tool=tool,
                    reason=str(raw.get("reason", ""))[:240],
                    depends_on=depends,
                )
            )

        if not steps:
            return self._fallback_plan(query, context), "The plan contained no usable steps."
        return steps, str(payload.get("synthesis_note", ""))[:400]

    @staticmethod
    def _fallback_plan(query: str, context: RetrievalContext) -> list[AgentStep]:
        """Deterministic plan when the planner is unavailable.

        Uses the analyzer's sub-questions when it produced any, otherwise probes
        the query with two complementary tools.
        """
        analysis = context.analysis
        sub_questions = list(getattr(analysis, "sub_questions", []) or [])[:3]
        steps: list[AgentStep] = []
        if sub_questions:
            for index, sub_question in enumerate(sub_questions, start=1):
                steps.append(
                    AgentStep(
                        step=index,
                        sub_question=sub_question,
                        tool="hybrid_search",
                        reason="Sub-question identified during query analysis.",
                    )
                )
        else:
            steps.append(
                AgentStep(step=1, sub_question=query, tool="hybrid_search", reason="Primary retrieval pass.")
            )
            if analysis is not None and (analysis.relationship_query or analysis.multi_hop):
                steps.append(
                    AgentStep(
                        step=2,
                        sub_question=query,
                        tool="graph_search",
                        reason="The query needs relationship traversal.",
                    )
                )
        return steps

    # ------------------------------------------------------------------- act
    async def _run_step(
        self,
        step: AgentStep,
        context: RetrievalContext,
        config: RetrievalConfig,
        prior_findings: dict[int, str],
    ) -> tuple[AgentStep, list[RetrievedChunk]]:
        query = step.sub_question
        if step.depends_on:
            hints = " ".join(prior_findings.get(d, "") for d in step.depends_on).strip()
            if hints:
                # Feed the earlier step's finding forward -- this is what makes
                # a dependent hop actually multi-hop rather than two lookups.
                query = f"{query} (context: {truncate_words(hints, 40)})"

        strategy = get_strategy(TOOLS.get(step.tool, StrategyName.HYBRID))
        step_config = config.copy_with(top_k=max(4, config.top_k // 2), rerank=False)

        try:
            result = await strategy.run(query, context, step_config)
        except Exception as exc:
            step.status = "failed"
            step.error = str(exc)[:200]
            log.warning("agentic.step_failed", step=step.step, tool=step.tool, error=step.error)
            return step, []

        grading = await grade_retrieval(
            step.sub_question,
            result.chunks,
            get_settings().corrective_relevance_floor,
            trace=context.trace,
            allow_llm=False,  # heuristic only: the loop already costs enough calls
        )

        step.status = "completed"
        step.chunks_found = len(result.chunks)
        step.top_score = result.top_score
        step.evidence_quality = grading.overall
        step.latency_ms = result.latency_ms
        step.finding = truncate_words(result.chunks[0].content, 60) if result.chunks else ""

        for chunk in result.chunks:
            chunk.metadata["agent_step"] = step.step
            chunk.metadata["agent_sub_question"] = step.sub_question
        return step, result.chunks

    # -------------------------------------------------------------- reflect
    async def reflect(
        self, query: str, steps: list[AgentStep], chunks: list[RetrievedChunk], context: RetrievalContext
    ) -> dict[str, Any]:
        if not self.gateway.any_configured or not steps:
            return {"sufficient": True, "reason": "Reflection was unavailable; the collected evidence was used as-is."}

        progress = "\n".join(
            f"{s.step}. [{s.tool}] {s.sub_question} -> "
            f"{'no evidence' if not s.chunks_found else f'{s.chunks_found} passages, quality {s.evidence_quality:.2f}'}"
            for s in steps
        )
        evidence = "\n\n".join(
            f"({c.document_name}{', ' + c.location if c.location else ''}) {truncate_words(c.content, 55)}"
            for c in chunks[:10]
        ) or "(no evidence collected)"

        try:
            payload, _ = await self.gateway.complete_json(
                [
                    Message.system(AGENT_REFLECTION_SYSTEM),
                    Message.user(
                        AGENT_REFLECTION_USER.format(
                            question=query, progress=progress, evidence=evidence, count=len(chunks)
                        )
                    ),
                ],
                Purpose.AGENT_PLANNING,
                default={},
                temperature=0.0,
                max_output_tokens=450,
                trace=context.trace,
            )
        except Exception as exc:
            log.warning("agentic.reflection_failed", error=str(exc)[:160])
            return {"sufficient": True, "reason": "Reflection failed; the collected evidence was used as-is."}

        if not isinstance(payload, dict):
            return {"sufficient": True, "reason": "Reflection returned no verdict."}
        return {
            "sufficient": bool(payload.get("sufficient", True)),
            "missing": [str(m)[:160] for m in (payload.get("missing") or [])][:5],
            "next_query": (str(payload.get("next_query")) if payload.get("next_query") else None),
            "next_tool": (str(payload.get("next_tool")) if payload.get("next_tool") else None),
            "reason": str(payload.get("reason", ""))[:400],
        }

    # ------------------------------------------------------------------ main
    async def retrieve(
        self, query: str, context: RetrievalContext, config: RetrievalConfig
    ) -> RetrievalResult:
        settings = get_settings()
        max_steps = max(1, min(settings.agentic_max_steps, settings.agentic_max_subqueries + 2))

        with context.trace.span("agent_planning", category="agentic"):
            steps, synthesis_note = await self.build_plan(query, context, max_steps)

        prior_findings: dict[int, str] = {}
        collected: list[tuple[str, list[RetrievedChunk]]] = []
        executed: list[AgentStep] = []
        retrieval_calls = 0

        # Execute in dependency waves: everything with satisfied dependencies
        # runs concurrently.
        pending = list(steps)
        guard = 0
        while pending and guard < max_steps + 2:
            guard += 1
            ready = [s for s in pending if all(d in prior_findings for d in s.depends_on)]
            if not ready:
                ready = pending[:1]  # break a dependency cycle rather than stall
            outcomes = await asyncio.gather(
                *(self._run_step(s, context, config, prior_findings) for s in ready)
            )
            for step, chunks in outcomes:
                executed.append(step)
                retrieval_calls += 1
                if chunks:
                    collected.append((f"step_{step.step}", chunks))
                    prior_findings[step.step] = step.finding
                else:
                    prior_findings[step.step] = ""
            pending = [s for s in pending if s not in ready]

        # -- reflection ------------------------------------------------------
        interim = deduplicate(
            reciprocal_rank_fusion(collected) if collected else []
        )
        with context.trace.span("agent_reflection", category="agentic"):
            reflection = await self.reflect(query, executed, interim, context)

        follow_up_ran = False
        if not reflection.get("sufficient") and reflection.get("next_query") and len(executed) < max_steps:
            tool = (reflection.get("next_tool") or "hybrid_search").lower()
            follow_up = AgentStep(
                step=len(executed) + 1,
                sub_question=str(reflection["next_query"])[:400],
                tool=tool if tool in TOOLS else "hybrid_search",
                reason="Reflection identified a specific evidence gap: "
                + "; ".join(reflection.get("missing", []))[:200],
            )
            step, chunks = await self._run_step(follow_up, context, config, prior_findings)
            executed.append(step)
            retrieval_calls += 1
            follow_up_ran = True
            if chunks:
                collected.append((f"step_{step.step}", chunks))

        # -- fuse ------------------------------------------------------------
        if not collected:
            return RetrievalResult(
                chunks=[],
                strategy=self.name.value,
                effective_query=query,
                retrieval_calls=retrieval_calls,
                notes=["The agent's retrieval plan returned no evidence."],
                diagnostics={
                    "agentic": {
                        "plan": [s.as_dict() for s in executed],
                        "reflection": reflection,
                        "synthesis_note": synthesis_note,
                    }
                },
            )

        # Steps that found high-quality evidence get more weight in the fusion.
        weights = {
            f"step_{s.step}": 0.6 + 0.6 * max(0.0, min(1.0, s.evidence_quality)) for s in executed
        }
        fused = deduplicate(reciprocal_rank_fusion(collected, weights=weights))

        if config.rerank and fused:
            fused, rerank_diagnostics = await rerank(
                query,
                fused,
                use_llm=True,
                prefer_modalities=config.modalities,
                top_n=settings.rerank_top_n,
                trace=context.trace,
            )
        else:
            rerank_diagnostics = {"applied": False}

        for chunk in fused:
            chunk.add_source(self.name.value, chunk.score)

        strategies_used = list(
            dict.fromkeys(
                [self.name.value] + [TOOLS[s.tool].value for s in executed if s.tool in TOOLS]
            )
        )

        result = RetrievalResult(
            chunks=fused[: config.top_k],
            strategy=self.name.value,
            strategies_used=strategies_used,
            effective_query=query,
            retrieval_calls=retrieval_calls,
            reranked=bool(rerank_diagnostics.get("applied", config.rerank)),
            notes=[synthesis_note] if synthesis_note else [],
            diagnostics={
                "agentic": {
                    "planned_steps": len(steps),
                    "executed_steps": len(executed),
                    "plan": [s.as_dict() for s in executed],
                    "tools_used": sorted({s.tool for s in executed}),
                    "reflection": reflection,
                    "follow_up_step_ran": follow_up_ran,
                    "synthesis_note": synthesis_note,
                    "failed_steps": [s.step for s in executed if s.status == "failed"],
                },
                "rerank": rerank_diagnostics,
            },
        )
        result.rerank_positions()
        return result


__all__ = ["AgenticRAG", "AgentStep", "TOOLS"]
