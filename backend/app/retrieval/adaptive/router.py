"""The Adaptive RAG Router.

This is the component the whole project exists to demonstrate. Given a
:class:`QueryAnalysis` it decides:

* which retrieval strategy is the **primary** one,
* which strategies (if any) should run **alongside** it and be fused,
* whether the result should be wrapped in **Corrective RAG**,
* whether the query warrants the **Agentic** loop,
* and the retrieval configuration (top-k, pool size, dense/sparse weighting,
  graph depth, reranking) those strategies should run with.

Two design rules matter more than any individual heuristic:

1. **Never run everything.** Each additional strategy costs latency, tokens and
   money. The router must justify every strategy it adds, and the justification
   is recorded in ``RoutingDecision.rules_fired`` and surfaced verbatim in the
   "Why this answer?" panel.
2. **Escalate, don't guess.** Corrective RAG is the recovery path for weak
   retrieval, so the router does not need to over-provision up front. A simple
   query starts cheap; if retrieval actually fails, correction repairs it.

The rules below are ordered from most to least specific. Each rule that fires
records its name and reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger
from app.retrieval.adaptive.analyzer import Complexity, Intent, QueryAnalysis
from app.retrieval.base import RetrievalConfig, StrategyName

log = get_logger("ragx.adaptive.router")


@dataclass
class RoutingDecision:
    primary: StrategyName = StrategyName.NAIVE
    parallel: list[StrategyName] = field(default_factory=list)
    use_corrective: bool = False
    use_agentic: bool = False
    config: RetrievalConfig = field(default_factory=RetrievalConfig)
    rules_fired: list[dict[str, str]] = field(default_factory=list)
    reason: str = ""
    estimated_llm_calls: int = 0
    mode: str = "single"  # single | composed | agentic

    @property
    def strategies(self) -> list[StrategyName]:
        ordered = [self.primary, *[s for s in self.parallel if s != self.primary]]
        if self.use_corrective and StrategyName.CORRECTIVE not in ordered:
            ordered.append(StrategyName.CORRECTIVE)
        if self.use_agentic and StrategyName.AGENTIC not in ordered:
            ordered.insert(0, StrategyName.AGENTIC)
        return ordered

    @property
    def strategy_values(self) -> list[str]:
        return [s.value for s in self.strategies]

    @property
    def strategy_labels(self) -> list[str]:
        return [s.label for s in self.strategies]

    def as_dict(self) -> dict[str, Any]:
        return {
            "primary": self.primary.value,
            "parallel": [s.value for s in self.parallel],
            "strategies": self.strategy_values,
            "strategy_labels": self.strategy_labels,
            "use_corrective": self.use_corrective,
            "use_agentic": self.use_agentic,
            "mode": self.mode,
            "reason": self.reason,
            "rules_fired": self.rules_fired,
            "estimated_llm_calls": self.estimated_llm_calls,
            "config": {
                "top_k": self.config.top_k,
                "candidate_pool": self.config.candidate_pool,
                "dense_weight": round(self.config.dense_weight, 3),
                "sparse_weight": round(self.config.sparse_weight, 3),
                "graph_depth": self.config.graph_depth,
                "rerank": self.config.rerank,
                "modalities": self.config.modalities,
                "min_score": self.config.min_score,
            },
        }


class AdaptiveRouter:
    """Rule-based strategy selection over the query analysis."""

    def __init__(self) -> None:
        self.settings = get_settings()

    # ------------------------------------------------------------------ main
    def route(
        self,
        analysis: QueryAnalysis,
        *,
        document_count: int = 0,
        graph_available: bool = True,
        multimodal_available: bool = True,
        forced_strategies: list[str] | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> RoutingDecision:
        settings = self.settings
        decision = RoutingDecision()
        rules: list[dict[str, str]] = []

        config = RetrievalConfig(
            top_k=settings.default_top_k,
            candidate_pool=settings.candidate_pool_size,
            rerank=settings.rerank_enabled,
            min_score=settings.min_relevance_score,
            graph_depth=2,
        )

        # -- explicit override: the user pinned strategies in Settings --------
        if forced_strategies:
            return self._forced(forced_strategies, analysis, config, overrides)

        # ================================================================
        # Rule 1 -- Visual / tabular content is a hard requirement.
        # A question about "Figure 3" cannot be answered from prose alone,
        # so Multimodal retrieval leads regardless of other signals.
        # ================================================================
        if (analysis.requires_visual or analysis.requires_tabular) and multimodal_available:
            decision.primary = StrategyName.MULTIMODAL
            config.modalities = analysis.modalities

            # Name the modality that actually triggered this, and note when the
            # signal came from the LLM rather than the query's own wording --
            # otherwise the explanation claims the query "references figures"
            # for a query where the lexical check found no such reference.
            kinds = [
                k
                for k, on in (("visual", analysis.requires_visual), ("tabular", analysis.requires_tabular))
                if on
            ]
            lexical = analysis.signals.get("markers", {})
            inferred = (analysis.requires_visual and not lexical.get("visual")) or (
                analysis.requires_tabular and not lexical.get("tabular")
            )
            reason = (
                f"The query needs {' and '.join(kinds)} evidence, so retrieval must cover "
                f"non-text modalities."
            )
            if inferred:
                reason += (
                    " This was inferred by query analysis rather than stated literally in the "
                    "query wording."
                )
            rules.append({"rule": "visual_or_tabular_requirement", "reason": reason})
            # Figure captions are terse; keyword matching on the label matters.
            if analysis.keyword_requirement >= 0.5:
                decision.parallel.append(StrategyName.HYBRID)
                rules.append(
                    {
                        "rule": "visual_with_exact_terms",
                        "reason": "The visual query also contains exact identifiers, so BM25 runs alongside.",
                    }
                )

        # ================================================================
        # Rule 2 -- Relationship / multi-hop questions need the graph.
        # These are precisely the questions no single passage answers.
        # ================================================================
        elif (analysis.relationship_query or analysis.multi_hop or analysis.intent is Intent.RELATIONSHIP) and graph_available:
            decision.primary = StrategyName.GRAPH
            config.graph_depth = 3 if analysis.multi_hop else 2
            rules.append(
                {
                    "rule": "relationship_or_multi_hop",
                    "reason": (
                        "The query asks how entities relate or requires chaining facts, which "
                        "graph traversal answers and single-passage retrieval does not."
                    ),
                }
            )
            if analysis.keyword_requirement >= 0.45:
                decision.parallel.append(StrategyName.HYBRID)
                rules.append(
                    {
                        "rule": "graph_with_exact_terms",
                        "reason": "Named entities must match exactly, so keyword search runs alongside the graph.",
                    }
                )

        # ================================================================
        # Rule 3 -- Keyword-dominant queries: exact terms beat semantics.
        # ================================================================
        elif analysis.keyword_requirement >= 0.55 and analysis.keyword_requirement > analysis.semantic_requirement:
            decision.primary = StrategyName.HYBRID
            total = analysis.keyword_requirement + analysis.semantic_requirement or 1.0
            config.sparse_weight = round(min(0.75, analysis.keyword_requirement / total), 3)
            config.dense_weight = round(1.0 - config.sparse_weight, 3)
            rules.append(
                {
                    "rule": "keyword_dominant",
                    "reason": (
                        f"The query contains exact technical terms "
                        f"(keyword requirement {analysis.keyword_requirement:.2f}), so BM25 is "
                        f"weighted above dense search."
                    ),
                }
            )

        # ================================================================
        # Rule 4 -- Semantically hard queries: HyDE bridges the vocabulary gap.
        # Gated on complexity so it does not spend an LLM call on easy lookups.
        # ================================================================
        elif (
            analysis.semantic_requirement >= 0.6
            and analysis.complexity is not Complexity.SIMPLE
            and analysis.intent in {Intent.ANALYSIS, Intent.EXPLORATORY, Intent.DEFINITION, Intent.SUMMARIZATION, Intent.PROCEDURAL}
        ):
            decision.primary = StrategyName.HYDE
            rules.append(
                {
                    "rule": "high_semantic_difficulty",
                    "reason": (
                        f"The query is conceptual (semantic requirement "
                        f"{analysis.semantic_requirement:.2f}) and its wording is unlikely to match "
                        f"the source text, so a hypothetical passage is used as the retrieval probe."
                    ),
                }
            )
            if analysis.keyword_requirement >= 0.5:
                decision.parallel.append(StrategyName.HYBRID)
                rules.append(
                    {
                        "rule": "hyde_with_exact_terms",
                        "reason": "Specific terms are present, so keyword search backs up the hypothesis probe.",
                    }
                )

        # ================================================================
        # Rule 5 -- Comparison across documents: breadth over depth.
        # ================================================================
        elif analysis.intent is Intent.COMPARISON or (analysis.cross_document and analysis.expected_documents > 1):
            decision.primary = StrategyName.HYBRID
            config.top_k = max(config.top_k, 10)
            config.candidate_pool = int(config.candidate_pool * 1.5)
            config.extra["max_per_document"] = max(2, config.top_k // max(2, analysis.expected_documents))
            rules.append(
                {
                    "rule": "cross_document_comparison",
                    "reason": (
                        f"The query compares across roughly {analysis.expected_documents} documents, "
                        f"so the candidate pool is widened and per-document contribution is capped."
                    ),
                }
            )

        # ================================================================
        # Rule 6 -- Default: a simple lookup gets the cheapest pipeline.
        # This is the rule that keeps the system inexpensive.
        # ================================================================
        else:
            decision.primary = StrategyName.NAIVE
            rules.append(
                {
                    "rule": "simple_lookup_default",
                    "reason": (
                        "The query is a direct lookup with no multi-hop, relationship or visual "
                        "requirement, so single-shot dense retrieval is sufficient. No expensive "
                        "strategy is used."
                    ),
                }
            )
            if analysis.complexity is Complexity.SIMPLE:
                config.top_k = min(config.top_k, 6)
                config.candidate_pool = min(config.candidate_pool, 24)
                config.rerank = False
                rules.append(
                    {
                        "rule": "cost_guard_simple_query",
                        "reason": "Simple query: reranking is skipped and the candidate pool is reduced.",
                    }
                )

        # ================================================================
        # Rule 7 -- Corrective wrapping.
        # Applied when a wrong answer would be costly, or when the query is
        # ambiguous enough that first-pass retrieval is likely to miss.
        # ================================================================
        if analysis.requires_verification or analysis.complexity is Complexity.COMPLEX or analysis.ambiguity >= 0.4:
            decision.use_corrective = True
            reason_bits = []
            if analysis.requires_verification:
                reason_bits.append("the answer must be verifiable against sources")
            if analysis.complexity is Complexity.COMPLEX:
                reason_bits.append("the query is complex")
            if analysis.ambiguity >= 0.4:
                reason_bits.append(f"the query is ambiguous ({analysis.ambiguity:.2f})")
            rules.append(
                {
                    "rule": "corrective_wrapping",
                    "reason": (
                        "Retrieval quality is graded and repaired before generation because "
                        + " and ".join(reason_bits)
                        + "."
                    ),
                }
            )
            config.rerank = True

        # ================================================================
        # Rule 8 -- Agentic escalation.
        # Reserved for genuinely decomposable research questions: the loop costs
        # several LLM calls, so it must clear a high bar.
        # ================================================================
        agentic_signals = sum(
            [
                analysis.complexity is Complexity.COMPLEX,
                analysis.multi_hop,
                len(analysis.sub_questions) >= 2,
                analysis.expected_documents >= 3,
                analysis.intent in {Intent.MULTI_HOP, Intent.ANALYSIS},
            ]
        )
        if agentic_signals >= 3 and document_count >= 1:
            decision.use_agentic = True
            decision.mode = "agentic"
            config.top_k = max(config.top_k, 10)
            rules.append(
                {
                    "rule": "agentic_escalation",
                    "reason": (
                        f"{agentic_signals} of 5 complexity signals fired (multi-hop, decomposable "
                        f"sub-questions, multiple documents), so the query is planned and answered "
                        f"step by step rather than in a single retrieval pass."
                    ),
                }
            )
        elif decision.parallel:
            decision.mode = "composed"

        # -- knowledge-base-aware guards --------------------------------------
        if document_count <= 1 and StrategyName.GRAPH in ([decision.primary] + decision.parallel):
            # A one-document graph rarely has cross-document structure worth the hop.
            config.graph_depth = min(config.graph_depth, 2)
            rules.append(
                {
                    "rule": "small_kb_graph_depth_cap",
                    "reason": "Only one document is indexed, so graph traversal depth is capped.",
                }
            )

        if overrides:
            config = self._apply_overrides(config, overrides)
            rules.append({"rule": "request_overrides", "reason": "Per-request retrieval overrides were applied."})

        decision.config = config
        decision.rules_fired = rules
        decision.reason = self._compose_reason(decision, analysis)
        decision.estimated_llm_calls = self._estimate_llm_calls(decision)

        log.info(
            "router.decision",
            strategies=decision.strategy_values,
            intent=analysis.intent.value,
            complexity=analysis.complexity.value,
            mode=decision.mode,
            estimated_llm_calls=decision.estimated_llm_calls,
        )
        return decision

    # -------------------------------------------------------------- helpers
    def _forced(
        self,
        forced: list[str],
        analysis: QueryAnalysis,
        config: RetrievalConfig,
        overrides: dict[str, Any] | None,
    ) -> RoutingDecision:
        valid: list[StrategyName] = []
        for name in forced:
            try:
                valid.append(StrategyName(name.lower()))
            except ValueError:
                continue
        if not valid:
            valid = [StrategyName.NAIVE]

        decision = RoutingDecision()
        decision.use_corrective = StrategyName.CORRECTIVE in valid
        decision.use_agentic = StrategyName.AGENTIC in valid
        core = [s for s in valid if s not in {StrategyName.CORRECTIVE, StrategyName.AGENTIC, StrategyName.ADAPTIVE}]
        decision.primary = core[0] if core else StrategyName.NAIVE
        decision.parallel = core[1:]
        if analysis.modalities and decision.primary is StrategyName.MULTIMODAL:
            config.modalities = analysis.modalities
        if overrides:
            config = self._apply_overrides(config, overrides)
        decision.config = config
        decision.mode = "agentic" if decision.use_agentic else ("composed" if decision.parallel else "single")
        decision.rules_fired = [
            {
                "rule": "explicit_strategy_override",
                "reason": (
                    "Strategies were pinned by the caller, so adaptive routing was bypassed. "
                    "This is the mode the evaluation harness uses to benchmark a single strategy."
                ),
            }
        ]
        decision.reason = (
            f"Strategy selection was overridden by the caller: {', '.join(s.label for s in decision.strategies)}."
        )
        decision.estimated_llm_calls = self._estimate_llm_calls(decision)
        return decision

    @staticmethod
    def _apply_overrides(config: RetrievalConfig, overrides: dict[str, Any]) -> RetrievalConfig:
        allowed = {
            "top_k", "candidate_pool", "document_ids", "modalities", "rerank",
            "min_score", "dense_weight", "sparse_weight", "graph_depth",
        }
        clean = {k: v for k, v in overrides.items() if k in allowed and v is not None}
        return config.copy_with(**clean) if clean else config

    @staticmethod
    def _compose_reason(decision: RoutingDecision, analysis: QueryAnalysis) -> str:
        labels = decision.strategy_labels
        selected = " + ".join(labels)
        drivers: list[str] = []
        if analysis.multi_hop:
            drivers.append("multi-hop reasoning")
        if analysis.relationship_query:
            drivers.append("relationship traversal")
        if analysis.keyword_requirement >= 0.55:
            drivers.append("exact terminology matching")
        if analysis.semantic_requirement >= 0.6:
            drivers.append("conceptual/semantic matching")
        if analysis.requires_visual:
            drivers.append("figure and chart evidence")
        if analysis.requires_tabular:
            drivers.append("tabular evidence")
        if analysis.cross_document:
            drivers.append("cross-document synthesis")
        if decision.use_corrective:
            drivers.append("evidence verification")
        if not drivers:
            drivers.append("a direct single-fact lookup")
        return f"Selected {selected} because the query requires {', '.join(drivers)}."

    @staticmethod
    def _estimate_llm_calls(decision: RoutingDecision) -> int:
        """Rough forward cost estimate, used for the cost-awareness display."""
        calls = 1  # query analysis
        if StrategyName.HYDE in decision.strategies:
            calls += 1
        if decision.use_corrective:
            calls += 2  # grading + rewriting, when triggered
        if decision.use_agentic:
            calls += 4  # plan + reflection + sub-steps
        if decision.config.rerank:
            calls += 1
        calls += 1  # generation
        calls += 2  # claim extraction + evidence matching
        return calls


__all__ = ["AdaptiveRouter", "RoutingDecision"]
