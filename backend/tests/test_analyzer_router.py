"""Query analyzer and adaptive router.

These are the behavioural guarantees the whole project rests on: the router must
pick the right strategy for a query class, and -- just as importantly -- must
*not* over-provision for simple ones.
"""

from __future__ import annotations

import pytest

from app.retrieval.adaptive.analyzer import Complexity, Intent, QueryAnalyzer, detect_visual
from app.retrieval.adaptive.router import AdaptiveRouter
from app.retrieval.base import StrategyName


@pytest.fixture(scope="module")
def analyzer() -> QueryAnalyzer:
    return QueryAnalyzer()


@pytest.fixture(scope="module")
def router() -> AdaptiveRouter:
    return AdaptiveRouter()


def route(analyzer: QueryAnalyzer, router: AdaptiveRouter, query: str, **kwargs):
    analysis = analyzer.heuristic(query)
    return analysis, router.route(analysis, document_count=kwargs.pop("document_count", 3), **kwargs)


# ---------------------------------------------------------------- visual
@pytest.mark.parametrize(
    "query,expected",
    [
        ("What does Figure 3 show?", True),
        ("Explain the architecture diagram.", True),
        ("What trend is visible in the chart?", True),
        ("Show me the image of the pipeline.", True),
        # False positives the naive keyword check gets wrong:
        ("What were the quarterly revenue figures in 2019?", False),
        ("How is the graph database configured?", False),
        ("What are the headline figures for the study?", False),
        ("What optimizer was used during training?", False),
    ],
)
def test_visual_detection(query: str, expected: bool) -> None:
    is_visual, reason = detect_visual(query)
    assert is_visual is expected, f"{query!r} -> {reason}"


# --------------------------------------------------------------- analysis
def test_simple_query_is_simple(analyzer: QueryAnalyzer) -> None:
    analysis = analyzer.heuristic("What optimizer was used?")
    assert analysis.complexity is Complexity.SIMPLE
    assert analysis.multi_hop is False
    assert analysis.intent is Intent.FACTUAL_LOOKUP


def test_chained_interrogative_is_multi_hop(analyzer: QueryAnalyzer) -> None:
    analysis = analyzer.heuristic(
        "Which model achieved the best result, and what data was that model trained on?"
    )
    assert analysis.multi_hop is True
    assert analysis.intent is Intent.MULTI_HOP


def test_keyword_heavy_query_raises_keyword_requirement(analyzer: QueryAnalyzer) -> None:
    technical = analyzer.heuristic("What mAP does MobileNetV2 reach on NEU-DET?")
    conceptual = analyzer.heuristic("Why is this approach considered more efficient overall?")
    assert technical.keyword_requirement > conceptual.keyword_requirement
    assert conceptual.semantic_requirement > technical.semantic_requirement


def test_relationship_query_detected(analyzer: QueryAnalyzer) -> None:
    analysis = analyzer.heuristic("How does DefectNet relate to MobileNetV2?")
    assert analysis.relationship_query is True


def test_adversarial_query_does_not_claim_visual(analyzer: QueryAnalyzer) -> None:
    analysis = analyzer.heuristic("What were the quarterly revenue figures in 2019?")
    assert analysis.requires_visual is False


# ----------------------------------------------------------------- routing
def test_simple_query_routes_to_naive_only(analyzer: QueryAnalyzer, router: AdaptiveRouter) -> None:
    _, decision = route(analyzer, router, "What optimizer was used?")
    assert decision.primary is StrategyName.NAIVE
    assert decision.use_agentic is False
    assert not decision.parallel


def test_simple_query_skips_expensive_work(analyzer: QueryAnalyzer, router: AdaptiveRouter) -> None:
    """The core cost claim: a simple lookup must not trigger the heavy path."""
    _, decision = route(analyzer, router, "What optimizer was used?")
    expensive = {StrategyName.HYDE, StrategyName.AGENTIC, StrategyName.GRAPH, StrategyName.MULTIMODAL}
    assert not expensive.intersection(decision.strategies)
    assert decision.config.rerank is False


def test_keyword_query_routes_hybrid_with_sparse_bias(analyzer: QueryAnalyzer, router: AdaptiveRouter) -> None:
    analysis, decision = route(analyzer, router, 'Find the exact term "NEU-DET" and its mAP value')
    assert decision.primary is StrategyName.HYBRID
    assert decision.config.sparse_weight >= decision.config.dense_weight


def test_visual_query_routes_multimodal(analyzer: QueryAnalyzer, router: AdaptiveRouter) -> None:
    _, decision = route(analyzer, router, "What does Figure 2 show about accuracy?")
    assert decision.primary is StrategyName.MULTIMODAL
    assert decision.config.modalities


def test_relationship_query_routes_graph(analyzer: QueryAnalyzer, router: AdaptiveRouter) -> None:
    _, decision = route(analyzer, router, "How does DefectNet relate to the prior work it builds on?")
    assert decision.primary is StrategyName.GRAPH


def test_graph_not_selected_when_graph_is_empty(analyzer: QueryAnalyzer, router: AdaptiveRouter) -> None:
    _, decision = route(
        analyzer, router, "How does DefectNet relate to MobileNetV2?", graph_available=False
    )
    assert decision.primary is not StrategyName.GRAPH


def test_complex_query_escalates_to_agentic(analyzer: QueryAnalyzer, router: AdaptiveRouter) -> None:
    _, decision = route(
        analyzer,
        router,
        "What are the main limitations, what evidence supports each limitation, and what "
        "would be required to address them across all the indexed papers?",
    )
    assert decision.use_agentic is True
    assert decision.mode == "agentic"


def test_router_never_selects_every_strategy(analyzer: QueryAnalyzer, router: AdaptiveRouter) -> None:
    """The explicit anti-requirement: no query should fan out to all eight."""
    queries = [
        "What optimizer was used?",
        "What does Figure 2 show?",
        "How does A relate to B?",
        "Compare the approaches across the papers.",
        "What are the limitations, what causes them, and how could they be fixed?",
    ]
    for query in queries:
        _, decision = route(analyzer, router, query)
        assert len(decision.strategies) < len(StrategyName), (
            f"{query!r} selected {decision.strategy_values}"
        )


def test_verification_query_enables_corrective(analyzer: QueryAnalyzer, router: AdaptiveRouter) -> None:
    _, decision = route(analyzer, router, "What is the exact reported accuracy percentage?")
    assert decision.use_corrective is True


def test_forced_strategies_bypass_routing(analyzer: QueryAnalyzer, router: AdaptiveRouter) -> None:
    analysis = analyzer.heuristic("Anything at all")
    decision = router.route(analysis, document_count=3, forced_strategies=["graph"])
    assert decision.primary is StrategyName.GRAPH
    assert decision.rules_fired[0]["rule"] == "explicit_strategy_override"


def test_every_decision_is_explained(analyzer: QueryAnalyzer, router: AdaptiveRouter) -> None:
    for query in ["What optimizer was used?", "What does Figure 1 show?", "How does A relate to B?"]:
        _, decision = route(analyzer, router, query)
        assert decision.rules_fired, f"no rule recorded for {query!r}"
        assert decision.reason
        for rule in decision.rules_fired:
            assert rule["rule"] and rule["reason"]


def test_cost_estimate_grows_with_complexity(analyzer: QueryAnalyzer, router: AdaptiveRouter) -> None:
    _, simple = route(analyzer, router, "What optimizer was used?")
    _, complex_ = route(
        analyzer,
        router,
        "What are the main limitations, what evidence supports each, and what would fix them?",
    )
    assert complex_.estimated_llm_calls > simple.estimated_llm_calls
