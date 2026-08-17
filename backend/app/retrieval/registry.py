"""Strategy registry.

Instances are created lazily and cached, and the composite strategies
(Corrective, Adaptive, Agentic) are constructed through factory functions so
they can wrap other strategies without import cycles.
"""

from __future__ import annotations

from typing import Any

from app.retrieval.base import RetrievalStrategy, StrategyName

_cache: dict[str, RetrievalStrategy] = {}


def _build(name: StrategyName) -> RetrievalStrategy:
    if name is StrategyName.NAIVE:
        from app.retrieval.naive.strategy import NaiveRAG  # noqa: PLC0415

        return NaiveRAG()
    if name is StrategyName.HYBRID:
        from app.retrieval.hybrid.strategy import HybridRAG  # noqa: PLC0415

        return HybridRAG()
    if name is StrategyName.HYDE:
        from app.retrieval.hyde.strategy import HyDERAG  # noqa: PLC0415

        return HyDERAG()
    if name is StrategyName.MULTIMODAL:
        from app.retrieval.multimodal.strategy import MultimodalRAG  # noqa: PLC0415

        return MultimodalRAG()
    if name is StrategyName.GRAPH:
        from app.retrieval.graph.strategy import GraphRAG  # noqa: PLC0415

        return GraphRAG()
    if name is StrategyName.CORRECTIVE:
        # Default inner retriever for a bare "corrective" request; the router
        # normally supplies its own base via ``wrap_corrective``.
        from app.retrieval.corrective.strategy import CorrectiveRAG  # noqa: PLC0415

        return CorrectiveRAG(get_strategy(StrategyName.HYBRID))
    if name is StrategyName.ADAPTIVE:
        from app.retrieval.adaptive.strategy import AdaptiveRAG  # noqa: PLC0415

        return AdaptiveRAG()
    if name is StrategyName.AGENTIC:
        from app.retrieval.agentic.strategy import AgenticRAG  # noqa: PLC0415

        return AgenticRAG()
    raise ValueError(f"Unknown strategy '{name}'.")


def get_strategy(name: StrategyName | str) -> RetrievalStrategy:
    key = name.value if isinstance(name, StrategyName) else str(name).lower()
    if key not in _cache:
        _cache[key] = _build(StrategyName(key))
    return _cache[key]


def wrap_corrective(base: RetrievalStrategy) -> RetrievalStrategy:
    from app.retrieval.corrective.strategy import CorrectiveRAG  # noqa: PLC0415

    return CorrectiveRAG(base)


def reset_registry() -> None:
    _cache.clear()


def describe_strategies() -> list[dict[str, Any]]:
    """Catalogue for the Settings and Evaluation pages."""
    return [get_strategy(name).info() for name in StrategyName]


__all__ = ["get_strategy", "wrap_corrective", "reset_registry", "describe_strategies"]
