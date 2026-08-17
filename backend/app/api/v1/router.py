"""v1 API router aggregation."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1 import documents, evaluation, graph, query, system

api_router = APIRouter()
api_router.include_router(system.router)
api_router.include_router(documents.router)
api_router.include_router(query.router)
api_router.include_router(query.evidence_router)
api_router.include_router(graph.router)
api_router.include_router(evaluation.router)

__all__ = ["api_router"]
