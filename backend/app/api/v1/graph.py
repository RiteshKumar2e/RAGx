"""Knowledge-graph exploration endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Query

from app.api.deps import SessionDep
from app.schemas.graph import (
    EntitySearchResult,
    GraphPathsResponse,
    GraphResponse,
    GraphStats,
    NeighborhoodResponse,
)
from app.services.graph_service import get_graph_service

router = APIRouter(prefix="/graph", tags=["graph"])


@router.get(
    "",
    response_model=GraphResponse,
    summary="Export the knowledge graph",
    description=(
        "Returns nodes and edges shaped for React Flow, ordered by degree so the most "
        "connected entities are always present when the result is truncated."
    ),
)
async def export_graph(
    session: SessionDep,
    limit: int = Query(250, ge=10, le=1000),
    document_id: str | None = Query(None, description="Restrict to entities from one document."),
) -> GraphResponse:
    return GraphResponse(**await get_graph_service().export(session, limit=limit, document_id=document_id))


@router.get("/stats", response_model=GraphStats, summary="Graph statistics")
async def graph_stats() -> GraphStats:
    return GraphStats(**await get_graph_service().stats())


@router.get("/search", response_model=list[EntitySearchResult], summary="Search entities")
async def search_entities(
    q: str = Query(min_length=1, description="Entity name or fragment."),
    limit: int = Query(20, ge=1, le=100),
) -> list[EntitySearchResult]:
    return [EntitySearchResult(**e) for e in await get_graph_service().search(q, limit=limit)]


@router.get(
    "/neighborhood",
    response_model=NeighborhoodResponse,
    summary="Sub-graph around one entity",
)
async def neighborhood(
    entity: str = Query(min_length=1),
    depth: int = Query(2, ge=1, le=4),
    limit: int = Query(50, ge=5, le=200),
) -> NeighborhoodResponse:
    return NeighborhoodResponse(**await get_graph_service().neighborhood(entity, depth=depth, limit=limit))


@router.get(
    "/paths",
    response_model=GraphPathsResponse,
    summary="Find relationship paths between two entities",
    description="The traversal that answers multi-hop relationship questions.",
)
async def paths(
    source: str = Query(min_length=1),
    target: str = Query(min_length=1),
    max_depth: int = Query(4, ge=1, le=5),
) -> GraphPathsResponse:
    return GraphPathsResponse(**await get_graph_service().paths(source, target, max_depth=max_depth))


@router.get("/entity/{entity}/documents", summary="Documents an entity appears in")
async def entity_documents(session: SessionDep, entity: str) -> list[dict]:
    return await get_graph_service().documents_for_entity(session, entity)


__all__ = ["router"]
