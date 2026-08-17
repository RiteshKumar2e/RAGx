"""Knowledge-graph schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class GraphNode(BaseModel):
    id: str
    name: str
    type: str = "CONCEPT"
    description: str = ""
    documents: list[str] = Field(default_factory=list)
    chunk_ids: list[str] = Field(default_factory=list)
    mentions: int = 1
    degree: int = 0


class GraphEdge(BaseModel):
    id: str
    source: str
    target: str
    type: str = "RELATED_TO"
    confidence: float = 0.5
    document_id: str = ""
    chunk_id: str = ""
    context: str = ""


class GraphResponse(BaseModel):
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    truncated: bool = False
    backend: str = "networkx"


class GraphStats(BaseModel):
    backend: str = "networkx"
    entities: int = 0
    relations: int = 0
    entity_types: dict[str, int] = Field(default_factory=dict)
    relation_types: dict[str, int] = Field(default_factory=dict)
    top_entities: list[dict[str, Any]] = Field(default_factory=list)


class EntitySearchResult(BaseModel):
    id: str
    name: str
    type: str = "CONCEPT"
    description: str = ""
    documents: list[str] = Field(default_factory=list)
    chunk_ids: list[str] = Field(default_factory=list)
    mentions: int = 1
    degree: int = 0
    score: float = 0.0


class GraphPathSchema(BaseModel):
    entities: list[str] = Field(default_factory=list)
    relations: list[dict[str, Any]] = Field(default_factory=list)
    score: float = 0.0
    description: str = ""
    chunk_ids: list[str] = Field(default_factory=list)


class GraphPathsResponse(BaseModel):
    source: str
    target: str
    paths: list[GraphPathSchema] = Field(default_factory=list)
    found: bool = False


class NeighborhoodResponse(BaseModel):
    entity: str
    depth: int = 2
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    paths: list[GraphPathSchema] = Field(default_factory=list)


__all__ = [
    "GraphNode",
    "GraphEdge",
    "GraphResponse",
    "GraphStats",
    "EntitySearchResult",
    "GraphPathSchema",
    "GraphPathsResponse",
    "NeighborhoodResponse",
]
