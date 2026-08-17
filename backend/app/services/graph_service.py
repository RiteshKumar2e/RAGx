"""Knowledge-graph exploration for the frontend graph page."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import NotFoundError
from app.indexing.graph_store import get_graph_store
from app.models.document import Document


class GraphService:
    def __init__(self) -> None:
        self.graph = get_graph_store()

    async def export(
        self, session: AsyncSession, limit: int = 250, document_id: str | None = None
    ) -> dict[str, Any]:
        if document_id:
            document = await session.get(Document, document_id)
            if document is None:
                raise NotFoundError(f"Document '{document_id}' was not found.")

        payload = await self.graph.export(limit=limit, document_id=document_id)
        payload["backend"] = self.graph.backend
        return payload

    async def stats(self) -> dict[str, Any]:
        return await self.graph.stats()

    async def search(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        return await self.graph.search_entities(query, limit=limit)

    async def neighborhood(self, entity: str, depth: int = 2, limit: int = 50) -> dict[str, Any]:
        """Sub-graph around one entity, shaped for React Flow."""
        matches = await self.graph.search_entities(entity, limit=1)
        if not matches:
            return {"entity": entity, "depth": depth, "nodes": [], "edges": [], "paths": []}

        root = matches[0]
        paths = await self.graph.neighborhood([root["name"]], depth=depth, limit=limit)

        nodes: dict[str, dict[str, Any]] = {
            root["id"]: {
                "id": root["id"],
                "name": root["name"],
                "type": root.get("type", "CONCEPT"),
                "description": root.get("description", ""),
                "documents": root.get("documents", []),
                "chunk_ids": root.get("chunk_ids", []),
                "mentions": root.get("mentions", 1),
                "degree": root.get("degree", 0),
            }
        }
        edges: list[dict[str, Any]] = []
        seen_edges: set[tuple[str, str, str]] = set()

        for path in paths:
            for index, name in enumerate(path.entities):
                key = _key(name)
                nodes.setdefault(
                    key,
                    {
                        "id": key,
                        "name": name,
                        "type": "CONCEPT",
                        "description": "",
                        "documents": [],
                        "chunk_ids": [],
                        "mentions": 1,
                        "degree": 0,
                        "hop": index,
                    },
                )
            for index, relation in enumerate(path.relations):
                if index + 1 >= len(path.entities):
                    break
                source = _key(path.entities[index])
                target = _key(path.entities[index + 1])
                signature = (source, target, relation.get("type", "RELATED_TO"))
                if signature in seen_edges:
                    continue
                seen_edges.add(signature)
                edges.append(
                    {
                        "id": f"{source}->{target}-{len(edges)}",
                        "source": source,
                        "target": target,
                        "type": relation.get("type", "RELATED_TO"),
                        "confidence": relation.get("confidence", 0.5),
                        "document_id": relation.get("document_id", ""),
                        "chunk_id": relation.get("chunk_id", ""),
                        "context": relation.get("context", ""),
                    }
                )

        return {
            "entity": root["name"],
            "depth": depth,
            "nodes": list(nodes.values()),
            "edges": edges,
            "paths": [
                {
                    "entities": p.entities,
                    "relations": p.relations,
                    "score": p.score,
                    "description": p.describe(),
                    "chunk_ids": p.chunk_ids,
                }
                for p in paths[:25]
            ],
        }

    async def paths(self, source: str, target: str, max_depth: int = 4) -> dict[str, Any]:
        paths = await self.graph.paths_between(source, target, max_depth=max_depth)
        return {
            "source": source,
            "target": target,
            "found": bool(paths),
            "paths": [
                {
                    "entities": p.entities,
                    "relations": p.relations,
                    "score": p.score,
                    "description": p.describe(),
                    "chunk_ids": p.chunk_ids,
                }
                for p in paths
            ],
        }

    async def documents_for_entity(self, session: AsyncSession, entity: str) -> list[dict[str, Any]]:
        matches = await self.graph.search_entities(entity, limit=1)
        if not matches:
            return []
        document_ids = matches[0].get("documents") or []
        if not document_ids:
            return []
        rows = await session.scalars(select(Document).where(Document.id.in_(document_ids)))
        return [
            {"id": d.id, "filename": d.filename, "title": d.title, "status": d.status.value}
            for d in rows
        ]


def _key(name: str) -> str:
    from app.indexing.graph_store import normalize_entity  # noqa: PLC0415

    return normalize_entity(name)


_service: GraphService | None = None


def get_graph_service() -> GraphService:
    global _service
    if _service is None:
        _service = GraphService()
    return _service


__all__ = ["GraphService", "get_graph_service"]
