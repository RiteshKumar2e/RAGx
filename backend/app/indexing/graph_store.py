"""Knowledge-graph storage for Graph RAG.

Two interchangeable backends implement :class:`GraphStore`:

* :class:`Neo4jGraphStore` -- the production backend, using Cypher traversal.
* :class:`NetworkXGraphStore` -- an embedded backend persisted to JSON, used
  when no Neo4j instance is configured. It implements the same traversal
  semantics (typed edges, bounded-depth neighbourhood expansion, shortest paths
  between entities) in-process so Graph RAG genuinely works without external
  infrastructure.

Both store the same shape: ``Entity`` nodes connected by typed ``Relation``
edges, each edge carrying the chunk it was extracted from so graph hits remain
citable.
"""

from __future__ import annotations

import asyncio
import json
import re
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.text import tokenize

log = get_logger("ragx.graph")

_NORMALIZE = re.compile(r"[^a-z0-9]+")


def normalize_entity(name: str) -> str:
    return _NORMALIZE.sub(" ", (name or "").lower()).strip()


@dataclass(slots=True)
class GraphEntity:
    name: str
    entity_type: str = "CONCEPT"
    description: str = ""
    salience: float = 0.5
    document_id: str = ""
    chunk_ids: list[str] = field(default_factory=list)

    @property
    def normalized(self) -> str:
        return normalize_entity(self.name)


@dataclass(slots=True)
class GraphRelation:
    source: str
    target: str
    relation_type: str = "RELATED_TO"
    confidence: float = 0.5
    document_id: str = ""
    chunk_id: str = ""
    context: str = ""


@dataclass(slots=True)
class GraphPath:
    """A traversal result: the entity chain plus the edges that connect it."""

    entities: list[str]
    relations: list[dict[str, Any]]
    score: float = 0.0

    @property
    def chunk_ids(self) -> list[str]:
        seen: list[str] = []
        for rel in self.relations:
            cid = rel.get("chunk_id")
            if cid and cid not in seen:
                seen.append(cid)
        return seen

    def describe(self) -> str:
        if not self.entities:
            return ""
        parts = [self.entities[0]]
        for i, rel in enumerate(self.relations):
            target = self.entities[i + 1] if i + 1 < len(self.entities) else "?"
            parts.append(f"-[{rel.get('type', 'RELATED_TO')}]->")
            parts.append(target)
        return " ".join(parts)


class GraphStore(ABC):
    backend: str = "base"

    @abstractmethod
    async def ensure_ready(self) -> None: ...

    @abstractmethod
    async def upsert(self, entities: list[GraphEntity], relations: list[GraphRelation]) -> dict[str, int]: ...

    @abstractmethod
    async def delete_document(self, document_id: str) -> None: ...

    @abstractmethod
    async def search_entities(self, query: str, limit: int = 10) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def neighborhood(self, names: list[str], depth: int = 2, limit: int = 60) -> list[GraphPath]: ...

    @abstractmethod
    async def paths_between(self, source: str, target: str, max_depth: int = 4) -> list[GraphPath]: ...

    @abstractmethod
    async def export(self, limit: int = 300, document_id: str | None = None) -> dict[str, Any]: ...

    @abstractmethod
    async def stats(self) -> dict[str, Any]: ...

    @abstractmethod
    async def health(self) -> dict[str, Any]: ...

    async def close(self) -> None:  # pragma: no cover - optional
        return None


# ===========================================================================
# NetworkX backend (embedded)
# ===========================================================================
class NetworkXGraphStore(GraphStore):
    backend = "networkx"

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or get_settings().graph_fallback_path)
        self._graph: Any = None
        self._lock = asyncio.Lock()
        self._loaded = False
        # Set when this store was chosen because Neo4j was misconfigured, so
        # health reporting can say *why* the embedded store is in use.
        self.config_warning: str | None = None

    def _new_graph(self) -> Any:
        import networkx as nx  # noqa: PLC0415

        return nx.MultiDiGraph()

    def _load_sync(self) -> None:
        import networkx as nx  # noqa: PLC0415

        self._graph = nx.MultiDiGraph()
        if not self.path.exists():
            self._loaded = True
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            for node in payload.get("nodes", []):
                key = node.pop("id")
                self._graph.add_node(key, **node)
            for edge in payload.get("edges", []):
                source = edge.pop("source")
                target = edge.pop("target")
                self._graph.add_edge(source, target, **edge)
            log.info(
                "graph.loaded",
                backend=self.backend,
                nodes=self._graph.number_of_nodes(),
                edges=self._graph.number_of_edges(),
            )
        except Exception as exc:  # pragma: no cover
            log.warning("graph.load_failed", error=str(exc))
            self._graph = nx.MultiDiGraph()
        self._loaded = True

    def _persist_sync(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "nodes": [{"id": n, **d} for n, d in self._graph.nodes(data=True)],
            "edges": [{"source": u, "target": v, **d} for u, v, d in self._graph.edges(data=True)],
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(self.path)

    async def ensure_ready(self) -> None:
        if self._loaded:
            return
        async with self._lock:
            if not self._loaded:
                await asyncio.to_thread(self._load_sync)

    # ---------------------------------------------------------------- writes
    async def upsert(self, entities: list[GraphEntity], relations: list[GraphRelation]) -> dict[str, int]:
        await self.ensure_ready()

        def _upsert() -> dict[str, int]:
            for entity in entities:
                key = entity.normalized
                if not key:
                    continue
                if self._graph.has_node(key):
                    node = self._graph.nodes[key]
                    node["mentions"] = node.get("mentions", 0) + 1
                    node["salience"] = max(node.get("salience", 0.0), entity.salience)
                    docs = set(node.get("documents", []))
                    docs.add(entity.document_id)
                    node["documents"] = sorted(d for d in docs if d)
                    chunks = set(node.get("chunk_ids", [])) | set(entity.chunk_ids)
                    node["chunk_ids"] = sorted(c for c in chunks if c)[:50]
                    if entity.description and len(entity.description) > len(node.get("description", "")):
                        node["description"] = entity.description
                    if entity.entity_type and node.get("type") == "CONCEPT":
                        node["type"] = entity.entity_type
                else:
                    self._graph.add_node(
                        key,
                        name=entity.name,
                        type=entity.entity_type or "CONCEPT",
                        description=entity.description or "",
                        salience=entity.salience,
                        mentions=1,
                        documents=[entity.document_id] if entity.document_id else [],
                        chunk_ids=list(entity.chunk_ids)[:50],
                    )

            added_edges = 0
            for relation in relations:
                source, target = normalize_entity(relation.source), normalize_entity(relation.target)
                if not source or not target or source == target:
                    continue
                if not self._graph.has_node(source) or not self._graph.has_node(target):
                    continue
                # Collapse duplicate (source, type, target) triples, keeping the
                # highest-confidence evidence.
                existing = None
                for key, data in self._graph.get_edge_data(source, target, default={}).items():
                    if data.get("type") == relation.relation_type:
                        existing = (key, data)
                        break
                if existing:
                    _, data = existing
                    data["confidence"] = max(data.get("confidence", 0.0), relation.confidence)
                    data["weight"] = data.get("weight", 1) + 1
                    continue
                self._graph.add_edge(
                    source,
                    target,
                    type=relation.relation_type or "RELATED_TO",
                    confidence=relation.confidence,
                    document_id=relation.document_id,
                    chunk_id=relation.chunk_id,
                    context=(relation.context or "")[:400],
                    weight=1,
                )
                added_edges += 1

            self._persist_sync()
            return {"entities": len(entities), "relations": added_edges}

        async with self._lock:
            return await asyncio.to_thread(_upsert)

    async def delete_document(self, document_id: str) -> None:
        await self.ensure_ready()

        def _delete() -> None:
            drop_edges = [
                (u, v, k)
                for u, v, k, d in self._graph.edges(keys=True, data=True)
                if d.get("document_id") == document_id
            ]
            self._graph.remove_edges_from(drop_edges)

            orphan_nodes: list[str] = []
            for node, data in list(self._graph.nodes(data=True)):
                docs = [d for d in data.get("documents", []) if d != document_id]
                data["documents"] = docs
                if not docs:
                    orphan_nodes.append(node)
            self._graph.remove_nodes_from(orphan_nodes)
            self._persist_sync()

        async with self._lock:
            await asyncio.to_thread(_delete)

    # ---------------------------------------------------------------- search
    async def search_entities(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        await self.ensure_ready()

        def _search() -> list[dict[str, Any]]:
            terms = set(tokenize(query, drop_stopwords=True))
            normalized_query = normalize_entity(query)
            scored: list[tuple[float, dict[str, Any]]] = []
            for node, data in self._graph.nodes(data=True):
                name = data.get("name", node)
                node_terms = set(tokenize(name, drop_stopwords=False))
                score = 0.0
                if normalized_query and normalized_query == node:
                    score = 1.0
                elif normalized_query and normalized_query in node:
                    score = 0.85
                elif node and node in normalized_query:
                    score = 0.8
                elif terms and node_terms:
                    overlap = len(terms & node_terms) / len(node_terms)
                    score = overlap * 0.7
                if score <= 0.05:
                    continue
                score += min(0.1, data.get("mentions", 1) * 0.01)
                scored.append(
                    (
                        score,
                        {
                            "id": node,
                            "name": name,
                            "type": data.get("type", "CONCEPT"),
                            "description": data.get("description", ""),
                            "documents": data.get("documents", []),
                            "chunk_ids": data.get("chunk_ids", []),
                            "mentions": data.get("mentions", 1),
                            "degree": self._graph.degree(node),
                            "score": round(score, 4),
                        },
                    )
                )
            scored.sort(key=lambda x: x[0], reverse=True)
            return [item for _, item in scored[:limit]]

        return await asyncio.to_thread(_search)

    async def neighborhood(self, names: list[str], depth: int = 2, limit: int = 60) -> list[GraphPath]:
        """Breadth-first expansion from seed entities, returning each traversed
        path so the caller can cite the chunk every hop came from."""
        await self.ensure_ready()

        def _expand() -> list[GraphPath]:
            seeds = [normalize_entity(n) for n in names]
            seeds = [s for s in seeds if s and self._graph.has_node(s)]
            if not seeds:
                return []

            paths: list[GraphPath] = []
            visited: set[tuple[str, ...]] = set()
            queue: deque[tuple[list[str], list[dict[str, Any]]]] = deque(
                ([seed], []) for seed in seeds
            )

            while queue and len(paths) < limit:
                entity_chain, relation_chain = queue.popleft()
                current = entity_chain[-1]
                if len(entity_chain) - 1 >= depth:
                    continue
                for _, neighbor, data in self._graph.out_edges(current, data=True):
                    self._maybe_extend(entity_chain, relation_chain, neighbor, data, "out", visited, paths, queue)
                for neighbor, _, data in self._graph.in_edges(current, data=True):
                    self._maybe_extend(entity_chain, relation_chain, neighbor, data, "in", visited, paths, queue)

            paths.sort(key=lambda p: p.score, reverse=True)
            return paths[:limit]

        return await asyncio.to_thread(_expand)

    def _maybe_extend(
        self,
        entity_chain: list[str],
        relation_chain: list[dict[str, Any]],
        neighbor: str,
        data: dict[str, Any],
        direction: str,
        visited: set[tuple[str, ...]],
        paths: list[GraphPath],
        queue: deque,
    ) -> None:
        if neighbor in entity_chain:
            return
        new_entities = entity_chain + [neighbor]
        signature = tuple(new_entities)
        if signature in visited:
            return
        visited.add(signature)
        edge = {
            "type": data.get("type", "RELATED_TO"),
            "confidence": data.get("confidence", 0.5),
            "chunk_id": data.get("chunk_id", ""),
            "document_id": data.get("document_id", ""),
            "context": data.get("context", ""),
            "direction": direction,
        }
        new_relations = relation_chain + [edge]
        # Confidence decays with hop count so 1-hop facts outrank 3-hop chains.
        score = data.get("confidence", 0.5) * (0.75 ** (len(new_relations) - 1))
        display = [self._graph.nodes[n].get("name", n) for n in new_entities]
        paths.append(GraphPath(entities=display, relations=new_relations, score=round(score, 4)))
        queue.append((new_entities, new_relations))

    async def paths_between(self, source: str, target: str, max_depth: int = 4) -> list[GraphPath]:
        await self.ensure_ready()

        def _paths() -> list[GraphPath]:
            import networkx as nx  # noqa: PLC0415

            src, dst = normalize_entity(source), normalize_entity(target)
            if not self._graph.has_node(src) or not self._graph.has_node(dst):
                return []
            undirected = self._graph.to_undirected(as_view=False)
            try:
                raw = list(nx.all_simple_paths(undirected, src, dst, cutoff=max_depth))
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return []
            out: list[GraphPath] = []
            for node_path in raw[:10]:
                relations: list[dict[str, Any]] = []
                for a, b in zip(node_path, node_path[1:]):
                    data = self._graph.get_edge_data(a, b) or self._graph.get_edge_data(b, a) or {}
                    best = max(data.values(), key=lambda d: d.get("confidence", 0.0)) if data else {}
                    relations.append(
                        {
                            "type": best.get("type", "RELATED_TO"),
                            "confidence": best.get("confidence", 0.4),
                            "chunk_id": best.get("chunk_id", ""),
                            "document_id": best.get("document_id", ""),
                            "context": best.get("context", ""),
                        }
                    )
                score = round(0.9 ** (len(relations) - 1), 4)
                display = [self._graph.nodes[n].get("name", n) for n in node_path]
                out.append(GraphPath(entities=display, relations=relations, score=score))
            out.sort(key=lambda p: p.score, reverse=True)
            return out

        return await asyncio.to_thread(_paths)

    async def export(self, limit: int = 300, document_id: str | None = None) -> dict[str, Any]:
        await self.ensure_ready()

        def _export() -> dict[str, Any]:
            nodes_data = list(self._graph.nodes(data=True))
            if document_id:
                nodes_data = [
                    (n, d) for n, d in nodes_data if document_id in (d.get("documents") or [])
                ]
            nodes_data.sort(key=lambda item: self._graph.degree(item[0]), reverse=True)
            selected = nodes_data[:limit]
            keys = {n for n, _ in selected}
            nodes = [
                {
                    "id": n,
                    "name": d.get("name", n),
                    "type": d.get("type", "CONCEPT"),
                    "description": d.get("description", ""),
                    "documents": d.get("documents", []),
                    "chunk_ids": d.get("chunk_ids", []),
                    "mentions": d.get("mentions", 1),
                    "degree": self._graph.degree(n),
                }
                for n, d in selected
            ]
            edges = [
                {
                    "id": f"{u}->{v}-{i}",
                    "source": u,
                    "target": v,
                    "type": d.get("type", "RELATED_TO"),
                    "confidence": d.get("confidence", 0.5),
                    "document_id": d.get("document_id", ""),
                    "chunk_id": d.get("chunk_id", ""),
                    "context": d.get("context", ""),
                }
                for i, (u, v, d) in enumerate(self._graph.edges(data=True))
                if u in keys and v in keys
            ]
            return {"nodes": nodes, "edges": edges, "truncated": len(nodes_data) > limit}

        return await asyncio.to_thread(_export)

    async def stats(self) -> dict[str, Any]:
        await self.ensure_ready()

        def _stats() -> dict[str, Any]:
            type_counts: dict[str, int] = defaultdict(int)
            for _, data in self._graph.nodes(data=True):
                type_counts[data.get("type", "CONCEPT")] += 1
            relation_counts: dict[str, int] = defaultdict(int)
            for _, _, data in self._graph.edges(data=True):
                relation_counts[data.get("type", "RELATED_TO")] += 1
            top = sorted(self._graph.degree(), key=lambda x: x[1], reverse=True)[:10]
            return {
                "backend": self.backend,
                "entities": self._graph.number_of_nodes(),
                "relations": self._graph.number_of_edges(),
                "entity_types": dict(sorted(type_counts.items(), key=lambda x: -x[1])),
                "relation_types": dict(sorted(relation_counts.items(), key=lambda x: -x[1])),
                "top_entities": [
                    {"name": self._graph.nodes[n].get("name", n), "degree": deg} for n, deg in top
                ],
            }

        return await asyncio.to_thread(_stats)

    async def health(self) -> dict[str, Any]:
        try:
            stats = await self.stats()
            return {
                "store": "graph",
                "backend": self.backend,
                "healthy": True,
                "status_text": "healthy",
                "mode": "embedded",
                "entities": stats["entities"],
                "relations": stats["relations"],
                "note": (
                    self.config_warning
                    or "Neo4j is not configured; using the embedded NetworkX graph store."
                ),
                "misconfigured_neo4j": bool(self.config_warning),
            }
        except Exception as exc:  # pragma: no cover
            return {"store": "graph", "backend": self.backend, "healthy": False, "error": str(exc)[:200]}


# ===========================================================================
# Neo4j backend
# ===========================================================================
class Neo4jGraphStore(GraphStore):
    backend = "neo4j"

    def __init__(self) -> None:
        settings = get_settings()
        self._uri = settings.neo4j_uri
        self._auth = (settings.neo4j_user, settings.neo4j_password)
        self._database = settings.neo4j_database
        self._driver: Any = None

    async def ensure_ready(self) -> None:
        if self._driver is not None:
            return
        from neo4j import AsyncGraphDatabase  # noqa: PLC0415

        self._driver = AsyncGraphDatabase.driver(self._uri, auth=self._auth)
        await self._driver.verify_connectivity()
        async with self._driver.session(database=self._database) as session:
            await session.run(
                "CREATE CONSTRAINT ragx_entity_key IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE e.key IS UNIQUE"
            )
            await session.run(
                "CREATE INDEX ragx_entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)"
            )
        log.info("graph.neo4j_ready", uri=self._uri.split("@")[-1])

    async def _run(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        await self.ensure_ready()
        async with self._driver.session(database=self._database) as session:
            result = await session.run(cypher, **params)
            return [record.data() async for record in result]

    async def upsert(self, entities: list[GraphEntity], relations: list[GraphRelation]) -> dict[str, int]:
        if entities:
            # Plain Cypher only -- no APOC dependency. List de-duplication uses
            # a comprehension over the concatenated lists.
            await self._run(
                """
                UNWIND $rows AS row
                MERGE (e:Entity {key: row.key})
                ON CREATE SET e.name = row.name,
                              e.type = row.type,
                              e.description = row.description,
                              e.salience = row.salience,
                              e.mentions = 1,
                              e.documents = row.documents,
                              e.chunk_ids = row.chunk_ids
                ON MATCH  SET e.mentions = coalesce(e.mentions, 0) + 1,
                              e.salience = CASE WHEN row.salience > coalesce(e.salience, 0.0)
                                                THEN row.salience ELSE e.salience END,
                              e.description = CASE WHEN size(row.description) > size(coalesce(e.description, ''))
                                                   THEN row.description ELSE e.description END,
                              e.type = CASE WHEN coalesce(e.type, 'CONCEPT') = 'CONCEPT'
                                            THEN row.type ELSE e.type END
                WITH e, row
                UNWIND (coalesce(e.documents, []) + row.documents) AS doc
                WITH e, row, collect(DISTINCT doc) AS merged_docs
                UNWIND (coalesce(e.chunk_ids, []) + row.chunk_ids) AS chunk
                WITH e, merged_docs, collect(DISTINCT chunk) AS merged_chunks
                SET e.documents = [d IN merged_docs WHERE d <> ''],
                    e.chunk_ids = [c IN merged_chunks WHERE c <> ''][0..50]
                """,
                rows=[
                    {
                        "key": e.normalized,
                        "name": e.name,
                        "type": e.entity_type or "CONCEPT",
                        "description": e.description or "",
                        "salience": float(e.salience),
                        "documents": [e.document_id] if e.document_id else [],
                        "chunk_ids": list(e.chunk_ids)[:50],
                    }
                    for e in entities
                    if e.normalized
                ],
            )

        added = 0
        if relations:
            rows = [
                {
                    "source": normalize_entity(r.source),
                    "target": normalize_entity(r.target),
                    "type": (r.relation_type or "RELATED_TO").upper(),
                    "confidence": float(r.confidence),
                    "document_id": r.document_id,
                    "chunk_id": r.chunk_id,
                    "context": (r.context or "")[:400],
                }
                for r in relations
                if normalize_entity(r.source) and normalize_entity(r.target)
                and normalize_entity(r.source) != normalize_entity(r.target)
            ]
            if rows:
                # A single generic edge label with a ``type`` property keeps the
                # schema stable while remaining fully queryable.
                await self._run(
                    """
                    UNWIND $rows AS row
                    MATCH (a:Entity {key: row.source})
                    MATCH (b:Entity {key: row.target})
                    MERGE (a)-[r:REL {type: row.type}]->(b)
                    ON CREATE SET r.confidence = row.confidence, r.document_id = row.document_id,
                                  r.chunk_id = row.chunk_id, r.context = row.context, r.weight = 1
                    ON MATCH  SET r.weight = coalesce(r.weight,1) + 1,
                                  r.confidence = CASE WHEN row.confidence > r.confidence
                                                      THEN row.confidence ELSE r.confidence END
                    """,
                    rows=rows,
                )
                added = len(rows)
        return {"entities": len(entities), "relations": added}

    async def delete_document(self, document_id: str) -> None:
        await self._run("MATCH ()-[r:REL {document_id: $doc}]-() DELETE r", doc=document_id)
        await self._run(
            """
            MATCH (e:Entity)
            SET e.documents = [d IN coalesce(e.documents, []) WHERE d <> $doc]
            WITH e WHERE size(coalesce(e.documents, [])) = 0
            DETACH DELETE e
            """,
            doc=document_id,
        )

    async def search_entities(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        rows = await self._run(
            """
            MATCH (e:Entity)
            WHERE toLower(e.name) CONTAINS toLower($q) OR e.key CONTAINS $key
            RETURN e.key AS id, e.name AS name, e.type AS type,
                   e.description AS description, e.documents AS documents,
                   e.chunk_ids AS chunk_ids, coalesce(e.mentions,1) AS mentions,
                   size([(e)--() | 1]) AS degree
            ORDER BY degree DESC LIMIT $limit
            """,
            q=query,
            key=normalize_entity(query),
            limit=limit,
        )
        for row in rows:
            row["score"] = 1.0
        return rows

    async def neighborhood(self, names: list[str], depth: int = 2, limit: int = 60) -> list[GraphPath]:
        keys = [normalize_entity(n) for n in names if normalize_entity(n)]
        if not keys:
            return []
        depth = max(1, min(depth, 4))
        rows = await self._run(
            f"""
            MATCH path = (a:Entity)-[rels:REL*1..{depth}]-(b:Entity)
            WHERE a.key IN $keys
            RETURN [n IN nodes(path) | n.name] AS entities,
                   [r IN relationships(path) | {{type: r.type, confidence: r.confidence,
                                                 chunk_id: r.chunk_id, document_id: r.document_id,
                                                 context: r.context}}] AS relations,
                   length(path) AS hops
            LIMIT $limit
            """,
            keys=keys,
            limit=limit,
        )
        paths: list[GraphPath] = []
        for row in rows:
            relations = row["relations"]
            base = min((r.get("confidence") or 0.5) for r in relations) if relations else 0.5
            paths.append(
                GraphPath(
                    entities=row["entities"],
                    relations=relations,
                    score=round(base * (0.75 ** max(0, row["hops"] - 1)), 4),
                )
            )
        paths.sort(key=lambda p: p.score, reverse=True)
        return paths

    async def paths_between(self, source: str, target: str, max_depth: int = 4) -> list[GraphPath]:
        max_depth = max(1, min(max_depth, 5))
        rows = await self._run(
            f"""
            MATCH path = shortestPath((a:Entity {{key: $src}})-[:REL*1..{max_depth}]-(b:Entity {{key: $dst}}))
            RETURN [n IN nodes(path) | n.name] AS entities,
                   [r IN relationships(path) | {{type: r.type, confidence: r.confidence,
                                                 chunk_id: r.chunk_id, document_id: r.document_id,
                                                 context: r.context}}] AS relations
            LIMIT 10
            """,
            src=normalize_entity(source),
            dst=normalize_entity(target),
        )
        return [
            GraphPath(
                entities=row["entities"],
                relations=row["relations"],
                score=round(0.9 ** max(0, len(row["relations"]) - 1), 4),
            )
            for row in rows
        ]

    async def export(self, limit: int = 300, document_id: str | None = None) -> dict[str, Any]:
        node_rows = await self._run(
            """
            MATCH (e:Entity)
            WHERE $doc IS NULL OR $doc IN coalesce(e.documents, [])
            RETURN e.key AS id, e.name AS name, e.type AS type, e.description AS description,
                   coalesce(e.documents, []) AS documents, coalesce(e.chunk_ids, []) AS chunk_ids,
                   coalesce(e.mentions,1) AS mentions, size([(e)--() | 1]) AS degree
            ORDER BY degree DESC LIMIT $limit
            """,
            doc=document_id,
            limit=limit,
        )
        keys = [row["id"] for row in node_rows]
        edge_rows = await self._run(
            """
            MATCH (a:Entity)-[r:REL]->(b:Entity)
            WHERE a.key IN $keys AND b.key IN $keys
            RETURN a.key AS source, b.key AS target, r.type AS type,
                   r.confidence AS confidence, r.document_id AS document_id,
                   r.chunk_id AS chunk_id, r.context AS context
            """,
            keys=keys,
        )
        for i, edge in enumerate(edge_rows):
            edge["id"] = f"{edge['source']}->{edge['target']}-{i}"
        return {"nodes": node_rows, "edges": edge_rows, "truncated": len(node_rows) >= limit}

    async def stats(self) -> dict[str, Any]:
        counts = await self._run(
            "MATCH (e:Entity) RETURN count(e) AS entities"
        )
        rel_counts = await self._run("MATCH ()-[r:REL]->() RETURN count(r) AS relations")
        types = await self._run(
            "MATCH (e:Entity) RETURN e.type AS type, count(*) AS n ORDER BY n DESC LIMIT 20"
        )
        rel_types = await self._run(
            "MATCH ()-[r:REL]->() RETURN r.type AS type, count(*) AS n ORDER BY n DESC LIMIT 20"
        )
        top = await self._run(
            "MATCH (e:Entity) RETURN e.name AS name, size([(e)--() | 1]) AS degree "
            "ORDER BY degree DESC LIMIT 10"
        )
        return {
            "backend": self.backend,
            "entities": counts[0]["entities"] if counts else 0,
            "relations": rel_counts[0]["relations"] if rel_counts else 0,
            "entity_types": {r["type"]: r["n"] for r in types},
            "relation_types": {r["type"]: r["n"] for r in rel_types},
            "top_entities": top,
        }

    async def health(self) -> dict[str, Any]:
        try:
            stats = await self.stats()
            return {
                "store": "graph",
                "backend": self.backend,
                "healthy": True,
                "status_text": "healthy",
                "mode": "server",
                "entities": stats["entities"],
                "relations": stats["relations"],
            }
        except Exception as exc:
            return {
                "store": "graph",
                "backend": self.backend,
                "healthy": False,
                "status_text": "unhealthy",
                "error": str(exc)[:200],
            }

    async def close(self) -> None:
        if self._driver is not None:
            await self._driver.close()
            self._driver = None


# ===========================================================================
_store: GraphStore | None = None

# Schemes the Neo4j driver accepts. Anything else cannot connect.
NEO4J_SCHEMES = ("bolt://", "bolt+s://", "bolt+ssc://", "neo4j://", "neo4j+s://", "neo4j+ssc://")

# Neo4j Aura shows an instance ID (e.g. "b7f9149b") next to the connection URI.
# Pasting the ID alone is a common mistake, so it gets a targeted message.
_AURA_ID = re.compile(r"^[0-9a-f]{8}$", re.IGNORECASE)


def validate_neo4j_uri(uri: str) -> str | None:
    """Return a human-readable problem with ``uri``, or ``None`` if it is usable."""
    value = (uri or "").strip()
    if not value:
        return "NEO4J_URI is empty."
    if value.startswith(NEO4J_SCHEMES):
        return None
    if _AURA_ID.match(value):
        return (
            f"NEO4J_URI is set to '{value}', which looks like a Neo4j Aura *instance ID*, "
            f"not a connection URI. Use the full URI instead:  "
            f"neo4j+s://{value}.databases.neo4j.io"
        )
    return (
        f"NEO4J_URI='{value}' has no supported scheme. It must start with one of: "
        f"{', '.join(s.rstrip('://') for s in NEO4J_SCHEMES)}. "
        f"Example: neo4j+s://<id>.databases.neo4j.io or bolt://localhost:7687"
    )


def get_graph_store() -> GraphStore:
    """Neo4j when correctly configured, otherwise the embedded NetworkX store.

    A malformed ``NEO4J_URI`` degrades to the embedded store with a loud warning
    rather than selecting a Neo4j backend that fails on every call — which would
    silently disable Graph RAG while reporting the graph as "configured".
    """
    global _store
    if _store is not None:
        return _store

    settings = get_settings()
    if settings.neo4j_uri:
        problem = validate_neo4j_uri(settings.neo4j_uri)
        if problem is None:
            _store = Neo4jGraphStore()
            log.info("graph.backend_selected", backend="neo4j")
            return _store
        log.warning(
            "graph.neo4j_uri_invalid_using_embedded_store",
            detail=problem,
            action="Fix NEO4J_URI, or leave it empty to use the embedded store deliberately.",
        )
        _store = NetworkXGraphStore()
        _store.config_warning = problem
        return _store

    _store = NetworkXGraphStore()
    log.info("graph.backend_selected", backend="networkx", reason="NEO4J_URI is not set")
    return _store


async def close_graph_store() -> None:
    global _store
    if _store is not None:
        await _store.close()
    _store = None


__all__ = [
    "GraphStore",
    "Neo4jGraphStore",
    "NetworkXGraphStore",
    "GraphEntity",
    "GraphRelation",
    "GraphPath",
    "get_graph_store",
    "close_graph_store",
    "normalize_entity",
]
