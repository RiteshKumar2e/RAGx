"""System health and safe configuration reporting.

The payloads here are what the Settings page and the LLM provider indicator
render. They deliberately expose *whether* a credential is configured, never the
credential itself.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import text

from app.core.cache import ALL_CACHES
from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.session import get_engine
from app.indexing.bm25_index import get_bm25_index
from app.indexing.graph_store import get_graph_store
from app.indexing.vector_store import get_vector_store
from app.ingestion.ocr import tesseract_available
from app.llm.embeddings import get_embedding_provider
from app.llm.gateway import get_gateway
from app.retrieval.registry import describe_strategies
from app.storage import get_object_store

log = get_logger("ragx.health")

VERSION = "1.0.0"


async def _database_health() -> dict[str, Any]:
    settings = get_settings()
    flavour = settings.database_flavour
    try:
        engine = get_engine()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return {"engine": flavour, "healthy": True, "status_text": "healthy"}
    except Exception as exc:
        # The Turso driver message is long and actionable; keep more of it.
        limit = 600 if flavour == "turso" else 200
        return {
            "engine": flavour,
            "healthy": False,
            "status_text": "unhealthy",
            "error": str(exc)[:limit],
        }


class HealthService:
    async def health(self, probe_llm: bool = False) -> dict[str, Any]:
        settings = get_settings()
        database, vector, graph, bm25, objects, llm = await asyncio.gather(
            _database_health(),
            get_vector_store().health(),
            get_graph_store().health(),
            get_bm25_index().health(),
            get_object_store().health(),
            get_gateway().health(probe=probe_llm),
            return_exceptions=True,
        )

        def _safe(value: Any, name: str) -> dict[str, Any]:
            if isinstance(value, Exception):
                return {"healthy": False, "status_text": "unhealthy", "error": str(value)[:200], "component": name}
            return value

        database = _safe(database, "database")
        vector = _safe(vector, "vector_store")
        graph = _safe(graph, "graph_store")
        bm25 = _safe(bm25, "bm25_index")
        objects = _safe(objects, "object_storage")
        llm = _safe(llm, "llm")

        components = [
            {"name": "database", "status": database.get("status_text", "unknown"), "healthy": database.get("healthy"), "detail": database},
            {"name": "vector_store", "status": vector.get("status_text", "unknown"), "healthy": vector.get("healthy"), "detail": vector},
            {"name": "graph_store", "status": graph.get("status_text", "unknown"), "healthy": graph.get("healthy"), "detail": graph},
            {"name": "bm25_index", "status": bm25.get("status_text", "unknown"), "healthy": bm25.get("healthy"), "detail": bm25},
            {"name": "object_storage", "status": objects.get("status", objects.get("status_text", "unknown")), "healthy": objects.get("status") == "healthy" if "status" in objects else objects.get("healthy"), "detail": objects},
            {
                "name": "llm_providers",
                "status": "configured" if llm.get("any_configured") else "not_configured",
                "healthy": llm.get("any_configured"),
                "detail": llm,
            },
        ]

        warnings = self._warnings(llm)
        critical = [c for c in components if c["name"] in {"database", "vector_store"} and c["healthy"] is False]
        status = "unhealthy" if critical else ("degraded" if warnings else "healthy")

        return {
            "status": status,
            "version": VERSION,
            "environment": settings.environment,
            "timestamp": datetime.now(timezone.utc),
            "components": components,
            "warnings": warnings,
        }

    @staticmethod
    def _warnings(llm: dict[str, Any]) -> list[str]:
        settings = get_settings()
        warnings: list[str] = []
        if not llm.get("any_configured"):
            warnings.append(
                "No cloud LLM provider is configured. Set GEMINI_API_KEY and/or GROQ_API_KEY. "
                "Retrieval works without them, but answers, query analysis and verification do not."
            )
        embedder = get_embedding_provider()
        if not embedder.production_ready:
            warnings.append(
                "The development hashing embedder is active. It matches text lexically, not "
                "semantically, so retrieval quality and any benchmark numbers are not representative. "
                "Set EMBEDDING_PROVIDER=gemini with a GEMINI_API_KEY for real embeddings."
            )
        if settings.database_flavour == "sqlite":
            warnings.append(
                "Running on a local SQLite file. This is fine for development; set "
                "TURSO_DATABASE_URL and TURSO_AUTH_TOKEN for a hosted libSQL database "
                "in production."
            )
        if settings.neo4j_uri:
            from app.indexing.graph_store import validate_neo4j_uri  # noqa: PLC0415

            problem = validate_neo4j_uri(settings.neo4j_uri)
            if problem:
                warnings.append(
                    f"{problem} Graph RAG is running on the embedded NetworkX store until this "
                    "is corrected."
                )
        else:
            warnings.append(
                "NEO4J_URI is not set, so Graph RAG uses the embedded NetworkX store. "
                "Traversal works; Neo4j adds scale and Cypher access."
            )
        if settings.enable_ocr and not tesseract_available():
            warnings.append(
                "OCR is enabled but the Tesseract binary was not found. Scanned pages fall back "
                "to Gemini vision, which requires GEMINI_API_KEY."
            )
        return warnings

    async def settings_payload(self) -> dict[str, Any]:
        settings = get_settings()
        gateway = get_gateway()
        embedder = get_embedding_provider()

        llm, vector, graph, bm25, objects, database = await asyncio.gather(
            gateway.health(probe=False),
            get_vector_store().health(),
            get_graph_store().health(),
            get_bm25_index().health(),
            get_object_store().health(),
            _database_health(),
            return_exceptions=True,
        )

        def _safe(value: Any) -> dict[str, Any]:
            return {"error": str(value)[:200], "healthy": False} if isinstance(value, Exception) else value

        embedding_health = await embedder.health()

        return {
            "app_name": settings.app_name,
            "version": VERSION,
            "environment": settings.environment,
            "llm": _safe(llm),
            "embeddings": {
                "provider": embedding_health.get("provider", embedder.name),
                "model": embedding_health.get("model"),
                "dimension": embedding_health.get("dimension", embedder.dimension),
                "production_ready": embedding_health.get("production_ready", True),
                "healthy": embedding_health.get("healthy"),
                "status_text": embedding_health.get("status_text", "unknown"),
                "warning": embedding_health.get("warning"),
            },
            "storage": {
                "vector_store": _safe(vector),
                "graph_store": _safe(graph),
                "bm25_index": _safe(bm25),
                "relational": _safe(database),
                "object_storage": _safe(objects),
            },
            "retrieval": {
                "default_top_k": settings.default_top_k,
                "candidate_pool_size": settings.candidate_pool_size,
                "rerank_enabled": settings.rerank_enabled,
                "min_relevance_score": settings.min_relevance_score,
                "corrective_relevance_floor": settings.corrective_relevance_floor,
                "corrective_max_rounds": settings.corrective_max_rounds,
                "agentic_max_steps": settings.agentic_max_steps,
            },
            "verification": {
                "enabled": settings.verification_enabled,
                "min_claim_support_score": settings.min_claim_support_score,
                "insufficient_evidence_threshold": settings.insufficient_evidence_threshold,
            },
            "ingestion": {
                "max_upload_mb": settings.max_upload_mb,
                "allowed_extensions": sorted(settings.allowed_extension_set),
                "ocr_enabled": settings.enable_ocr,
                "ocr_available": tesseract_available(),
                "entity_extraction": settings.extract_entities,
                "chunk_target_tokens": settings.chunk_target_tokens,
                "chunk_overlap_tokens": settings.chunk_overlap_tokens,
            },
            "strategies": describe_strategies(),
            "cache": {name: cache.stats() for name, cache in ALL_CACHES.items()},
            "warnings": self._warnings(_safe(llm)),
        }


_service: HealthService | None = None


def get_health_service() -> HealthService:
    global _service
    if _service is None:
        _service = HealthService()
    return _service


__all__ = ["HealthService", "get_health_service", "VERSION"]
