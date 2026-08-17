"""RAGX FastAPI application entrypoint."""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from app.api.v1.router import api_router
from app.core.config import get_settings
from app.core.errors import register_exception_handlers
from app.core.logging import configure_logging, get_logger, request_id_var
from app.db.init_db import init_db
from app.db.session import dispose_engine
from app.indexing.bm25_index import get_bm25_index
from app.indexing.graph_store import close_graph_store, get_graph_store
from app.indexing.vector_store import close_vector_store, get_vector_store
from app.llm.embeddings import get_embedding_provider
from app.llm.gateway import get_gateway
from app.services.health_service import VERSION

configure_logging()
log = get_logger("ragx.app")

DESCRIPTION = """
**RAGX — Adaptive Multi-Strategy Research Intelligence System**

RAGX analyses each question, selects the retrieval strategies that can actually
answer it, verifies the resulting answer against its evidence, and reports why it
made every decision.

**Eight retrieval strategies**
`naive` · `hybrid` · `hyde` · `multimodal` · `corrective` · `graph` · `adaptive` · `agentic`

The Adaptive Router composes them per query — it does not run all of them.
`POST /api/v1/query/analyze` shows the routing decision without spending a
retrieval or generation call.

**Cloud LLMs only.** Gemini and Groq are reached through an internal gateway;
no local model runtime is supported. API keys live in the backend environment
and are never returned by any endpoint.

**Grounding.** Every answer carries citations resolvable to a document, page,
section and figure/table. When the evidence is insufficient, RAGX says so
instead of answering.
"""

TAGS_METADATA = [
    {"name": "system", "description": "Health, analytics, settings and the strategy catalogue."},
    {"name": "documents", "description": "Upload, processing status, inspection and deletion."},
    {"name": "query", "description": "Query execution, routing inspection and history."},
    {"name": "evidence", "description": "Citation drill-down: full passages and figure images."},
    {"name": "graph", "description": "Knowledge-graph traversal and export."},
    {"name": "evaluation", "description": "Benchmarks, experiment runs and strategy comparison."},
]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    log.info("app.starting", environment=settings.environment, version=VERSION)

    await init_db()

    # Warm the indexes so the first query does not pay initialisation cost.
    try:
        await get_vector_store().ensure_ready(dimension=get_embedding_provider().dimension)
    except Exception as exc:
        log.error("app.vector_store_unavailable", error=str(exc)[:200])
    try:
        await get_bm25_index().ensure_loaded()
        await get_graph_store().ensure_ready()
    except Exception as exc:
        log.warning("app.index_warmup_partial", error=str(exc)[:200])

    gateway = get_gateway()
    embedder = get_embedding_provider()
    log.info(
        "app.ready",
        llm_providers=gateway.configured_providers or ["none"],
        embedding_provider=embedder.name,
        embedding_production_ready=embedder.production_ready,
        graph_backend=get_graph_store().backend,
        database=settings.database_flavour,
    )
    if not gateway.any_configured:
        log.warning(
            "app.no_llm_configured",
            detail="Set GEMINI_API_KEY and/or GROQ_API_KEY. Retrieval works; generation does not.",
        )

    yield

    log.info("app.shutting_down")
    await close_vector_store()
    await close_graph_store()
    await dispose_engine()


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="RAGX API",
        description=DESCRIPTION,
        version=VERSION,
        openapi_tags=TAGS_METADATA,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # CORS is restricted to the configured origins; credentials are not used
    # because the frontend never holds a provider key.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "X-Ragx-Key", "Accept"],
        expose_headers=["X-Request-ID", "X-Process-Time-Ms"],
        max_age=600,
    )
    app.add_middleware(GZipMiddleware, minimum_size=1024)

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"
        if not request.url.path.startswith(("/docs", "/openapi", "/redoc")):
            log.info(
                "http.request",
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                duration_ms=round(elapsed_ms, 1),
            )
        return response

    register_exception_handlers(app)
    app.include_router(api_router, prefix=settings.api_v1_prefix)

    @app.get("/", tags=["system"], summary="Service banner")
    async def root() -> dict:
        return {
            "name": "RAGX",
            "tagline": "One query. Multiple retrieval strategies. Verified answers.",
            "version": VERSION,
            "docs": "/docs",
            "api": settings.api_v1_prefix,
            "strategies": [
                "naive", "hybrid", "hyde", "multimodal",
                "corrective", "graph", "adaptive", "agentic",
            ],
            "llm": "cloud-only (Gemini, Groq)",
        }

    return app


app = create_app()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
