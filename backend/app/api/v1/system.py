"""Health, analytics, settings and strategy-catalogue endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Query

from app.api.deps import ApiKeyDep, SessionDep
from app.core.cache import ALL_CACHES
from app.core.config import get_settings, reload_settings
from app.llm.gateway import reset_gateway
from app.schemas.common import Acknowledgement, HealthResponse
from app.schemas.evaluation import AnalyticsResponse
from app.schemas.settings import (
    LLMStatusResponse,
    RetrievalSettingsUpdate,
    SettingsResponse,
    StrategyInfo,
)
from app.services.analytics_service import get_analytics_service
from app.services.health_service import get_health_service
from app.retrieval.registry import describe_strategies

router = APIRouter(tags=["system"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System health",
    description=(
        "Reports the status of every component. Pass `probe=true` to make a live request to each "
        "configured LLM provider (this costs a few tokens); the default is a configuration-only "
        "check, which is free."
    ),
)
async def health(probe: bool = Query(False, description="Make a live LLM provider request.")) -> HealthResponse:
    return HealthResponse(**await get_health_service().health(probe_llm=probe))


@router.get(
    "/llm/status",
    response_model=LLMStatusResponse,
    summary="Cloud LLM provider status",
    description=(
        "Which cloud providers are configured and which models are in use. API keys are never "
        "returned. RAGX supports cloud providers only; no local model runtime is available."
    ),
)
async def llm_status(probe: bool = Query(False)) -> LLMStatusResponse:
    from app.llm.gateway import get_gateway  # noqa: PLC0415

    return LLMStatusResponse(**await get_gateway().health(probe=probe))


@router.get("/analytics", response_model=AnalyticsResponse, summary="Dashboard analytics")
async def analytics(session: SessionDep, days: int = Query(30, ge=1, le=365)) -> AnalyticsResponse:
    return AnalyticsResponse(**await get_analytics_service().dashboard(session, days=days))


@router.get(
    "/settings",
    response_model=SettingsResponse,
    summary="Safe runtime configuration",
    description="Non-sensitive configuration and component status for the Settings page.",
)
async def read_settings() -> SettingsResponse:
    return SettingsResponse(**await get_health_service().settings_payload())


@router.patch(
    "/settings",
    response_model=SettingsResponse,
    summary="Update runtime settings",
    description=(
        "Updates tunable retrieval, verification and provider-selection settings for this "
        "process. Credentials cannot be set here -- they come from the environment only. "
        "Changes are not persisted across restarts; put permanent values in `.env`."
    ),
)
async def update_settings(_: ApiKeyDep, payload: RetrievalSettingsUpdate) -> SettingsResponse:
    settings = get_settings()
    for field, value in payload.model_dump(exclude_none=True).items():
        if hasattr(settings, field):
            setattr(settings, field, value)
    # Provider preference changes require the gateway to re-resolve its routing.
    reset_gateway()
    return SettingsResponse(**await get_health_service().settings_payload())


@router.get("/strategies", response_model=list[StrategyInfo], summary="Available RAG strategies")
async def strategies() -> list[StrategyInfo]:
    return [StrategyInfo(**s) for s in describe_strategies()]


@router.post("/cache/clear", response_model=Acknowledgement, summary="Clear in-process caches")
async def clear_cache(_: ApiKeyDep) -> Acknowledgement:
    for cache in ALL_CACHES.values():
        await cache.clear()
    return Acknowledgement(ok=True, message="Embedding, analysis and answer caches were cleared.")


@router.post(
    "/settings/reload",
    response_model=SettingsResponse,
    summary="Re-read configuration from the environment",
)
async def reload_configuration(_: ApiKeyDep) -> SettingsResponse:
    reload_settings()
    reset_gateway()
    return SettingsResponse(**await get_health_service().settings_payload())


__all__ = ["router"]
