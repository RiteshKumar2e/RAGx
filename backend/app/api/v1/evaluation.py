"""Evaluation endpoints: benchmark inspection, experiment runs and comparison."""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Query

from app.api.deps import ApiKeyDep, SessionDep
from app.schemas.common import Acknowledgement
from app.schemas.evaluation import (
    BenchmarkInfo,
    EvaluationComparison,
    EvaluationResultItem,
    EvaluationRunDetail,
    EvaluationRunRequest,
    EvaluationRunSummary,
    EvaluationStartResponse,
)
from app.services.evaluation_service import get_evaluation_service

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


@router.get(
    "/benchmark",
    response_model=BenchmarkInfo,
    summary="Inspect the benchmark dataset",
    description=(
        "Returns the benchmark questions grouped by category. `has_relevance_labels` reports "
        "whether manual labels ship with the dataset; when false, retrieval metrics are computed "
        "from pooled LLM judgements at run time."
    ),
)
async def get_benchmark(dataset: str = Query("ragx_benchmark")) -> BenchmarkInfo:
    return BenchmarkInfo(**get_evaluation_service().benchmark(dataset))


@router.get("/datasets", response_model=list[str], summary="List available datasets")
async def list_datasets() -> list[str]:
    return get_evaluation_service().datasets()


@router.post(
    "/run",
    response_model=EvaluationStartResponse,
    summary="Run a benchmark experiment",
    description=(
        "Starts one run per requested strategy over the same questions, in the background. "
        "Conditions: a named strategy pins that strategy (router bypassed); `adaptive` uses the "
        "router with verification disabled; `ragx` is the full pipeline. Results are written to "
        "the database as each run completes -- nothing is displayed until then."
    ),
)
async def run_evaluation(
    session: SessionDep,
    _: ApiKeyDep,
    background: BackgroundTasks,
    request: EvaluationRunRequest,
) -> EvaluationStartResponse:
    service = get_evaluation_service()
    result = await service.create_runs(session, request)
    run_ids_map = result.pop("_run_ids_map")
    background.add_task(service.execute_runs, run_ids_map, request)
    return EvaluationStartResponse(**result)


@router.get("/runs", response_model=list[EvaluationRunSummary], summary="List evaluation runs")
async def list_runs(
    session: SessionDep,
    limit: int = Query(50, ge=1, le=200),
    strategy: str | None = None,
) -> list[EvaluationRunSummary]:
    runs = await get_evaluation_service().list_runs(session, limit=limit, strategy=strategy)
    return [EvaluationRunSummary.model_validate(r) for r in runs]


@router.get(
    "/comparison",
    response_model=EvaluationComparison,
    summary="Compare strategies",
    description=(
        "The latest completed run for each strategy, side by side, with the best performer "
        "flagged per metric. Returns `has_data: false` when no run has completed -- the "
        "dashboard shows an empty state rather than placeholder numbers."
    ),
)
async def comparison(session: SessionDep) -> EvaluationComparison:
    result = await get_evaluation_service().comparison(session)
    return EvaluationComparison(
        runs=[EvaluationRunSummary.model_validate(r) for r in result["runs"]],
        metrics=result["metrics"],
        best_by_metric=result["best_by_metric"],
        generated_at=result["generated_at"],
        has_data=result["has_data"],
        message=result["message"],
    )


@router.get("/runs/{run_id}", response_model=EvaluationRunDetail, summary="Evaluation run detail")
async def get_run(session: SessionDep, run_id: str) -> EvaluationRunDetail:
    run = await get_evaluation_service().get_run(session, run_id)
    detail = EvaluationRunDetail.model_validate(run)
    detail.results = [EvaluationResultItem.model_validate(r) for r in run.results]
    return detail


@router.get(
    "/runs/{run_id}/results",
    response_model=list[EvaluationResultItem],
    summary="Per-question results for a run",
)
async def run_results(
    session: SessionDep, run_id: str, limit: int = Query(200, ge=1, le=500)
) -> list[EvaluationResultItem]:
    results = await get_evaluation_service().results(session, run_id, limit=limit)
    return [EvaluationResultItem.model_validate(r) for r in results]


@router.delete("/runs/{run_id}", response_model=Acknowledgement, summary="Delete an evaluation run")
async def delete_run(session: SessionDep, _: ApiKeyDep, run_id: str) -> Acknowledgement:
    return Acknowledgement(**await get_evaluation_service().delete_run(session, run_id))


__all__ = ["router"]
