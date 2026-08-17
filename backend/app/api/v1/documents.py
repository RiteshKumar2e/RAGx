"""Document upload, knowledge-base management and evidence inspection."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, File, Form, Query, Response, UploadFile

from app.api.deps import ApiKeyDep, SessionDep
from app.core.config import get_settings
from app.core.errors import NotFoundError, RagxError, ValidationError
from app.core.logging import get_logger
from app.schemas.common import Acknowledgement
from app.schemas.document import (
    BulkUploadResponse,
    DocumentDetail,
    DocumentListResponse,
    DocumentSummary,
    KnowledgeBaseStats,
    ReindexResponse,
    UploadResponse,
    WebIngestRequest,
)
from app.services.document_service import get_document_service
from app.storage import get_object_store

log = get_logger("ragx.api.documents")
router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "/upload",
    response_model=BulkUploadResponse,
    summary="Upload one or more documents",
    description=(
        "Accepts PDF, DOCX, TXT, MD, CSV and image files. Each accepted file is validated "
        "(extension, declared MIME type, magic bytes and size), stored in object storage and "
        "queued for the ingestion pipeline. Files that fail validation are reported in "
        "`rejected` while valid files still proceed."
    ),
)
async def upload_documents(
    session: SessionDep,
    _: ApiKeyDep,
    background: BackgroundTasks,
    files: Annotated[list[UploadFile], File(description="Files to ingest.")],
) -> BulkUploadResponse:
    settings = get_settings()
    service = get_document_service()
    uploaded: list[UploadResponse] = []
    rejected: list[dict[str, str]] = []

    for upload in files:
        try:
            data = await upload.read()
            if len(data) > settings.max_upload_bytes:
                raise ValidationError(
                    f"'{upload.filename}' is {len(data) / 1_048_576:.1f} MB; "
                    f"the limit is {settings.max_upload_mb} MB."
                )
            result = await service.upload(session, upload.filename or "upload", upload.content_type, data)
            uploaded.append(UploadResponse(**result))
            if not result["duplicate_of"]:
                background.add_task(service.process_in_background, result["document_id"])
        except RagxError as exc:
            rejected.append({"filename": upload.filename or "unknown", "reason": exc.message})
        except Exception as exc:  # pragma: no cover - defensive
            log.error("upload.failed", filename=upload.filename, error=str(exc))
            rejected.append({"filename": upload.filename or "unknown", "reason": "The file could not be read."})
        finally:
            await upload.close()

    if not uploaded and rejected:
        raise ValidationError("No files could be accepted.", detail=rejected)
    return BulkUploadResponse(uploaded=uploaded, rejected=rejected)


@router.post(
    "/ingest-url",
    response_model=UploadResponse,
    summary="Ingest a web page",
    description=(
        "Fetches an http(s) URL and ingests its readable content. Only text/html and plain-text "
        "responses are accepted, and the response is subject to the same size limit as uploads."
    ),
)
async def ingest_url(
    session: SessionDep,
    _: ApiKeyDep,
    background: BackgroundTasks,
    payload: WebIngestRequest,
) -> UploadResponse:
    import httpx  # noqa: PLC0415

    settings = get_settings()
    service = get_document_service()

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=25.0) as client:
            response = await client.get(payload.url, headers={"User-Agent": "RAGX/1.0 (+research indexer)"})
            response.raise_for_status()
    except Exception as exc:
        raise ValidationError(f"The URL could not be fetched: {str(exc)[:160]}") from exc

    content_type = (response.headers.get("content-type") or "").split(";")[0].strip().lower()
    if content_type not in {"text/html", "application/xhtml+xml", "text/plain", "text/markdown"}:
        raise ValidationError(
            f"Only HTML and plain-text pages can be ingested; this URL returned '{content_type or 'unknown'}'."
        )

    data = response.content
    if len(data) > settings.max_upload_bytes:
        raise ValidationError("The fetched page exceeds the configured size limit.")

    extension = ".html" if "html" in content_type else ".txt"
    host = payload.url.split("//", 1)[-1].split("/", 1)[0]
    filename = (payload.title or host or "web-page").replace("/", "_")[:120] + extension

    # ``ingest-url`` bypasses magic-byte checks by design (the content came from
    # an HTTP response, not a user file), but the extension and size limits still
    # apply and the parser is chosen by extension.
    result = await service.upload(session, filename, "text/plain", data, source_url=payload.url)
    if not result["duplicate_of"]:
        background.add_task(service.process_in_background, result["document_id"])
    return UploadResponse(**result)


@router.get("", response_model=DocumentListResponse, summary="List indexed documents")
async def list_documents(
    session: SessionDep,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: str | None = Query(None, description="Filter by processing status."),
    file_type: str | None = Query(None, description="Filter by extension, e.g. '.pdf'."),
    search: str | None = Query(None, description="Substring match on filename or title."),
) -> DocumentListResponse:
    result = await get_document_service().list_documents(
        session, page=page, page_size=page_size, status=status, file_type=file_type, search=search
    )
    return DocumentListResponse(
        items=[DocumentSummary.model_validate(d) for d in result["items"]],
        total=result["total"],
        page=result["page"],
        page_size=result["page_size"],
        status_counts=result["status_counts"],
    )


@router.get("/stats", response_model=KnowledgeBaseStats, summary="Knowledge-base statistics")
async def knowledge_base_stats(session: SessionDep) -> KnowledgeBaseStats:
    return KnowledgeBaseStats(**await get_document_service().stats(session))


@router.get("/{document_id}", response_model=DocumentDetail, summary="Document detail")
async def get_document(
    session: SessionDep,
    document_id: str,
    chunk_limit: int = Query(50, ge=1, le=500),
) -> DocumentDetail:
    result = await get_document_service().get_document(session, document_id, chunk_limit)
    detail = DocumentDetail.model_validate(result["document"])
    detail.chunks = [c for c in result["chunks"]]  # validated by pydantic from_attributes
    detail.entities = [e for e in result["entities"]]
    detail.modality_breakdown = result["modality_breakdown"]
    return detail


@router.get(
    "/{document_id}/file",
    summary="Download the original file",
    response_class=Response,
)
async def download_document(session: SessionDep, document_id: str) -> Response:
    result = await get_document_service().get_document(session, document_id, chunk_limit=1)
    document = result["document"]
    if not document.object_key:
        raise NotFoundError("The original file is no longer available in object storage.")
    data = await get_object_store().get(document.object_key)
    return Response(
        content=data,
        media_type=document.mime_type or "application/octet-stream",
        headers={"Content-Disposition": f'inline; filename="{document.filename}"'},
    )


@router.post("/{document_id}/reindex", response_model=ReindexResponse, summary="Reprocess a document")
async def reindex_document(
    session: SessionDep, _: ApiKeyDep, background: BackgroundTasks, document_id: str
) -> ReindexResponse:
    service = get_document_service()
    result = await service.reindex(session, document_id)
    background.add_task(service.process_in_background, document_id)
    return ReindexResponse(**result)


@router.delete("/{document_id}", response_model=Acknowledgement, summary="Delete a document")
async def delete_document(session: SessionDep, _: ApiKeyDep, document_id: str) -> Acknowledgement:
    result = await get_document_service().delete_document(session, document_id)
    return Acknowledgement(**result)


@router.post(
    "/rebuild-indexes",
    response_model=Acknowledgement,
    summary="Rebuild the BM25 index from the database",
)
async def rebuild_indexes(session: SessionDep, _: ApiKeyDep) -> Acknowledgement:
    return Acknowledgement(**await get_document_service().rebuild_indexes(session))


__all__ = ["router"]
