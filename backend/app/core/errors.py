"""Application error hierarchy and FastAPI exception handlers.

Users receive a stable ``{code, message, detail}`` envelope. Internal stack
traces are logged, never returned.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logging import get_logger

log = get_logger("ragx.errors")


class RagxError(Exception):
    """Base class for all expected application failures."""

    code = "ragx_error"
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    message = "An unexpected error occurred."

    def __init__(self, message: str | None = None, detail: Any = None):
        self.message = message or self.message
        self.detail = detail
        super().__init__(self.message)

    def to_payload(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "detail": self.detail}


class NotFoundError(RagxError):
    code = "not_found"
    status_code = status.HTTP_404_NOT_FOUND
    message = "The requested resource was not found."


class ValidationError(RagxError):
    code = "validation_error"
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    message = "The request payload is invalid."


class UnsupportedFileError(RagxError):
    code = "unsupported_file"
    status_code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
    message = "This file type is not supported."


class FileTooLargeError(RagxError):
    code = "file_too_large"
    status_code = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
    message = "The uploaded file exceeds the configured size limit."


class AuthError(RagxError):
    code = "unauthorized"
    status_code = status.HTTP_401_UNAUTHORIZED
    message = "Missing or invalid API key."


class ProviderNotConfiguredError(RagxError):
    code = "provider_not_configured"
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    message = "No cloud LLM provider is configured. Set GEMINI_API_KEY or GROQ_API_KEY."


class ProviderError(RagxError):
    code = "provider_error"
    status_code = status.HTTP_502_BAD_GATEWAY
    message = "The upstream LLM provider failed to respond."


class StorageError(RagxError):
    code = "storage_error"
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    message = "A storage backend is unavailable."


class IngestionError(RagxError):
    code = "ingestion_failed"
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    message = "The document could not be processed."


class RetrievalError(RagxError):
    code = "retrieval_failed"
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    message = "Retrieval failed."


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(RagxError)
    async def _ragx(request: Request, exc: RagxError) -> JSONResponse:
        log.warning("request.failed", code=exc.code, path=request.url.path, message=exc.message)
        return JSONResponse(status_code=exc.status_code, content={"error": exc.to_payload()})

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": {
                    "code": "validation_error",
                    "message": "The request payload is invalid.",
                    "detail": [
                        {"field": ".".join(str(p) for p in e.get("loc", [])), "issue": e.get("msg")}
                        for e in exc.errors()
                    ],
                }
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": "http_error", "message": str(exc.detail), "detail": None}},
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        # Log the trace server-side; return an opaque message to the client.
        log.error("request.unhandled", path=request.url.path, error=str(exc), exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": "internal_error",
                    "message": "An internal error occurred. Please retry or check the server logs.",
                    "detail": None,
                }
            },
        )


__all__ = [
    "RagxError",
    "NotFoundError",
    "ValidationError",
    "UnsupportedFileError",
    "FileTooLargeError",
    "AuthError",
    "ProviderNotConfiguredError",
    "ProviderError",
    "StorageError",
    "IngestionError",
    "RetrievalError",
    "register_exception_handlers",
]
