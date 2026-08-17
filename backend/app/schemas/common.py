"""Shared response envelopes and primitives."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class ErrorDetail(BaseModel):
    code: str
    message: str
    detail: Any = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


class Page(BaseModel, Generic[T]):
    items: list[T]
    total: int
    page: int = 1
    page_size: int = 20

    @property
    def pages(self) -> int:
        return max(1, -(-self.total // self.page_size))


class Acknowledgement(BaseModel):
    ok: bool = True
    message: str = ""
    detail: dict[str, Any] = Field(default_factory=dict)


class HealthComponent(BaseModel):
    name: str
    status: str
    healthy: bool | None = None
    detail: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    timestamp: datetime
    components: list[HealthComponent]
    warnings: list[str] = Field(default_factory=list)


__all__ = [
    "ORMModel",
    "ErrorDetail",
    "ErrorResponse",
    "Page",
    "Acknowledgement",
    "HealthComponent",
    "HealthResponse",
]
