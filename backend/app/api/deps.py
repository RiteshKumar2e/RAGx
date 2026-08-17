"""Shared FastAPI dependencies."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import require_api_key
from app.db.session import get_session

SessionDep = Annotated[AsyncSession, Depends(get_session)]
ApiKeyDep = Annotated[None, Depends(require_api_key)]


class Pagination:
    def __init__(
        self,
        page: int = Query(default=1, ge=1, description="1-indexed page number."),
        page_size: int = Query(default=20, ge=1, le=100, description="Items per page."),
    ):
        self.page = page
        self.page_size = page_size

    @property
    def offset(self) -> int:
        return (self.page - 1) * self.page_size


PaginationDep = Annotated[Pagination, Depends(Pagination)]

__all__ = ["SessionDep", "ApiKeyDep", "Pagination", "PaginationDep"]
