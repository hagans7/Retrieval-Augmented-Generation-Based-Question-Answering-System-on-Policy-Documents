"""Pagination metadata and paginated response wrapper."""
from __future__ import annotations
from typing import Generic, TypeVar
from pydantic import BaseModel

T = TypeVar("T")

class PaginationMeta(BaseModel):
    total: int | None = None   # None when total count not calculated
    page: int
    page_size: int
    has_next: bool

class PaginatedResponse(BaseModel, Generic[T]):
    success: bool = True
    data: list[T]
    pagination: PaginationMeta
    error: None = None
