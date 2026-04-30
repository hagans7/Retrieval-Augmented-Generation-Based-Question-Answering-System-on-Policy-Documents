"""
Standard API response envelopes used across all endpoints.

Every endpoint returns either SuccessResponse or ErrorResponse.
This consistency allows clients to always check `success` first.
"""
from __future__ import annotations
from typing import Generic, TypeVar
from pydantic import BaseModel

T = TypeVar("T")

class ErrorDetail(BaseModel):
    code: str        # UPPER_SNAKE error code, e.g. "CONVERSATION_NOT_FOUND"
    message: str     # Human-readable message safe to return to clients

class SuccessResponse(BaseModel, Generic[T]):
    success: bool = True
    data: T
    error: None = None

class ErrorResponse(BaseModel):
    success: bool = False
    data: None = None
    error: ErrorDetail
