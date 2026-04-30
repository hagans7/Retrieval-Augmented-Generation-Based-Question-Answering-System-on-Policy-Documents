"""Conversation request schemas."""
from __future__ import annotations
from pydantic import BaseModel, field_validator

class CreateConversationRequest(BaseModel):
    title: str = "New Conversation"

    @field_validator("title")
    @classmethod
    def title_not_empty(cls, v: str) -> str:
        stripped = v.strip()
        return stripped if stripped else "New Conversation"

class UpdateConversationRequest(BaseModel):
    title: str

    @field_validator("title")
    @classmethod
    def title_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("title cannot be empty")
        return v.strip()
