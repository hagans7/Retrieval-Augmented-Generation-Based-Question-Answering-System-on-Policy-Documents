"""System prompt request schemas."""
from __future__ import annotations
from pydantic import BaseModel, field_validator

class CreatePromptRequest(BaseModel):
    name: str
    content: str
    is_default: bool = False

    @field_validator("name", "content")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("field cannot be empty")
        return v.strip()

class UpdatePromptRequest(BaseModel):
    name: str | None = None
    content: str | None = None
