"""Chat request schema."""
from __future__ import annotations
from uuid import UUID
from pydantic import BaseModel, field_validator


class ChatRequest(BaseModel):
    conversation_id: UUID | None = None  # None → service auto-creates
    message: str
    model_id: UUID | None = None
    system_prompt_id: UUID | None = None
    stream: bool = False

    @field_validator("message")
    @classmethod
    def message_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message cannot be empty")
        return v.strip()