"""Conversation and message response schemas."""
from __future__ import annotations
from datetime import datetime
from pydantic import BaseModel

class MessageResponse(BaseModel):
    message_id: str
    conversation_id: str
    role: str
    content: str
    system_prompt_id: str | None = None
    model_id: str | None = None
    token_count: int | None = None
    created_at: datetime

class ConversationResponse(BaseModel):
    conversation_id: str
    user_id: str | None = None
    title: str
    is_active: bool
    created_at: datetime
    updated_at: datetime
