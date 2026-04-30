"""System prompt response schema."""
from __future__ import annotations
from datetime import datetime
from pydantic import BaseModel

class SystemPromptResponse(BaseModel):
    system_prompt_id: str
    user_id: str | None = None
    name: str
    content: str
    is_default: bool
    is_active: bool
    created_at: datetime
    updated_at: datetime
