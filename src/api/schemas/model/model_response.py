"""Model response schema."""
from __future__ import annotations
from datetime import datetime
from pydantic import BaseModel

class ModelResponse(BaseModel):
    model_id: str
    model_name: str
    provider: str
    display_name: str
    description: str | None = None
    context_window: int | None = None
    is_active: bool
    created_at: datetime
