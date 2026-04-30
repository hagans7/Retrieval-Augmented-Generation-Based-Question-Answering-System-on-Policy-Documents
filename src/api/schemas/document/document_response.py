"""Document response schema."""
from __future__ import annotations
from datetime import datetime
from pydantic import BaseModel

class DocumentResponse(BaseModel):
    document_id: str
    user_id: str | None = None
    file_name: str
    file_type: str
    storage_key: str
    file_size_bytes: int | None = None
    ingestion_status: str
    error_message: str | None = None
    chunk_count: int | None = None
    created_at: datetime
    updated_at: datetime
