"""
Chat response schema.

SourceItem mirrors the source contract from ChatResult entity:
  chunk_id, document_id, score, preview.
All fields are always present in the response — no optional fields.
"""
from __future__ import annotations
from datetime import datetime
from pydantic import BaseModel


class SourceItem(BaseModel):
    """
    One cited evidence chunk in the chat response.

    Fields match the Weaviate chunk metadata returned by hybrid_search:
      chunk_id:    Weaviate object UUID (str).
      document_id: Parent document UUID (str).
      score:       Reranker relevance score, 0.0–1.0 (float).
      preview:     First 100 chars of chunk content, newlines stripped (str).
    """
    chunk_id: str
    document_id: str
    score: float
    preview: str


class ChatResponse(BaseModel):
    conversation_id: str
    message_id: str
    answer: str
    sources: list[SourceItem] = []
    model_name: str
    system_prompt_id: str | None = None
    created_at: datetime