"""
ChatResult entity — boundary object between service layer and API layer.

This is the data contract for a completed chat request.
The service layer produces this; the API layer consumes it.
Pure Python dataclass. Zero external dependencies.

Source item contract:
    Each source is a dict with:
      - chunk_id:    str  — Weaviate UUID of the chunk
      - document_id: str  — parent document UUID
      - score:       float — reranker relevance score (0.0–1.0)
      - preview:     str  — first 100 chars of chunk content (no newlines)
    All fields always present. Empty string / 0.0 if unavailable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ChatResult:
    """
    Represents the result of a single completed chat request.

    This is the boundary object between ProcessChatService and the route handler.
    Field names are the contract — renaming any field requires updating all consumers.

    Fields:
        conversation_id: The conversation this result belongs to.
        message_id:      UUID of the newly created assistant message in the DB.
        answer:          The final generated answer text.
        sources:         Structured source references. Each item is a dict:
                         {chunk_id, document_id, score, preview}.
                         Empty list when query type is simple (no retrieval).
        model_name:      The model name used for generation.
        system_prompt_id: The system prompt active during generation.
        token_count:     Estimated tokens consumed. None if unavailable.
        created_at:      UTC timestamp of generation.
    """

    conversation_id: str
    message_id: str
    answer: str
    sources: list[dict] = field(default_factory=list)
    model_name: str = ""
    system_prompt_id: str | None = None
    token_count: int | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    def has_sources(self) -> bool:
        """Return True if this result includes cited source references."""
        return len(self.sources) > 0

    def to_sse_done_payload(self) -> dict:
        """
        Build the flat payload dict for the SSE 'done' event.

        All fields are serializable. Datetime is ISO 8601 string.
        sources list[dict] is already JSON-serializable.
        """
        return {
            "conversation_id": self.conversation_id,
            "message_id": self.message_id,
            "answer": self.answer,
            "sources": self.sources,
            "model_name": self.model_name,
            "system_prompt_id": self.system_prompt_id,
            "token_count": self.token_count,
            "created_at": self.created_at.isoformat(),
        }