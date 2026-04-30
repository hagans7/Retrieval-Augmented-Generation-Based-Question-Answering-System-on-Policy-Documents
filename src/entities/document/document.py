"""
Document entity — domain object for an ingested document's metadata.

The actual file is stored in object storage (MinIO/S3).
This entity tracks ingestion status and processing metadata.
Pure Python dataclass. Zero external dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Document:
    """
    Represents metadata for a document uploaded to the system.

    The physical file is stored in object storage at storage_key.
    Chunks are stored in Weaviate. Entity relations are in Neo4j.
    This entity is the coordination record across all stores.

    Fields:
        document_id: Unique UUID identifier.
        file_name: Original filename as uploaded.
        file_type: Type: "pdf", "docx", or "txt".
        storage_key: Path key in object storage for retrieving the file.
        ingestion_status: Current pipeline status.
        user_id: Uploader. Nullable while auth is deferred.
        file_size_bytes: File size if known.
        error_message: Set when ingestion_status is "failed".
        chunk_count: Number of chunks generated. Set when "completed".
        created_at: UTC-aware upload timestamp.
        updated_at: UTC-aware last status update timestamp.
    """

    document_id: str
    file_name: str
    file_type: str
    storage_key: str
    ingestion_status: str = "pending"
    user_id: str | None = None
    file_size_bytes: int | None = None
    error_message: str | None = None
    chunk_count: int | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def is_ready(self) -> bool:
        """Return True if ingestion completed successfully."""
        return self.ingestion_status == "completed"

    def has_failed(self) -> bool:
        """Return True if ingestion failed."""
        return self.ingestion_status == "failed"

    def is_processing(self) -> bool:
        """Return True if ingestion is currently in progress."""
        return self.ingestion_status == "processing"

    def is_pending(self) -> bool:
        """Return True if ingestion has not started yet."""
        return self.ingestion_status == "pending"
