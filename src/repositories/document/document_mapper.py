"""Document mapper — ORM to entity."""
from __future__ import annotations
from src.db_models.document.document_orm import DocumentORM
from src.entities.document.document import Document

def to_entity(orm: DocumentORM) -> Document:
    return Document(
        document_id=orm.document_id,
        user_id=orm.user_id,
        file_name=orm.file_name,
        file_type=orm.file_type,
        storage_key=orm.storage_key,
        file_size_bytes=orm.file_size_bytes,
        ingestion_status=orm.ingestion_status,
        error_message=orm.error_message,
        chunk_count=orm.chunk_count,
        created_at=orm.created_at,
        updated_at=orm.updated_at,
    )
