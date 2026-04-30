"""Document upload and management routes."""
from __future__ import annotations
import base64
import uuid

from fastapi import APIRouter, Depends, File, Query, UploadFile, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.schemas.common.pagination import PaginatedResponse, PaginationMeta
from src.api.schemas.common.response_envelope import SuccessResponse
from src.api.schemas.document.document_response import DocumentResponse
from src.core.config.settings import settings
from src.core.constants.ingestion import MAX_FILE_SIZE_BYTES, SUPPORTED_FILE_TYPES
from src.core.constants.pagination import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE
from src.core.exceptions.domain import ValidationError
from src.core.logging.logger import get_logger
from src.entities.document.document import Document
from src.providers import get_db_session, get_document_repo

router = APIRouter(prefix="/documents", tags=["documents"])
logger = get_logger(__name__)


def _to_response(d: Document) -> DocumentResponse:
    return DocumentResponse(
        document_id=d.document_id, user_id=d.user_id, file_name=d.file_name,
        file_type=d.file_type, storage_key=d.storage_key, file_size_bytes=d.file_size_bytes,
        ingestion_status=d.ingestion_status, error_message=d.error_message,
        chunk_count=d.chunk_count, created_at=d.created_at, updated_at=d.updated_at,
    )


@router.post("", status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db_session),
):
    """Upload a document and enqueue it for ingestion. Returns 202 + document_id."""
    if not file.filename:
        raise ValidationError(field="file", reason="Filename is required.")

    file_ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if file_ext not in SUPPORTED_FILE_TYPES:
        raise ValidationError(
            field="file",
            reason=f"Unsupported file type '{file_ext}'. Supported: {SUPPORTED_FILE_TYPES}.",
        )

    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise ValidationError(
            field="file",
            reason=f"File exceeds maximum size of {MAX_FILE_SIZE_BYTES // 1_048_576}MB.",
        )

    user_id = settings.DEFAULT_USER_ID or None
    repo = get_document_repo(db)

    from src.clients.storage.key_builder import StorageKeyBuilder
    document_id = str(uuid.uuid4())
    storage_key = StorageKeyBuilder.document_key(file_ext, document_id, file.filename)

    doc = await repo.create(
        user_id=user_id,
        file_name=file.filename,
        file_type=file_ext,
        storage_key=storage_key,
        file_size_bytes=len(file_bytes),
    )

    # base64 encode for JSON-safe Celery dispatch (bytes not JSON serializable)
    file_b64 = base64.b64encode(file_bytes).decode("utf-8")

    try:
        from src.celery_app import ingest_document_task
        ingest_document_task.delay(
            file_b64=file_b64,
            file_name=file.filename,
            file_type=file_ext,
            user_id=user_id,
            model_name=settings.LLM_DEFAULT_MODEL,
            document_id=doc.document_id,
            storage_key=storage_key,
        )
        logger.info(
            "Ingestion task dispatched",
            extra={"document_id": doc.document_id, "file_name": file.filename},
        )
    except Exception as exc:
        logger.error(
            "Failed to dispatch ingestion task — Celery broker may be unavailable.",
            extra={"document_id": doc.document_id, "error": str(exc)},
            exc_info=True,
        )

    return JSONResponse(
        status_code=202,
        content={
            "success": True,
            "data": {"document_id": doc.document_id, "status": "pending"},
            "error": None,
        },
    )


@router.get("", response_model=PaginatedResponse[DocumentResponse])
async def list_documents(
    limit: int = Query(default=DEFAULT_PAGE_SIZE, le=MAX_PAGE_SIZE, ge=1),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db_session),
):
    """List all documents for the current user, newest first."""
    user_id = settings.DEFAULT_USER_ID or None
    repo = get_document_repo(db)
    docs = await repo.get_all_by_user(user_id=user_id, limit=limit, offset=offset)
    return PaginatedResponse(
        data=[_to_response(d) for d in docs],
        pagination=PaginationMeta(
            total=None,
            page=(offset // limit) + 1,
            page_size=limit,
            has_next=len(docs) == limit,
        ),
    )


@router.get("/{document_id}", response_model=SuccessResponse[DocumentResponse])
async def get_document(document_id: str, db: AsyncSession = Depends(get_db_session)):
    """Get document status and metadata by ID."""
    repo = get_document_repo(db)
    doc = await repo.get_by_id(document_id)
    if not doc:
        raise ValidationError(
            field="document_id", reason=f"Document '{document_id}' not found."
        )
    return SuccessResponse(data=_to_response(doc))


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: str,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Soft-delete a document record.

    Sets ingestion_status='failed' with error_message indicating deletion.
    Does not remove data from MinIO storage or Weaviate vector store.
    """
    repo = get_document_repo(db)
    doc = await repo.get_by_id(document_id)
    if not doc:
        raise ValidationError(
            field="document_id", reason=f"Document '{document_id}' not found."
        )
    # Soft delete: marks document as deleted in DB.
    # Does not remove data from MinIO storage or Weaviate (handled by cleanup jobs).
    await repo.soft_delete(document_id=document_id)