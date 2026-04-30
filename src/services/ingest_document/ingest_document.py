"""
IngestDocumentService — orchestrates the full document ingestion pipeline.

IMPORTANT: This service does NOT create a new document record.
The document record is pre-created by the HTTP route (document_routes.py)
with status=pending. This service only updates the existing record's status
and executes the ingestion steps.

Pipeline: update_status(processing) → upload → OCR → chunk → embed → vector_upsert → entity_extract → graph_upsert → update_status(completed)

Called exclusively by the Celery worker (ingest_document_task).
Retry policy is managed at the Celery task level.
"""
from __future__ import annotations

from src.core.logging.logger import get_logger
from src.interfaces.clients.base_embedding_client import BaseEmbeddingClient
from src.interfaces.clients.base_graph_client import BaseGraphClient
from src.interfaces.clients.base_llm_client import BaseLLMClient
from src.interfaces.clients.base_ocr_client import BaseOCRClient
from src.interfaces.clients.base_storage_client import BaseStorageClient
from src.interfaces.clients.base_vector_store_client import BaseVectorStoreClient
from src.interfaces.repositories.base_document_repository import BaseDocumentRepository
from src.services.ingest_document.chunker import DocumentChunker
from src.services.ingest_document.entity_extractor import EntityExtractor


class IngestDocumentService:
    """
    Orchestrates the end-to-end document ingestion pipeline.

    Receives a pre-created document_id and updates its status through the pipeline.
    Does not create new document records.
    """

    def __init__(
        self,
        document_repo: BaseDocumentRepository,
        storage_client: BaseStorageClient,
        ocr_client: BaseOCRClient,
        embedding_client: BaseEmbeddingClient,
        vector_client: BaseVectorStoreClient,
        graph_client: BaseGraphClient,
        llm_client: BaseLLMClient,
    ) -> None:
        self._document_repo = document_repo
        self._storage = storage_client
        self._ocr = ocr_client
        self._embedding = embedding_client
        self._vector = vector_client
        self._graph = graph_client
        self._llm = llm_client
        self._logger = get_logger(__name__)

    async def execute(
        self,
        document_id: str,
        storage_key: str,
        file_bytes: bytes,
        file_name: str,
        file_type: str,
        user_id: str | None,
        model_name: str,
    ) -> str:
        """
        Run the complete ingestion pipeline for a pre-created document.

        Args:
            document_id: Pre-created document UUID (status=pending).
            storage_key: Pre-computed storage key for this document.
            file_bytes: Raw file bytes.
            file_name: Original filename.
            file_type: File extension: "pdf", "docx", or "txt".
            user_id: Uploader UUID.
            model_name: LLM model name for entity extraction.

        Returns:
            document_id on success.

        Raises:
            Any exception — Celery task handles status=failed update and retry.
        """
        content_type_map = {
            "pdf": "application/pdf",
            "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "txt": "text/plain",
        }

        # ── 1. Upload raw file ───────────────────────────────
        await self._storage.upload_file(
            key=storage_key,
            data=file_bytes,
            content_type=content_type_map.get(file_type, "application/octet-stream"),
        )
        self._logger.info("File uploaded", extra={"document_id": document_id, "key": storage_key})

        # ── 2. Extract text (OCR / Docling) ──────────────────
        text = await self._ocr.extract_text(file_bytes, file_type)
        self._logger.info("Text extracted", extra={"document_id": document_id, "chars": len(text)})

        # ── 3. Chunk ─────────────────────────────────────────
        chunker = DocumentChunker(document_id=document_id, file_name=file_name)
        chunks = chunker.chunk(text)
        self._logger.info("Chunking complete", extra={"document_id": document_id, "chunk_count": len(chunks)})

        # ── 4. Embed ─────────────────────────────────────────
        chunk_dicts = [c.to_dict() for c in chunks]
        texts = [c["content"] for c in chunk_dicts]
        vectors = await self._embedding.embed(texts)
        for i, chunk_dict in enumerate(chunk_dicts):
            chunk_dict["vector"] = vectors[i]

        # ── 5. Upsert to Weaviate ─────────────────────────────
        await self._vector.upsert_chunks(chunk_dicts)
        self._logger.info("Chunks upserted to Weaviate", extra={"document_id": document_id})

        # ── 6. Extract entities ───────────────────────────────
        extractor = EntityExtractor(llm_client=self._llm, model_name=model_name)
        entities, relations = await extractor.extract(chunk_dicts)

        # ── 7. Upsert to Neo4j ────────────────────────────────
        if entities:
            await self._graph.upsert_entities(entities)
        if relations:
            await self._graph.upsert_relations(relations)
        self._logger.info("Graph updated", extra={"document_id": document_id, "entities": len(entities)})

        return document_id