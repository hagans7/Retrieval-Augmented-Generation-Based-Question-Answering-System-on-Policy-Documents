"""
Infrastructure exception variants for all external service failures.

All exceptions here are critical and cause the current workflow to fail,
EXCEPT CacheError which is non-critical and allows graceful degradation.

HTTP mapping: all map to 503 Service Unavailable in the API layer.
"""

from __future__ import annotations

from src.core.exceptions.base import AppBaseError


class LLMInferenceError(AppBaseError):
    """
    Raised when the LLM provider fails to generate a response.

    Critical — the chat workflow cannot continue without LLM output.
    Raised by: clients/llm/llm_client.py after all retries are exhausted.
    """


class VectorSearchError(AppBaseError):
    """
    Raised when the Weaviate vector store fails to respond.

    Critical — retrieval cannot proceed without the vector store.
    Raised by: clients/vector_store/vector_store_client.py
    """


class GraphQueryError(AppBaseError):
    """
    Raised when the Neo4j graph store fails to respond.

    Critical — graph-based retrieval cannot proceed.
    Raised by: clients/graph/graph_client.py
    """


class EmbeddingError(AppBaseError):
    """
    Raised when the TEI embedding service fails to generate vectors.

    Critical — ingestion cannot proceed without embeddings.
    Raised by: clients/embedding/embedding_client.py
    """


class RerankerError(AppBaseError):
    """
    Raised when the TEI reranker service fails to score documents.

    Critical — reranking is part of the retrieval pipeline.
    Raised by: clients/reranker/reranker_client.py
    """


class StorageError(AppBaseError):
    """
    Raised when the S3-compatible object storage (MinIO/S3) fails.

    Critical — document upload/download cannot proceed.
    Raised by: clients/storage/storage_client.py
    """


class OCRError(AppBaseError):
    """
    Raised when OCR model initialization or text extraction fails.

    Critical for ingestion — document cannot be processed without text extraction.
    Raised by: clients/ocr/ocr_client.py and clients/ocr/model_loader.py
    """


class CacheError(AppBaseError):
    """
    Raised when the Redis cache fails.

    NON-CRITICAL — service layer catches this, logs a WARNING,
    and continues the workflow without cache (degraded mode).
    Raised by: clients/cache/cache_client.py
    """
