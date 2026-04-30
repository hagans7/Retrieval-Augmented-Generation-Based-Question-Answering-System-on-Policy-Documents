"""Abstract base for all vector store clients."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseVectorStoreClient(ABC):
    """Contract for all vector store clients (Weaviate, etc.)."""

    @abstractmethod
    async def hybrid_search(
        self,
        query_vector: list[float],
        query_text: str,
        top_k: int,
        filters: dict | None = None,
    ) -> list[dict]:
        """
        Perform hybrid search combining vector similarity and BM25.

        Args:
            query_vector: Dense vector for similarity search.
            query_text: Raw text for BM25 keyword search.
            top_k: Maximum results to return.
            filters: Optional metadata filters (e.g. {"document_id": "..."}).

        Returns:
            List of dicts with keys: chunk_id, content, score, document_id, metadata.

        Raises:
            VectorSearchError: If the search fails.
        """
        ...

    @abstractmethod
    async def upsert_chunks(self, chunks: list[dict]) -> None:
        """
        Insert or update document chunks in the vector store.

        Each chunk dict must contain: chunk_id, content, document_id,
        vector (list[float]), and optional metadata fields.

        Args:
            chunks: List of chunk dicts to upsert.

        Raises:
            VectorSearchError: If the upsert fails for all chunks.
        """
        ...

    @abstractmethod
    async def delete_by_document_id(self, document_id: str) -> None:
        """
        Delete all chunks belonging to a specific document.

        Args:
            document_id: UUID of the document whose chunks to delete.

        Raises:
            VectorSearchError: If the deletion fails.
        """
        ...
