"""
Weaviate vector store client — entry point implementing BaseVectorStoreClient.

Uses Weaviate Python SDK v4. Breaking change from v3 — no weaviate.Client().
"""

from __future__ import annotations

import weaviate

from src.clients.vector_store.collection_manager import CollectionManager
from src.clients.vector_store.query_builder import WeaviateQueryBuilder
from src.clients.vector_store.result_mapper import WeaviateResultMapper
from src.core.exceptions.infrastructure import VectorSearchError
from src.core.logging.logger import get_logger
from src.interfaces.clients.base_vector_store_client import BaseVectorStoreClient


class WeaviateVectorStoreClient(BaseVectorStoreClient):
    """
    Async-capable Weaviate v4 client for hybrid search and chunk management.

    Args:
        weaviate_url: Weaviate HTTP endpoint URL.
        collection_name: Name of the Weaviate collection for document chunks.
    """

    def __init__(self, weaviate_url: str, collection_name: str) -> None:
        self._weaviate_url = weaviate_url
        self._collection_name = collection_name
        self._client: weaviate.WeaviateClient | None = None
        self._logger = get_logger(__name__)

    def connect(self) -> None:
        """
        Open the Weaviate connection. Called during application startup.

        Raises:
            VectorSearchError: If connection fails.
        """
        try:
            self._client = weaviate.connect_to_custom(
                http_host=self._weaviate_url.replace("http://", "").replace("https://", "").split(":")[0],
                http_port=int(self._weaviate_url.split(":")[-1]) if ":" in self._weaviate_url.split("//")[-1] else 80,
                http_secure=self._weaviate_url.startswith("https"),
                grpc_host=self._weaviate_url.replace("http://", "").replace("https://", "").split(":")[0],
                grpc_port=50051,
                grpc_secure=False,
            )
            manager = CollectionManager(self._client, self._collection_name)
            manager.ensure_collection_exists()
            self._logger.info("Weaviate connection established", extra={"url": self._weaviate_url})
        except Exception as exc:
            raise VectorSearchError(
                message="Failed to connect to Weaviate.",
                context={"url": self._weaviate_url, "error": str(exc)},
            ) from exc

    def close(self) -> None:
        """Close the Weaviate connection during application shutdown."""
        if self._client:
            self._client.close()

    def _get_collection(self):
        """Return the Weaviate collection object. Raises if not connected."""
        if not self._client:
            raise VectorSearchError(
                message="Weaviate client is not connected. Call connect() first.",
            )
        return self._client.collections.get(self._collection_name)

    async def hybrid_search(
        self,
        query_vector: list[float],
        query_text: str,
        top_k: int,
        filters: dict | None = None,
    ) -> list[dict]:
        """
        Hybrid vector + BM25 search.

        Returns:
            List of dicts with chunk_id, content, score, document_id, metadata.

        Raises:
            VectorSearchError: If the search fails.
        """
        try:
            collection = self._get_collection()
            results = WeaviateQueryBuilder.build_hybrid_query(
                collection=collection,
                query_text=query_text,
                query_vector=query_vector,
                top_k=top_k,
                filters=filters,
            )
            return WeaviateResultMapper.map_search_results(results.objects)
        except VectorSearchError:
            raise
        except Exception as exc:
            raise VectorSearchError(
                message="Hybrid search failed.",
                context={"query_text": query_text[:100], "error": str(exc)},
            ) from exc

    async def upsert_chunks(self, chunks: list[dict]) -> None:
        """
        Upsert document chunks into Weaviate.

        Each chunk dict must include: chunk_id (uuid), content, document_id,
        vector (list[float]), and optional metadata properties.

        Raises:
            VectorSearchError: If the upsert fails entirely.
        """
        if not chunks:
            return
        try:
            collection = self._get_collection()
            objects = []
            import weaviate.classes as wvc
            for chunk in chunks:
                props = {k: v for k, v in chunk.items() if k not in ("chunk_id", "vector")}
                objects.append(
                    wvc.data.DataObject(
                        uuid=chunk["chunk_id"],
                        properties=props,
                        vector=chunk["vector"],
                    )
                )
            collection.data.insert_many(objects)
            self._logger.info("Chunks upserted", extra={"count": len(chunks)})
        except Exception as exc:
            raise VectorSearchError(
                message=f"Failed to upsert {len(chunks)} chunks.",
                context={"error": str(exc)},
            ) from exc

    async def delete_by_document_id(self, document_id: str) -> None:
        """
        Delete all chunks belonging to a document.

        Raises:
            VectorSearchError: If deletion fails.
        """
        try:
            import weaviate.classes as wvc
            collection = self._get_collection()
            collection.data.delete_many(
                where=wvc.query.Filter.by_property("document_id").equal(document_id)
            )
            self._logger.info("Chunks deleted by document_id", extra={"document_id": document_id})
        except Exception as exc:
            raise VectorSearchError(
                message=f"Failed to delete chunks for document '{document_id}'.",
                context={"document_id": document_id, "error": str(exc)},
            ) from exc
