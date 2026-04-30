"""
Weaviate collection manager — setup and verification of collection schema.

Ensures the DocumentChunk collection exists with the correct schema
before the application begins serving requests.
"""

from __future__ import annotations

import weaviate
import weaviate.classes as wvc

from src.core.logging.logger import get_logger

logger = get_logger(__name__)

# Schema definition for DocumentChunk collection
DOCUMENT_CHUNK_PROPERTIES = [
    wvc.config.Property(name="content", data_type=wvc.config.DataType.TEXT),
    wvc.config.Property(name="document_id", data_type=wvc.config.DataType.TEXT),
    wvc.config.Property(name="chunk_index", data_type=wvc.config.DataType.INT),
    wvc.config.Property(name="file_name", data_type=wvc.config.DataType.TEXT),
    wvc.config.Property(name="file_type", data_type=wvc.config.DataType.TEXT),
    wvc.config.Property(name="section_type", data_type=wvc.config.DataType.TEXT),
    wvc.config.Property(name="parent_section", data_type=wvc.config.DataType.TEXT),
    wvc.config.Property(name="token_count", data_type=wvc.config.DataType.INT),
]


class CollectionManager:
    """
    Manages Weaviate collection schema lifecycle.

    Args:
        client: Active Weaviate v4 client.
        collection_name: Name of the collection to manage.
    """

    def __init__(self, client: weaviate.WeaviateClient, collection_name: str) -> None:
        self._client = client
        self._collection_name = collection_name

    def ensure_collection_exists(self) -> None:
        """
        Create the collection if it does not already exist.

        Called once at application startup by the lifespan handler.
        Idempotent — safe to call even if collection already exists.
        """
        if self._client.collections.exists(self._collection_name):
            logger.info(
                "Weaviate collection already exists",
                extra={"collection": self._collection_name},
            )
            return

        self._client.collections.create(
            name=self._collection_name,
            vectorizer_config=wvc.config.Configure.Vectorizer.none(),
            properties=DOCUMENT_CHUNK_PROPERTIES,
        )
        logger.info(
            "Weaviate collection created",
            extra={"collection": self._collection_name},
        )
