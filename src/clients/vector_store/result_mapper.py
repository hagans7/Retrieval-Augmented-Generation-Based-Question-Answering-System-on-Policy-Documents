"""
Weaviate result mapper — maps Weaviate v4 objects to standard dicts.

All Weaviate-specific object types are consumed here.
Service layer only ever sees the standard dict schema.
"""

from __future__ import annotations


class WeaviateResultMapper:
    """Maps Weaviate v4 QueryReturn objects to standardized dicts."""

    @staticmethod
    def map_search_results(weaviate_objects: list) -> list[dict]:
        """
        Map a list of Weaviate search result objects to standard dicts.

        Standard output schema:
            chunk_id: str — Weaviate UUID of the chunk
            content: str — The chunk text content
            score: float — Hybrid search relevance score
            document_id: str — Source document UUID
            metadata: dict — All other properties

        Args:
            weaviate_objects: List of Weaviate result objects.

        Returns:
            List of standardized dicts.
        """
        results = []
        for obj in weaviate_objects:
            properties = obj.properties or {}
            results.append({
                "chunk_id": str(obj.uuid),
                "content": properties.get("content", ""),
                "score": float(obj.metadata.score) if obj.metadata and obj.metadata.score else 0.0,
                "document_id": properties.get("document_id", ""),
                "metadata": {
                    k: v for k, v in properties.items()
                    if k not in ("content", "document_id")
                },
            })
        return results
