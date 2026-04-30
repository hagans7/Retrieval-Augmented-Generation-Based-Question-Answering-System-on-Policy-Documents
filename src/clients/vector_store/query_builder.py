"""
Weaviate query builder — constructs hybrid search queries for Weaviate v4.
"""

from __future__ import annotations

import weaviate.classes as wvc


class WeaviateQueryBuilder:
    """
    Builds Weaviate v4 hybrid search queries.

    Uses Weaviate's nearVector + bm25 hybrid approach.
    alpha controls the balance: 0.0 = pure BM25, 1.0 = pure vector.
    """

    DEFAULT_ALPHA = 0.5  # Equal weight: vector and keyword

    @staticmethod
    def build_hybrid_query(
        collection,
        query_text: str,
        query_vector: list[float],
        top_k: int,
        filters: dict | None = None,
        alpha: float = DEFAULT_ALPHA,
    ):
        """
        Build and return a Weaviate hybrid query object.

        Args:
            collection: Weaviate collection object.
            query_text: BM25 keyword query string.
            query_vector: Dense vector for similarity component.
            top_k: Maximum results.
            filters: Optional Weaviate filter object.
            alpha: Balance between vector (1.0) and keyword (0.0).

        Returns:
            Weaviate query result object.
        """
        query = collection.query.hybrid(
            query=query_text,
            vector=query_vector,
            alpha=alpha,
            limit=top_k,
            return_metadata=wvc.query.MetadataQuery(score=True),
        )
        return query
