
"""Abstract base for all reranker clients."""
from __future__ import annotations
from abc import ABC, abstractmethod


class BaseRerankerClient(ABC):
    """Contract for all cross-encoder reranker clients."""

    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int,
        min_score: float = 0.0,
    ) -> list[dict]:
        """
        Rerank documents by relevance to the query using a cross-encoder.

        Args:
            query: The search query.
            documents: List of document text strings to rank.
            top_k: Maximum number of results to return.
            min_score: Minimum relevance_score threshold.
                       Always returns at least 1 result (highest-scored)
                       even if below threshold.

        Returns:
            List of dicts: {text: str, score: float}
            Sorted by score DESC. Never empty if documents is non-empty.
        """
        ...