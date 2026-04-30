"""
Embedding response validator — validates OpenAI-compatible vector shape.

Validates the final parsed vector list before it reaches the vector store.
Works on the already-parsed list of float lists (post batch_processor parsing).
"""

from __future__ import annotations

from src.core.exceptions.infrastructure import EmbeddingError


class EmbeddingResponseValidator:
    """Validates that parsed embedding vectors have the expected shape."""

    @staticmethod
    def validate(vectors: list, expected_count: int) -> list[list[float]]:
        """
        Validate parsed vector list before use.

        Args:
            vectors: Parsed list of float lists from batch_processor.
            expected_count: Expected number of vectors (must match input count).

        Returns:
            Validated list of float vectors.

        Raises:
            EmbeddingError: If count mismatch or vectors are malformed.
        """
        if not isinstance(vectors, list):
            raise EmbeddingError(
                message="Parsed embedding result is not a list.",
                context={"received_type": type(vectors).__name__},
            )
        if len(vectors) != expected_count:
            raise EmbeddingError(
                message=f"Expected {expected_count} vectors, got {len(vectors)}.",
                context={"expected": expected_count, "received": len(vectors)},
            )
        for i, vec in enumerate(vectors):
            if not isinstance(vec, list) or len(vec) == 0:
                raise EmbeddingError(
                    message=f"Vector at index {i} is empty or not a list.",
                    context={"index": i, "type": type(vec).__name__},
                )
            # Allow both float and int (some APIs return int 0 for zero vectors)
            if not all(isinstance(v, (float, int)) for v in vec):
                raise EmbeddingError(
                    message=f"Vector at index {i} contains non-numeric values.",
                    context={"index": i},
                )
        return vectors