"""
Reranker score processor — parses OpenRouter/Cohere rerank response.

OpenRouter rerank response format:
    {
      "id": "...",
      "results": [
        {"index": 0, "relevance_score": 0.95, "document": {"text": "..."}},
        {"index": 1, "relevance_score": 0.72, "document": {"text": "..."}}
      ]
    }

Results are already sorted by relevance_score DESC by the API.
We sort explicitly to guarantee ordering regardless of provider.

Return format change (from previous version):
    Previously: list[tuple[str, float]]
    Now:        list[dict] with keys: text, score
    Reason: callers (agent_runner) need score for auditor decision-making.

Threshold policy:
    RERANKER_MIN_SCORE filters low-relevance results, but always returns
    at least 1 result (the highest-scored one) even if it's below threshold.
    This prevents empty evidence on legitimate but poorly-phrased queries.
"""
from __future__ import annotations

from src.core.exceptions.infrastructure import RerankerError


class ScoreProcessor:
    """Parses OpenRouter/Cohere rerank API response into scored result dicts."""

    @staticmethod
    def process(
        raw_response: dict,
        documents: list[str],
        top_k: int,
        min_score: float = 0.0,
    ) -> list[dict]:
        """
        Extract, filter, and sort rerank results from the API response.

        Args:
            raw_response: Full JSON response dict from the rerank API.
            documents: Original list of document strings sent in the request.
                       Used to retrieve text by index when "document.text"
                       is absent from the response.
            top_k: Maximum number of results to return.
            min_score: Minimum relevance_score to include a result.
                       Always returns at least 1 result (highest-scored)
                       even if it falls below this threshold.

        Returns:
            List of dicts: {text: str, score: float}
            Sorted by score DESC. Never empty if input documents is non-empty.

        Raises:
            RerankerError: If the response structure is unexpected.
        """
        try:
            results = raw_response.get("results", [])
            if not results:
                return []

            sorted_results = sorted(
                results,
                key=lambda x: x.get("relevance_score", 0.0),
                reverse=True,
            )

            output: list[dict] = []
            for item in sorted_results[:top_k]:
                score = float(item.get("relevance_score", 0.0))
                index = item.get("index", 0)

                # Prefer document.text from response; fall back to original list by index
                doc_obj = item.get("document", {})
                if isinstance(doc_obj, dict):
                    text = doc_obj.get("text") or (documents[index] if index < len(documents) else "")
                else:
                    text = documents[index] if index < len(documents) else ""

                output.append({"text": text, "score": score})

            # Apply min_score filter — but always keep at least 1 (the top result)
            if min_score > 0.0 and output:
                filtered = [r for r in output if r["score"] >= min_score]
                # Guarantee minimum 1 result even if all fall below threshold
                return filtered if filtered else [output[0]]

            return output

        except (KeyError, TypeError, IndexError) as exc:
            raise RerankerError(
                message="Failed to parse reranker response.",
                context={"error": str(exc), "response_keys": list(raw_response.keys())},
            ) from exc