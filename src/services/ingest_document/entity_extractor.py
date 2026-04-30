"""
Entity extractor — LLM-based named entity and relation extraction.

Extracts legal entities (Pasal, UU, pihak, konsep) and their relations
from document chunks. Used to populate the Neo4j knowledge graph.

Only chunks above ENTITY_EXTRACTION_MIN_TOKENS are sent to the LLM
to avoid wasting tokens on headings and very short fragments.
"""
from __future__ import annotations

import json

from src.core.constants.ingestion import ENTITY_EXTRACTION_MIN_TOKENS
from src.core.logging.logger import get_logger
from src.interfaces.clients.base_llm_client import BaseLLMClient

logger = get_logger(__name__)

_EXTRACTION_PROMPT = """\
Kamu adalah ekstraksi entitas untuk dokumen hukum Indonesia.
Dari teks berikut, ekstrak:
1. Entitas hukum: pasal, undang-undang, peraturan, pihak, konsep hukum
2. Relasi antar entitas

Kembalikan JSON dengan format:
{
  "entities": [
    {"canonical_name": "...", "entity_type": "...", "mention": "..."}
  ],
  "relations": [
    {"from_entity": "...", "to_entity": "...", "relation_type": "..."}
  ]
}

Hanya kembalikan JSON, tanpa penjelasan atau markdown.

TEKS:
"""


class EntityExtractor:
    """
    Extracts entities and relations from text chunks using an LLM.

    Args:
        llm_client: LLM client for extraction inference.
        model_name: Model to use for extraction calls.
    """

    def __init__(self, llm_client: BaseLLMClient, model_name: str) -> None:
        self._llm = llm_client
        self._model = model_name

    async def extract(
        self,
        chunks: list[dict],
    ) -> tuple[list[dict], list[dict]]:
        """
        Extract entities and relations from a list of chunk dicts.

        Only processes chunks with sufficient token count.
        Failed extractions are logged and skipped — not fatal.

        Args:
            chunks: List of chunk dicts with at least 'content' and 'chunk_id'.

        Returns:
            Tuple of (entities_list, relations_list) ready for Neo4j upsert.
        """
        all_entities: list[dict] = []
        all_relations: list[dict] = []

        eligible = [
            c for c in chunks
            if len(c.get("content", "")) // 4 >= ENTITY_EXTRACTION_MIN_TOKENS
        ]

        logger.info(
            "Starting entity extraction",
            extra={"total_chunks": len(chunks), "eligible_chunks": len(eligible)},
        )

        for chunk in eligible:
            try:
                entities, relations = await self._extract_from_chunk(chunk["content"])
                all_entities.extend(entities)
                all_relations.extend(relations)
            except Exception as exc:
                logger.warning(
                    "Entity extraction failed for chunk, skipping",
                    extra={"chunk_id": chunk.get("chunk_id"), "error": str(exc)},
                )

        # Deduplicate entities by canonical_name
        seen: set[str] = set()
        deduped_entities = []
        for e in all_entities:
            name = e.get("canonical_name", "")
            if name and name not in seen:
                seen.add(name)
                deduped_entities.append(e)

        logger.info(
            "Entity extraction complete",
            extra={"entities": len(deduped_entities), "relations": len(all_relations)},
        )
        return deduped_entities, all_relations

    async def _extract_from_chunk(self, content: str) -> tuple[list[dict], list[dict]]:
        """Run LLM extraction on a single chunk and parse the JSON result."""
        prompt = _EXTRACTION_PROMPT + content[:2000]  # Truncate very long chunks
        messages = [{"role": "user", "content": prompt}]
        response = await self._llm.generate(messages=messages, model=self._model)

        # Strip markdown fences if present
        clean = response.strip().strip("```json").strip("```").strip()
        parsed = json.loads(clean)
        return parsed.get("entities", []), parsed.get("relations", [])
