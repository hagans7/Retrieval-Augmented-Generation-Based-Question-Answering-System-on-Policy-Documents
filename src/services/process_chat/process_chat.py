"""
ProcessChatService — orchestrates one full chat request lifecycle.

Responsibilities:
  1. Auto-create conversation if conversation_id is None
  2. Resolve all context via ContextBuilder
  3. Persist the user message
  4. Run the LangGraph agent (stream or non-stream)
  5. Build sources from agent result — deduplicated, consistent list[str]
  6. Persist the assistant response
  7. Return a ChatResult (non-stream) or yield SSE events (stream)

Sources contract:
  ChatResult.sources is list[str] where each item is a short content preview
  (first 200 chars) of a unique evidence chunk used in the answer.
  Built from agent_runner result["evidence"] — deduplicated by content hash.
  If generator tool call succeeded: only evidence whose index is in citations
  is included. If generator fell back to plain text: top-scored evidence is used.
"""
from __future__ import annotations

from typing import AsyncIterator

from src.core.config.settings import settings
from src.core.logging.logger import get_logger
from src.entities.chat_result.chat_result import ChatResult
from src.interfaces.clients.base_cache_client import BaseCacheClient
from src.interfaces.clients.base_embedding_client import BaseEmbeddingClient
from src.interfaces.clients.base_graph_client import BaseGraphClient
from src.interfaces.clients.base_llm_client import BaseLLMClient
from src.interfaces.clients.base_reranker_client import BaseRerankerClient
from src.interfaces.clients.base_vector_store_client import BaseVectorStoreClient
from src.interfaces.repositories.base_available_model_repository import BaseAvailableModelRepository
from src.interfaces.repositories.base_conversation_repository import BaseConversationRepository
from src.interfaces.repositories.base_message_repository import BaseMessageRepository
from src.interfaces.repositories.base_system_prompt_repository import BaseSystemPromptRepository
from src.services.process_chat.agent_runner import AgentRunner
from src.services.process_chat.context_builder import ContextBuilder


def _build_sources(
    evidence: list[dict],
    citations: list[str],
) -> list[dict]:
    """
    Build a deduplicated list of structured source items.

    Each source item carries the full chunk metadata needed by the API consumer:
      chunk_id, document_id, score, preview (first 100 chars of content).

    Deduplication:
      - By chunk_id if non-empty.
      - Fallback: by preview string (prevents same content with different IDs).

    Citation resolution:
      - citations is a list of 1-based index strings (e.g. ["1", "3"]).
      - These indices refer to evidence sorted by score DESC — the same order
        presented to the generator in the prompt.
      - If citations resolve to valid items, only those are included.
      - If citations is empty or all indices are out of range,
        all readable evidence is included (up to MAX_EVIDENCE_SLOTS, sorted by score).

    Graph evidence exclusion:
      - Evidence from Neo4j graph queries has source="graph" and content is
        JSON-serialized entity data — not human-readable, excluded from sources.

    Args:
        evidence: list[dict] from agent state.
                  Each item: {content, score, source, chunk_id, document_id, _hash}
        citations: list[str] of 1-based index strings from generator tool call.

    Returns:
        list[dict] of source items: {chunk_id, document_id, score, preview}.
        May be empty if no human-readable evidence exists.
        No size limit — all cited/available sources are included.
    """
    # Filter to human-readable sources only (exclude graph JSON)
    readable = [
        e for e in evidence
        if e.get("source", "vector") != "graph" and e.get("content", "").strip()
    ]

    if not readable:
        return []

    # Sort by score DESC — this is the same order shown to generator in prompt
    sorted_evidence = sorted(readable, key=lambda e: e.get("score", 0.0), reverse=True)

    selected: list[dict] = []

    # Try to resolve cited indices first
    if citations:
        for ref in citations:
            try:
                idx = int(ref) - 1  # citations are 1-based
                if 0 <= idx < len(sorted_evidence):
                    selected.append(sorted_evidence[idx])
            except (ValueError, TypeError):
                continue  # skip non-integer citation strings

    # Fallback: use all readable evidence sorted by score
    if not selected:
        selected = sorted_evidence

    # Build structured source items — deduplicated by chunk_id then by preview
    seen_chunk_ids: set[str] = set()
    seen_previews: set[str] = set()
    sources: list[dict] = []

    for item in selected:
        chunk_id = item.get("chunk_id", "")
        content = item.get("content", "").strip()
        preview = " ".join(content[:100].split())  # first 100 chars, whitespace normalized

        if not preview:
            continue

        # Dedup by chunk_id if available
        if chunk_id and chunk_id in seen_chunk_ids:
            continue

        # Dedup by preview as fallback (catches same content with empty/different IDs)
        if preview in seen_previews:
            continue

        if chunk_id:
            seen_chunk_ids.add(chunk_id)
        seen_previews.add(preview)

        sources.append({
            "chunk_id": chunk_id,
            "document_id": item.get("document_id", ""),
            "score": round(float(item.get("score", 0.0)), 4),
            "preview": preview,
        })

    return sources


class ProcessChatService:
    def __init__(
        self,
        conversation_repo: BaseConversationRepository,
        message_repo: BaseMessageRepository,
        system_prompt_repo: BaseSystemPromptRepository,
        model_repo: BaseAvailableModelRepository,
        llm_client: BaseLLMClient,
        vector_client: BaseVectorStoreClient,
        graph_client: BaseGraphClient,
        embedding_client: BaseEmbeddingClient,
        reranker_client: BaseRerankerClient,
        cache_client: BaseCacheClient,
    ) -> None:
        self._conv_repo = conversation_repo
        self._message_repo = message_repo
        self._cache = cache_client
        self._logger = get_logger(__name__)

        self._context_builder = ContextBuilder(
            conversation_repo=conversation_repo,
            message_repo=message_repo,
            system_prompt_repo=system_prompt_repo,
            model_repo=model_repo,
        )
        self._agent_runner = AgentRunner(
            llm_client=llm_client,
            vector_client=vector_client,
            graph_client=graph_client,
            embedding_client=embedding_client,
            reranker_client=reranker_client,
        )

    async def execute(
        self,
        conversation_id: str | None,
        user_message: str,
        model_id: str | None,
        system_prompt_id: str | None,
        stream: bool,
        correlation_id: str = "",
    ) -> "ChatResult | AsyncIterator":
        """
        Process one chat request end-to-end.

        If conversation_id is None, a new conversation is auto-created.
        The created/resolved conversation_id is always returned in ChatResult.

        Returns:
            ChatResult for non-stream, or AsyncIterator for stream mode.
        """
        # ── Auto-create conversation if not provided ─────────
        if conversation_id is None:
            user_id = settings.DEFAULT_USER_ID or None
            raw = user_message.strip()
            title = (raw[:57] + "...") if len(raw) > 60 else raw or "New Conversation"
            new_conv = await self._conv_repo.create(user_id=user_id, title=title)
            conversation_id = new_conv.conversation_id
            self._logger.info(
                "Auto-created conversation",
                extra={"conversation_id": conversation_id, "correlation_id": correlation_id},
            )

        # ── Resolve context ───────────────────────────────────
        context = await self._context_builder.resolve_all(
            conversation_id=conversation_id,
            user_message=user_message,
            model_id=model_id,
            system_prompt_id=system_prompt_id,
        )

        # ── Persist user message ──────────────────────────────
        await self._message_repo.save_user_message(
            conversation_id=conversation_id,
            content=user_message,
            system_prompt_id=context.system_prompt_id,
            model_id=context.model_id,
        )

        if stream:
            return self._execute_stream(context, correlation_id)

        # ── Non-stream: run agent ─────────────────────────────
        # agent_runner.run() returns dict:
        #   {final_answer, citations, confidence, evidence}
        agent_result = await self._agent_runner.run(context, correlation_id)

        final_answer: str = agent_result.get("final_answer", "")
        citations: list[str] = agent_result.get("citations", [])
        evidence: list[dict] = agent_result.get("evidence", [])

        # ── Build sources — deduplicated content previews ─────
        sources = _build_sources(evidence=evidence, citations=citations)

        # ── Persist assistant message ─────────────────────────
        saved_msg = await self._message_repo.save_assistant_message(
            conversation_id=conversation_id,
            content=final_answer,
            system_prompt_id=context.system_prompt_id,
            model_id=context.model_id,
            token_count=None,
        )

        self._logger.info(
            "Chat request completed",
            extra={
                "conversation_id": conversation_id,
                "message_id": saved_msg.message_id,
                "sources_count": len(sources),
                "citations_count": len(citations),
                "evidence_count": len(evidence),
                "correlation_id": correlation_id,
            },
        )

        return ChatResult(
            conversation_id=conversation_id,
            message_id=saved_msg.message_id,
            answer=final_answer,
            sources=sources,
            model_name=context.model_name,
            system_prompt_id=context.system_prompt_id,
        )

    async def _execute_stream(self, context, correlation_id: str) -> AsyncIterator:
        from src.core.constants.agent import (
            AUDITOR_NODE, EXECUTOR_NODE, GENERATOR_NODE, PLANNER_NODE, ROUTER_NODE,
        )

        accumulated_answer = ""
        citations: list[str] = []
        evidence: list[dict] = []

        async for node_name, node_output in self._agent_runner.run_stream(context, correlation_id):
            if node_name == ROUTER_NODE:
                yield "conversation_started", {
                    "conversation_id": context.conversation_id,
                    "model_name": context.model_name,
                }
            elif node_name == PLANNER_NODE:
                yield "plan_created", {"steps": node_output.get("plan_steps", [])}
            elif node_name == EXECUTOR_NODE:
                executed = node_output.get("executed_steps", [])
                if executed:
                    yield "tool_called", {"tool": executed[-1].get("tool_call", {})}
                # Accumulate evidence from each executor output
                node_evidence = node_output.get("evidence", [])
                if node_evidence:
                    evidence = node_evidence  # replace with latest full state
                    yield "evidence_found", {"source_count": len(evidence)}
            elif node_name == GENERATOR_NODE:
                yield "generation_started", {}
                accumulated_answer = node_output.get("final_answer", "")
                citations = node_output.get("citations", [])

        # Build deduplicated sources from final evidence + citations
        sources = _build_sources(evidence=evidence, citations=citations)

        saved_msg = await self._message_repo.save_assistant_message(
            conversation_id=context.conversation_id,
            content=accumulated_answer,
            system_prompt_id=context.system_prompt_id,
            model_id=context.model_id,
            token_count=None,
        )

        result = ChatResult(
            conversation_id=context.conversation_id,
            message_id=saved_msg.message_id,
            answer=accumulated_answer,
            sources=sources,
            model_name=context.model_name,
            system_prompt_id=context.system_prompt_id,
        )
        yield "done", result.to_sse_done_payload()