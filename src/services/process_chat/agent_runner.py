
"""
Agent runner — builds and invokes the LangGraph agentic orchestration graph.

Constructs the StateGraph with ROUTER → PLANNER → EXECUTOR → AUDITOR → GENERATOR
flow. Each node is a closure over the injected clients.
Supports both stream and non-stream execution modes.

Key behavioral contracts:
  1. Reranker score capture: all evidence items carry a 'score' field from
     the reranker. This score is visible to auditor and generator nodes.
  2. Evidence deduplication: chunks are deduplicated by content hash before
     appending to state, preventing the same chunk from inflating evidence count.
  3. Evidence cap: total evidence is capped at MAX_EVIDENCE_SLOTS. When cap is
     reached, lower-scored items are dropped in favor of higher-scored ones.
  4. Reranker threshold with min-1 guarantee: RERANKER_MIN_SCORE filters
     low-relevance results, but always keeps at least 1 result to prevent
     empty evidence on legitimate queries.
  5. Loop guard: AGENT_MAX_EXECUTOR_LOOPS caps how many times auditor can
     restart executor. When reached, routing forces GENERATOR regardless of
     auditor verdict.
  6. Citations fallback: if generator tool call fails, citations are extracted
     from evidence state (by index number) to ensure sources is never empty
     when evidence exists.
  7. Grounding: generator prompt explicitly forbids using parametric knowledge.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, AsyncIterator, TypedDict

from langgraph.graph import END, StateGraph

from src.core.config.settings import settings
from src.core.constants.agent import (
    AUDITOR_NODE,
    EXECUTOR_NODE,
    GENERATOR_NODE,
    PLANNER_NODE,
    ROUTER_NODE,
)
from src.core.logging.logger import get_logger
from src.interfaces.clients.base_embedding_client import BaseEmbeddingClient
from src.interfaces.clients.base_graph_client import BaseGraphClient
from src.interfaces.clients.base_llm_client import BaseLLMClient
from src.interfaces.clients.base_reranker_client import BaseRerankerClient
from src.interfaces.clients.base_vector_store_client import BaseVectorStoreClient
from src.prompts.agents import (
    auditor_prompt,
    executor_prompt,
    generator_prompt,
    planner_prompt,
    router_prompt,
)
from src.services.process_chat.context_builder import ChatContext

logger = get_logger(__name__)


def _content_hash(text: str) -> str:
    """Short hash for evidence deduplication."""
    return hashlib.md5(text.encode("utf-8"), usedforsecurity=False).hexdigest()[:12]


class AgentState(TypedDict):
    """Shared state dict passed between all LangGraph nodes."""
    conversation_id: str
    user_message: str
    chat_history: list[dict]
    user_system_prompt: str
    model_name: str
    correlation_id: str
    # Router output
    query_type: str
    # Planner output
    plan_steps: list[str]
    current_step_index: int
    # Executor output
    executed_steps: list[dict]
    evidence: list[dict]           # items: {content, score, source, chunk_id, _hash}
    seen_hashes: list[str]         # dedup tracker — content hashes already in evidence
    # Auditor output
    audit_verdict: str
    audit_retry_count: int         # how many times auditor has restarted executor
    top_evidence_score: float      # best reranker score in current evidence set
    # Generator output
    final_answer: str
    citations: list[str]
    confidence: str


class AgentRunner:
    """
    Builds and invokes the LangGraph orchestration graph.

    Args:
        llm_client: For all LLM calls.
        vector_client: For vector/hybrid search.
        graph_client: For Neo4j graph queries.
        embedding_client: To embed user queries before vector search.
        reranker_client: To rerank retrieved chunks (returns scored dicts).
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        vector_client: BaseVectorStoreClient,
        graph_client: BaseGraphClient,
        embedding_client: BaseEmbeddingClient,
        reranker_client: BaseRerankerClient,
    ) -> None:
        self._llm = llm_client
        self._vector = vector_client
        self._graph = graph_client
        self._embedding = embedding_client
        self._reranker = reranker_client

    def _build_graph(self) -> Any:
        """Construct and compile the LangGraph StateGraph."""
        graph = StateGraph(AgentState)
        graph.add_node(ROUTER_NODE, self._router_node)
        graph.add_node(PLANNER_NODE, self._planner_node)
        graph.add_node(EXECUTOR_NODE, self._executor_node)
        graph.add_node(AUDITOR_NODE, self._auditor_node)
        graph.add_node(GENERATOR_NODE, self._generator_node)

        graph.set_entry_point(ROUTER_NODE)

        graph.add_conditional_edges(
            ROUTER_NODE,
            lambda s: GENERATOR_NODE if s.get("query_type") == "simple" else PLANNER_NODE,
        )
        graph.add_edge(PLANNER_NODE, EXECUTOR_NODE)
        graph.add_conditional_edges(EXECUTOR_NODE, self._executor_routing)
        graph.add_conditional_edges(AUDITOR_NODE, self._auditor_routing)
        graph.add_edge(GENERATOR_NODE, END)

        return graph.compile()

    # ── Node implementations ─────────────────────────────────

    async def _router_node(self, state: AgentState) -> dict:
        """Classify query into simple/retrieval/graph/hybrid."""
        messages = [
            {"role": "system", "content": router_prompt.SYSTEM_PROMPT},
            {"role": "user", "content": state["user_message"]},
        ]
        result = await self._llm.generate_with_tool_call(
            messages=messages,
            model=state["model_name"],
            tools=router_prompt.TOOL_SCHEMA,
        )
        query_type = (result or {}).get("query_type", "retrieval")
        logger.debug(
            "Router classified query",
            extra={"query_type": query_type, "conversation_id": state["conversation_id"]},
        )
        return {"query_type": query_type}

    async def _planner_node(self, state: AgentState) -> dict:
        """Create ordered retrieval plan."""
        messages = [
            {"role": "system", "content": planner_prompt.SYSTEM_PROMPT},
            {"role": "user", "content": f"Query: {state['user_message']}\nTipe: {state['query_type']}"},
        ]
        result = await self._llm.generate_with_tool_call(
            messages=messages,
            model=state["model_name"],
            tools=planner_prompt.TOOL_SCHEMA,
        )
        steps = (result or {}).get("steps", [state["user_message"]])
        return {"plan_steps": steps, "current_step_index": 0}

    async def _executor_node(self, state: AgentState) -> dict:
        """
        Execute one retrieval step, accumulate evidence.

        Evidence management:
          - New results from reranker carry {text, score} format.
          - Each item is converted to {content, score, source, _hash}.
          - Items already seen (by content hash) are skipped.
          - Total evidence is capped at MAX_EVIDENCE_SLOTS.
            When cap is reached, lowest-scored items are replaced
            if new items have higher scores.
        """
        steps = state.get("plan_steps", [])
        idx = state.get("current_step_index", 0)
        if idx >= len(steps):
            return {"current_step_index": idx}

        current_step = steps[idx]
        messages = [
            {"role": "system", "content": executor_prompt.SYSTEM_PROMPT},
            {"role": "user", "content": f"Langkah: {current_step}"},
        ]
        tool_call = await self._llm.generate_with_tool_call(
            messages=messages,
            model=state["model_name"],
            tools=executor_prompt.TOOL_SCHEMA,
        )

        evidence: list[dict] = list(state.get("evidence", []))
        seen_hashes: list[str] = list(state.get("seen_hashes", []))
        executed: list[dict] = list(state.get("executed_steps", []))
        new_evidence_count = 0

        if tool_call:
            raw_results = await self._dispatch_tool(tool_call, state["user_message"])

            for item in raw_results:
                content = item.get("content", "")
                h = _content_hash(content)
                if h in seen_hashes:
                    continue  # skip duplicate

                evidence_item = {
                    "content": content,
                    "score": item.get("score", 0.0),
                    "source": item.get("source", "vector"),
                    "chunk_id": item.get("chunk_id", ""),
                    "document_id": item.get("document_id", ""),
                    "_hash": h,
                }

                if len(evidence) < settings.MAX_EVIDENCE_SLOTS:
                    evidence.append(evidence_item)
                    seen_hashes.append(h)
                    new_evidence_count += 1
                else:
                    # Cap reached: replace lowest-scored item if new item scores higher
                    min_idx = min(range(len(evidence)), key=lambda i: evidence[i].get("score", 0.0))
                    if evidence_item["score"] > evidence[min_idx].get("score", 0.0):
                        old_hash = evidence[min_idx].get("_hash", "")
                        if old_hash in seen_hashes:
                            seen_hashes.remove(old_hash)
                        evidence[min_idx] = evidence_item
                        seen_hashes.append(h)
                        new_evidence_count += 1

            executed.append({
                "step": current_step,
                "tool_call": tool_call,
                "result_count": new_evidence_count,
            })
            logger.debug(
                "Executor step done",
                extra={"step": idx, "evidence_added": new_evidence_count},
            )

        # Track best reranker score across all evidence
        top_score = max((e.get("score", 0.0) for e in evidence), default=0.0)

        return {
            "evidence": evidence,
            "seen_hashes": seen_hashes,
            "executed_steps": executed,
            "current_step_index": idx + 1,
            "top_evidence_score": top_score,
        }

    async def _auditor_node(self, state: AgentState) -> dict:
        """
        Evaluate evidence sufficiency with score-aware context.

        Loop guard: audit_retry_count is checked in _auditor_routing.
        When AGENT_MAX_EXECUTOR_LOOPS is reached, routing bypasses
        auditor verdict and forces GENERATOR.
        """
        evidence = state.get("evidence", [])
        loop_count = state.get("audit_retry_count", 0)

        # Build evidence summary with scores for auditor to evaluate
        evidence_summary_lines = []
        for i, e in enumerate(evidence[:10], 1):
            score = e.get("score", 0.0)
            preview = e.get("content", "")[:150].replace("\n", " ")
            evidence_summary_lines.append(f"[{i}] score={score:.3f} | {preview}")
        evidence_summary = "\n".join(evidence_summary_lines) if evidence_summary_lines else "Tidak ada evidence."

        system_content = auditor_prompt.SYSTEM_PROMPT_TEMPLATE.format(
            query=state["user_message"],
            loop_count=loop_count,
            max_loops=settings.AGENT_MAX_EXECUTOR_LOOPS,
            evidence_summary=evidence_summary,
        )
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": "Evaluasi evidence di atas."},
        ]
        result = await self._llm.generate_with_tool_call(
            messages=messages,
            model=state["model_name"],
            tools=auditor_prompt.TOOL_SCHEMA,
        )

        verdict = (result or {}).get("verdict", "sufficient")
        top_score = (result or {}).get("top_evidence_score", state.get("top_evidence_score", 0.0))

        new_retry_count = loop_count + (1 if verdict == "retry" else 0)

        logger.debug(
            "Auditor verdict",
            extra={
                "verdict": verdict,
                "top_score": top_score,
                "loop_count": loop_count,
                "evidence_count": len(evidence),
                "conversation_id": state["conversation_id"],
            },
        )

        return {
            "audit_verdict": verdict,
            "audit_retry_count": new_retry_count,
            "top_evidence_score": top_score,
        }

    async def _generator_node(self, state: AgentState) -> dict:
        """
        Generate the final answer, grounded exclusively in evidence.

        Evidence grounding:
          - Evidence is formatted as numbered blocks [1], [2], ...
          - Each block shows score so the model can weight by confidence.
          - Prompt explicitly prohibits using parametric knowledge.
          - Tool call is attempted first (structured output with citations).
          - Fallback: plain text generation with citations extracted from state.

        Citations fallback guarantee:
          If tool call fails (model returns plain text), citations are populated
          from the top-scored evidence items by index number, ensuring sources
          is never empty when evidence exists.
        """
        evidence = state.get("evidence", [])

        # Sort evidence by score DESC for generator context
        sorted_evidence = sorted(evidence, key=lambda e: e.get("score", 0.0), reverse=True)

        # Build numbered evidence blocks with score info
        evidence_blocks = []
        for i, e in enumerate(sorted_evidence[:12], 1):
            score = e.get("score", 0.0)
            content = e.get("content", "")
            source = e.get("source", "")
            evidence_blocks.append(f"[{i}] (relevance={score:.3f}, source={source})\n{content}")

        evidence_context = "\n\n".join(evidence_blocks) if evidence_blocks else "Tidak ada evidence yang tersedia."

        # Quality summary for generator awareness
        top_score = state.get("top_evidence_score", 0.0)
        evidence_count = len(evidence)
        if evidence_count == 0:
            quality_summary = "Tidak ada evidence — jawab bahwa informasi tidak tersedia."
        elif top_score >= 0.7:
            quality_summary = f"{evidence_count} evidence tersedia. Skor tertinggi: {top_score:.3f} (tinggi — jawab dengan confident)."
        elif top_score >= 0.4:
            quality_summary = f"{evidence_count} evidence tersedia. Skor tertinggi: {top_score:.3f} (sedang — jawab dengan hati-hati)."
        else:
            quality_summary = f"{evidence_count} evidence tersedia. Skor tertinggi: {top_score:.3f} (rendah — nyatakan ketidakpastian dalam jawaban)."

        system_content = generator_prompt.SYSTEM_PROMPT_TEMPLATE.format(
            user_system_prompt=state.get("user_system_prompt", ""),
            evidence_context=evidence_context,
            evidence_quality_summary=quality_summary,
        )
        messages = (
            [{"role": "system", "content": system_content}]
            + state.get("chat_history", [])
            + [{"role": "user", "content": state["user_message"]}]
        )

        # Attempt structured tool call first
        result = await self._llm.generate_with_tool_call(
            messages=messages,
            model=state["model_name"],
            tools=generator_prompt.TOOL_SCHEMA,
        )

        if result is not None:
            answer = result.get("answer", "")
            citations = result.get("citations", [])
            confidence = result.get("confidence", "low")
        else:
            # Fallback: plain text generation
            logger.debug(
                "Generator: no tool call returned, falling back to generate()",
                extra={"conversation_id": state["conversation_id"]},
            )
            answer = await self._llm.generate(
                messages=messages,
                model=state["model_name"],
            )
            confidence = "medium"
            # Citations fallback: use top-scored evidence indices as source references
            # This ensures sources is never empty when evidence exists
            citations = [
                str(i)
                for i, _ in enumerate(sorted_evidence[:5], 1)
            ] if sorted_evidence else []

        return {"final_answer": answer, "citations": citations, "confidence": confidence}

    # ── Conditional routing ──────────────────────────────────

    def _executor_routing(self, state: AgentState) -> str:
        """Route to next executor step or to auditor when plan is done."""
        idx = state.get("current_step_index", 0)
        steps = state.get("plan_steps", [])
        if idx < len(steps):
            return EXECUTOR_NODE
        return AUDITOR_NODE

    def _auditor_routing(self, state: AgentState) -> str:
        """
        Route back to planner for retry, or forward to generator.

        Loop guard: if audit_retry_count >= AGENT_MAX_EXECUTOR_LOOPS,
        force GENERATOR regardless of auditor verdict.
        This prevents infinite retry loops when retrieval cannot improve.
        """
        verdict = state.get("audit_verdict", "sufficient")
        retry_count = state.get("audit_retry_count", 0)

        # Hard cap — force generator when loop limit reached
        if retry_count >= settings.AGENT_MAX_EXECUTOR_LOOPS:
            logger.debug(
                "Auditor: loop limit reached, forcing GENERATOR",
                extra={
                    "retry_count": retry_count,
                    "max_loops": settings.AGENT_MAX_EXECUTOR_LOOPS,
                    "verdict": verdict,
                },
            )
            return GENERATOR_NODE

        if verdict == "retry":
            return PLANNER_NODE

        return GENERATOR_NODE

    # ── Tool dispatch ────────────────────────────────────────

    async def _dispatch_tool(self, tool_call: dict, user_query: str) -> list[dict]:
        """Route executor tool calls to the appropriate client."""
        if "entity_name" in tool_call:
            return await self._run_graph_query(tool_call)
        if "entity_hint" in tool_call:
            return await self._run_hybrid_search(tool_call, user_query)
        return await self._run_vector_search(tool_call, user_query)

    async def _run_vector_search(self, params: dict, user_query: str) -> list[dict]:
        """
        Vector search + rerank.

        Returns list[dict] with {content, score, source, chunk_id}.
        Score comes from reranker (not vector similarity).
        """
        query_text = params.get("query", user_query)
        top_k = params.get("top_k", settings.RERANKER_TOP_K)

        query_vector = await self._embedding.embed_single(query_text)
        results = await self._vector.hybrid_search(
            query_vector=query_vector,
            query_text=query_text,
            top_k=top_k * 2,  # fetch more, reranker will filter down
        )
        if not results:
            return []

        texts = [r.get("content", "") for r in results]
        reranked = await self._reranker.rerank(
            query=query_text,
            documents=texts,
            top_k=top_k,
            min_score=settings.RERANKER_MIN_SCORE,
        )

        # Merge reranker scores back with original metadata
        text_to_meta: dict[str, dict] = {}
        for r in results:
            text_to_meta[r.get("content", "")] = r

        output = []
        for item in reranked:
            text = item["text"]
            score = item["score"]
            meta = text_to_meta.get(text, {})
            output.append({
                "content": text,
                "score": score,
                "source": "vector",
                "chunk_id": meta.get("chunk_id", meta.get("id", "")),
                "document_id": meta.get("document_id", ""),
            })
        return output

    async def _run_graph_query(self, params: dict) -> list[dict]:
        """Graph traversal — results carry score=1.0 (graph is always exact match)."""
        entity_name = params.get("entity_name", "")
        max_hops = params.get("max_hops", 2)
        results = await self._graph.multi_hop_query(entity_name, max_hops)
        return [
            {
                "content": json.dumps(r, ensure_ascii=False),
                "score": 1.0,
                "source": "graph",
                "chunk_id": r.get("chunk_id", ""),
                "document_id": r.get("document_id", ""),
            }
            for r in results
        ]

    async def _run_hybrid_search(self, params: dict, user_query: str) -> list[dict]:
        """Hybrid: vector search + graph enrichment."""
        vector_results = await self._run_vector_search(
            {"query": params.get("query", user_query), "top_k": params.get("top_k", settings.RERANKER_TOP_K)},
            user_query,
        )
        graph_results: list[dict] = []
        if params.get("entity_hint"):
            graph_results = await self._run_graph_query({"entity_name": params["entity_hint"]})
        return vector_results + graph_results

    # ── Public interface ─────────────────────────────────────

    def _make_initial_state(self, context: ChatContext, correlation_id: str) -> AgentState:
        return {
            "conversation_id": context.conversation_id,
            "user_message": context.user_message,
            "chat_history": context.chat_history,
            "user_system_prompt": context.user_system_prompt,
            "model_name": context.model_name,
            "correlation_id": correlation_id,
            "query_type": "",
            "plan_steps": [],
            "current_step_index": 0,
            "executed_steps": [],
            "evidence": [],
            "seen_hashes": [],
            "audit_verdict": "",
            "audit_retry_count": 0,
            "top_evidence_score": 0.0,
            "final_answer": "",
            "citations": [],
            "confidence": "",
        }

    async def run(self, context: ChatContext, correlation_id: str) -> dict:
        """
        Run the agent graph in non-stream mode.

        Returns:
            dict with keys:
              - final_answer: str
              - citations: list[str]  — index references like ["1", "2"]
              - confidence: str
              - evidence: list[dict]  — full evidence items for source extraction
        """
        graph = self._build_graph()
        final_state = await graph.ainvoke(self._make_initial_state(context, correlation_id))
        return {
            "final_answer": final_state.get("final_answer", ""),
            "citations": final_state.get("citations", []),
            "confidence": final_state.get("confidence", ""),
            "evidence": final_state.get("evidence", []),
        }

    async def run_stream(
        self, context: ChatContext, correlation_id: str
    ) -> AsyncIterator[tuple[str, dict]]:
        """Run in stream mode, yielding (node_name, node_output) tuples."""
        graph = self._build_graph()
        async for event in graph.astream(self._make_initial_state(context, correlation_id)):
            for node_name, node_output in event.items():
                yield node_name, node_output