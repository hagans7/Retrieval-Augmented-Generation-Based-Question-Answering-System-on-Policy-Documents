"""
Contract tests for BaseObservabilityClient → Langfuse backend.

Requires a running Langfuse instance (docker-compose up langfuse langfuse-db)
AND LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY set in .env.

Skip condition: keys absent or Langfuse unreachable.

What is tested:
- LangfuseObservabilityClient satisfies BaseObservabilityClient contract
- start_trace / end_trace lifecycle does not raise
- start_generation / end_generation does not raise
- start_span / end_span does not raise
- score_trace does not raise
- log_event does not raise
- flush does not raise
- Full agent lifecycle trace (trace → spans → gens → scores → flush)

To run:
    pytest tests/contracts/test_observability_contract.py -v
"""
from __future__ import annotations

import pytest


def _should_skip() -> tuple[bool, str]:
    try:
        from src.core.config.settings import settings
        if not settings.LANGFUSE_SECRET_KEY or not settings.LANGFUSE_PUBLIC_KEY:
            return True, "LANGFUSE keys not set in .env"
        import httpx
        try:
            r = httpx.get(f"{settings.LANGFUSE_HOST}/api/public/health", timeout=3)
            if r.status_code not in (200, 401):
                return True, f"Langfuse not reachable at {settings.LANGFUSE_HOST}"
        except Exception:
            return True, f"Langfuse not reachable at {settings.LANGFUSE_HOST}"
        return False, ""
    except Exception as exc:
        return True, f"Settings load failed: {exc}"


_SKIP, _REASON = _should_skip()
pytestmark = pytest.mark.skipif(_SKIP, reason=_REASON)


@pytest.fixture(scope="module")
def obs_client():
    from src.core.config.settings import settings
    from src.core.observability.langfuse_client import LangfuseObservabilityClient
    return LangfuseObservabilityClient(
        public_key=settings.LANGFUSE_PUBLIC_KEY,
        secret_key=settings.LANGFUSE_SECRET_KEY,
        host=settings.LANGFUSE_HOST,
    )


# ── Interface contract ─────────────────────────────────────────────────────────

def test_implements_base_interface(obs_client):
    from src.interfaces.clients.base_observability_client import BaseObservabilityClient
    assert isinstance(obs_client, BaseObservabilityClient)


def test_enabled_is_true(obs_client):
    assert obs_client.enabled is True


# ── Lifecycle — does not raise ─────────────────────────────────────────────────

def test_trace_lifecycle(obs_client):
    trace = obs_client.start_trace(
        name="contract_test_trace",
        session_id="contract-session",
        input={"query": "test query"},
        tags=["contract-test"],
    )
    assert trace is not None
    obs_client.end_trace(trace, output={"answer": "test answer"})


def test_generation_lifecycle(obs_client):
    trace = obs_client.start_trace("contract_test_generation")
    gen = obs_client.start_generation(
        trace, "test_llm_call",
        model="qwen/qwen-turbo",
        input=[{"role": "user", "content": "test"}],
    )
    assert gen is not None
    obs_client.end_generation(gen, output={"answer": "ok"},
                              usage={"input": 10, "output": 5, "total": 15})
    obs_client.end_trace(trace)


def test_span_lifecycle(obs_client):
    trace = obs_client.start_trace("contract_test_span")
    span = obs_client.start_span(trace, "test_retrieval",
                                 input={"query": "bpjs"})
    assert span is not None
    obs_client.end_span(span, output={"evidence_count": 1})
    obs_client.end_trace(trace)


def test_log_event(obs_client):
    trace = obs_client.start_trace("contract_test_event")
    obs_client.log_event(trace, "auditor_decision",
                         input={"evidence_count": 3},
                         output={"verdict": "sufficient", "score": 0.85})
    obs_client.end_trace(trace)


def test_score_trace(obs_client):
    trace = obs_client.start_trace("contract_test_score")
    obs_client.score_trace(trace, "top_evidence_score",
                           value=0.85, comment="contract test score")
    obs_client.end_trace(trace)


def test_flush(obs_client):
    obs_client.flush()


# ── Full agent lifecycle ───────────────────────────────────────────────────────

def test_full_agent_request_lifecycle(obs_client):
    """
    Simulate a complete chat request trace as agent_runner would produce.

    Expected trace structure in Langfuse UI:
      chat_request (trace)
      ├── router_llm (generation)
      ├── executor_llm (generation)
      │   └── retrieval (span)
      ├── auditor_llm (generation)
      │   └── auditor_decision (event)
      └── generator_llm (generation)
    """
    trace = obs_client.start_trace(
        name="chat_request",
        session_id="contract-conv-1",
        input={"query": "jelaskan BPJS", "model": "qwen/qwen-turbo"},
        tags=["contract-test", "full-lifecycle"],
    )

    # Router
    router_gen = obs_client.start_generation(trace, "router_llm",
        model="qwen/qwen-turbo",
        input=[{"role": "system", "content": "classify"}, {"role": "user", "content": "bpjs"}])
    obs_client.end_generation(router_gen, output={"query_type": "retrieval"})

    # Executor — retrieval span
    exec_gen = obs_client.start_generation(trace, "executor_llm", model="qwen/qwen-turbo",
        input=[{"role": "user", "content": "plan step"}])
    obs_client.end_generation(exec_gen, output={"query": "bpjs kesehatan", "top_k": 5})

    retrieval_span = obs_client.start_span(trace, "retrieval",
        input={"query": "bpjs kesehatan", "top_k": 5})
    obs_client.end_span(retrieval_span, output={
        "raw_results_count": 5, "evidence_added": 1})

    # Auditor
    auditor_gen = obs_client.start_generation(trace, "auditor_llm", model="qwen/qwen-turbo",
        input=[{"role": "system", "content": "audit"}, {"role": "user", "content": "eval"}],
        metadata={"loop_count": 0, "evidence_count": 1})
    obs_client.end_generation(auditor_gen,
        output={"verdict": "sufficient", "top_evidence_score": 0.85},
        metadata={"verdict": "sufficient", "top_score": 0.85, "loop_count": 0})
    obs_client.log_event(trace, "auditor_decision",
        input={"evidence_count": 1, "loop_count": 0},
        output={"verdict": "sufficient", "top_score": 0.85})
    obs_client.score_trace(trace, "auditor_evidence_score", value=0.85,
                           comment="loop=0 verdict=sufficient")

    # Generator
    gen_gen = obs_client.start_generation(trace, "generator_llm", model="qwen/qwen-turbo",
        input=[{"role": "system", "content": "generate"}, {"role": "user", "content": "bpjs"}],
        metadata={"evidence_count": 1, "top_score": 0.85})
    obs_client.end_generation(gen_gen,
        output={"answer": "BPJS Kesehatan adalah...", "citations": ["1"], "confidence": "high"},
        metadata={"tool_call_success": True, "confidence": "high", "citations_count": 1})

    # End trace
    obs_client.end_trace(trace, output={
        "answer_preview": "BPJS Kesehatan adalah...",
        "citations_count": 1,
        "evidence_count": 1,
        "top_reranker_score": 0.85,
        "confidence": "high",
    })
    obs_client.score_trace(trace, "top_evidence_score", value=0.85,
                           comment="Reranker score of best evidence chunk")
    obs_client.flush()