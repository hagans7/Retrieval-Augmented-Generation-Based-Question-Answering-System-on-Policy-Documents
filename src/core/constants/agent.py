"""
Constants for the LangGraph agentic orchestration layer.

Node names and SSE event types are the shared vocabulary between
the agent runner and the SSE event emitter. Always reference these
constants — never use string literals in agent or SSE code.
"""

# ── LangGraph Node Names ─────────────────────────────────────
ROUTER_NODE: str = "router"
PLANNER_NODE: str = "planner"
EXECUTOR_NODE: str = "executor"
AUDITOR_NODE: str = "auditor"
GENERATOR_NODE: str = "generator"

# ── Agent Retry Limits ───────────────────────────────────────
MAX_AUDIT_RETRIES: int = 2   # Max times auditor can send flow back to planner
MAX_EXECUTOR_STEPS: int = 5  # Max tool calls executor can make in one invocation

# ── SSE Event Types ──────────────────────────────────────────
# All events emitted to client during a streaming chat request.
# Events 1–5 are emitted without LLM calls (from state parameters).
# Only SSE_EVENT_TOKEN involves LLM streaming output.
SSE_EVENT_CONVERSATION_STARTED: str = "conversation_started"
SSE_EVENT_PLAN_CREATED: str = "plan_created"
SSE_EVENT_TOOL_CALLED: str = "tool_called"
SSE_EVENT_EVIDENCE_FOUND: str = "evidence_found"
SSE_EVENT_GENERATION_STARTED: str = "generation_started"
SSE_EVENT_TOKEN: str = "token"
SSE_EVENT_DONE: str = "done"
SSE_EVENT_ERROR: str = "error"
