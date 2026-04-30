"""
Contract tests for BaseLLMClient (OpenAI-compatible, OpenRouter backend).

Reads configuration from src.core.config.settings (loads .env file)
NOT from os.environ directly — avoids stale system environment pollution.

SKIP: if LLM_API_KEY is empty in .env.

Model name must NOT have 'openrouter/' prefix:
  Correct:   qwen/qwen-turbo
  Wrong:     openrouter/qwen/qwen-turbo  ← causes HTTP 400

To run:
    pytest tests/contracts/test_llm_client_contract.py -v
"""
from __future__ import annotations

import pytest


def _should_skip() -> tuple[bool, str]:
    try:
        from src.core.config.settings import settings
        if not settings.LLM_API_KEY:
            return True, "LLM_API_KEY is empty in .env"
        if settings.LLM_DEFAULT_MODEL.startswith("openrouter/"):
            return True, (
                f"LLM_DEFAULT_MODEL='{settings.LLM_DEFAULT_MODEL}' has 'openrouter/' prefix. "
                f"OpenRouter rejects this. Change to: "
                f"{settings.LLM_DEFAULT_MODEL.replace('openrouter/', '')}"
            )
        return False, ""
    except Exception as exc:
        return True, f"Could not load settings: {exc}"


_SKIP, _REASON = _should_skip()
pytestmark = pytest.mark.skipif(_SKIP, reason=_REASON)


@pytest.fixture
def llm_client():
    """Build LLM client from .env settings."""
    from src.clients.llm.llm_client import OpenAICompatibleLLMClient
    from src.core.config.settings import settings
    return OpenAICompatibleLLMClient(
        base_url=settings.LLM_BASE_URL,
        api_key=settings.LLM_API_KEY,
        default_model=settings.LLM_DEFAULT_MODEL,
        timeout=30,
        max_retries=1,
    )


@pytest.fixture
def model_name():
    from src.core.config.settings import settings
    return settings.LLM_DEFAULT_MODEL


@pytest.mark.asyncio
async def test_generate_returns_non_empty_string(llm_client, model_name):
    messages = [{"role": "user", "content": "Say 'hello' in one word."}]
    result = await llm_client.generate(messages=messages, model=model_name)
    assert isinstance(result, str)
    assert len(result.strip()) > 0


@pytest.mark.asyncio
async def test_generate_stream_yields_strings(llm_client, model_name):
    messages = [{"role": "user", "content": "Count to 3."}]
    chunks = []
    async for chunk in llm_client.generate_stream(messages=messages, model=model_name):
        assert isinstance(chunk, str)
        chunks.append(chunk)
    assert len(chunks) > 0


@pytest.mark.asyncio
async def test_health_check_returns_bool(llm_client):
    result = await llm_client.health_check()
    assert isinstance(result, bool)


@pytest.mark.asyncio
async def test_generate_with_tool_call_returns_dict_or_none(llm_client, model_name):
    from src.prompts.agents.router_prompt import TOOL_SCHEMA, SYSTEM_PROMPT
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Apa itu wanprestasi?"},
    ]
    result = await llm_client.generate_with_tool_call(
        messages=messages,
        model=model_name,
        tools=TOOL_SCHEMA,
    )
    assert result is None or isinstance(result, dict)