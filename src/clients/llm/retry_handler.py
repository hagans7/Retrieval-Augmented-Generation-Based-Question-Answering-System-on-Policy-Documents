"""
LLM retry handler — tenacity-based retry configuration.

Defines when and how to retry failed LLM requests.
Separating retry logic from request execution keeps each file focused.
"""

from __future__ import annotations

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
    RetryCallState,
)

from src.core.logging.logger import get_logger

logger = get_logger(__name__)


def _is_retryable(exc: BaseException) -> bool:
    """
    Determine whether an exception warrants a retry attempt.

    Retryable: network timeouts, connection errors, HTTP 429, HTTP 503.
    Not retryable: HTTP 400, 401, 403, 404 (client errors — retrying won't help).

    Args:
        exc: The exception to evaluate.

    Returns:
        True if the request should be retried.
    """
    if isinstance(exc, httpx.ConnectError):
        return True
    if isinstance(exc, httpx.TimeoutException):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 503)
    return False


def _log_retry_attempt(retry_state: RetryCallState) -> None:
    """Log a warning each time a retry is triggered."""
    logger.warning(
        "LLM request failed, retrying",
        extra={
            "attempt": retry_state.attempt_number,
            "error": str(retry_state.outcome.exception()) if retry_state.outcome else "unknown",
        },
    )


def build_llm_retry(max_retries: int):
    """
    Build a tenacity retry decorator for LLM calls.

    Strategy: exponential backoff with jitter.
    Base wait doubles each attempt: 1s → 2s → 4s + random jitter.

    Args:
        max_retries: Maximum number of attempts (including first attempt).

    Returns:
        A tenacity retry decorator configured for LLM calls.
    """
    return retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential_jitter(initial=1, max=30),
        after=_log_retry_attempt,
        reraise=True,
    )
