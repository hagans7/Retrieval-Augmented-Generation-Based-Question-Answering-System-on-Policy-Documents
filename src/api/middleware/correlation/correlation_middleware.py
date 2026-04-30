"""
Correlation ID middleware.

Attaches a correlation_id to every request for end-to-end log tracing.
Reads X-Correlation-ID from incoming headers; generates UUID if absent.
Adds the ID to the response headers.
"""
from __future__ import annotations
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

_HEADER_NAME = "X-Correlation-ID"


class CorrelationMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        correlation_id = request.headers.get(_HEADER_NAME) or str(uuid.uuid4())
        request.state.correlation_id = correlation_id
        response = await call_next(request)
        response.headers[_HEADER_NAME] = correlation_id
        return response
