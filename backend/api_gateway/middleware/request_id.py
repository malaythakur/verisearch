"""Request ID middleware — generates/propagates X-Request-Id.

Per R15.1, request_id must be 16–64 code points. This middleware:
- Accepts an incoming X-Request-Id header if it meets length requirements
- Generates a UUID-based request ID if none is provided
- Stores request_id in request.state for downstream use
- Sets X-Request-Id on the response header
"""

from __future__ import annotations

import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

_MIN_REQUEST_ID_LEN = 16
_MAX_REQUEST_ID_LEN = 64


def _generate_request_id() -> str:
    """Generate a UUID-based request ID (32 hex chars = 32 code points)."""
    return uuid.uuid4().hex


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Middleware that generates or propagates X-Request-Id on every request.

    If the incoming request has an X-Request-Id header with a valid length
    (16–64 code points per R15.1), it is reused. Otherwise, a new UUID-based
    ID is generated.

    The request_id is stored in request.state.request_id for downstream
    middleware and route handlers.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Check for incoming request ID header
        incoming_id = request.headers.get("x-request-id", "")

        if _MIN_REQUEST_ID_LEN <= len(incoming_id) <= _MAX_REQUEST_ID_LEN:
            request_id = incoming_id
        else:
            request_id = _generate_request_id()

        # Store in request state for downstream use
        request.state.request_id = request_id

        # Call next middleware/handler
        response = await call_next(request)

        # Set response header
        response.headers["X-Request-Id"] = request_id

        return response
