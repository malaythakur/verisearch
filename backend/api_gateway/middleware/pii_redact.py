"""PII redaction middleware — placeholder stub.

This is a placeholder that passes through all requests. The actual
PII redaction implementation is in Task 6.

When fully implemented, this middleware will redact PII patterns
(email, phone, SSN, EU national ID, credit card) from query parameters
before they reach logging/audit systems.
"""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response


class PiiRedactMiddleware(BaseHTTPMiddleware):
    """Placeholder PII redaction middleware.

    Passes through all requests without modification. The actual PII
    detection and redaction per R15.2 will be added in Task 6.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Mark that PII redaction middleware has been executed (for testing middleware order)
        request.state.pii_redacted = True

        return await call_next(request)
