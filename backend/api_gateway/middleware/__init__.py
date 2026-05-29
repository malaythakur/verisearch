"""API Gateway middleware package.

Middleware chain order (outermost first = executed first on request):
1. RequestIdMiddleware — generates/propagates X-Request-Id
2. AuthMiddleware — extracts bearer, resolves tenant_id, rejects unauthorized
3. RateLimitMiddleware — checks rate limits (placeholder)
4. PiiRedactMiddleware — redacts PII from query params before logging (placeholder)
"""

from backend.api_gateway.middleware.auth import AuthMiddleware
from backend.api_gateway.middleware.pii_redact import PiiRedactMiddleware
from backend.api_gateway.middleware.rate_limit import RateLimitMiddleware
from backend.api_gateway.middleware.request_id import RequestIdMiddleware

__all__ = [
    "AuthMiddleware",
    "PiiRedactMiddleware",
    "RateLimitMiddleware",
    "RequestIdMiddleware",
]
