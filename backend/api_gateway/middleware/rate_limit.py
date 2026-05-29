"""Rate limit middleware — enforces per-tenant rate limits and adds headers.

Integrates with the TokenBucketRateLimiter to enforce per-(tenant_id, endpoint)
rate limits. Adds X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset
headers on every response (R14.4).

If no real limiter is configured (e.g., in tests), uses a pass-through that
always allows requests with placeholder values.
"""

from __future__ import annotations

from typing import Protocol

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response


# Paths that skip rate limiting (health, docs, OpenAPI)
_SKIP_RATE_LIMIT_PATHS: set[str] = {
    "/health",
    "/docs",
    "/redoc",
    "/v1/openapi.json",
    "/openapi.json",
}


class RateLimitChecker(Protocol):
    """Protocol for rate limit checkers.

    Allows the middleware to work with both the real TokenBucketRateLimiter
    and a pass-through implementation for testing.
    """

    async def check_rate_limit(
        self,
        tenant_id: str,
        endpoint: str,
        limit_per_minute: int | None = None,
    ) -> object:
        """Check rate limit and return a result with allowed, limit, remaining, reset_at."""
        ...


class PassThroughRateLimiter:
    """A pass-through rate limiter that always allows requests.

    Used when no real Redis-backed limiter is configured (e.g., in tests
    or local development without Redis).
    """

    _DEFAULT_LIMIT = 1000
    _DEFAULT_REMAINING = 999
    _DEFAULT_RESET = 0

    async def check_rate_limit(
        self,
        tenant_id: str,
        endpoint: str,
        limit_per_minute: int | None = None,
    ) -> _PassThroughResult:
        """Always allow with placeholder values."""
        import time

        limit = limit_per_minute if limit_per_minute is not None else self._DEFAULT_LIMIT
        return _PassThroughResult(
            allowed=True,
            limit=limit,
            remaining=limit - 1,
            reset_at=int(time.time()) + 60,
            retry_after=None,
        )


class _PassThroughResult:
    """Result object for the pass-through limiter."""

    __slots__ = ("allowed", "limit", "remaining", "reset_at", "retry_after")

    def __init__(
        self,
        *,
        allowed: bool,
        limit: int,
        remaining: int,
        reset_at: int,
        retry_after: int | None,
    ) -> None:
        self.allowed = allowed
        self.limit = limit
        self.remaining = remaining
        self.reset_at = reset_at
        self.retry_after = retry_after


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limit middleware that enforces per-tenant rate limits.

    On each request to a protected endpoint:
    - Extracts tenant_id from request.state (set by auth middleware)
    - Calls the rate limiter to check/consume a token
    - If allowed: sets request.state.rate_limit_checked = True and adds
      X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset headers
    - If denied: returns 429 with Retry-After header

    Skips rate limiting for health/docs endpoints.

    Args:
        app: The ASGI application.
        limiter: A rate limit checker instance. If None, uses a pass-through
            that always allows requests (useful for testing without Redis).
    """

    def __init__(self, app, limiter: RateLimitChecker | None = None) -> None:  # noqa: ANN001
        super().__init__(app)
        self._limiter: RateLimitChecker = limiter if limiter is not None else PassThroughRateLimiter()

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Skip rate limiting for health/docs endpoints
        if request.url.path in _SKIP_RATE_LIMIT_PATHS:
            return await call_next(request)

        # Extract tenant_id from request state (set by auth middleware upstream)
        tenant_id = getattr(request.state, "tenant_id", None)
        if tenant_id is None:
            # If no tenant_id is set (e.g., auth failed or skipped), pass through
            # Auth middleware will have already rejected unauthorized requests
            return await call_next(request)

        # Determine the endpoint for rate limiting
        endpoint = request.url.path

        # Check rate limit
        result = await self._limiter.check_rate_limit(tenant_id, endpoint)

        if not result.allowed:
            # Return 429 Too Many Requests with rate limit headers
            retry_after = result.retry_after if result.retry_after is not None else 60
            # Clamp retry_after to [1, 3600] per R14.1
            retry_after = max(1, min(3600, retry_after))

            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "code": "rate_limited",
                        "message": "Rate limit exceeded. Please retry after the specified time.",
                    }
                },
                headers={
                    "X-RateLimit-Limit": str(result.limit),
                    "X-RateLimit-Remaining": str(result.remaining),
                    "X-RateLimit-Reset": str(result.reset_at),
                    "Retry-After": str(retry_after),
                },
            )

        # Mark that rate limit middleware has been executed
        request.state.rate_limit_checked = True

        # Proceed with the request
        response = await call_next(request)

        # Add rate limit headers to the response (R14.4)
        response.headers["X-RateLimit-Limit"] = str(result.limit)
        response.headers["X-RateLimit-Remaining"] = str(result.remaining)
        response.headers["X-RateLimit-Reset"] = str(result.reset_at)

        return response
