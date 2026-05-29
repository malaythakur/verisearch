"""FastAPI application factory for the API Gateway.

Creates a configured FastAPI instance with the middleware chain:
  request_id → auth → rate_limit → pii_redact → route

Middleware is registered outermost-first, meaning the first middleware added
is the outermost (executed first on request, last on response). In Starlette/FastAPI,
middleware added later wraps earlier middleware, so we add them in reverse order:
  - PII redact (innermost, closest to route handlers)
  - Rate limit
  - Auth
  - Request ID (outermost, first to execute)

This ensures the execution order on request is:
  Request → RequestId → Auth → RateLimit → PiiRedact → Route Handler
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from backend.api_gateway.errors import ErrorHandlingMiddleware, register_exception_handlers
from backend.api_gateway.middleware.auth import AuthMiddleware
from backend.api_gateway.middleware.pii_redact import PiiRedactMiddleware
from backend.api_gateway.middleware.rate_limit import RateLimitMiddleware
from backend.api_gateway.middleware.request_id import RequestIdMiddleware

if TYPE_CHECKING:
    from backend.auth.service import AuthService
    from backend.api_gateway.middleware.rate_limit import RateLimitChecker


def create_app(
    *,
    auth_service: AuthService | None = None,
    rate_limiter: RateLimitChecker | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        auth_service: The AuthService instance for authentication middleware.
            If None, a no-op auth service is used (useful for testing health endpoints).
        rate_limiter: The rate limiter instance for rate limit middleware.
            If None, a pass-through limiter is used (useful for testing without Redis).

    Returns:
        A configured FastAPI application with middleware chain and routes.
    """
    app = FastAPI(
        title="Agentic Research Search Engine",
        version="0.1.0",
        docs_url="/docs",
        openapi_url="/v1/openapi.json",
    )

    # Register routes
    _register_routes(app)

    # Register exception handlers for uniform error responses
    register_exception_handlers(app)

    # Register middleware in reverse order (last added = outermost = first executed)
    # Execution order on request: RequestId → Auth → RateLimit → PiiRedact → ErrorHandling → Handler
    #
    # Starlette processes middleware in LIFO order (last added middleware runs first),
    # so we add them in reverse of desired execution order:
    app.add_middleware(ErrorHandlingMiddleware)
    app.add_middleware(PiiRedactMiddleware)
    app.add_middleware(RateLimitMiddleware, limiter=rate_limiter)

    if auth_service is not None:
        app.add_middleware(AuthMiddleware, auth_service=auth_service)
    else:
        app.add_middleware(AuthMiddleware, auth_service=_NoOpAuthService())

    app.add_middleware(RequestIdMiddleware)

    return app


def _register_routes(app: FastAPI) -> None:
    """Register API routes on the FastAPI application."""
    from backend.api_gateway.routes import router as v1_router

    app.include_router(v1_router)

    @app.get("/health")
    async def health_check() -> JSONResponse:
        """Health check endpoint — returns 200 with status info."""
        return JSONResponse(
            status_code=200,
            content={"status": "healthy", "version": "0.1.0"},
        )


class _NoOpAuthService:
    """A no-op auth service that always authenticates successfully.

    Used when no real AuthService is provided (e.g., for testing health endpoints
    without a database connection).
    """

    async def authenticate(self, headers: dict[str, str], **kwargs) -> object:  # noqa: ANN003, ANN001
        """Always return a successful auth result."""
        from backend.auth.service import AuthResult

        return AuthResult(tenant_id="test-tenant", api_key_id="test-key")
