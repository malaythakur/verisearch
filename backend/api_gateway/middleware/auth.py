"""Auth middleware — extracts bearer token, resolves tenant_id, rejects unauthorized.

Per R13.1, no business handler is invoked before Auth_Service resolves a tenant_id.
This middleware:
- Skips auth for health check and OpenAPI endpoints
- Extracts the Authorization header
- Calls AuthService.authenticate()
- On success: stores tenant_id and api_key_id in request.state
- On failure: returns 401 with uniform error shape
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

if TYPE_CHECKING:
    from backend.auth.service import AuthService

# Paths that skip authentication
_PUBLIC_PATHS: set[str] = {
    "/health",
    "/v1/openapi.json",
    "/docs",
    "/openapi.json",
    "/redoc",
}


class AuthMiddleware(BaseHTTPMiddleware):
    """Middleware that authenticates requests via the AuthService.

    On successful authentication, stores tenant_id and api_key_id in
    request.state for downstream handlers.

    On failure, returns a 401 JSON response with a uniform error shape:
    {"error": {"code": "<error_code>", "message": "<description>"}}

    Skips authentication for health check and documentation endpoints.
    """

    def __init__(self, app, auth_service: AuthService) -> None:  # noqa: ANN001
        super().__init__(app)
        self._auth_service = auth_service

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Skip auth for public endpoints
        if request.url.path in _PUBLIC_PATHS:
            return await call_next(request)

        # Get request_id from state (set by RequestIdMiddleware upstream)
        request_id = getattr(request.state, "request_id", "")

        # Build headers dict for AuthService
        headers = {k.lower(): v for k, v in request.headers.items()}

        # Authenticate
        result = await self._auth_service.authenticate(
            headers,
            request_id=request_id,
            resource=request.url.path,
        )

        # Check if authentication failed
        from backend.auth.service import AuthError

        if isinstance(result, AuthError):
            return JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "code": result.code,
                        "message": result.message,
                    }
                },
            )

        # Store auth context in request state for downstream use
        request.state.tenant_id = result.tenant_id
        request.state.api_key_id = result.api_key_id

        return await call_next(request)
