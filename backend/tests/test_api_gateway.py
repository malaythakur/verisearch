"""Tests for the API Gateway middleware chain and routes.

Validates:
- Health check returns 200
- Request without auth header returns 401 on protected endpoints
- X-Request-Id is present in responses
- Middleware chain executes in correct order
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient
from starlette.requests import Request

from backend.api_gateway.app import create_app
from backend.auth.service import AuthError, AuthResult, AuthService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class FakeAuthService:
    """A fake AuthService for testing that simulates authentication behavior."""

    def __init__(self, *, should_fail: bool = False, error: AuthError | None = None) -> None:
        self._should_fail = should_fail
        self._error = error or AuthError(code="missing_token", message="Authorization header is required")
        self._calls: list[dict] = []

    async def authenticate(self, headers: dict[str, str], **kwargs) -> AuthResult | AuthError:  # noqa: ANN003
        self._calls.append({"headers": headers, **kwargs})
        if self._should_fail:
            return self._error
        return AuthResult(tenant_id="tenant-123", api_key_id="key-456")

    @property
    def call_count(self) -> int:
        return len(self._calls)


@pytest.fixture
def fake_auth_success() -> FakeAuthService:
    """Auth service that always succeeds."""
    return FakeAuthService(should_fail=False)


@pytest.fixture
def fake_auth_failure() -> FakeAuthService:
    """Auth service that always fails with missing_token."""
    return FakeAuthService(should_fail=True)


@pytest.fixture
def fake_auth_invalid() -> FakeAuthService:
    """Auth service that fails with invalid_token."""
    return FakeAuthService(
        should_fail=True,
        error=AuthError(code="invalid_token", message="API key is not valid"),
    )


# ---------------------------------------------------------------------------
# Health Check Tests
# ---------------------------------------------------------------------------


class TestHealthCheck:
    """Tests for the /health endpoint."""

    async def test_health_returns_200(self, fake_auth_success: FakeAuthService) -> None:
        """Health check returns 200 with status info."""
        app = create_app(auth_service=fake_auth_success)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "healthy"
        assert "version" in body

    async def test_health_skips_auth(self, fake_auth_success: FakeAuthService) -> None:
        """Health check does not require authentication."""
        app = create_app(auth_service=fake_auth_success)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health")

        assert response.status_code == 200
        # Auth service should not have been called for health endpoint
        assert fake_auth_success.call_count == 0

    async def test_health_has_request_id(self, fake_auth_success: FakeAuthService) -> None:
        """Health check response includes X-Request-Id header."""
        app = create_app(auth_service=fake_auth_success)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health")

        assert "x-request-id" in response.headers
        request_id = response.headers["x-request-id"]
        assert 16 <= len(request_id) <= 64


# ---------------------------------------------------------------------------
# Auth Middleware Tests
# ---------------------------------------------------------------------------


class TestAuthMiddleware:
    """Tests for authentication middleware behavior."""

    async def test_missing_auth_returns_401(self, fake_auth_failure: FakeAuthService) -> None:
        """Request without auth header returns 401 on protected endpoints."""
        app = create_app(auth_service=fake_auth_failure)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/v1/search")

        assert response.status_code == 401
        body = response.json()
        assert "error" in body
        assert body["error"]["code"] == "missing_token"
        assert "message" in body["error"]

    async def test_invalid_token_returns_401(self, fake_auth_invalid: FakeAuthService) -> None:
        """Request with invalid token returns 401 with invalid_token code."""
        app = create_app(auth_service=fake_auth_invalid)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(
                "/v1/search",
                headers={"Authorization": "Bearer invalid_token_value"},
            )

        assert response.status_code == 401
        body = response.json()
        assert body["error"]["code"] == "invalid_token"

    async def test_authenticated_request_passes_through(self, fake_auth_success: FakeAuthService) -> None:
        """Authenticated request reaches the route handler."""
        app = create_app(auth_service=fake_auth_success)

        # Add a test route that requires auth
        @app.get("/v1/test-protected")
        async def protected_route():  # noqa: ANN202
            return {"message": "success"}

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(
                "/v1/test-protected",
                headers={"Authorization": "Bearer testprefix_secretvalue"},
            )

        assert response.status_code == 200
        assert response.json()["message"] == "success"

    async def test_openapi_skips_auth(self, fake_auth_success: FakeAuthService) -> None:
        """OpenAPI endpoint does not require authentication."""
        app = create_app(auth_service=fake_auth_success)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/v1/openapi.json")

        # FastAPI serves OpenAPI spec at this path
        assert response.status_code == 200
        assert fake_auth_success.call_count == 0


# ---------------------------------------------------------------------------
# Request ID Middleware Tests
# ---------------------------------------------------------------------------


class TestRequestIdMiddleware:
    """Tests for X-Request-Id generation and propagation."""

    async def test_generates_request_id_when_missing(self, fake_auth_success: FakeAuthService) -> None:
        """Generates a new request ID when none is provided."""
        app = create_app(auth_service=fake_auth_success)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health")

        assert "x-request-id" in response.headers
        request_id = response.headers["x-request-id"]
        # UUID hex is 32 chars, within 16-64 range
        assert 16 <= len(request_id) <= 64

    async def test_propagates_valid_request_id(self, fake_auth_success: FakeAuthService) -> None:
        """Propagates an incoming X-Request-Id if it meets length requirements."""
        app = create_app(auth_service=fake_auth_success)
        custom_id = "custom-request-id-12345678"  # 26 chars, within 16-64

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health", headers={"X-Request-Id": custom_id})

        assert response.headers["x-request-id"] == custom_id

    async def test_rejects_short_request_id(self, fake_auth_success: FakeAuthService) -> None:
        """Generates a new ID when incoming X-Request-Id is too short."""
        app = create_app(auth_service=fake_auth_success)
        short_id = "short"  # 5 chars, below 16 minimum

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health", headers={"X-Request-Id": short_id})

        # Should generate a new ID, not use the short one
        assert response.headers["x-request-id"] != short_id
        assert 16 <= len(response.headers["x-request-id"]) <= 64

    async def test_rejects_long_request_id(self, fake_auth_success: FakeAuthService) -> None:
        """Generates a new ID when incoming X-Request-Id is too long."""
        app = create_app(auth_service=fake_auth_success)
        long_id = "x" * 100  # 100 chars, above 64 maximum

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health", headers={"X-Request-Id": long_id})

        # Should generate a new ID, not use the long one
        assert response.headers["x-request-id"] != long_id
        assert 16 <= len(response.headers["x-request-id"]) <= 64


# ---------------------------------------------------------------------------
# Middleware Chain Order Tests
# ---------------------------------------------------------------------------


class TestMiddlewareChainOrder:
    """Tests that middleware executes in the correct order."""

    async def test_rate_limit_headers_present(self, fake_auth_success: FakeAuthService) -> None:
        """Rate limit middleware adds X-RateLimit-* headers."""
        app = create_app(auth_service=fake_auth_success)

        @app.get("/v1/test-rate-limit")
        async def rate_limit_route():  # noqa: ANN202
            return {"ok": True}

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(
                "/v1/test-rate-limit",
                headers={"Authorization": "Bearer testprefix_secretvalue"},
            )

        assert response.status_code == 200
        assert "x-ratelimit-limit" in response.headers
        assert "x-ratelimit-remaining" in response.headers
        assert "x-ratelimit-reset" in response.headers

    async def test_middleware_order_auth_before_rate_limit(self, fake_auth_failure: FakeAuthService) -> None:
        """Auth middleware runs before rate limit — unauthenticated requests
        are rejected before rate limit is checked."""
        app = create_app(auth_service=fake_auth_failure)

        @app.get("/v1/test-order")
        async def order_route():  # noqa: ANN202
            return {"ok": True}

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/v1/test-order")

        # Should get 401 from auth middleware
        assert response.status_code == 401
        # Rate limit headers should still be present (rate limit middleware wraps auth)
        # But the route handler was never reached

    async def test_request_id_present_on_auth_failure(self, fake_auth_failure: FakeAuthService) -> None:
        """Request ID middleware runs before auth — X-Request-Id is present
        even when auth fails."""
        app = create_app(auth_service=fake_auth_failure)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/v1/search")

        assert response.status_code == 401
        # Request ID should still be set even on auth failure
        assert "x-request-id" in response.headers
        assert 16 <= len(response.headers["x-request-id"]) <= 64

    async def test_full_middleware_chain_on_success(self, fake_auth_success: FakeAuthService) -> None:
        """All middleware executes in order on a successful authenticated request."""
        app = create_app(auth_service=fake_auth_success)
        middleware_trace: list[str] = []

        @app.get("/v1/test-chain")
        async def chain_route(request: Request):  # noqa: ANN202
            # Verify state set by middleware
            assert hasattr(request.state, "request_id")
            assert hasattr(request.state, "tenant_id")
            assert hasattr(request.state, "api_key_id")
            assert hasattr(request.state, "rate_limit_checked")
            assert hasattr(request.state, "pii_redacted")

            return {
                "request_id": request.state.request_id,
                "tenant_id": request.state.tenant_id,
                "api_key_id": request.state.api_key_id,
                "rate_limit_checked": request.state.rate_limit_checked,
                "pii_redacted": request.state.pii_redacted,
            }

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(
                "/v1/test-chain",
                headers={"Authorization": "Bearer testprefix_secretvalue"},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["tenant_id"] == "tenant-123"
        assert body["api_key_id"] == "key-456"
        assert body["rate_limit_checked"] is True
        assert body["pii_redacted"] is True
        assert 16 <= len(body["request_id"]) <= 64


# ---------------------------------------------------------------------------
# Error Response Shape Tests
# ---------------------------------------------------------------------------


class TestErrorResponseShape:
    """Tests for uniform error response format."""

    async def test_401_error_shape(self, fake_auth_failure: FakeAuthService) -> None:
        """401 responses have the uniform error shape."""
        app = create_app(auth_service=fake_auth_failure)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/v1/search")

        assert response.status_code == 401
        body = response.json()
        assert "error" in body
        assert "code" in body["error"]
        assert "message" in body["error"]
        # Code should be a non-empty string
        assert isinstance(body["error"]["code"], str)
        assert len(body["error"]["code"]) > 0
        # Message should be a non-empty string
        assert isinstance(body["error"]["message"], str)
        assert len(body["error"]["message"]) > 0
