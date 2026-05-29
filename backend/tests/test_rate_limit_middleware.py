"""Tests for the rate limit middleware (Task 7.2).

Validates:
- X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset headers on every response (R14.4)
- Rate limiting skipped for health/docs endpoints
- 429 response when rate limit is exceeded
- Pass-through limiter works when no real limiter is configured
- Integration with TokenBucketRateLimiter result shape
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from backend.api_gateway.app import create_app
from backend.api_gateway.middleware.rate_limit import PassThroughRateLimiter, RateLimitMiddleware
from backend.auth.service import AuthError, AuthResult
from backend.rate_limiter.token_bucket import RateLimitResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class FakeAuthService:
    """A fake AuthService for testing that always succeeds."""

    def __init__(self, *, should_fail: bool = False) -> None:
        self._should_fail = should_fail

    async def authenticate(self, headers: dict[str, str], **kwargs) -> AuthResult | AuthError:  # noqa: ANN003
        if self._should_fail:
            return AuthError(code="missing_token", message="Authorization header is required")
        return AuthResult(tenant_id="tenant-123", api_key_id="key-456")


class FakeRateLimiter:
    """A fake rate limiter that returns configurable results."""

    def __init__(
        self,
        *,
        allowed: bool = True,
        limit: int = 100,
        remaining: int = 99,
        reset_at: int = 1700000000,
        retry_after: int | None = None,
    ) -> None:
        self._result = RateLimitResult(
            allowed=allowed,
            limit=limit,
            remaining=remaining,
            reset_at=reset_at,
            retry_after=retry_after,
        )
        self.calls: list[tuple[str, str]] = []

    async def check_rate_limit(
        self,
        tenant_id: str,
        endpoint: str,
        limit_per_minute: int | None = None,
    ) -> RateLimitResult:
        self.calls.append((tenant_id, endpoint))
        return self._result


@pytest.fixture
def fake_auth() -> FakeAuthService:
    """Auth service that always succeeds."""
    return FakeAuthService()


@pytest.fixture
def fake_auth_failure() -> FakeAuthService:
    """Auth service that always fails."""
    return FakeAuthService(should_fail=True)


@pytest.fixture
def allowed_limiter() -> FakeRateLimiter:
    """Rate limiter that always allows."""
    return FakeRateLimiter(allowed=True, limit=100, remaining=95, reset_at=1700000060)


@pytest.fixture
def denied_limiter() -> FakeRateLimiter:
    """Rate limiter that always denies."""
    return FakeRateLimiter(
        allowed=False, limit=100, remaining=0, reset_at=1700000060, retry_after=30
    )


# ---------------------------------------------------------------------------
# Tests: X-RateLimit-* headers on allowed responses (R14.4)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRateLimitHeaders:
    """Tests that X-RateLimit-* headers are present on every response."""

    async def test_headers_present_on_allowed_response(
        self, fake_auth: FakeAuthService, allowed_limiter: FakeRateLimiter
    ) -> None:
        """X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset headers
        are present on successful responses."""
        app = create_app(auth_service=fake_auth, rate_limiter=allowed_limiter)

        @app.get("/v1/test-endpoint")
        async def test_route():  # noqa: ANN202
            return {"ok": True}

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(
                "/v1/test-endpoint",
                headers={"Authorization": "Bearer testprefix_secretvalue"},
            )

        assert response.status_code == 200
        assert response.headers["x-ratelimit-limit"] == "100"
        assert response.headers["x-ratelimit-remaining"] == "95"
        assert response.headers["x-ratelimit-reset"] == "1700000060"

    async def test_headers_reflect_limiter_values(
        self, fake_auth: FakeAuthService
    ) -> None:
        """Headers reflect the actual values returned by the rate limiter."""
        limiter = FakeRateLimiter(allowed=True, limit=60, remaining=42, reset_at=1700001234)
        app = create_app(auth_service=fake_auth, rate_limiter=limiter)

        @app.get("/v1/test-values")
        async def test_route():  # noqa: ANN202
            return {"ok": True}

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(
                "/v1/test-values",
                headers={"Authorization": "Bearer testprefix_secretvalue"},
            )

        assert response.status_code == 200
        assert response.headers["x-ratelimit-limit"] == "60"
        assert response.headers["x-ratelimit-remaining"] == "42"
        assert response.headers["x-ratelimit-reset"] == "1700001234"

    async def test_limiter_receives_correct_tenant_and_endpoint(
        self, fake_auth: FakeAuthService, allowed_limiter: FakeRateLimiter
    ) -> None:
        """Rate limiter is called with the correct tenant_id and endpoint."""
        app = create_app(auth_service=fake_auth, rate_limiter=allowed_limiter)

        @app.get("/v1/search")
        async def search_route():  # noqa: ANN202
            return {"results": []}

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.get(
                "/v1/search",
                headers={"Authorization": "Bearer testprefix_secretvalue"},
            )

        assert len(allowed_limiter.calls) == 1
        tenant_id, endpoint = allowed_limiter.calls[0]
        assert tenant_id == "tenant-123"
        assert endpoint == "/v1/search"


# ---------------------------------------------------------------------------
# Tests: Skip rate limiting for health/docs endpoints
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRateLimitSkipPaths:
    """Tests that rate limiting is skipped for health/docs endpoints."""

    async def test_health_endpoint_skips_rate_limit(
        self, fake_auth: FakeAuthService, allowed_limiter: FakeRateLimiter
    ) -> None:
        """Health endpoint does not invoke the rate limiter."""
        app = create_app(auth_service=fake_auth, rate_limiter=allowed_limiter)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health")

        assert response.status_code == 200
        # Rate limiter should not have been called
        assert len(allowed_limiter.calls) == 0
        # Rate limit headers should NOT be present on skipped paths
        assert "x-ratelimit-limit" not in response.headers

    async def test_docs_endpoint_skips_rate_limit(
        self, fake_auth: FakeAuthService, allowed_limiter: FakeRateLimiter
    ) -> None:
        """Docs endpoint does not invoke the rate limiter."""
        app = create_app(auth_service=fake_auth, rate_limiter=allowed_limiter)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/docs")

        # Rate limiter should not have been called
        assert len(allowed_limiter.calls) == 0


# ---------------------------------------------------------------------------
# Tests: 429 response when rate limit exceeded
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRateLimitDenied:
    """Tests that denied requests return 429 with proper headers."""

    async def test_denied_returns_429(
        self, fake_auth: FakeAuthService, denied_limiter: FakeRateLimiter
    ) -> None:
        """When rate limit is exceeded, returns 429 Too Many Requests."""
        app = create_app(auth_service=fake_auth, rate_limiter=denied_limiter)

        @app.get("/v1/test-denied")
        async def test_route():  # noqa: ANN202
            return {"ok": True}

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(
                "/v1/test-denied",
                headers={"Authorization": "Bearer testprefix_secretvalue"},
            )

        assert response.status_code == 429

    async def test_denied_has_rate_limit_headers(
        self, fake_auth: FakeAuthService, denied_limiter: FakeRateLimiter
    ) -> None:
        """429 response includes X-RateLimit-* headers."""
        app = create_app(auth_service=fake_auth, rate_limiter=denied_limiter)

        @app.get("/v1/test-denied-headers")
        async def test_route():  # noqa: ANN202
            return {"ok": True}

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(
                "/v1/test-denied-headers",
                headers={"Authorization": "Bearer testprefix_secretvalue"},
            )

        assert response.status_code == 429
        assert response.headers["x-ratelimit-limit"] == "100"
        assert response.headers["x-ratelimit-remaining"] == "0"
        assert response.headers["x-ratelimit-reset"] == "1700000060"

    async def test_denied_has_retry_after_header(
        self, fake_auth: FakeAuthService, denied_limiter: FakeRateLimiter
    ) -> None:
        """429 response includes Retry-After header."""
        app = create_app(auth_service=fake_auth, rate_limiter=denied_limiter)

        @app.get("/v1/test-retry-after")
        async def test_route():  # noqa: ANN202
            return {"ok": True}

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(
                "/v1/test-retry-after",
                headers={"Authorization": "Bearer testprefix_secretvalue"},
            )

        assert response.status_code == 429
        assert response.headers["retry-after"] == "30"

    async def test_denied_has_error_body(
        self, fake_auth: FakeAuthService, denied_limiter: FakeRateLimiter
    ) -> None:
        """429 response body contains error code and message."""
        app = create_app(auth_service=fake_auth, rate_limiter=denied_limiter)

        @app.get("/v1/test-error-body")
        async def test_route():  # noqa: ANN202
            return {"ok": True}

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(
                "/v1/test-error-body",
                headers={"Authorization": "Bearer testprefix_secretvalue"},
            )

        assert response.status_code == 429
        body = response.json()
        assert body["error"]["code"] == "rate_limited"
        assert "message" in body["error"]

    async def test_retry_after_clamped_to_valid_range(
        self, fake_auth: FakeAuthService
    ) -> None:
        """Retry-After is clamped to [1, 3600] per R14.1."""
        # Test with retry_after = 0 (should be clamped to 1)
        limiter = FakeRateLimiter(
            allowed=False, limit=100, remaining=0, reset_at=1700000060, retry_after=0
        )
        app = create_app(auth_service=fake_auth, rate_limiter=limiter)

        @app.get("/v1/test-clamp")
        async def test_route():  # noqa: ANN202
            return {"ok": True}

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(
                "/v1/test-clamp",
                headers={"Authorization": "Bearer testprefix_secretvalue"},
            )

        assert response.status_code == 429
        retry_after = int(response.headers["retry-after"])
        assert 1 <= retry_after <= 3600


# ---------------------------------------------------------------------------
# Tests: Pass-through limiter (no Redis configured)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestPassThroughRateLimiter:
    """Tests for the pass-through rate limiter used when no Redis is available."""

    async def test_pass_through_always_allows(self) -> None:
        """PassThroughRateLimiter always returns allowed=True."""
        limiter = PassThroughRateLimiter()
        result = await limiter.check_rate_limit("any-tenant", "/v1/search")
        assert result.allowed is True

    async def test_pass_through_returns_valid_values(self) -> None:
        """PassThroughRateLimiter returns valid limit, remaining, reset_at."""
        limiter = PassThroughRateLimiter()
        result = await limiter.check_rate_limit("any-tenant", "/v1/search")
        assert result.limit > 0
        assert result.remaining >= 0
        assert result.remaining < result.limit
        assert result.reset_at > 0

    async def test_default_app_uses_pass_through(self, fake_auth: FakeAuthService) -> None:
        """When no limiter is provided, the app uses a pass-through and adds headers."""
        app = create_app(auth_service=fake_auth)

        @app.get("/v1/test-default")
        async def test_route():  # noqa: ANN202
            return {"ok": True}

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(
                "/v1/test-default",
                headers={"Authorization": "Bearer testprefix_secretvalue"},
            )

        assert response.status_code == 200
        # Headers should still be present with pass-through values
        assert "x-ratelimit-limit" in response.headers
        assert "x-ratelimit-remaining" in response.headers
        assert "x-ratelimit-reset" in response.headers
        # Values should be reasonable
        assert int(response.headers["x-ratelimit-limit"]) > 0
        assert int(response.headers["x-ratelimit-remaining"]) >= 0
        assert int(response.headers["x-ratelimit-reset"]) > 0


# ---------------------------------------------------------------------------
# Tests: Auth failure does not trigger rate limiting
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRateLimitWithAuthFailure:
    """Tests that rate limiting is not applied when auth fails."""

    async def test_auth_failure_skips_rate_limit(
        self, fake_auth_failure: FakeAuthService, allowed_limiter: FakeRateLimiter
    ) -> None:
        """When auth fails, rate limiter is not invoked (no tenant_id available)."""
        app = create_app(auth_service=fake_auth_failure, rate_limiter=allowed_limiter)

        @app.get("/v1/test-no-auth")
        async def test_route():  # noqa: ANN202
            return {"ok": True}

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/v1/test-no-auth")

        # Auth should fail with 401
        assert response.status_code == 401
        # Rate limiter should not have been called
        assert len(allowed_limiter.calls) == 0
