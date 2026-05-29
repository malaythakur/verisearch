"""Tests for the token-bucket rate limiter.

Validates R14.1: Per-tenant rate limits using token-bucket algorithm.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from backend.rate_limiter.token_bucket import RateLimitResult, TokenBucketRateLimiter


@pytest.fixture
async def redis_client():
    """Create a fakeredis async client for testing with Lua scripting support."""
    import fakeredis.aioredis

    client = fakeredis.aioredis.FakeRedis(lua_modules={"cjson", "struct"}, retry_on_timeout=False)
    yield client
    await client.flushall()
    await client.aclose()


@pytest.fixture
def limiter(redis_client) -> TokenBucketRateLimiter:
    """Create a rate limiter with a small limit for testing."""
    return TokenBucketRateLimiter(redis_client, default_limit_per_minute=10)


class TestRateLimitResult:
    """Tests for the RateLimitResult dataclass."""

    def test_allowed_result_has_correct_fields(self):
        result = RateLimitResult(allowed=True, limit=60, remaining=59, reset_at=1700000060)
        assert result.allowed is True
        assert result.limit == 60
        assert result.remaining == 59
        assert result.reset_at == 1700000060
        assert result.retry_after is None

    def test_denied_result_has_retry_after(self):
        result = RateLimitResult(allowed=False, limit=60, remaining=0, reset_at=1700000060, retry_after=5)
        assert result.allowed is False
        assert result.remaining == 0
        assert result.retry_after == 5

    def test_result_is_frozen(self):
        result = RateLimitResult(allowed=True, limit=60, remaining=59, reset_at=1700000060)
        with pytest.raises(Exception):
            result.allowed = False  # type: ignore[misc]


class TestTokenBucketRateLimiter:
    """Tests for the TokenBucketRateLimiter class."""

    async def test_requests_within_limit_are_allowed(self, limiter: TokenBucketRateLimiter):
        """Requests within the configured limit should be allowed."""
        results = []
        for _ in range(10):
            result = await limiter.check_rate_limit("tenant-1", "/v1/search")
            results.append(result)

        # All 10 requests should be allowed (limit is 10/min)
        assert all(r.allowed for r in results)

    async def test_requests_exceeding_limit_are_denied(self, limiter: TokenBucketRateLimiter):
        """Requests exceeding the limit should be denied."""
        # Exhaust the bucket
        for _ in range(10):
            await limiter.check_rate_limit("tenant-1", "/v1/search")

        # Next request should be denied
        result = await limiter.check_rate_limit("tenant-1", "/v1/search")
        assert result.allowed is False
        assert result.remaining == 0
        assert result.retry_after is not None
        assert result.retry_after > 0

    async def test_bucket_refills_over_time(self, limiter: TokenBucketRateLimiter):
        """Bucket should refill tokens over time based on the refill rate."""
        # Exhaust the bucket
        for _ in range(10):
            await limiter.check_rate_limit("tenant-1", "/v1/search")

        # Verify denied
        result = await limiter.check_rate_limit("tenant-1", "/v1/search")
        assert result.allowed is False

        # Simulate time passing (enough for at least 1 token to refill)
        # Refill rate = 10/60 = 1/6 tokens per second, so 7 seconds = ~1.16 tokens
        future_time = time.time() + 7
        with patch("time.time", return_value=future_time):
            result = await limiter.check_rate_limit("tenant-1", "/v1/search")
            assert result.allowed is True

    async def test_different_tenants_have_independent_buckets(self, limiter: TokenBucketRateLimiter):
        """Different tenants should have completely independent rate limit buckets."""
        # Exhaust tenant-1's bucket
        for _ in range(10):
            await limiter.check_rate_limit("tenant-1", "/v1/search")

        # tenant-1 should be denied
        result = await limiter.check_rate_limit("tenant-1", "/v1/search")
        assert result.allowed is False

        # tenant-2 should still be allowed
        result = await limiter.check_rate_limit("tenant-2", "/v1/search")
        assert result.allowed is True

    async def test_different_endpoints_have_independent_buckets(self, limiter: TokenBucketRateLimiter):
        """Different endpoints for the same tenant should have independent buckets."""
        # Exhaust the /v1/search bucket for tenant-1
        for _ in range(10):
            await limiter.check_rate_limit("tenant-1", "/v1/search")

        # /v1/search should be denied
        result = await limiter.check_rate_limit("tenant-1", "/v1/search")
        assert result.allowed is False

        # /v1/answer should still be allowed
        result = await limiter.check_rate_limit("tenant-1", "/v1/answer")
        assert result.allowed is True

    async def test_result_has_correct_limit_value(self, limiter: TokenBucketRateLimiter):
        """RateLimitResult should report the configured limit."""
        result = await limiter.check_rate_limit("tenant-1", "/v1/search")
        assert result.limit == 10

    async def test_remaining_decreases_with_each_request(self, limiter: TokenBucketRateLimiter):
        """Remaining tokens should decrease with each request."""
        result1 = await limiter.check_rate_limit("tenant-1", "/v1/search")
        result2 = await limiter.check_rate_limit("tenant-1", "/v1/search")

        # remaining should decrease (first request gets limit-1, second gets limit-2)
        assert result1.remaining > result2.remaining

    async def test_reset_at_is_future_timestamp(self, limiter: TokenBucketRateLimiter):
        """reset_at should be a Unix timestamp in the future (or now if bucket is full)."""
        result = await limiter.check_rate_limit("tenant-1", "/v1/search")
        # reset_at should be >= current time
        assert result.reset_at >= int(time.time()) - 1

    async def test_custom_limit_per_minute_override(self, limiter: TokenBucketRateLimiter):
        """Per-tenant limit override should be respected."""
        # Use a custom limit of 2
        result1 = await limiter.check_rate_limit("tenant-1", "/v1/search", limit_per_minute=2)
        result2 = await limiter.check_rate_limit("tenant-1", "/v1/search", limit_per_minute=2)
        result3 = await limiter.check_rate_limit("tenant-1", "/v1/search", limit_per_minute=2)

        assert result1.allowed is True
        assert result2.allowed is True
        assert result3.allowed is False
        assert result1.limit == 2
        assert result3.retry_after is not None

    async def test_redis_key_format(self, redis_client, limiter: TokenBucketRateLimiter):
        """Redis key should follow the format ratelimit:{tenant_id}:{endpoint}."""
        await limiter.check_rate_limit("my-tenant", "/v1/search")

        # Check that the key exists in Redis
        keys = await redis_client.keys("ratelimit:*")
        assert len(keys) == 1
        assert keys[0] == b"ratelimit:my-tenant:/v1/search"

    async def test_denied_result_retry_after_is_positive(self, limiter: TokenBucketRateLimiter):
        """When denied, retry_after should be a positive integer."""
        # Exhaust the bucket
        for _ in range(10):
            await limiter.check_rate_limit("tenant-1", "/v1/search")

        result = await limiter.check_rate_limit("tenant-1", "/v1/search")
        assert result.allowed is False
        assert result.retry_after is not None
        assert result.retry_after >= 1
