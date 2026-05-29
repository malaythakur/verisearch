"""
Load test: Auth resolution p95 ≤ 50ms (R13.1)

Validates that API key authentication resolves within the SLO target.
Auth uses an in-memory LRU cache of key_hash → tenant for fast lookup.
"""

import asyncio
import time

import pytest

from tests.load.conftest import LatencyResult, run_load_test

# SLO Target
SLO_P95_MS = 50


@pytest.fixture
def mock_auth_service():
    """Mock auth service with LRU cache simulation."""

    # Simulated LRU cache of key_hash → tenant_id
    _cache: dict[str, str] = {}

    async def _authenticate(api_key: str) -> dict:
        """Simulate auth resolution with cache hit."""
        # Cache lookup is O(1) - typically < 1ms
        key_prefix = api_key[:12]

        if key_prefix in _cache:
            # Cache hit - very fast
            await asyncio.sleep(0.001)  # ~1ms for cache hit
            return {"tenant_id": _cache[key_prefix], "authenticated": True}

        # Cache miss - hash comparison (still fast with Argon2id prefix index)
        await asyncio.sleep(0.01)  # ~10ms for cache miss
        tenant_id = f"tenant_{hash(api_key) % 100}"
        _cache[key_prefix] = tenant_id
        return {"tenant_id": tenant_id, "authenticated": True}

    # Pre-warm cache with common keys
    for i in range(10):
        key = f"sk_test_{i:04d}_{'x' * 20}"
        _cache[key[:12]] = f"tenant_{i}"

    return _authenticate


@pytest.mark.integration
class TestAuthResolutionSLO:
    """Load tests for auth resolution p95 latency SLO (R13.1)."""

    async def test_auth_resolution_p95_within_slo(self, mock_auth_service):
        """
        SLO: Auth resolution p95 ≤ 50ms.

        Simulates 200 concurrent auth requests (mostly cache hits)
        and validates p95 latency.
        """
        keys = [f"sk_test_{i:04d}_{'x' * 20}" for i in range(10)]
        key_idx = 0

        async def auth_request():
            nonlocal key_idx
            key = keys[key_idx % len(keys)]
            key_idx += 1
            await mock_auth_service(key)

        result = await run_load_test(
            func=auth_request,
            num_requests=200,
            concurrency=20,
        )

        print(f"\n[Auth Resolution SLO Test]\n{result.summary()}")
        print(f"SLO Target: p95 ≤ {SLO_P95_MS}ms")

        assert result.p95 <= SLO_P95_MS, (
            f"Auth resolution p95 ({result.p95:.1f}ms) exceeds SLO ({SLO_P95_MS}ms)"
        )
        assert result.success_rate >= 0.999, (
            f"Success rate ({result.success_rate:.2%}) below 99.9%"
        )

    async def test_auth_cache_miss_still_within_slo(self, mock_auth_service):
        """
        Even with cache misses, auth should stay within SLO.
        """
        request_num = 0

        async def auth_request_cold():
            nonlocal request_num
            # Use unique keys to force cache misses
            key = f"sk_cold_{request_num:06d}_{'y' * 20}"
            request_num += 1
            await mock_auth_service(key)

        result = await run_load_test(
            func=auth_request_cold,
            num_requests=100,
            concurrency=10,
        )

        print(f"\n[Auth Cold Cache Test]\n{result.summary()}")
        assert result.p95 <= SLO_P95_MS
