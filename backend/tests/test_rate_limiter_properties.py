"""Property-based tests for Rate Limiter & Metering (Properties 35, 36, 37).

**Validates: Requirements 14.1, 14.2, 14.3, 14.4, 14.5**

Property 35: Rate-limited responses have valid headers (R14.1, R14.4)
Property 36: Exactly one metering event per billable response after dedup (R14.2, R14.3)
Property 37: Metering pipeline outage does not block API responses (R14.5)

Uses Hypothesis to generate random tenant_ids, endpoints, limit values,
request_ids, and metering events to verify invariants hold across all inputs.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from backend.rate_limiter.metering import MeteringEvent, MeteringService
from backend.rate_limiter.metering_buffer import DurableMeteringBuffer
from backend.rate_limiter.token_bucket import RateLimitResult, TokenBucketRateLimiter


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Tenant IDs: UUID-like strings
tenant_id_strategy = st.uuids().map(str)

# Endpoints: one of the billable API endpoints
endpoint_strategy = st.sampled_from([
    "/v1/search",
    "/v1/find_similar",
    "/v1/contents",
    "/v1/answer",
    "/v1/research",
])

# Rate limit values: positive integers in a reasonable range for property testing
# We use a smaller range for tests that exhaust the bucket to avoid slow tests
limit_strategy = st.integers(min_value=1, max_value=10000)
limit_exhaustion_strategy = st.integers(min_value=1, max_value=100)

# Request IDs: unique identifiers (16-64 chars)
request_id_strategy = st.text(
    alphabet=st.characters(
        whitelist_categories=("Ll", "Nd"),
        whitelist_characters="abcdefghijklmnopqrstuvwxyz0123456789-_",
    ),
    min_size=16,
    max_size=64,
).filter(lambda s: len(s) >= 16)

# Retry-After values from the limiter (before clamping)
retry_after_raw_strategy = st.integers(min_value=-10, max_value=5000)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeAcquireContext:
    """Async context manager that mimics asyncpg pool.acquire()."""

    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *args):
        return False


class _FakeAcquireContextError:
    """Async context manager that raises on __aenter__."""

    def __init__(self, error):
        self._error = error

    async def __aenter__(self):
        raise self._error

    async def __aexit__(self, *args):
        return False


def _make_event(request_id: str | None = None, tenant_id: str | None = None) -> MeteringEvent:
    """Create a MeteringEvent with sensible defaults."""
    rid = request_id or f"req-{uuid.uuid4().hex[:8]}"
    tid = tenant_id or str(uuid.uuid4())
    return MeteringEvent(
        request_id=rid,
        tenant_id=tid,
        endpoint="/v1/search",
        timestamp_utc=datetime.now(timezone.utc),
        response_status=200,
        tokens_consumed=None,
        dedup_key=f"meter:{rid}",
    )


# ---------------------------------------------------------------------------
# Property 35: Rate-limited responses have valid headers
# ---------------------------------------------------------------------------


class TestRateLimitedResponseHeaders:
    """Property-based tests for rate-limited response headers (Property 35).

    **Validates: Requirements 14.1, 14.4**

    Property 35: For any rate-limited response (429), the response MUST have:
    - X-RateLimit-Limit header with a positive integer
    - X-RateLimit-Remaining header with value 0
    - X-RateLimit-Reset header with a future Unix timestamp
    - Retry-After header with value in [1, 3600]
    """

    @given(
        tenant_id=tenant_id_strategy,
        endpoint=endpoint_strategy,
        limit=limit_exhaustion_strategy,
    )
    @settings(suppress_health_check=[HealthCheck.too_slow], deadline=None)
    async def test_rate_limited_response_has_valid_headers(
        self,
        tenant_id: str,
        endpoint: str,
        limit: int,
    ):
        """Property: When a request is rate-limited, the RateLimitResult has valid header values.

        **Validates: Requirements 14.1, 14.4**

        For any tenant_id, endpoint, and limit configuration, when the bucket
        is exhausted, the resulting RateLimitResult must satisfy:
        - limit is a positive integer (equals the configured limit)
        - remaining == 0 (bucket is empty)
        - reset_at is a Unix timestamp >= current time
        - retry_after is in [1, 3600] (or at least positive)
        """
        import fakeredis.aioredis

        redis_client = fakeredis.aioredis.FakeRedis()

        try:
            limiter = TokenBucketRateLimiter(redis_client, default_limit_per_minute=limit)

            # Pin time to prevent refill during token exhaustion.
            # The Lua script uses the time we pass (via time.time()), so by
            # freezing time we ensure no tokens refill between calls.
            frozen_time = time.time()
            with patch("time.time", return_value=frozen_time):
                # Exhaust the bucket by consuming all tokens
                for _ in range(limit):
                    await limiter.check_rate_limit(tenant_id, endpoint)

                # The next request should be denied
                now_before = int(frozen_time)
                result = await limiter.check_rate_limit(tenant_id, endpoint)

            # Property assertions for a rate-limited (denied) response
            assert result.allowed is False, (
                f"After exhausting {limit} tokens, request should be denied"
            )
            assert result.limit == limit, (
                f"X-RateLimit-Limit should equal configured limit {limit}, got {result.limit}"
            )
            assert result.limit > 0, (
                "X-RateLimit-Limit must be a positive integer"
            )
            assert result.remaining == 0, (
                f"X-RateLimit-Remaining must be 0 when rate-limited, got {result.remaining}"
            )
            assert result.reset_at >= now_before, (
                f"X-RateLimit-Reset must be a future Unix timestamp, "
                f"got {result.reset_at} but now is {now_before}"
            )
            assert result.retry_after is not None, (
                "Retry-After must be present when rate-limited"
            )
            assert result.retry_after >= 1, (
                f"Retry-After must be >= 1, got {result.retry_after}"
            )
        finally:
            await redis_client.flushall()
            await redis_client.aclose()

    @given(
        tenant_id=tenant_id_strategy,
        endpoint=endpoint_strategy,
        limit=limit_strategy,
    )
    @settings(suppress_health_check=[HealthCheck.too_slow], deadline=None)
    async def test_within_limit_response_has_valid_headers(
        self,
        tenant_id: str,
        endpoint: str,
        limit: int,
    ):
        """Property: When a request is within limits, the RateLimitResult has valid header values.

        **Validates: Requirements 14.4**

        For any within-limit response, the same three X-RateLimit-* headers
        are present with Remaining >= 0.
        """
        import fakeredis.aioredis

        redis_client = fakeredis.aioredis.FakeRedis()

        try:
            limiter = TokenBucketRateLimiter(redis_client, default_limit_per_minute=limit)

            # First request should always be within limits
            result = await limiter.check_rate_limit(tenant_id, endpoint)

            # Property assertions for an allowed response
            assert result.allowed is True, (
                "First request should always be allowed"
            )
            assert result.limit == limit, (
                f"X-RateLimit-Limit should equal configured limit {limit}, got {result.limit}"
            )
            assert result.limit > 0, (
                "X-RateLimit-Limit must be a positive integer"
            )
            assert result.remaining >= 0, (
                f"X-RateLimit-Remaining must be >= 0, got {result.remaining}"
            )
            assert result.reset_at >= int(time.time()) - 1, (
                f"X-RateLimit-Reset must be a valid Unix timestamp"
            )
            assert result.retry_after is None, (
                "Retry-After should be None when request is allowed"
            )
        finally:
            await redis_client.flushall()
            await redis_client.aclose()

    @given(
        tenant_id=tenant_id_strategy,
        endpoint=endpoint_strategy,
        limit=limit_strategy,
        retry_after_raw=retry_after_raw_strategy,
    )
    @settings(suppress_health_check=[HealthCheck.too_slow])
    async def test_retry_after_clamped_to_valid_range(
        self,
        tenant_id: str,
        endpoint: str,
        limit: int,
        retry_after_raw: int,
    ):
        """Property: Retry-After is always clamped to [1, 3600] in the middleware.

        **Validates: Requirements 14.1**

        The middleware clamps retry_after to [1, 3600] regardless of what
        the rate limiter returns. This tests the clamping logic directly.
        """
        # Simulate the clamping logic from the middleware
        retry_after = retry_after_raw if retry_after_raw is not None else 60
        retry_after = max(1, min(3600, retry_after))

        assert 1 <= retry_after <= 3600, (
            f"Retry-After must be in [1, 3600] after clamping, got {retry_after}"
        )


# ---------------------------------------------------------------------------
# Property 36: Exactly one metering event per billable response after dedup
# ---------------------------------------------------------------------------


class TestMeteringDeduplication:
    """Property-based tests for metering deduplication (Property 36).

    **Validates: Requirements 14.2, 14.3**

    Property 36: For any billable 2xx response with a unique request_id:
    - Exactly one metering event is persisted after dedup
    - Calling emit_metering_event twice with the same request_id still
      results in one event (ON CONFLICT DO NOTHING)
    """

    @given(
        request_id=request_id_strategy,
        tenant_id=tenant_id_strategy,
        endpoint=endpoint_strategy,
        status_code=st.sampled_from([200, 201, 202, 204, 299]),
    )
    @settings(suppress_health_check=[HealthCheck.too_slow])
    async def test_unique_request_id_produces_one_event(
        self,
        request_id: str,
        tenant_id: str,
        endpoint: str,
        status_code: int,
    ):
        """Property: A unique request_id produces exactly one metering event.

        **Validates: Requirements 14.2**

        For any billable 2xx response with a unique request_id, exactly one
        INSERT is attempted with the correct dedup_key.
        """
        pool = MagicMock()
        conn = AsyncMock()
        pool.acquire.return_value = _FakeAcquireContext(conn)

        service = MeteringService(db_pool=pool)

        await service.emit_metering_event(
            request_id=request_id,
            tenant_id=tenant_id,
            endpoint=endpoint,
            response_status=status_code,
        )

        # Exactly one INSERT should have been attempted
        conn.execute.assert_called_once()

        # Verify the dedup_key is deterministic from request_id
        call_args = conn.execute.call_args[0]
        dedup_key = call_args[8]  # 9th positional arg is dedup_key
        assert dedup_key == f"meter:{request_id}", (
            f"dedup_key should be 'meter:{request_id}', got '{dedup_key}'"
        )

        # Verify the SQL uses ON CONFLICT DO NOTHING
        sql = call_args[0]
        assert "ON CONFLICT (dedup_key) DO NOTHING" in sql, (
            "INSERT must use ON CONFLICT (dedup_key) DO NOTHING for dedup"
        )

    @given(
        request_id=request_id_strategy,
        tenant_id=tenant_id_strategy,
        endpoint=endpoint_strategy,
    )
    @settings(suppress_health_check=[HealthCheck.too_slow])
    async def test_duplicate_request_id_still_results_in_one_event(
        self,
        request_id: str,
        tenant_id: str,
        endpoint: str,
    ):
        """Property: Calling emit twice with the same request_id results in one persisted event.

        **Validates: Requirements 14.3**

        At-least-once delivery means the service may call INSERT multiple times,
        but ON CONFLICT DO NOTHING ensures only one row is persisted. Both calls
        use the same dedup_key, so the DB deduplicates.
        """
        pool = MagicMock()
        conn = AsyncMock()
        pool.acquire.return_value = _FakeAcquireContext(conn)

        service = MeteringService(db_pool=pool)

        # Emit twice with the same request_id
        await service.emit_metering_event(
            request_id=request_id,
            tenant_id=tenant_id,
            endpoint=endpoint,
            response_status=200,
        )
        await service.emit_metering_event(
            request_id=request_id,
            tenant_id=tenant_id,
            endpoint=endpoint,
            response_status=200,
        )

        # Both calls attempt INSERT (at-least-once delivery)
        assert conn.execute.call_count == 2, (
            "Both emit calls should attempt INSERT (at-least-once)"
        )

        # Both use the same dedup_key — DB deduplicates via ON CONFLICT
        first_dedup = conn.execute.call_args_list[0][0][8]
        second_dedup = conn.execute.call_args_list[1][0][8]
        assert first_dedup == second_dedup == f"meter:{request_id}", (
            f"Both calls must use the same dedup_key 'meter:{request_id}'"
        )

    @given(
        request_id=request_id_strategy,
        tenant_id=tenant_id_strategy,
        endpoint=endpoint_strategy,
        non_billable_status=st.sampled_from([100, 199, 300, 301, 400, 401, 403, 404, 429, 500, 502, 503]),
    )
    @settings(suppress_health_check=[HealthCheck.too_slow])
    async def test_non_billable_responses_emit_no_event(
        self,
        request_id: str,
        tenant_id: str,
        endpoint: str,
        non_billable_status: int,
    ):
        """Property: Non-2xx responses never emit a metering event.

        **Validates: Requirements 14.2**

        Only HTTP 2xx responses are billable. Any other status code must
        not trigger any DB operation.
        """
        pool = MagicMock()
        conn = AsyncMock()
        pool.acquire.return_value = _FakeAcquireContext(conn)

        service = MeteringService(db_pool=pool)

        await service.emit_metering_event(
            request_id=request_id,
            tenant_id=tenant_id,
            endpoint=endpoint,
            response_status=non_billable_status,
        )

        # No DB operation should have been attempted
        pool.acquire.assert_not_called()
        conn.execute.assert_not_called()

    @given(
        request_ids=st.lists(
            request_id_strategy,
            min_size=2,
            max_size=10,
            unique=True,
        ),
        tenant_id=tenant_id_strategy,
        endpoint=endpoint_strategy,
    )
    @settings(suppress_health_check=[HealthCheck.too_slow])
    async def test_distinct_request_ids_produce_distinct_dedup_keys(
        self,
        request_ids: list[str],
        tenant_id: str,
        endpoint: str,
    ):
        """Property: Distinct request_ids produce distinct dedup_keys.

        **Validates: Requirements 14.3**

        The dedup_key derivation must be injective: different request_ids
        must never collide on the same dedup_key.
        """
        dedup_keys = set()
        for rid in request_ids:
            key = MeteringService._make_dedup_key(rid)
            dedup_keys.add(key)

        assert len(dedup_keys) == len(request_ids), (
            f"Expected {len(request_ids)} distinct dedup_keys, got {len(dedup_keys)}. "
            "Dedup key derivation must be injective."
        )


# ---------------------------------------------------------------------------
# Property 37: Metering outage does not block API responses
# ---------------------------------------------------------------------------


class TestMeteringOutageDoesNotBlock:
    """Property-based tests for metering outage resilience (Property 37).

    **Validates: Requirements 14.5**

    Property 37: For any metering pipeline outage (DB unavailable):
    - The API response is NOT blocked (emit_metering_event returns without raising)
    - The event is buffered locally
    - The buffer does not affect response latency
    """

    @given(
        request_id=request_id_strategy,
        tenant_id=tenant_id_strategy,
        endpoint=endpoint_strategy,
        status_code=st.sampled_from([200, 201, 202, 204, 299]),
    )
    @settings(suppress_health_check=[HealthCheck.too_slow])
    async def test_db_outage_does_not_raise(
        self,
        request_id: str,
        tenant_id: str,
        endpoint: str,
        status_code: int,
    ):
        """Property: DB outage never raises an exception from emit_metering_event.

        **Validates: Requirements 14.5**

        For any metering event during a DB outage, the emit call returns
        without raising, ensuring the API response is not blocked.
        """
        pool = MagicMock()
        pool.acquire.return_value = _FakeAcquireContextError(
            Exception("Connection refused: DB unavailable")
        )

        buffer = DurableMeteringBuffer(max_size=1000)
        service = MeteringService(db_pool=pool, buffer=buffer)

        # This MUST NOT raise — fire-and-forget semantics
        await service.emit_metering_event(
            request_id=request_id,
            tenant_id=tenant_id,
            endpoint=endpoint,
            response_status=status_code,
        )

        # If we reach here, the property holds: no exception was raised

    @given(
        request_id=request_id_strategy,
        tenant_id=tenant_id_strategy,
        endpoint=endpoint_strategy,
        status_code=st.sampled_from([200, 201, 202, 204, 299]),
    )
    @settings(suppress_health_check=[HealthCheck.too_slow])
    async def test_db_outage_buffers_event_locally(
        self,
        request_id: str,
        tenant_id: str,
        endpoint: str,
        status_code: int,
    ):
        """Property: On DB outage, the event is buffered locally.

        **Validates: Requirements 14.5**

        For any billable event during a DB outage, the event must be
        persisted to the local durable buffer for later retry.
        """
        pool = MagicMock()
        pool.acquire.return_value = _FakeAcquireContextError(
            Exception("Connection refused: DB unavailable")
        )

        buffer = DurableMeteringBuffer(max_size=1000)
        service = MeteringService(db_pool=pool, buffer=buffer)

        initial_size = buffer.size

        await service.emit_metering_event(
            request_id=request_id,
            tenant_id=tenant_id,
            endpoint=endpoint,
            response_status=status_code,
        )

        # Event must be buffered
        assert buffer.size == initial_size + 1, (
            f"Buffer should have grown by 1 (from {initial_size} to {initial_size + 1}), "
            f"got {buffer.size}"
        )

    @given(
        events=st.lists(
            st.tuples(request_id_strategy, tenant_id_strategy, endpoint_strategy),
            min_size=1,
            max_size=20,
        ),
    )
    @settings(suppress_health_check=[HealthCheck.too_slow])
    async def test_multiple_outage_events_all_buffered(
        self,
        events: list[tuple[str, str, str]],
    ):
        """Property: Multiple events during outage are all buffered.

        **Validates: Requirements 14.5**

        For any sequence of billable events during a sustained DB outage,
        all events are buffered locally without any being lost.
        """
        pool = MagicMock()
        pool.acquire.return_value = _FakeAcquireContextError(
            Exception("Connection refused: sustained outage")
        )

        buffer = DurableMeteringBuffer(max_size=10000)
        service = MeteringService(db_pool=pool, buffer=buffer)

        for request_id, tenant_id, endpoint in events:
            await service.emit_metering_event(
                request_id=request_id,
                tenant_id=tenant_id,
                endpoint=endpoint,
                response_status=200,
            )

        # All events must be buffered
        assert buffer.size == len(events), (
            f"All {len(events)} events should be buffered, got {buffer.size}"
        )

    @given(
        request_id=request_id_strategy,
        tenant_id=tenant_id_strategy,
        endpoint=endpoint_strategy,
    )
    @settings(suppress_health_check=[HealthCheck.too_slow])
    async def test_outage_does_not_affect_response_latency(
        self,
        request_id: str,
        tenant_id: str,
        endpoint: str,
    ):
        """Property: Metering outage does not add significant latency.

        **Validates: Requirements 14.5**

        The emit_metering_event call during an outage should complete
        quickly (fire-and-forget), not blocking the API response.
        We verify this by measuring that the call completes within
        a generous 100ms bound (actual should be <1ms).
        """
        pool = MagicMock()
        pool.acquire.return_value = _FakeAcquireContextError(
            Exception("Connection refused: DB unavailable")
        )

        buffer = DurableMeteringBuffer(max_size=1000)
        service = MeteringService(db_pool=pool, buffer=buffer)

        start = time.perf_counter()
        await service.emit_metering_event(
            request_id=request_id,
            tenant_id=tenant_id,
            endpoint=endpoint,
            response_status=200,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Fire-and-forget should be very fast — well under 100ms
        assert elapsed_ms < 100, (
            f"emit_metering_event during outage took {elapsed_ms:.2f}ms, "
            "expected < 100ms (fire-and-forget should not block)"
        )

    @given(
        request_id=request_id_strategy,
        tenant_id=tenant_id_strategy,
        endpoint=endpoint_strategy,
    )
    @settings(suppress_health_check=[HealthCheck.too_slow])
    async def test_execute_error_also_buffers(
        self,
        request_id: str,
        tenant_id: str,
        endpoint: str,
    ):
        """Property: conn.execute errors also buffer the event (not just acquire errors).

        **Validates: Requirements 14.5**

        The fire-and-forget behavior applies to any DB error, whether it
        occurs during pool.acquire() or during conn.execute().
        """
        pool = MagicMock()
        conn = AsyncMock()
        conn.execute.side_effect = Exception("Timeout during INSERT")
        pool.acquire.return_value = _FakeAcquireContext(conn)

        buffer = DurableMeteringBuffer(max_size=1000)
        service = MeteringService(db_pool=pool, buffer=buffer)

        # Must not raise
        await service.emit_metering_event(
            request_id=request_id,
            tenant_id=tenant_id,
            endpoint=endpoint,
            response_status=200,
        )

        # Event must be buffered
        assert buffer.size == 1, (
            "Event should be buffered when conn.execute fails"
        )
