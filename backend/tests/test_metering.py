"""Tests for the MeteringService (R14.2, R14.3).

Validates:
- 2xx responses emit a metering event
- Non-2xx responses do NOT emit
- Duplicate request_id results in only one event (dedup via ON CONFLICT)
- DB errors do not block (fire-and-forget)
- All fields are correctly populated
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.rate_limiter.metering import MeteringEvent, MeteringService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _FakeAcquireContext:
    """Async context manager that mimics asyncpg pool.acquire()."""

    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *args):
        return False


@pytest.fixture
def mock_pool():
    """Create a mock asyncpg pool with an async context manager for acquire."""
    pool = MagicMock()
    conn = AsyncMock()
    pool.acquire.return_value = _FakeAcquireContext(conn)
    return pool, conn


@pytest.fixture
def metering_service(mock_pool):
    """Create a MeteringService with a mocked pool."""
    pool, _ = mock_pool
    return MeteringService(db_pool=pool)


# ---------------------------------------------------------------------------
# Tests: 2xx responses emit a metering event
# ---------------------------------------------------------------------------


class TestBillableEmission:
    """2xx responses should emit a metering event."""

    @pytest.mark.parametrize("status", [200, 201, 202, 204, 299])
    async def test_2xx_responses_emit_event(self, mock_pool, status):
        """All 2xx status codes should trigger an INSERT."""
        pool, conn = mock_pool
        service = MeteringService(db_pool=pool)

        await service.emit_metering_event(
            request_id="req-abc123",
            tenant_id=str(uuid.uuid4()),
            endpoint="/v1/search",
            response_status=status,
        )

        conn.execute.assert_called_once()
        call_args = conn.execute.call_args
        sql = call_args[0][0]
        assert "INSERT INTO metering_events" in sql
        assert "ON CONFLICT (dedup_key) DO NOTHING" in sql

    async def test_emitted_event_has_correct_fields(self, mock_pool):
        """Verify all fields are passed correctly to the INSERT."""
        pool, conn = mock_pool
        service = MeteringService(db_pool=pool)

        tenant_id = str(uuid.uuid4())
        await service.emit_metering_event(
            request_id="req-xyz789",
            tenant_id=tenant_id,
            endpoint="/v1/answer",
            response_status=200,
            tokens_consumed=150,
        )

        conn.execute.assert_called_once()
        call_args = conn.execute.call_args[0]
        # Positional args after SQL: metering_event_id, request_id, tenant_id,
        # endpoint, timestamp_utc, response_status, tokens_consumed, dedup_key
        assert isinstance(call_args[1], uuid.UUID)  # metering_event_id
        assert call_args[2] == "req-xyz789"  # request_id
        assert call_args[3] == uuid.UUID(tenant_id)  # tenant_id
        assert call_args[4] == "/v1/answer"  # endpoint
        assert isinstance(call_args[5], datetime)  # timestamp_utc
        assert call_args[5].tzinfo is not None  # timezone-aware
        assert call_args[6] == 200  # response_status
        assert call_args[7] == 150  # tokens_consumed
        assert call_args[8] == "meter:req-xyz789"  # dedup_key

    async def test_tokens_consumed_defaults_to_none(self, mock_pool):
        """When tokens_consumed is not provided, it should be None."""
        pool, conn = mock_pool
        service = MeteringService(db_pool=pool)

        await service.emit_metering_event(
            request_id="req-no-tokens",
            tenant_id=str(uuid.uuid4()),
            endpoint="/v1/search",
            response_status=200,
        )

        call_args = conn.execute.call_args[0]
        assert call_args[7] is None  # tokens_consumed


# ---------------------------------------------------------------------------
# Tests: Non-2xx responses do NOT emit
# ---------------------------------------------------------------------------


class TestNonBillableSkipped:
    """Non-2xx responses should NOT emit a metering event."""

    @pytest.mark.parametrize("status", [100, 199, 300, 301, 400, 401, 403, 404, 429, 500, 502, 503])
    async def test_non_2xx_responses_do_not_emit(self, mock_pool, status):
        """Non-2xx status codes should not trigger any DB operation."""
        pool, conn = mock_pool
        service = MeteringService(db_pool=pool)

        await service.emit_metering_event(
            request_id="req-non-billable",
            tenant_id=str(uuid.uuid4()),
            endpoint="/v1/search",
            response_status=status,
        )

        pool.acquire.assert_not_called()
        conn.execute.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: Dedup via ON CONFLICT DO NOTHING
# ---------------------------------------------------------------------------


class TestDeduplication:
    """Duplicate request_id results in only one event (dedup via ON CONFLICT)."""

    async def test_dedup_key_derived_from_request_id(self):
        """dedup_key should be deterministic: 'meter:{request_id}'."""
        assert MeteringService._make_dedup_key("req-123") == "meter:req-123"
        assert MeteringService._make_dedup_key("req-abc") == "meter:req-abc"

    async def test_same_request_id_produces_same_dedup_key(self):
        """Calling with the same request_id always produces the same dedup_key."""
        key1 = MeteringService._make_dedup_key("req-duplicate")
        key2 = MeteringService._make_dedup_key("req-duplicate")
        assert key1 == key2

    async def test_insert_uses_on_conflict_do_nothing(self, mock_pool):
        """The SQL should use ON CONFLICT (dedup_key) DO NOTHING."""
        pool, conn = mock_pool
        service = MeteringService(db_pool=pool)

        await service.emit_metering_event(
            request_id="req-dedup-test",
            tenant_id=str(uuid.uuid4()),
            endpoint="/v1/search",
            response_status=200,
        )

        sql = conn.execute.call_args[0][0]
        assert "ON CONFLICT (dedup_key) DO NOTHING" in sql

    async def test_duplicate_emit_calls_insert_twice_but_db_deduplicates(self, mock_pool):
        """Two calls with the same request_id both attempt INSERT, relying on DB dedup."""
        pool, conn = mock_pool
        service = MeteringService(db_pool=pool)

        tenant_id = str(uuid.uuid4())
        for _ in range(2):
            await service.emit_metering_event(
                request_id="req-same",
                tenant_id=tenant_id,
                endpoint="/v1/search",
                response_status=200,
            )

        # Both calls attempt the INSERT (at-least-once delivery)
        assert conn.execute.call_count == 2
        # Both use the same dedup_key
        first_dedup = conn.execute.call_args_list[0][0][8]
        second_dedup = conn.execute.call_args_list[1][0][8]
        assert first_dedup == second_dedup == "meter:req-same"


# ---------------------------------------------------------------------------
# Tests: DB errors do not block (fire-and-forget)
# ---------------------------------------------------------------------------


class _FakeAcquireContextError:
    """Async context manager that raises on __aenter__."""

    def __init__(self, error):
        self._error = error

    async def __aenter__(self):
        raise self._error

    async def __aexit__(self, *args):
        return False


class TestFireAndForget:
    """DB errors should be logged but never raise to the caller."""

    async def test_db_connection_error_does_not_raise(self):
        """If the pool raises on acquire, the method should not propagate."""
        pool = MagicMock()
        pool.acquire.return_value = _FakeAcquireContextError(Exception("Connection refused"))
        service = MeteringService(db_pool=pool)

        # Should NOT raise
        await service.emit_metering_event(
            request_id="req-db-error",
            tenant_id=str(uuid.uuid4()),
            endpoint="/v1/search",
            response_status=200,
        )

    async def test_db_execute_error_does_not_raise(self, mock_pool):
        """If conn.execute raises, the method should not propagate."""
        pool, conn = mock_pool
        conn.execute.side_effect = Exception("Unique violation or timeout")
        service = MeteringService(db_pool=pool)

        # Should NOT raise
        await service.emit_metering_event(
            request_id="req-exec-error",
            tenant_id=str(uuid.uuid4()),
            endpoint="/v1/search",
            response_status=200,
        )

    async def test_db_error_is_logged(self, mock_pool):
        """DB errors should be logged with relevant context."""
        pool, conn = mock_pool
        conn.execute.side_effect = Exception("DB timeout")
        service = MeteringService(db_pool=pool)

        with patch("backend.rate_limiter.metering.logger") as mock_logger:
            await service.emit_metering_event(
                request_id="req-log-test",
                tenant_id=str(uuid.uuid4()),
                endpoint="/v1/search",
                response_status=200,
            )

            mock_logger.exception.assert_called_once()
            log_args = mock_logger.exception.call_args[0]
            assert "req-log-test" in str(log_args)


# ---------------------------------------------------------------------------
# Tests: MeteringEvent dataclass
# ---------------------------------------------------------------------------


class TestMeteringEventDataclass:
    """Verify the MeteringEvent dataclass structure."""

    def test_metering_event_fields(self):
        """MeteringEvent should have all required fields."""
        now = datetime.now(timezone.utc)
        event = MeteringEvent(
            request_id="req-001",
            tenant_id="tenant-abc",
            endpoint="/v1/search",
            timestamp_utc=now,
            response_status=200,
            tokens_consumed=42,
            dedup_key="meter:req-001",
        )

        assert event.request_id == "req-001"
        assert event.tenant_id == "tenant-abc"
        assert event.endpoint == "/v1/search"
        assert event.timestamp_utc == now
        assert event.response_status == 200
        assert event.tokens_consumed == 42
        assert event.dedup_key == "meter:req-001"

    def test_metering_event_is_frozen(self):
        """MeteringEvent should be immutable (frozen dataclass)."""
        event = MeteringEvent(
            request_id="req-002",
            tenant_id="tenant-xyz",
            endpoint="/v1/answer",
            timestamp_utc=datetime.now(timezone.utc),
            response_status=201,
            tokens_consumed=None,
            dedup_key="meter:req-002",
        )

        with pytest.raises(AttributeError):
            event.request_id = "modified"  # type: ignore[misc]

    def test_metering_event_tokens_consumed_nullable(self):
        """tokens_consumed can be None for non-LLM endpoints."""
        event = MeteringEvent(
            request_id="req-003",
            tenant_id="tenant-123",
            endpoint="/v1/search",
            timestamp_utc=datetime.now(timezone.utc),
            response_status=200,
            tokens_consumed=None,
            dedup_key="meter:req-003",
        )

        assert event.tokens_consumed is None


# ---------------------------------------------------------------------------
# Tests: Billable status boundary
# ---------------------------------------------------------------------------


class TestBillableStatusBoundary:
    """Test the exact boundary of billable status codes."""

    def test_199_is_not_billable(self):
        assert not MeteringService._is_billable_status(199)

    def test_200_is_billable(self):
        assert MeteringService._is_billable_status(200)

    def test_299_is_billable(self):
        assert MeteringService._is_billable_status(299)

    def test_300_is_not_billable(self):
        assert not MeteringService._is_billable_status(300)
