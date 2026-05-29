"""Tests for the DurableMeteringBuffer (R14.5).

Validates:
- Events are buffered when added
- Flush sends events to DB
- 80% threshold triggers audit event
- Audit is only emitted once per threshold crossing
- Buffer respects max_size (oldest events dropped when full)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.rate_limiter.metering import MeteringEvent
from backend.rate_limiter.metering_buffer import DurableMeteringBuffer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_event(request_id: str | None = None) -> MeteringEvent:
    """Create a MeteringEvent with sensible defaults."""
    rid = request_id or f"req-{uuid.uuid4().hex[:8]}"
    return MeteringEvent(
        request_id=rid,
        tenant_id=str(uuid.uuid4()),
        endpoint="/v1/search",
        timestamp_utc=datetime.now(timezone.utc),
        response_status=200,
        tokens_consumed=None,
        dedup_key=f"meter:{rid}",
    )


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


# ---------------------------------------------------------------------------
# Tests: Events are buffered when added
# ---------------------------------------------------------------------------


class TestBufferEvent:
    """Events should be stored in the buffer when added."""

    async def test_buffer_event_increases_size(self):
        """Adding an event should increase the buffer size."""
        buf = DurableMeteringBuffer(max_size=100)
        assert buf.size == 0

        await buf.buffer_event(_make_event())
        assert buf.size == 1

    async def test_buffer_multiple_events(self):
        """Multiple events should all be stored."""
        buf = DurableMeteringBuffer(max_size=100)

        for _ in range(5):
            await buf.buffer_event(_make_event())

        assert buf.size == 5

    async def test_fill_ratio_reflects_current_state(self):
        """fill_ratio should be size / max_size."""
        buf = DurableMeteringBuffer(max_size=10)

        await buf.buffer_event(_make_event())
        assert buf.fill_ratio == pytest.approx(0.1)

        for _ in range(4):
            await buf.buffer_event(_make_event())
        assert buf.fill_ratio == pytest.approx(0.5)

    async def test_empty_buffer_has_zero_fill_ratio(self):
        """An empty buffer should have fill_ratio 0.0."""
        buf = DurableMeteringBuffer(max_size=100)
        assert buf.fill_ratio == 0.0

    async def test_full_buffer_has_fill_ratio_one(self):
        """A full buffer should have fill_ratio 1.0."""
        buf = DurableMeteringBuffer(max_size=5)

        for _ in range(5):
            await buf.buffer_event(_make_event())

        assert buf.fill_ratio == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Tests: Flush sends events to DB
# ---------------------------------------------------------------------------


class TestFlush:
    """Flush should send buffered events to the database."""

    async def test_flush_sends_all_events(self):
        """All buffered events should be flushed to DB."""
        buf = DurableMeteringBuffer(max_size=100)
        for _ in range(3):
            await buf.buffer_event(_make_event())

        pool = MagicMock()
        conn = AsyncMock()
        pool.acquire.return_value = _FakeAcquireContext(conn)

        flushed = await buf.flush(pool)

        assert flushed == 3
        assert buf.size == 0
        assert conn.execute.call_count == 3

    async def test_flush_returns_zero_on_empty_buffer(self):
        """Flushing an empty buffer should return 0."""
        buf = DurableMeteringBuffer(max_size=100)

        pool = MagicMock()
        conn = AsyncMock()
        pool.acquire.return_value = _FakeAcquireContext(conn)

        flushed = await buf.flush(pool)

        assert flushed == 0
        conn.execute.assert_not_called()

    async def test_flush_stops_on_db_error(self):
        """If DB fails mid-flush, stop and return count flushed so far."""
        buf = DurableMeteringBuffer(max_size=100)
        for _ in range(5):
            await buf.buffer_event(_make_event())

        pool = MagicMock()
        conn = AsyncMock()
        # Succeed twice, then fail
        conn.execute.side_effect = [None, None, Exception("DB down")]
        pool.acquire.return_value = _FakeAcquireContext(conn)

        flushed = await buf.flush(pool)

        assert flushed == 2
        assert buf.size == 3  # 5 - 2 = 3 remaining

    async def test_flush_uses_correct_sql(self):
        """Flush should use INSERT with ON CONFLICT DO NOTHING."""
        buf = DurableMeteringBuffer(max_size=100)
        await buf.buffer_event(_make_event("req-flush-test"))

        pool = MagicMock()
        conn = AsyncMock()
        pool.acquire.return_value = _FakeAcquireContext(conn)

        await buf.flush(pool)

        sql = conn.execute.call_args[0][0]
        assert "INSERT INTO metering_events" in sql
        assert "ON CONFLICT (dedup_key) DO NOTHING" in sql

    async def test_flush_preserves_fifo_order(self):
        """Events should be flushed in the order they were buffered."""
        buf = DurableMeteringBuffer(max_size=100)
        events = [_make_event(f"req-order-{i}") for i in range(3)]
        for e in events:
            await buf.buffer_event(e)

        pool = MagicMock()
        conn = AsyncMock()
        pool.acquire.return_value = _FakeAcquireContext(conn)

        await buf.flush(pool)

        # Check request_ids in order (arg index 1 is request_id)
        for i, call in enumerate(conn.execute.call_args_list):
            assert call[0][2] == f"req-order-{i}"


# ---------------------------------------------------------------------------
# Tests: 80% threshold triggers audit event
# ---------------------------------------------------------------------------


class TestDegradedAudit:
    """When buffer reaches 80% capacity, emit metering_delivery_degraded audit."""

    async def test_audit_emitted_at_80_percent(self):
        """Audit should fire when fill_ratio crosses 0.8."""
        audit_emitter = AsyncMock()
        buf = DurableMeteringBuffer(max_size=10, audit_emitter=audit_emitter)

        # Fill to 7 events (70%) — no audit
        for _ in range(7):
            await buf.buffer_event(_make_event())
        audit_emitter.assert_not_called()

        # 8th event crosses 80%
        await buf.buffer_event(_make_event())
        audit_emitter.assert_called_once()

        call_kwargs = audit_emitter.call_args[1]
        assert call_kwargs["action"] == "metering_delivery_degraded"
        assert call_kwargs["resource"] == "metering_pipeline"
        assert call_kwargs["detail"]["buffer_size"] == 8
        assert call_kwargs["detail"]["buffer_max_size"] == 10
        assert call_kwargs["detail"]["fill_ratio"] == pytest.approx(0.8)

    async def test_audit_not_emitted_below_threshold(self):
        """Audit should NOT fire when buffer is below 80%."""
        audit_emitter = AsyncMock()
        buf = DurableMeteringBuffer(max_size=10, audit_emitter=audit_emitter)

        for _ in range(7):
            await buf.buffer_event(_make_event())

        audit_emitter.assert_not_called()

    async def test_no_audit_emitter_does_not_crash(self):
        """If no audit_emitter is configured, crossing 80% should not crash."""
        buf = DurableMeteringBuffer(max_size=10, audit_emitter=None)

        # Fill to 80% — should not raise
        for _ in range(8):
            await buf.buffer_event(_make_event())

        assert buf.size == 8


# ---------------------------------------------------------------------------
# Tests: Audit is only emitted once per threshold crossing
# ---------------------------------------------------------------------------


class TestAuditOncePerCrossing:
    """Audit should only be emitted once per threshold crossing."""

    async def test_audit_emitted_only_once_while_above_threshold(self):
        """Adding more events above 80% should NOT re-emit audit."""
        audit_emitter = AsyncMock()
        buf = DurableMeteringBuffer(max_size=10, audit_emitter=audit_emitter)

        # Fill to 100%
        for _ in range(10):
            await buf.buffer_event(_make_event())

        # Audit should have been called exactly once (at the 8th event)
        assert audit_emitter.call_count == 1

    async def test_audit_resets_after_full_flush(self):
        """After a full flush, the audit flag resets and can fire again."""
        audit_emitter = AsyncMock()
        buf = DurableMeteringBuffer(max_size=10, audit_emitter=audit_emitter)

        # Fill to 80% — triggers audit
        for _ in range(8):
            await buf.buffer_event(_make_event())
        assert audit_emitter.call_count == 1

        # Flush all events
        pool = MagicMock()
        conn = AsyncMock()
        pool.acquire.return_value = _FakeAcquireContext(conn)
        await buf.flush(pool)
        assert buf.size == 0

        # Fill again to 80% — should trigger audit again
        for _ in range(8):
            await buf.buffer_event(_make_event())
        assert audit_emitter.call_count == 2

    async def test_audit_does_not_reset_on_partial_flush(self):
        """A partial flush (buffer still above 80%) should NOT reset the flag."""
        audit_emitter = AsyncMock()
        buf = DurableMeteringBuffer(max_size=10, audit_emitter=audit_emitter)

        # Fill to 90%
        for _ in range(9):
            await buf.buffer_event(_make_event())
        assert audit_emitter.call_count == 1

        # Partial flush — only flush 1 event, leaving 8 (80%)
        pool = MagicMock()
        conn = AsyncMock()
        conn.execute.side_effect = [None, Exception("DB down")]
        pool.acquire.return_value = _FakeAcquireContext(conn)
        await buf.flush(pool)
        assert buf.size == 8  # Still at 80%

        # Adding more events should NOT re-emit audit (flag not reset)
        await buf.buffer_event(_make_event())
        assert audit_emitter.call_count == 1


# ---------------------------------------------------------------------------
# Tests: Buffer respects max_size (oldest events dropped when full)
# ---------------------------------------------------------------------------


class TestMaxSizeBehavior:
    """Buffer should drop oldest events when full (deque maxlen)."""

    async def test_buffer_does_not_exceed_max_size(self):
        """Buffer size should never exceed max_size."""
        buf = DurableMeteringBuffer(max_size=5)

        for _ in range(10):
            await buf.buffer_event(_make_event())

        assert buf.size == 5

    async def test_oldest_events_dropped_when_full(self):
        """When full, the oldest events should be dropped."""
        buf = DurableMeteringBuffer(max_size=3)

        events = [_make_event(f"req-{i}") for i in range(5)]
        for e in events:
            await buf.buffer_event(e)

        # Buffer should contain only the last 3 events
        assert buf.size == 3

        # Flush and verify we get events 2, 3, 4 (oldest 0, 1 dropped)
        pool = MagicMock()
        conn = AsyncMock()
        pool.acquire.return_value = _FakeAcquireContext(conn)
        await buf.flush(pool)

        flushed_request_ids = [
            call[0][2] for call in conn.execute.call_args_list
        ]
        assert flushed_request_ids == ["req-2", "req-3", "req-4"]

    async def test_max_size_property(self):
        """max_size property should return the configured maximum."""
        buf = DurableMeteringBuffer(max_size=500)
        assert buf.max_size == 500


# ---------------------------------------------------------------------------
# Tests: MeteringService integration with buffer
# ---------------------------------------------------------------------------


class TestMeteringServiceBufferIntegration:
    """MeteringService should buffer events on DB failure when buffer is configured."""

    async def test_db_error_buffers_event(self):
        """On DB error, the event should be buffered locally."""
        from backend.rate_limiter.metering import MeteringService

        pool = MagicMock()
        pool.acquire.return_value = _FakeAcquireContextError(Exception("DB down"))

        buf = DurableMeteringBuffer(max_size=100)
        service = MeteringService(db_pool=pool, buffer=buf)

        await service.emit_metering_event(
            request_id="req-buffered",
            tenant_id=str(uuid.uuid4()),
            endpoint="/v1/search",
            response_status=200,
        )

        assert buf.size == 1

    async def test_db_success_does_not_buffer(self):
        """On DB success, nothing should be buffered."""
        from backend.rate_limiter.metering import MeteringService

        pool = MagicMock()
        conn = AsyncMock()
        pool.acquire.return_value = _FakeAcquireContext(conn)

        buf = DurableMeteringBuffer(max_size=100)
        service = MeteringService(db_pool=pool, buffer=buf)

        await service.emit_metering_event(
            request_id="req-success",
            tenant_id=str(uuid.uuid4()),
            endpoint="/v1/search",
            response_status=200,
        )

        assert buf.size == 0

    async def test_no_buffer_configured_still_fire_and_forget(self):
        """Without a buffer, DB errors are still fire-and-forget (no crash)."""
        from backend.rate_limiter.metering import MeteringService

        pool = MagicMock()
        pool.acquire.return_value = _FakeAcquireContextError(Exception("DB down"))

        service = MeteringService(db_pool=pool, buffer=None)

        # Should NOT raise
        await service.emit_metering_event(
            request_id="req-no-buffer",
            tenant_id=str(uuid.uuid4()),
            endpoint="/v1/search",
            response_status=200,
        )
