"""Unit tests for AuditLogService configurable retention (Task 5.4).

Validates:
- Valid retention_days accepted (365, 730, 2555)
- Invalid retention_days rejected (0, 364, 2556, negative)
- cleanup_expired calls DELETE with correct interval
- Default retention is 365 days

**Validates: Requirements 15.4**
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest

from audit_log.service import AuditLogService, AuditLogUnavailableError


# ---------------------------------------------------------------------------
# Mock pool infrastructure
# ---------------------------------------------------------------------------


class _MockConnection:
    """Mock asyncpg connection that records execute calls."""

    def __init__(self, *, execute_return: str = "DELETE 0"):
        self._execute_return = execute_return
        self.execute_calls: list[tuple] = []

    async def execute(self, query: str, *args):
        self.execute_calls.append((query, args))
        return self._execute_return


class _MockPool:
    """Mock asyncpg pool that yields a mock connection."""

    def __init__(self, conn: _MockConnection | None = None):
        self.conn = conn or _MockConnection()

    @asynccontextmanager
    async def acquire(self):
        yield self.conn


# ---------------------------------------------------------------------------
# Tests: Default retention is 365 days
# ---------------------------------------------------------------------------


class TestDefaultRetention:
    """Verify the default retention_days is 365."""

    def test_default_retention_days(self):
        """AuditLogService defaults to 365 days retention."""
        pool = _MockPool()
        service = AuditLogService(db_pool=pool)
        assert service.retention_days == 365

    def test_retention_days_property(self):
        """retention_days property returns the configured value."""
        pool = _MockPool()
        service = AuditLogService(db_pool=pool, retention_days=730)
        assert service.retention_days == 730


# ---------------------------------------------------------------------------
# Tests: Valid retention_days accepted (365, 730, 2555)
# ---------------------------------------------------------------------------


class TestValidRetentionDays:
    """Verify that valid retention_days values are accepted."""

    @pytest.mark.parametrize("days", [365, 730, 1000, 1825, 2555])
    def test_valid_retention_days_accepted(self, days: int):
        """retention_days in [365, 2555] should be accepted without error."""
        pool = _MockPool()
        service = AuditLogService(db_pool=pool, retention_days=days)
        assert service.retention_days == days

    def test_min_boundary_accepted(self):
        """retention_days=365 (minimum) should be accepted."""
        pool = _MockPool()
        service = AuditLogService(db_pool=pool, retention_days=365)
        assert service.retention_days == 365

    def test_max_boundary_accepted(self):
        """retention_days=2555 (maximum) should be accepted."""
        pool = _MockPool()
        service = AuditLogService(db_pool=pool, retention_days=2555)
        assert service.retention_days == 2555


# ---------------------------------------------------------------------------
# Tests: Invalid retention_days rejected (0, 364, 2556, negative)
# ---------------------------------------------------------------------------


class TestInvalidRetentionDays:
    """Verify that invalid retention_days values raise ValueError."""

    @pytest.mark.parametrize("days", [0, 1, 100, 364])
    def test_below_minimum_rejected(self, days: int):
        """retention_days below 365 should raise ValueError."""
        pool = _MockPool()
        with pytest.raises(ValueError, match="retention_days must be in"):
            AuditLogService(db_pool=pool, retention_days=days)

    @pytest.mark.parametrize("days", [2556, 3000, 10000])
    def test_above_maximum_rejected(self, days: int):
        """retention_days above 2555 should raise ValueError."""
        pool = _MockPool()
        with pytest.raises(ValueError, match="retention_days must be in"):
            AuditLogService(db_pool=pool, retention_days=days)

    def test_negative_rejected(self):
        """Negative retention_days should raise ValueError."""
        pool = _MockPool()
        with pytest.raises(ValueError, match="retention_days must be in"):
            AuditLogService(db_pool=pool, retention_days=-1)

    def test_zero_rejected(self):
        """retention_days=0 should raise ValueError."""
        pool = _MockPool()
        with pytest.raises(ValueError, match="retention_days must be in"):
            AuditLogService(db_pool=pool, retention_days=0)


# ---------------------------------------------------------------------------
# Tests: cleanup_expired calls DELETE with correct interval
# ---------------------------------------------------------------------------


class TestCleanupExpired:
    """Verify cleanup_expired deletes old events with the correct interval."""

    async def test_cleanup_expired_executes_delete(self):
        """cleanup_expired should execute a DELETE query."""
        conn = _MockConnection(execute_return="DELETE 5")
        pool = _MockPool(conn)
        service = AuditLogService(db_pool=pool, retention_days=365)

        deleted = await service.cleanup_expired()

        assert deleted == 5
        assert len(conn.execute_calls) == 1
        query, args = conn.execute_calls[0]
        assert "DELETE FROM audit_events" in query
        assert "timestamp_utc" in query

    async def test_cleanup_expired_uses_configured_retention(self):
        """cleanup_expired should pass the configured retention_days to the query."""
        conn = _MockConnection(execute_return="DELETE 10")
        pool = _MockPool(conn)
        service = AuditLogService(db_pool=pool, retention_days=730)

        await service.cleanup_expired()

        query, args = conn.execute_calls[0]
        # The retention_days value should be passed as a parameter
        assert 730 in args

    async def test_cleanup_expired_returns_zero_when_no_rows(self):
        """cleanup_expired should return 0 when no rows are deleted."""
        conn = _MockConnection(execute_return="DELETE 0")
        pool = _MockPool(conn)
        service = AuditLogService(db_pool=pool, retention_days=365)

        deleted = await service.cleanup_expired()

        assert deleted == 0

    async def test_cleanup_expired_returns_row_count(self):
        """cleanup_expired should return the number of deleted rows."""
        conn = _MockConnection(execute_return="DELETE 42")
        pool = _MockPool(conn)
        service = AuditLogService(db_pool=pool, retention_days=365)

        deleted = await service.cleanup_expired()

        assert deleted == 42

    async def test_cleanup_expired_db_error_raises(self):
        """cleanup_expired should raise AuditLogUnavailableError on DB error."""
        import asyncpg

        pool = _MockPool()
        service = AuditLogService(db_pool=pool, retention_days=365)

        # Patch the pool to raise on acquire
        async def failing_execute(query, *args):
            raise asyncpg.PostgresError("connection lost")

        pool.conn.execute = failing_execute  # type: ignore[assignment]

        # Re-mock the connection to fail
        class _FailingConn:
            async def execute(self, query, *args):
                raise asyncpg.PostgresError("connection lost")

        class _FailingPool:
            @asynccontextmanager
            async def acquire(self):
                yield _FailingConn()

        service._pool = _FailingPool()

        with pytest.raises(AuditLogUnavailableError, match="cleanup failed"):
            await service.cleanup_expired()

    async def test_cleanup_expired_os_error_raises(self):
        """cleanup_expired should raise AuditLogUnavailableError on connection error."""

        class _FailingConn:
            async def execute(self, query, *args):
                raise OSError("Connection reset")

        class _FailingPool:
            @asynccontextmanager
            async def acquire(self):
                yield _FailingConn()

        pool = _MockPool()
        service = AuditLogService(db_pool=pool, retention_days=365)
        service._pool = _FailingPool()

        with pytest.raises(AuditLogUnavailableError, match="cleanup failed"):
            await service.cleanup_expired()

    async def test_cleanup_expired_uses_interval_syntax(self):
        """cleanup_expired should use INTERVAL-based date comparison."""
        conn = _MockConnection(execute_return="DELETE 0")
        pool = _MockPool(conn)
        service = AuditLogService(db_pool=pool, retention_days=2555)

        await service.cleanup_expired()

        query, args = conn.execute_calls[0]
        # Should use interval-based comparison
        assert "INTERVAL" in query or "interval" in query.lower()
        assert 2555 in args
