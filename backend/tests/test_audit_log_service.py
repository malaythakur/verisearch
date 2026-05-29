"""Unit tests for AuditLogService (Task 5.1).

Validates:
- Successful append inserts into DB
- Timeout raises AuditLogUnavailableError
- DB error raises AuditLogUnavailableError
- Entry shape is correct (all fields present)
- request_id validation (16-64 code points)
- AuditEntry dataclass and append() interface

**Validates: Requirements 15.1**
"""

from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from audit_log.service import (
    AuditAppendError,
    AuditEntry,
    AuditLogService,
    AuditLogUnavailableError,
)


# ---------------------------------------------------------------------------
# Mock pool infrastructure
# ---------------------------------------------------------------------------


class _MockConnection:
    """Mock asyncpg connection that records execute calls."""

    def __init__(self, *, fail_with: Exception | None = None, delay: float = 0.0):
        self._fail_with = fail_with
        self._delay = delay
        self.execute_calls: list[tuple] = []

    async def execute(self, query: str, *args):
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        if self._fail_with is not None:
            raise self._fail_with
        self.execute_calls.append((query, args))


class _MockPool:
    """Mock asyncpg pool that yields a mock connection."""

    def __init__(self, conn: _MockConnection | None = None):
        self.conn = conn or _MockConnection()

    @asynccontextmanager
    async def acquire(self):
        yield self.conn


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_conn():
    """Create a mock connection."""
    return _MockConnection()


@pytest.fixture
def mock_pool(mock_conn):
    """Create a mock pool with a default connection."""
    return _MockPool(mock_conn)


@pytest.fixture
def service(mock_pool):
    """Create an AuditLogService with mock pool and default timeout."""
    return AuditLogService(db_pool=mock_pool, timeout_seconds=5.0)


@pytest.fixture
def valid_request_id():
    """A valid request_id (16-64 code points)."""
    return f"req-{uuid.uuid4().hex}"  # 36 chars


@pytest.fixture
def tenant_id():
    """A valid tenant UUID string."""
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Tests: Successful append inserts into DB
# ---------------------------------------------------------------------------


class TestSuccessfulAppend:
    """Verify that successful emit() and append() insert into the DB."""

    async def test_emit_inserts_row(self, service, mock_conn, valid_request_id, tenant_id):
        """emit() should execute SET LOCAL and INSERT statements."""
        await service.emit(
            action="auth_failure",
            tenant_id=tenant_id,
            actor="anonymous",
            resource="/v1/search",
            request_id=valid_request_id,
            detail={"reason": "invalid_token"},
        )

        # Should have two execute calls: SET LOCAL + INSERT
        assert len(mock_conn.execute_calls) == 2

        # First call is SET LOCAL statement_timeout
        set_call = mock_conn.execute_calls[0]
        assert "SET LOCAL statement_timeout" in set_call[0]

        # Second call is the INSERT
        insert_call = mock_conn.execute_calls[1]
        assert "INSERT INTO audit_events" in insert_call[0]

    async def test_emit_passes_correct_values(self, service, mock_conn, valid_request_id, tenant_id):
        """emit() should pass all field values to the INSERT statement."""
        await service.emit(
            action="session_expired",
            tenant_id=tenant_id,
            actor="system",
            resource="session/abc",
            request_id=valid_request_id,
            detail={"session_id": "abc"},
        )

        insert_call = mock_conn.execute_calls[1]
        args = insert_call[1]

        # args: (audit_id, tenant_uuid, actor, action, resource, timestamp_utc, request_id, detail)
        assert isinstance(args[0], uuid.UUID)  # audit_id
        assert args[1] == uuid.UUID(tenant_id)  # tenant_uuid
        assert args[2] == "system"  # actor
        assert args[3] == "session_expired"  # action
        assert args[4] == "session/abc"  # resource
        assert isinstance(args[5], datetime)  # timestamp_utc
        assert args[6] == valid_request_id  # request_id
        assert args[7] == {"session_id": "abc"}  # detail

    async def test_emit_with_none_tenant_id(self, service, mock_conn, valid_request_id):
        """emit() with tenant_id=None passes None as tenant_uuid."""
        await service.emit(
            action="auth_failure",
            tenant_id=None,
            actor="anonymous",
            resource="/v1/search",
            request_id=valid_request_id,
            detail={},
        )

        insert_call = mock_conn.execute_calls[1]
        args = insert_call[1]
        assert args[1] is None  # tenant_uuid is None

    async def test_append_inserts_row(self, service, mock_conn, valid_request_id, tenant_id):
        """append() with AuditEntry should execute SET LOCAL and INSERT."""
        entry = AuditEntry(
            action="pipeline_created",
            tenant_id=tenant_id,
            actor="user-123",
            resource="pipeline/xyz",
            request_id=valid_request_id,
            detail={"pipeline_name": "my-pipeline"},
        )

        await service.append(entry)

        # Should have two execute calls: SET LOCAL + INSERT
        assert len(mock_conn.execute_calls) == 2
        insert_call = mock_conn.execute_calls[1]
        assert "INSERT INTO audit_events" in insert_call[0]

    async def test_append_passes_entry_fields(self, service, mock_conn, valid_request_id, tenant_id):
        """append() should pass AuditEntry fields to the INSERT."""
        ts = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        entry = AuditEntry(
            action="api_key_revoked",
            tenant_id=tenant_id,
            actor="admin",
            resource="api_key/key-1",
            request_id=valid_request_id,
            detail={"key_prefix": "testpfx"},
            timestamp_utc=ts,
        )

        await service.append(entry)

        insert_call = mock_conn.execute_calls[1]
        args = insert_call[1]

        assert args[1] == uuid.UUID(tenant_id)
        assert args[2] == "admin"
        assert args[3] == "api_key_revoked"
        assert args[4] == "api_key/key-1"
        assert args[5] == ts
        assert args[6] == valid_request_id
        assert args[7] == {"key_prefix": "testpfx"}


# ---------------------------------------------------------------------------
# Tests: Timeout raises AuditLogUnavailableError
# ---------------------------------------------------------------------------


class TestTimeoutHandling:
    """Verify that timeouts raise AuditLogUnavailableError."""

    async def test_emit_timeout_raises_audit_log_unavailable(self, valid_request_id, tenant_id):
        """When the DB operation exceeds the timeout, AuditLogUnavailableError is raised."""
        # Use a very short timeout and a connection that delays
        slow_conn = _MockConnection(delay=1.0)
        pool = _MockPool(slow_conn)
        service = AuditLogService(db_pool=pool, timeout_seconds=0.01)

        with pytest.raises(AuditLogUnavailableError) as exc_info:
            await service.emit(
                action="test_action",
                tenant_id=tenant_id,
                actor="tester",
                resource="/test",
                request_id=valid_request_id,
                detail={},
            )

        assert exc_info.value.code == "audit_log_unavailable"
        assert "timed out" in exc_info.value.message

    async def test_append_timeout_raises_audit_log_unavailable(self, valid_request_id, tenant_id):
        """append() also raises AuditLogUnavailableError on timeout."""
        slow_conn = _MockConnection(delay=1.0)
        pool = _MockPool(slow_conn)
        service = AuditLogService(db_pool=pool, timeout_seconds=0.01)

        entry = AuditEntry(
            action="test_action",
            tenant_id=tenant_id,
            actor="tester",
            resource="/test",
            request_id=valid_request_id,
        )

        with pytest.raises(AuditLogUnavailableError) as exc_info:
            await service.append(entry)

        assert exc_info.value.code == "audit_log_unavailable"

    async def test_audit_append_error_is_alias(self):
        """AuditAppendError is an alias for AuditLogUnavailableError."""
        assert AuditAppendError is AuditLogUnavailableError


# ---------------------------------------------------------------------------
# Tests: DB error raises AuditLogUnavailableError
# ---------------------------------------------------------------------------


class TestDBErrorHandling:
    """Verify that DB errors raise AuditLogUnavailableError."""

    async def test_postgres_error_raises_audit_log_unavailable(self, valid_request_id, tenant_id):
        """asyncpg.PostgresError during insert raises AuditLogUnavailableError."""
        import asyncpg

        error_conn = _MockConnection(fail_with=asyncpg.PostgresError("connection refused"))
        pool = _MockPool(error_conn)
        service = AuditLogService(db_pool=pool)

        with pytest.raises(AuditLogUnavailableError) as exc_info:
            await service.emit(
                action="test_action",
                tenant_id=tenant_id,
                actor="tester",
                resource="/test",
                request_id=valid_request_id,
                detail={},
            )

        assert exc_info.value.code == "audit_log_unavailable"
        assert "failed" in exc_info.value.message

    async def test_os_error_raises_audit_log_unavailable(self, valid_request_id, tenant_id):
        """OSError (network issue) during insert raises AuditLogUnavailableError."""
        error_conn = _MockConnection(fail_with=OSError("Connection reset"))
        pool = _MockPool(error_conn)
        service = AuditLogService(db_pool=pool)

        with pytest.raises(AuditLogUnavailableError) as exc_info:
            await service.emit(
                action="test_action",
                tenant_id=tenant_id,
                actor="tester",
                resource="/test",
                request_id=valid_request_id,
                detail={},
            )

        assert exc_info.value.code == "audit_log_unavailable"
        assert "connection error" in exc_info.value.message

    async def test_append_db_error_raises_audit_log_unavailable(self, valid_request_id, tenant_id):
        """append() also raises AuditLogUnavailableError on DB error."""
        import asyncpg

        error_conn = _MockConnection(fail_with=asyncpg.PostgresError("disk full"))
        pool = _MockPool(error_conn)
        service = AuditLogService(db_pool=pool)

        entry = AuditEntry(
            action="test_action",
            tenant_id=tenant_id,
            actor="tester",
            resource="/test",
            request_id=valid_request_id,
        )

        with pytest.raises(AuditLogUnavailableError):
            await service.append(entry)


# ---------------------------------------------------------------------------
# Tests: Entry shape is correct (all fields present)
# ---------------------------------------------------------------------------


class TestEntryShape:
    """Verify AuditEntry dataclass has all required fields."""

    def test_audit_entry_has_all_fields(self):
        """AuditEntry should have action, tenant_id, actor, resource, request_id, detail, timestamp_utc."""
        entry = AuditEntry(
            action="auth_failure",
            tenant_id="tenant-123",
            actor="anonymous",
            resource="/v1/search",
            request_id="a" * 16,
            detail={"key": "value"},
        )

        assert entry.action == "auth_failure"
        assert entry.tenant_id == "tenant-123"
        assert entry.actor == "anonymous"
        assert entry.resource == "/v1/search"
        assert entry.request_id == "a" * 16
        assert entry.detail == {"key": "value"}
        assert isinstance(entry.timestamp_utc, datetime)
        assert entry.timestamp_utc.tzinfo is not None

    def test_audit_entry_default_timestamp(self):
        """AuditEntry should auto-set timestamp_utc to now(UTC) if not provided."""
        before = datetime.now(timezone.utc)
        entry = AuditEntry(
            action="test",
            tenant_id=None,
            actor="system",
            resource="/test",
            request_id="x" * 16,
        )
        after = datetime.now(timezone.utc)

        assert before <= entry.timestamp_utc <= after

    def test_audit_entry_custom_timestamp(self):
        """AuditEntry should accept a custom timestamp_utc."""
        ts = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        entry = AuditEntry(
            action="test",
            tenant_id=None,
            actor="system",
            resource="/test",
            request_id="y" * 16,
            timestamp_utc=ts,
        )

        assert entry.timestamp_utc == ts

    def test_audit_entry_default_detail(self):
        """AuditEntry should default detail to empty dict."""
        entry = AuditEntry(
            action="test",
            tenant_id=None,
            actor="system",
            resource="/test",
            request_id="z" * 16,
        )

        assert entry.detail == {}

    def test_audit_entry_is_frozen(self):
        """AuditEntry should be immutable (frozen dataclass)."""
        entry = AuditEntry(
            action="test",
            tenant_id=None,
            actor="system",
            resource="/test",
            request_id="a" * 16,
        )

        with pytest.raises(AttributeError):
            entry.action = "modified"  # type: ignore[misc]

    def test_audit_entry_none_tenant_id(self):
        """AuditEntry should accept None for tenant_id (unattributable events)."""
        entry = AuditEntry(
            action="auth_failure",
            tenant_id=None,
            actor="anonymous",
            resource="/v1/search",
            request_id="b" * 16,
        )

        assert entry.tenant_id is None


# ---------------------------------------------------------------------------
# Tests: request_id validation (16-64 code points)
# ---------------------------------------------------------------------------


class TestRequestIdValidation:
    """Verify request_id length validation per R15.1."""

    async def test_emit_rejects_short_request_id(self, service, tenant_id):
        """emit() raises ValueError for request_id shorter than 16 code points."""
        with pytest.raises(ValueError, match="16–64 code points"):
            await service.emit(
                action="test",
                tenant_id=tenant_id,
                actor="tester",
                resource="/test",
                request_id="short",  # 5 chars, too short
                detail={},
            )

    async def test_emit_rejects_long_request_id(self, service, tenant_id):
        """emit() raises ValueError for request_id longer than 64 code points."""
        with pytest.raises(ValueError, match="16–64 code points"):
            await service.emit(
                action="test",
                tenant_id=tenant_id,
                actor="tester",
                resource="/test",
                request_id="x" * 65,  # 65 chars, too long
                detail={},
            )

    async def test_emit_accepts_min_length_request_id(self, service, mock_conn, tenant_id):
        """emit() accepts request_id of exactly 16 code points."""
        await service.emit(
            action="test",
            tenant_id=tenant_id,
            actor="tester",
            resource="/test",
            request_id="a" * 16,
            detail={},
        )

        assert len(mock_conn.execute_calls) == 2  # SET LOCAL + INSERT

    async def test_emit_accepts_max_length_request_id(self, service, mock_conn, tenant_id):
        """emit() accepts request_id of exactly 64 code points."""
        await service.emit(
            action="test",
            tenant_id=tenant_id,
            actor="tester",
            resource="/test",
            request_id="b" * 64,
            detail={},
        )

        assert len(mock_conn.execute_calls) == 2

    async def test_append_rejects_short_request_id(self, service, tenant_id):
        """append() raises ValueError for request_id shorter than 16 code points."""
        entry = AuditEntry(
            action="test",
            tenant_id=tenant_id,
            actor="tester",
            resource="/test",
            request_id="short",
        )

        with pytest.raises(ValueError, match="16–64 code points"):
            await service.append(entry)

    async def test_append_rejects_long_request_id(self, service, tenant_id):
        """append() raises ValueError for request_id longer than 64 code points."""
        entry = AuditEntry(
            action="test",
            tenant_id=tenant_id,
            actor="tester",
            resource="/test",
            request_id="x" * 65,
        )

        with pytest.raises(ValueError, match="16–64 code points"):
            await service.append(entry)

    async def test_emit_request_id_unicode_code_points(self, service, mock_conn, tenant_id):
        """request_id validation counts Unicode code points, not bytes."""
        # Use multi-byte characters: 16 emoji = 16 code points but many bytes
        emoji_id = "🔑" * 16  # 16 code points
        await service.emit(
            action="test",
            tenant_id=tenant_id,
            actor="tester",
            resource="/test",
            request_id=emoji_id,
            detail={},
        )

        assert len(mock_conn.execute_calls) == 2


# ---------------------------------------------------------------------------
# Tests: Configurable timeout
# ---------------------------------------------------------------------------


class TestConfigurableTimeout:
    """Verify the timeout_seconds parameter works correctly."""

    def test_default_timeout_is_5_seconds(self):
        """Default timeout should be 5.0 seconds per R15.1."""
        pool = _MockPool()
        service = AuditLogService(db_pool=pool)
        assert service._timeout_seconds == 5.0

    def test_custom_timeout(self):
        """Custom timeout should be stored correctly."""
        pool = _MockPool()
        service = AuditLogService(db_pool=pool, timeout_seconds=3.0)
        assert service._timeout_seconds == 3.0

    async def test_custom_timeout_used_in_set_local(self, valid_request_id, tenant_id):
        """The configured timeout should be used in SET LOCAL statement_timeout."""
        conn = _MockConnection()
        pool = _MockPool(conn)
        service = AuditLogService(db_pool=pool, timeout_seconds=3.0)

        await service.emit(
            action="test",
            tenant_id=tenant_id,
            actor="tester",
            resource="/test",
            request_id=valid_request_id,
            detail={},
        )

        set_call = conn.execute_calls[0]
        assert "3000" in set_call[0]  # 3.0s = 3000ms
