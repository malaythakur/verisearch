"""Unit tests for TenantDeletionService (Task 5.5).

Validates:
- Deletion request emits audit entry and updates state
- Execute deletion removes data except legal-hold entries
- Legal hold preserves audit entries
- No legal hold → all audit entries deleted
- Deletion emits `deletion_completed` audit entry

**Validates: Requirements 15.3, 15.5**
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import pytest

from audit_log.deletion import DeletionResult, TenantDeletionService, _parse_delete_count
from audit_log.in_memory import InMemoryAuditEmitter


# ---------------------------------------------------------------------------
# Mock DB infrastructure
# ---------------------------------------------------------------------------


class _MockConnection:
    """Mock asyncpg connection that tracks queries and returns configurable results."""

    def __init__(
        self,
        *,
        legal_hold_until: datetime | None = None,
        audit_count: int = 0,
        audit_ids: list[uuid.UUID] | None = None,
        delete_counts: dict[str, int] | None = None,
    ):
        self._legal_hold_until = legal_hold_until
        self._audit_count = audit_count
        self._audit_ids = audit_ids or []
        self._delete_counts = delete_counts or {}
        self.execute_calls: list[tuple[str, tuple]] = []
        self.fetchrow_calls: list[tuple[str, tuple]] = []
        self.fetch_calls: list[tuple[str, tuple]] = []

    async def execute(self, query: str, *args):
        self.execute_calls.append((query, args))
        # Return DELETE count strings based on table name
        for table_name, count in self._delete_counts.items():
            if table_name in query and "DELETE" in query:
                return f"DELETE {count}"
        if "DELETE" in query:
            return "DELETE 0"
        if "UPDATE" in query:
            return "UPDATE 1"
        return "OK"

    async def fetchrow(self, query: str, *args):
        self.fetchrow_calls.append((query, args))
        if "legal_hold_until" in query:
            return {"legal_hold_until": self._legal_hold_until}
        if "COUNT" in query:
            return {"cnt": self._audit_count}
        return None

    async def fetch(self, query: str, *args):
        self.fetch_calls.append((query, args))
        if "audit_id" in query:
            return [{"audit_id": aid} for aid in self._audit_ids]
        return []


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
def audit_emitter():
    """In-memory audit emitter for test assertions."""
    return InMemoryAuditEmitter()


@pytest.fixture
def tenant_id():
    """A valid tenant UUID string."""
    return str(uuid.uuid4())


@pytest.fixture
def request_id():
    """A valid request_id (16-64 code points)."""
    return f"req-{uuid.uuid4().hex}"


# ---------------------------------------------------------------------------
# Tests: DeletionResult dataclass
# ---------------------------------------------------------------------------


class TestDeletionResult:
    """Verify DeletionResult dataclass shape and defaults."""

    def test_deletion_result_fields(self, tenant_id):
        """DeletionResult should have all required fields."""
        result = DeletionResult(
            tenant_id=tenant_id,
            status="pending_deletion",
            records_deleted=5,
            records_preserved_legal_hold=3,
        )
        assert result.tenant_id == tenant_id
        assert result.status == "pending_deletion"
        assert result.records_deleted == 5
        assert result.records_preserved_legal_hold == 3

    def test_deletion_result_defaults(self, tenant_id):
        """DeletionResult should default counts to 0."""
        result = DeletionResult(tenant_id=tenant_id, status="completed")
        assert result.records_deleted == 0
        assert result.records_preserved_legal_hold == 0
        assert result.preserved_records == []

    def test_deletion_result_is_frozen(self, tenant_id):
        """DeletionResult should be immutable."""
        result = DeletionResult(tenant_id=tenant_id, status="completed")
        with pytest.raises(AttributeError):
            result.status = "modified"  # type: ignore[misc]

    def test_deletion_result_with_preserved_records(self, tenant_id):
        """DeletionResult should carry preserved_records list."""
        preserved = [
            {"record_id": "abc-123", "reason": "retention_required"},
            {"record_id": "def-456", "reason": "retention_required"},
        ]
        result = DeletionResult(
            tenant_id=tenant_id,
            status="completed",
            records_deleted=10,
            records_preserved_legal_hold=2,
            preserved_records=preserved,
        )
        assert len(result.preserved_records) == 2
        assert result.preserved_records[0]["reason"] == "retention_required"


# ---------------------------------------------------------------------------
# Tests: Deletion request emits audit entry and updates state
# ---------------------------------------------------------------------------


class TestRequestDeletion:
    """Verify request_deletion() emits audit and updates tenant state."""

    async def test_request_deletion_emits_audit_entry(self, audit_emitter, tenant_id, request_id):
        """request_deletion() should emit a 'deletion_requested' audit entry."""
        conn = _MockConnection()
        pool = _MockPool(conn)
        service = TenantDeletionService(db_pool=pool, audit_emitter=audit_emitter)

        await service.request_deletion(tenant_id, request_id)

        assert len(audit_emitter.events) == 1
        event = audit_emitter.events[0]
        assert event.action == "deletion_requested"
        assert event.tenant_id == tenant_id
        assert event.resource == f"tenant/{tenant_id}"
        assert event.request_id == request_id

    async def test_request_deletion_updates_tenant_state(self, audit_emitter, tenant_id, request_id):
        """request_deletion() should UPDATE tenant deletion_state to 'pending_deletion'."""
        conn = _MockConnection()
        pool = _MockPool(conn)
        service = TenantDeletionService(db_pool=pool, audit_emitter=audit_emitter)

        await service.request_deletion(tenant_id, request_id)

        # Check that an UPDATE was executed on the tenants table
        update_calls = [c for c in conn.execute_calls if "UPDATE tenants" in c[0]]
        assert len(update_calls) == 1
        assert "pending_deletion" in update_calls[0][0]
        assert update_calls[0][1] == (uuid.UUID(tenant_id),)

    async def test_request_deletion_returns_pending_status(self, audit_emitter, tenant_id, request_id):
        """request_deletion() should return DeletionResult with status='pending_deletion'."""
        conn = _MockConnection()
        pool = _MockPool(conn)
        service = TenantDeletionService(db_pool=pool, audit_emitter=audit_emitter)

        result = await service.request_deletion(tenant_id, request_id)

        assert isinstance(result, DeletionResult)
        assert result.tenant_id == tenant_id
        assert result.status == "pending_deletion"
        assert result.records_deleted == 0
        assert result.records_preserved_legal_hold == 0

    async def test_request_deletion_audit_must_succeed_first(self, tenant_id, request_id):
        """If audit emit fails, request_deletion() should not update tenant state."""
        from audit_log.service import AuditLogUnavailableError

        class FailingEmitter:
            async def emit(self, **kwargs):
                raise AuditLogUnavailableError("DB down")

        conn = _MockConnection()
        pool = _MockPool(conn)
        service = TenantDeletionService(db_pool=pool, audit_emitter=FailingEmitter())

        with pytest.raises(AuditLogUnavailableError):
            await service.request_deletion(tenant_id, request_id)

        # No UPDATE should have been executed
        update_calls = [c for c in conn.execute_calls if "UPDATE" in c[0]]
        assert len(update_calls) == 0


# ---------------------------------------------------------------------------
# Tests: Execute deletion removes data except legal-hold entries
# ---------------------------------------------------------------------------


class TestExecuteDeletion:
    """Verify execute_deletion() removes data and respects legal hold."""

    async def test_execute_deletion_deletes_non_audit_data(self, audit_emitter, tenant_id, request_id):
        """execute_deletion() should delete sessions, research jobs, metering, pipelines, citations."""
        conn = _MockConnection(
            legal_hold_until=None,
            delete_counts={
                "sessions": 3,
                "research_jobs": 2,
                "metering_events": 10,
                "pipelines": 1,
                "citations": 5,
                "audit_events": 4,
            },
        )
        pool = _MockPool(conn)
        service = TenantDeletionService(db_pool=pool, audit_emitter=audit_emitter)

        result = await service.execute_deletion(tenant_id, request_id)

        # Verify DELETE queries were issued for each table
        delete_queries = [c[0] for c in conn.execute_calls if "DELETE" in c[0]]
        assert any("sessions" in q for q in delete_queries)
        assert any("research_jobs" in q for q in delete_queries)
        assert any("metering_events" in q for q in delete_queries)
        assert any("pipelines" in q for q in delete_queries)
        assert any("citations" in q for q in delete_queries)

    async def test_execute_deletion_updates_state_to_deleted(self, audit_emitter, tenant_id, request_id):
        """execute_deletion() should update tenant deletion_state to 'deleted'."""
        conn = _MockConnection(legal_hold_until=None)
        pool = _MockPool(conn)
        service = TenantDeletionService(db_pool=pool, audit_emitter=audit_emitter)

        await service.execute_deletion(tenant_id, request_id)

        update_calls = [c for c in conn.execute_calls if "UPDATE tenants" in c[0] and "deleted" in c[0]]
        assert len(update_calls) == 1

    async def test_execute_deletion_returns_completed_status(self, audit_emitter, tenant_id, request_id):
        """execute_deletion() should return DeletionResult with status='completed'."""
        conn = _MockConnection(
            legal_hold_until=None,
            delete_counts={"sessions": 2, "research_jobs": 1, "metering_events": 5, "pipelines": 0, "citations": 3, "audit_events": 4},
        )
        pool = _MockPool(conn)
        service = TenantDeletionService(db_pool=pool, audit_emitter=audit_emitter)

        result = await service.execute_deletion(tenant_id, request_id)

        assert result.status == "completed"
        assert result.tenant_id == tenant_id
        # Non-audit: 2 + 1 + 5 + 0 + 3 = 11, plus audit: 4 = 15
        assert result.records_deleted == 15
        assert result.records_preserved_legal_hold == 0


# ---------------------------------------------------------------------------
# Tests: Legal hold preserves audit entries
# ---------------------------------------------------------------------------


class TestLegalHoldPreservation:
    """Verify that legal hold prevents audit entry deletion."""

    async def test_legal_hold_preserves_all_audit_entries(self, audit_emitter, tenant_id, request_id):
        """When legal hold is active, ALL audit entries should be preserved."""
        future_hold = datetime.now(timezone.utc) + timedelta(days=30)
        audit_ids = [uuid.uuid4(), uuid.uuid4(), uuid.uuid4()]

        conn = _MockConnection(
            legal_hold_until=future_hold,
            audit_count=3,
            audit_ids=audit_ids,
            delete_counts={"sessions": 2, "research_jobs": 1, "metering_events": 0, "pipelines": 0, "citations": 0},
        )
        pool = _MockPool(conn)
        service = TenantDeletionService(db_pool=pool, audit_emitter=audit_emitter)

        result = await service.execute_deletion(tenant_id, request_id)

        assert result.records_preserved_legal_hold == 3
        assert len(result.preserved_records) == 3
        for record in result.preserved_records:
            assert record["reason"] == "retention_required"
            assert "record_id" in record

    async def test_legal_hold_does_not_prevent_non_audit_deletion(self, audit_emitter, tenant_id, request_id):
        """Legal hold should NOT prevent deletion of sessions, research jobs, etc."""
        future_hold = datetime.now(timezone.utc) + timedelta(days=30)

        conn = _MockConnection(
            legal_hold_until=future_hold,
            audit_count=5,
            audit_ids=[uuid.uuid4() for _ in range(5)],
            delete_counts={"sessions": 3, "research_jobs": 2, "metering_events": 4, "pipelines": 1, "citations": 2},
        )
        pool = _MockPool(conn)
        service = TenantDeletionService(db_pool=pool, audit_emitter=audit_emitter)

        result = await service.execute_deletion(tenant_id, request_id)

        # Non-audit data should still be deleted
        assert result.records_deleted == 12  # 3 + 2 + 4 + 1 + 2
        # Audit entries preserved
        assert result.records_preserved_legal_hold == 5

    async def test_legal_hold_no_audit_delete_query(self, audit_emitter, tenant_id, request_id):
        """When legal hold is active, no DELETE on audit_events should be issued."""
        future_hold = datetime.now(timezone.utc) + timedelta(days=30)

        conn = _MockConnection(
            legal_hold_until=future_hold,
            audit_count=2,
            audit_ids=[uuid.uuid4(), uuid.uuid4()],
        )
        pool = _MockPool(conn)
        service = TenantDeletionService(db_pool=pool, audit_emitter=audit_emitter)

        await service.execute_deletion(tenant_id, request_id)

        # No DELETE FROM audit_events should have been issued
        delete_audit_calls = [
            c for c in conn.execute_calls
            if "DELETE" in c[0] and "audit_events" in c[0]
        ]
        assert len(delete_audit_calls) == 0

    async def test_expired_legal_hold_allows_deletion(self, audit_emitter, tenant_id, request_id):
        """When legal_hold_until is in the past, audit entries should be deleted."""
        past_hold = datetime.now(timezone.utc) - timedelta(days=1)

        conn = _MockConnection(
            legal_hold_until=past_hold,
            delete_counts={"sessions": 1, "audit_events": 3, "research_jobs": 0, "metering_events": 0, "pipelines": 0, "citations": 0},
        )
        pool = _MockPool(conn)
        service = TenantDeletionService(db_pool=pool, audit_emitter=audit_emitter)

        result = await service.execute_deletion(tenant_id, request_id)

        # Audit entries should be deleted (legal hold expired)
        assert result.records_preserved_legal_hold == 0
        # 1 (sessions) + 3 (audit) = 4
        assert result.records_deleted == 4

    async def test_null_legal_hold_allows_deletion(self, audit_emitter, tenant_id, request_id):
        """When legal_hold_until is None, audit entries should be deleted."""
        conn = _MockConnection(
            legal_hold_until=None,
            delete_counts={"sessions": 0, "audit_events": 5, "research_jobs": 0, "metering_events": 0, "pipelines": 0, "citations": 0},
        )
        pool = _MockPool(conn)
        service = TenantDeletionService(db_pool=pool, audit_emitter=audit_emitter)

        result = await service.execute_deletion(tenant_id, request_id)

        assert result.records_preserved_legal_hold == 0
        assert result.records_deleted == 5


# ---------------------------------------------------------------------------
# Tests: No legal hold → all audit entries deleted
# ---------------------------------------------------------------------------


class TestNoLegalHoldDeletion:
    """Verify that without legal hold, all audit entries are deleted."""

    async def test_no_legal_hold_deletes_audit_entries(self, audit_emitter, tenant_id, request_id):
        """Without legal hold, DELETE FROM audit_events should be issued."""
        conn = _MockConnection(
            legal_hold_until=None,
            delete_counts={"audit_events": 7, "sessions": 0, "research_jobs": 0, "metering_events": 0, "pipelines": 0, "citations": 0},
        )
        pool = _MockPool(conn)
        service = TenantDeletionService(db_pool=pool, audit_emitter=audit_emitter)

        await service.execute_deletion(tenant_id, request_id)

        # DELETE FROM audit_events should have been issued
        delete_audit_calls = [
            c for c in conn.execute_calls
            if "DELETE" in c[0] and "audit_events" in c[0]
        ]
        assert len(delete_audit_calls) == 1

    async def test_no_legal_hold_counts_deleted_audit_entries(self, audit_emitter, tenant_id, request_id):
        """Without legal hold, records_deleted should include audit entry count."""
        conn = _MockConnection(
            legal_hold_until=None,
            delete_counts={"audit_events": 10, "sessions": 2, "research_jobs": 0, "metering_events": 0, "pipelines": 0, "citations": 0},
        )
        pool = _MockPool(conn)
        service = TenantDeletionService(db_pool=pool, audit_emitter=audit_emitter)

        result = await service.execute_deletion(tenant_id, request_id)

        assert result.records_deleted == 12  # 2 (sessions) + 10 (audit)
        assert result.records_preserved_legal_hold == 0
        assert result.preserved_records == []


# ---------------------------------------------------------------------------
# Tests: Deletion emits `deletion_completed` audit entry
# ---------------------------------------------------------------------------


class TestDeletionCompletedAudit:
    """Verify that execute_deletion() emits a deletion_completed audit entry."""

    async def test_emits_deletion_completed_event(self, audit_emitter, tenant_id, request_id):
        """execute_deletion() should emit a 'deletion_completed' audit entry."""
        conn = _MockConnection(legal_hold_until=None)
        pool = _MockPool(conn)
        service = TenantDeletionService(db_pool=pool, audit_emitter=audit_emitter)

        await service.execute_deletion(tenant_id, request_id)

        assert len(audit_emitter.events) == 1
        event = audit_emitter.events[0]
        assert event.action == "deletion_completed"
        assert event.tenant_id == tenant_id
        assert event.request_id == request_id

    async def test_deletion_completed_includes_counts(self, audit_emitter, tenant_id, request_id):
        """deletion_completed audit entry should include record counts in detail."""
        conn = _MockConnection(
            legal_hold_until=None,
            delete_counts={"sessions": 3, "audit_events": 2, "research_jobs": 0, "metering_events": 0, "pipelines": 0, "citations": 0},
        )
        pool = _MockPool(conn)
        service = TenantDeletionService(db_pool=pool, audit_emitter=audit_emitter)

        await service.execute_deletion(tenant_id, request_id)

        event = audit_emitter.events[0]
        assert event.detail["records_deleted"] == 5
        assert event.detail["records_preserved_legal_hold"] == 0
        assert event.detail["tenant_id"] == tenant_id

    async def test_deletion_completed_with_legal_hold_counts(self, audit_emitter, tenant_id, request_id):
        """deletion_completed should reflect preserved count when legal hold is active."""
        future_hold = datetime.now(timezone.utc) + timedelta(days=30)

        conn = _MockConnection(
            legal_hold_until=future_hold,
            audit_count=4,
            audit_ids=[uuid.uuid4() for _ in range(4)],
            delete_counts={"sessions": 1, "research_jobs": 0, "metering_events": 0, "pipelines": 0, "citations": 0},
        )
        pool = _MockPool(conn)
        service = TenantDeletionService(db_pool=pool, audit_emitter=audit_emitter)

        await service.execute_deletion(tenant_id, request_id)

        event = audit_emitter.events[0]
        assert event.detail["records_deleted"] == 1
        assert event.detail["records_preserved_legal_hold"] == 4

    async def test_deletion_completed_actor_is_system(self, audit_emitter, tenant_id, request_id):
        """deletion_completed audit entry should have actor='system'."""
        conn = _MockConnection(legal_hold_until=None)
        pool = _MockPool(conn)
        service = TenantDeletionService(db_pool=pool, audit_emitter=audit_emitter)

        await service.execute_deletion(tenant_id, request_id)

        event = audit_emitter.events[0]
        assert event.actor == "system"


# ---------------------------------------------------------------------------
# Tests: Helper function
# ---------------------------------------------------------------------------


class TestParseDeleteCount:
    """Verify _parse_delete_count helper."""

    def test_parses_standard_format(self):
        assert _parse_delete_count("DELETE 42") == 42

    def test_parses_zero(self):
        assert _parse_delete_count("DELETE 0") == 0

    def test_handles_empty_string(self):
        assert _parse_delete_count("") == 0

    def test_handles_unexpected_format(self):
        assert _parse_delete_count("UNEXPECTED") == 0


# ---------------------------------------------------------------------------
# Tests: Cross-tenant deletion request → 404 resource_not_found (R15.7)
# ---------------------------------------------------------------------------


class TestCrossTenantDeletion:
    """Verify cross-tenant deletion requests raise ResourceNotFoundError (R15.7).

    Cross-tenant access must return the same 404 shape as a genuine not-found,
    making it impossible to distinguish between "resource belongs to another
    tenant" and "resource does not exist".

    **Validates: Requirement 15.7**
    """

    async def test_cross_tenant_request_deletion_raises_resource_not_found(
        self, audit_emitter, tenant_id, request_id
    ):
        """request_deletion() with a different requesting_tenant_id raises ResourceNotFoundError."""
        from auth.service import ResourceNotFoundError

        other_tenant_id = str(uuid.uuid4())
        conn = _MockConnection()
        pool = _MockPool(conn)
        service = TenantDeletionService(db_pool=pool, audit_emitter=audit_emitter)

        with pytest.raises(ResourceNotFoundError):
            await service.request_deletion(
                tenant_id, request_id, requesting_tenant_id=other_tenant_id
            )

    async def test_cross_tenant_execute_deletion_raises_resource_not_found(
        self, audit_emitter, tenant_id, request_id
    ):
        """execute_deletion() with a different requesting_tenant_id raises ResourceNotFoundError."""
        from auth.service import ResourceNotFoundError

        other_tenant_id = str(uuid.uuid4())
        conn = _MockConnection()
        pool = _MockPool(conn)
        service = TenantDeletionService(db_pool=pool, audit_emitter=audit_emitter)

        with pytest.raises(ResourceNotFoundError):
            await service.execute_deletion(
                tenant_id, request_id, requesting_tenant_id=other_tenant_id
            )

    async def test_same_tenant_request_deletion_succeeds(
        self, audit_emitter, tenant_id, request_id
    ):
        """request_deletion() with matching requesting_tenant_id succeeds normally."""
        conn = _MockConnection()
        pool = _MockPool(conn)
        service = TenantDeletionService(db_pool=pool, audit_emitter=audit_emitter)

        result = await service.request_deletion(
            tenant_id, request_id, requesting_tenant_id=tenant_id
        )

        assert isinstance(result, DeletionResult)
        assert result.status == "pending_deletion"
        assert result.tenant_id == tenant_id

    async def test_same_tenant_execute_deletion_succeeds(
        self, audit_emitter, tenant_id, request_id
    ):
        """execute_deletion() with matching requesting_tenant_id succeeds normally."""
        conn = _MockConnection(legal_hold_until=None)
        pool = _MockPool(conn)
        service = TenantDeletionService(db_pool=pool, audit_emitter=audit_emitter)

        result = await service.execute_deletion(
            tenant_id, request_id, requesting_tenant_id=tenant_id
        )

        assert isinstance(result, DeletionResult)
        assert result.status == "completed"
        assert result.tenant_id == tenant_id

    async def test_cross_tenant_error_code_is_resource_not_found(
        self, audit_emitter, tenant_id, request_id
    ):
        """The error code must be 'resource_not_found' — indistinguishable from genuine not-found."""
        from auth.service import ResourceNotFoundError

        other_tenant_id = str(uuid.uuid4())
        conn = _MockConnection()
        pool = _MockPool(conn)
        service = TenantDeletionService(db_pool=pool, audit_emitter=audit_emitter)

        with pytest.raises(ResourceNotFoundError) as exc_info:
            await service.request_deletion(
                tenant_id, request_id, requesting_tenant_id=other_tenant_id
            )

        assert exc_info.value.code == "resource_not_found"
        assert "not found" in exc_info.value.message.lower()

    async def test_cross_tenant_no_audit_emitted(
        self, audit_emitter, tenant_id, request_id
    ):
        """Cross-tenant request should NOT emit any audit entry (blocked before action)."""
        other_tenant_id = str(uuid.uuid4())
        conn = _MockConnection()
        pool = _MockPool(conn)
        service = TenantDeletionService(db_pool=pool, audit_emitter=audit_emitter)

        from auth.service import ResourceNotFoundError

        with pytest.raises(ResourceNotFoundError):
            await service.request_deletion(
                tenant_id, request_id, requesting_tenant_id=other_tenant_id
            )

        # No audit events should have been emitted
        assert len(audit_emitter.events) == 0

    async def test_cross_tenant_no_db_operations(
        self, audit_emitter, tenant_id, request_id
    ):
        """Cross-tenant request should NOT perform any DB operations."""
        from auth.service import ResourceNotFoundError

        other_tenant_id = str(uuid.uuid4())
        conn = _MockConnection()
        pool = _MockPool(conn)
        service = TenantDeletionService(db_pool=pool, audit_emitter=audit_emitter)

        with pytest.raises(ResourceNotFoundError):
            await service.execute_deletion(
                tenant_id, request_id, requesting_tenant_id=other_tenant_id
            )

        # No DB operations should have been performed
        assert len(conn.execute_calls) == 0
        assert len(conn.fetchrow_calls) == 0
        assert len(conn.fetch_calls) == 0
