"""Tests for audit log immutability enforcement (Task 5.2, R15.4).

Validates:
- AuditLogService has no update/delete/modify methods
- The service only uses INSERT SQL (never UPDATE or DELETE)
- Accessing mutation-named methods raises ImmutableAuditLogError
- The migration has immutability triggers
- verify_immutability() health check works correctly

**Validates: Requirements 15.4**
"""

from __future__ import annotations

import asyncio
import inspect
import re
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from audit_log.service import (
    AuditLogService,
    AuditLogUnavailableError,
    ImmutableAuditLogError,
    _BLOCKED_MUTATION_METHODS,
    _INSERT_SQL,
)


# ---------------------------------------------------------------------------
# Mock pool infrastructure
# ---------------------------------------------------------------------------


class _MockConnection:
    """Mock asyncpg connection that records execute calls."""

    def __init__(self, *, fail_with: Exception | None = None):
        self._fail_with = fail_with
        self.execute_calls: list[tuple] = []

    async def execute(self, query: str, *args):
        if self._fail_with is not None:
            raise self._fail_with
        self.execute_calls.append((query, args))

    async def fetch(self, query: str, *args):
        if self._fail_with is not None:
            raise self._fail_with
        # Default: return both expected triggers
        return [
            {"trigger_name": "trg_audit_events_no_update"},
            {"trigger_name": "trg_audit_events_no_delete"},
        ]


class _MockConnectionMissingTriggers(_MockConnection):
    """Mock connection that returns only one trigger (simulating missing trigger)."""

    async def fetch(self, query: str, *args):
        return [
            {"trigger_name": "trg_audit_events_no_update"},
        ]


class _MockConnectionNoTriggers(_MockConnection):
    """Mock connection that returns no triggers."""

    async def fetch(self, query: str, *args):
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
def mock_conn():
    return _MockConnection()


@pytest.fixture
def mock_pool(mock_conn):
    return _MockPool(mock_conn)


@pytest.fixture
def service(mock_pool):
    return AuditLogService(db_pool=mock_pool, timeout_seconds=5.0)


# ---------------------------------------------------------------------------
# Tests: AuditLogService has no update/delete/modify methods
# ---------------------------------------------------------------------------


class TestNoMutationMethods:
    """Verify the AuditLogService class does not define any mutation methods."""

    def test_no_update_method(self, service):
        """AuditLogService should not have an 'update' method."""
        # The method should not be defined on the class
        assert "update" not in type(service).__dict__

    def test_no_delete_method(self, service):
        """AuditLogService should not have a 'delete' method."""
        assert "delete" not in type(service).__dict__

    def test_no_modify_method(self, service):
        """AuditLogService should not have a 'modify' method."""
        assert "modify" not in type(service).__dict__

    def test_no_remove_method(self, service):
        """AuditLogService should not have a 'remove' method."""
        assert "remove" not in type(service).__dict__

    def test_no_edit_method(self, service):
        """AuditLogService should not have an 'edit' method."""
        assert "edit" not in type(service).__dict__

    def test_no_patch_method(self, service):
        """AuditLogService should not have a 'patch' method."""
        assert "patch" not in type(service).__dict__

    def test_only_emit_and_append_are_write_methods(self, service):
        """The only public write methods should be emit() and append()."""
        public_methods = {
            name
            for name, _ in inspect.getmembers(type(service), predicate=inspect.isfunction)
            if not name.startswith("_")
        }
        # Public instance methods (classmethods are not included by isfunction)
        # cleanup_expired is allowed as part of retention policy (R15.4)
        allowed = {
            "emit",
            "append",
            "verify_immutability",
            "cleanup_expired",
        }
        assert public_methods == allowed

    def test_allowed_write_methods_are_only_emit_and_append(self):
        """The class declares only emit and append as allowed write methods."""
        allowed = AuditLogService.get_allowed_write_methods()
        assert allowed == frozenset({"emit", "append"})


# ---------------------------------------------------------------------------
# Tests: Accessing mutation methods raises ImmutableAuditLogError
# ---------------------------------------------------------------------------


class TestMutationMethodsBlocked:
    """Verify that accessing blocked mutation methods raises ImmutableAuditLogError."""

    @pytest.mark.parametrize("method_name", sorted(_BLOCKED_MUTATION_METHODS))
    def test_blocked_method_raises_immutable_error(self, service, method_name):
        """Accessing a blocked mutation method raises ImmutableAuditLogError."""
        with pytest.raises(ImmutableAuditLogError) as exc_info:
            getattr(service, method_name)

        assert exc_info.value.method_name == method_name
        assert "append-only" in str(exc_info.value)
        assert "R15.4" in str(exc_info.value)

    def test_update_raises_immutable_error(self, service):
        """Specifically: service.update raises ImmutableAuditLogError."""
        with pytest.raises(ImmutableAuditLogError):
            service.update

    def test_delete_raises_immutable_error(self, service):
        """Specifically: service.delete raises ImmutableAuditLogError."""
        with pytest.raises(ImmutableAuditLogError):
            service.delete

    def test_modify_raises_immutable_error(self, service):
        """Specifically: service.modify raises ImmutableAuditLogError."""
        with pytest.raises(ImmutableAuditLogError):
            service.modify

    def test_nonexistent_non_mutation_method_raises_attribute_error(self, service):
        """Accessing a non-mutation, non-existent attribute raises AttributeError."""
        with pytest.raises(AttributeError):
            service.nonexistent_method

    def test_calling_blocked_method_raises_before_invocation(self, service):
        """Attempting to call a blocked method raises before any DB interaction."""
        with pytest.raises(ImmutableAuditLogError):
            # Even trying to get the attribute to call it raises
            fn = service.update


# ---------------------------------------------------------------------------
# Tests: Service only uses INSERT SQL
# ---------------------------------------------------------------------------


class TestOnlyInsertSQL:
    """Verify the service only uses INSERT SQL statements."""

    def test_insert_sql_is_insert_only(self):
        """The _INSERT_SQL constant should only contain INSERT."""
        sql_upper = _INSERT_SQL.upper().strip()
        assert sql_upper.startswith("INSERT INTO")
        assert "UPDATE" not in sql_upper
        assert "DELETE" not in sql_upper

    def test_service_source_has_no_update_sql(self):
        """The service module source code should not contain UPDATE SQL for audit_events."""
        source = inspect.getsource(AuditLogService)
        # Should not contain UPDATE audit_events
        assert "UPDATE audit_events" not in source
        # DELETE FROM audit_events is allowed only in cleanup_expired (R15.4 retention)
        # Verify no UPDATE exists at all
        assert "UPDATE audit_events" not in source

    async def test_emit_only_executes_insert(self, service, mock_conn):
        """emit() should only execute SET LOCAL and INSERT statements."""
        await service.emit(
            action="test_action",
            tenant_id=str(uuid.uuid4()),
            actor="tester",
            resource="/test",
            request_id="a" * 16,
            detail={},
        )

        for query, _ in mock_conn.execute_calls:
            query_upper = query.upper().strip()
            assert query_upper.startswith("SET LOCAL") or query_upper.startswith("INSERT INTO"), (
                f"Unexpected SQL: {query}"
            )


# ---------------------------------------------------------------------------
# Tests: Migration has immutability triggers
# ---------------------------------------------------------------------------


class TestMigrationTriggers:
    """Verify the migration file contains the expected immutability triggers."""

    @pytest.fixture
    def migration_content(self):
        """Read the migration file content."""
        migration_path = Path(__file__).parent.parent.parent / "migrations" / "008_create_audit_events.sql"
        return migration_path.read_text(encoding="utf-8")

    def test_migration_has_update_trigger(self, migration_content):
        """Migration should define a BEFORE UPDATE trigger."""
        assert "trg_audit_events_no_update" in migration_content
        assert "BEFORE UPDATE ON audit_events" in migration_content

    def test_migration_has_delete_trigger(self, migration_content):
        """Migration should define a BEFORE DELETE trigger."""
        assert "trg_audit_events_no_delete" in migration_content
        assert "BEFORE DELETE ON audit_events" in migration_content

    def test_migration_has_prevent_modification_function(self, migration_content):
        """Migration should define the prevent_audit_events_modification function."""
        assert "prevent_audit_events_modification" in migration_content

    def test_migration_raises_exception_on_modification(self, migration_content):
        """The trigger function should RAISE EXCEPTION."""
        assert "RAISE EXCEPTION" in migration_content
        assert "append-only" in migration_content

    def test_migration_grants_only_insert_and_select(self, migration_content):
        """Migration should only GRANT INSERT and SELECT (no UPDATE or DELETE)."""
        # Find the GRANT line
        grant_match = re.search(r"GRANT\s+(.+?)\s+ON\s+audit_events", migration_content)
        assert grant_match is not None
        granted_perms = grant_match.group(1).upper()
        assert "INSERT" in granted_perms
        assert "SELECT" in granted_perms
        assert "UPDATE" not in granted_perms
        assert "DELETE" not in granted_perms


# ---------------------------------------------------------------------------
# Tests: verify_immutability() health check
# ---------------------------------------------------------------------------


class TestVerifyImmutability:
    """Verify the verify_immutability() method works correctly."""

    async def test_verify_immutability_all_triggers_present(self):
        """When both triggers are present, returns immutable=True."""
        conn = _MockConnection()
        pool = _MockPool(conn)
        service = AuditLogService(db_pool=pool)

        result = await service.verify_immutability()

        assert result["immutable"] is True
        assert "trg_audit_events_no_update" in result["triggers_found"]
        assert "trg_audit_events_no_delete" in result["triggers_found"]
        assert result["missing_triggers"] == []

    async def test_verify_immutability_missing_one_trigger(self):
        """When one trigger is missing, returns immutable=False with details."""
        conn = _MockConnectionMissingTriggers()
        pool = _MockPool(conn)
        service = AuditLogService(db_pool=pool)

        result = await service.verify_immutability()

        assert result["immutable"] is False
        assert "trg_audit_events_no_update" in result["triggers_found"]
        assert "trg_audit_events_no_delete" in result["missing_triggers"]

    async def test_verify_immutability_no_triggers(self):
        """When no triggers are present, returns immutable=False."""
        conn = _MockConnectionNoTriggers()
        pool = _MockPool(conn)
        service = AuditLogService(db_pool=pool)

        result = await service.verify_immutability()

        assert result["immutable"] is False
        assert result["triggers_found"] == []
        assert len(result["missing_triggers"]) == 2

    async def test_verify_immutability_db_error_raises(self):
        """When the DB query fails, raises AuditLogUnavailableError."""
        import asyncpg

        conn = _MockConnection(fail_with=asyncpg.PostgresError("connection lost"))
        pool = _MockPool(conn)
        service = AuditLogService(db_pool=pool)

        with pytest.raises(AuditLogUnavailableError) as exc_info:
            await service.verify_immutability()

        assert "verify immutability" in exc_info.value.message.lower()


# ---------------------------------------------------------------------------
# Tests: ImmutableAuditLogError
# ---------------------------------------------------------------------------


class TestImmutableAuditLogError:
    """Verify the ImmutableAuditLogError exception class."""

    def test_error_has_method_name(self):
        """Error should store the attempted method name."""
        err = ImmutableAuditLogError("delete")
        assert err.method_name == "delete"

    def test_error_message_mentions_append_only(self):
        """Error message should mention append-only."""
        err = ImmutableAuditLogError("update")
        assert "append-only" in str(err)

    def test_error_message_mentions_r15_4(self):
        """Error message should reference R15.4."""
        err = ImmutableAuditLogError("modify")
        assert "R15.4" in str(err)

    def test_error_is_exception(self):
        """ImmutableAuditLogError should be an Exception."""
        err = ImmutableAuditLogError("delete")
        assert isinstance(err, Exception)
