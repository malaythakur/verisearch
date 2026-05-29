"""Property-based tests for Audit Log — append-only invariant (Property 40).

**Validates: Requirements 15.4**

Property 40: Audit_Log is append-only.
For any sequence of operations on the audit log, the set of entries only grows
(never shrinks or mutates). Specifically:

1. After N emit() calls, the audit log contains exactly N entries (monotonically growing).
2. Attempting to access any blocked mutation method always raises ImmutableAuditLogError,
   regardless of the sequence of prior operations.
3. The entries in the log after N appends are exactly the N entries that were appended
   (no mutations, no losses).

Uses InMemoryAuditEmitter for growth/content verification and AuditLogService with
a mock pool for mutation-blocking verification.
"""

from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from audit_log.in_memory import InMemoryAuditEmitter
from audit_log.service import (
    AuditLogService,
    ImmutableAuditLogError,
    _BLOCKED_MUTATION_METHODS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockConnection:
    """Mock asyncpg connection that records execute calls."""

    async def execute(self, query: str, *args):
        pass

    async def fetch(self, query: str, *args):
        return [
            {"trigger_name": "trg_audit_events_no_update"},
            {"trigger_name": "trg_audit_events_no_delete"},
        ]


class _MockPool:
    """Mock asyncpg pool that yields a mock connection."""

    @asynccontextmanager
    async def acquire(self):
        yield _MockConnection()


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Generate valid action names
st_action = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "Pd")),
    min_size=1,
    max_size=50,
)

# Generate valid tenant IDs (UUID strings)
st_tenant_id = st.uuids().map(str)

# Generate valid actor names
st_actor = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "Pd", "Zs")),
    min_size=1,
    max_size=50,
)

# Generate valid resource paths
st_resource = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "Pd", "Po")),
    min_size=1,
    max_size=100,
)

# Generate valid request IDs (16-64 code points)
st_request_id = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N")),
    min_size=16,
    max_size=64,
)

# Generate detail dicts (simple JSON-serializable)
st_detail = st.fixed_dictionaries({}).map(lambda _: {}) | st.fixed_dictionaries(
    {"key": st.text(min_size=1, max_size=20)}
)

# A single audit operation (the kwargs for emit)
st_audit_op = st.fixed_dictionaries({
    "action": st_action,
    "tenant_id": st_tenant_id,
    "actor": st_actor,
    "resource": st_resource,
    "request_id": st_request_id,
    "detail": st_detail,
})

# A sequence of 1-50 audit operations
st_audit_ops = st.lists(st_audit_op, min_size=1, max_size=50)

# A blocked mutation method name
st_blocked_method = st.sampled_from(sorted(_BLOCKED_MUTATION_METHODS))

# A sequence of blocked method access attempts interspersed with emits
st_mixed_ops = st.lists(
    st.one_of(
        st_audit_op.map(lambda op: ("emit", op)),
        st_blocked_method.map(lambda m: ("blocked", m)),
    ),
    min_size=1,
    max_size=50,
)


# ---------------------------------------------------------------------------
# Property Tests
# ---------------------------------------------------------------------------


class TestAppendOnlyGrowth:
    """Property: After N emit() calls, the log contains exactly N entries."""

    @given(ops=st_audit_ops)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    async def test_log_grows_monotonically(self, ops: list[dict]):
        """
        For any sequence of N emit() calls, the audit log contains exactly N
        entries and the count never decreases between operations.

        **Validates: Requirements 15.4**
        """
        emitter = InMemoryAuditEmitter()
        previous_count = 0

        for i, op in enumerate(ops):
            await emitter.emit(**op)
            current_count = len(emitter.events)

            # Count must be strictly greater than previous (monotonically growing)
            assert current_count == previous_count + 1, (
                f"After emit #{i + 1}, expected {previous_count + 1} entries, got {current_count}"
            )
            previous_count = current_count

        # Final count must equal total number of emits
        assert len(emitter.events) == len(ops)


class TestMutationAlwaysBlocked:
    """Property: Blocked mutation methods always raise ImmutableAuditLogError."""

    @given(ops=st_mixed_ops)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    async def test_mutation_blocked_regardless_of_prior_operations(self, ops: list[tuple]):
        """
        For any sequence of operations (emits interspersed with mutation attempts),
        every access to a blocked mutation method raises ImmutableAuditLogError,
        regardless of how many successful emits preceded it.

        **Validates: Requirements 15.4**
        """
        pool = _MockPool()
        service = AuditLogService(db_pool=pool, timeout_seconds=5.0)

        for op_type, payload in ops:
            if op_type == "emit":
                await service.emit(**payload)
            elif op_type == "blocked":
                with pytest.raises(ImmutableAuditLogError) as exc_info:
                    getattr(service, payload)
                assert exc_info.value.method_name == payload
                assert "append-only" in str(exc_info.value)


class TestEntriesPreservedExactly:
    """Property: Entries after N appends are exactly the N entries appended."""

    @given(ops=st_audit_ops)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    async def test_entries_match_appended_data(self, ops: list[dict]):
        """
        For any sequence of N emit() calls with specific data, the entries
        stored in the log match exactly the data that was appended — no
        mutations, no losses, no reordering.

        **Validates: Requirements 15.4**
        """
        emitter = InMemoryAuditEmitter()

        for op in ops:
            await emitter.emit(**op)

        # Verify count
        assert len(emitter.events) == len(ops)

        # Verify each entry matches the corresponding emit call
        for i, (entry, op) in enumerate(zip(emitter.events, ops)):
            assert entry.action == op["action"], f"Entry {i}: action mismatch"
            assert entry.tenant_id == op["tenant_id"], f"Entry {i}: tenant_id mismatch"
            assert entry.actor == op["actor"], f"Entry {i}: actor mismatch"
            assert entry.resource == op["resource"], f"Entry {i}: resource mismatch"
            assert entry.request_id == op["request_id"], f"Entry {i}: request_id mismatch"
            assert entry.detail == op["detail"], f"Entry {i}: detail mismatch"

    @given(ops=st_audit_ops, blocked_methods=st.lists(st_blocked_method, min_size=1, max_size=10))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    async def test_entries_unchanged_after_failed_mutations(
        self, ops: list[dict], blocked_methods: list[str]
    ):
        """
        After N successful emits, attempting blocked mutations does not alter
        the existing entries — the log remains exactly as it was.

        **Validates: Requirements 15.4**
        """
        emitter = InMemoryAuditEmitter()

        # Emit all entries
        for op in ops:
            await emitter.emit(**op)

        # Snapshot the entries
        snapshot = list(emitter.events)

        # Attempt blocked mutations on the AuditLogService (which wraps the same protocol)
        pool = _MockPool()
        service = AuditLogService(db_pool=pool, timeout_seconds=5.0)

        for method_name in blocked_methods:
            with pytest.raises(ImmutableAuditLogError):
                getattr(service, method_name)

        # Verify the emitter's entries are unchanged
        assert len(emitter.events) == len(snapshot)
        for i, (current, original) in enumerate(zip(emitter.events, snapshot)):
            assert current == original, f"Entry {i} was mutated after blocked method access"


# ---------------------------------------------------------------------------
# Property 39: Privileged actions block on audit failure
# ---------------------------------------------------------------------------

"""
Property 39: Privileged actions block on audit-log append failure.

**Validates: Requirements 15.6**

For any privileged action wrapped with audit_or_block or require_audit,
if the audit emitter fails, the action MUST NOT execute. Specifically:

1. For any audit parameters and any failure mode, when the emitter fails:
   - audit_or_block() raises AuditLogUnavailableError
   - Code after audit_or_block() is never reached
   - The error code is always "audit_log_unavailable"

2. For any decorated function using @require_audit, when the emitter fails:
   - The function body is never executed
   - AuditLogUnavailableError propagates to the caller
   - No side effects from the function body occur
"""

from audit_log.guards import audit_or_block, require_audit
from audit_log.service import AuditLogUnavailableError


# ---------------------------------------------------------------------------
# Strategies for Property 39
# ---------------------------------------------------------------------------

# Failure modes that an audit emitter might encounter
st_failure_mode = st.sampled_from([
    "timeout",
    "db_error",
    "connection_error",
])

# Map failure modes to realistic error messages
_FAILURE_MESSAGES = {
    "timeout": "Audit append timed out after 5000ms",
    "db_error": "could not serialize access due to concurrent update",
    "connection_error": "connection refused: host=db port=5432",
}

# Random privileged action results (to verify they're never produced on failure)
st_privileged_result = st.one_of(
    st.dictionaries(
        keys=st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=("L",))),
        values=st.text(min_size=1, max_size=20),
        min_size=1,
        max_size=5,
    ),
    st.integers(),
    st.text(min_size=1, max_size=50),
    st.lists(st.integers(), min_size=1, max_size=10),
)


# ---------------------------------------------------------------------------
# Helpers for Property 39
# ---------------------------------------------------------------------------


class FailingAuditEmitter:
    """An audit emitter that always raises AuditLogUnavailableError.

    Simulates various failure modes (timeout, DB error, connection error).
    """

    def __init__(self, failure_mode: str = "timeout"):
        self._message = _FAILURE_MESSAGES.get(failure_mode, f"Unknown failure: {failure_mode}")

    async def emit(self, **kwargs) -> None:
        raise AuditLogUnavailableError(self._message)


class SideEffectTracker:
    """Tracks whether a privileged action body was executed."""

    def __init__(self):
        self.called = False
        self.call_count = 0
        self.results: list = []

    def mark_called(self, result=None):
        self.called = True
        self.call_count += 1
        if result is not None:
            self.results.append(result)


# ---------------------------------------------------------------------------
# Property Tests — audit_or_block blocks on failure
# ---------------------------------------------------------------------------


class TestAuditOrBlockBlocksOnFailure:
    """Property 39 (part 1): audit_or_block raises and blocks action on any failure."""

    @given(
        audit_params=st_audit_op,
        failure_mode=st_failure_mode,
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    async def test_raises_audit_log_unavailable_for_any_params_and_failure(
        self, audit_params: dict, failure_mode: str
    ):
        """
        For any audit parameters and any failure mode, audit_or_block()
        always raises AuditLogUnavailableError.

        **Validates: Requirements 15.6**
        """
        emitter = FailingAuditEmitter(failure_mode=failure_mode)

        with pytest.raises(AuditLogUnavailableError) as exc_info:
            await audit_or_block(
                emitter,
                action=audit_params["action"],
                tenant_id=audit_params["tenant_id"],
                actor=audit_params["actor"],
                resource=audit_params["resource"],
                request_id=audit_params["request_id"],
                detail=audit_params["detail"],
            )

        # Error code is always "audit_log_unavailable"
        assert exc_info.value.code == "audit_log_unavailable"

    @given(
        audit_params=st_audit_op,
        failure_mode=st_failure_mode,
        privileged_result=st_privileged_result,
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    async def test_code_after_audit_or_block_never_reached(
        self, audit_params: dict, failure_mode: str, privileged_result
    ):
        """
        For any audit parameters and any failure mode, code placed after
        audit_or_block() is never reached — the privileged action never
        produces its result.

        **Validates: Requirements 15.6**
        """
        emitter = FailingAuditEmitter(failure_mode=failure_mode)
        tracker = SideEffectTracker()

        with pytest.raises(AuditLogUnavailableError):
            await audit_or_block(
                emitter,
                action=audit_params["action"],
                tenant_id=audit_params["tenant_id"],
                actor=audit_params["actor"],
                resource=audit_params["resource"],
                request_id=audit_params["request_id"],
                detail=audit_params["detail"],
            )
            # This code must NEVER execute
            tracker.mark_called(privileged_result)

        # Verify the action body was never reached
        assert tracker.called is False
        assert tracker.call_count == 0
        assert tracker.results == []


# ---------------------------------------------------------------------------
# Property Tests — require_audit decorator blocks on failure
# ---------------------------------------------------------------------------


class TestRequireAuditBlocksOnFailure:
    """Property 39 (part 2): @require_audit blocks function execution on any failure."""

    @given(
        audit_params=st_audit_op,
        failure_mode=st_failure_mode,
        privileged_result=st_privileged_result,
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    async def test_decorated_function_never_executes_on_failure(
        self, audit_params: dict, failure_mode: str, privileged_result
    ):
        """
        For any decorated function using @require_audit, when the emitter
        fails, the function body is never executed regardless of the
        failure mode or audit parameters.

        **Validates: Requirements 15.6**
        """
        emitter = FailingAuditEmitter(failure_mode=failure_mode)
        tracker = SideEffectTracker()

        @require_audit(action=audit_params["action"])
        async def privileged_action(
            *, audit_emitter, tenant_id, actor, resource, request_id, detail=None
        ):
            tracker.mark_called(privileged_result)
            return privileged_result

        with pytest.raises(AuditLogUnavailableError):
            await privileged_action(
                audit_emitter=emitter,
                tenant_id=audit_params["tenant_id"],
                actor=audit_params["actor"],
                resource=audit_params["resource"],
                request_id=audit_params["request_id"],
                detail=audit_params["detail"],
            )

        # Function body was never executed
        assert tracker.called is False
        assert tracker.call_count == 0
        assert tracker.results == []

    @given(
        audit_params=st_audit_op,
        failure_mode=st_failure_mode,
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    async def test_audit_log_unavailable_propagates_to_caller(
        self, audit_params: dict, failure_mode: str
    ):
        """
        For any decorated function, when the emitter fails,
        AuditLogUnavailableError propagates to the caller with the
        correct error code.

        **Validates: Requirements 15.6**
        """
        emitter = FailingAuditEmitter(failure_mode=failure_mode)

        @require_audit(action=audit_params["action"])
        async def privileged_action(
            *, audit_emitter, tenant_id, actor, resource, request_id, detail=None
        ):
            return "should-never-return"

        with pytest.raises(AuditLogUnavailableError) as exc_info:
            await privileged_action(
                audit_emitter=emitter,
                tenant_id=audit_params["tenant_id"],
                actor=audit_params["actor"],
                resource=audit_params["resource"],
                request_id=audit_params["request_id"],
                detail=audit_params["detail"],
            )

        assert exc_info.value.code == "audit_log_unavailable"

    @given(
        audit_params=st_audit_op,
        failure_mode=st_failure_mode,
        num_side_effects=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    async def test_no_side_effects_from_function_body(
        self, audit_params: dict, failure_mode: str, num_side_effects: int
    ):
        """
        For any decorated function with multiple side effects in its body,
        when the emitter fails, none of the side effects occur.

        **Validates: Requirements 15.6**
        """
        emitter = FailingAuditEmitter(failure_mode=failure_mode)
        side_effects: list[str] = []

        @require_audit(action=audit_params["action"])
        async def privileged_action_with_side_effects(
            *, audit_emitter, tenant_id, actor, resource, request_id, detail=None
        ):
            # Multiple side effects that should never happen
            for i in range(num_side_effects):
                side_effects.append(f"effect_{i}")
            return {"effects": num_side_effects}

        with pytest.raises(AuditLogUnavailableError):
            await privileged_action_with_side_effects(
                audit_emitter=emitter,
                tenant_id=audit_params["tenant_id"],
                actor=audit_params["actor"],
                resource=audit_params["resource"],
                request_id=audit_params["request_id"],
                detail=audit_params["detail"],
            )

        # No side effects occurred
        assert side_effects == []


# ---------------------------------------------------------------------------
# Property 41: Deletion partitions correctly under legal hold
# ---------------------------------------------------------------------------

"""
Property 41: Tenant deletion partitions records correctly under legal hold.

**Validates: Requirements 15.3, 15.5**

For any tenant deletion request and any classification of the tenant's records
into (deletable, legally_held), executing the deletion:

1. When legal_hold_until is in the future (active hold):
   - records_preserved_legal_hold == total audit entry count
   - records_deleted includes only non-audit data
   - No DELETE on audit_events is issued

2. When legal_hold_until is None or in the past (no hold):
   - records_preserved_legal_hold == 0
   - records_deleted includes both audit and non-audit data
   - DELETE on audit_events IS issued

3. Non-audit data is ALWAYS deleted regardless of legal hold status
"""

from audit_log.deletion import DeletionResult, TenantDeletionService


# ---------------------------------------------------------------------------
# Strategies for Property 41
# ---------------------------------------------------------------------------

# Generate future timestamps (legal hold active)
st_future_hold = st.datetimes(
    min_value=datetime(2030, 1, 1),
    max_value=datetime(2099, 12, 31),
    timezones=st.just(timezone.utc),
)

# Generate past timestamps (legal hold expired)
st_past_hold = st.datetimes(
    min_value=datetime(2000, 1, 1),
    max_value=datetime(2020, 12, 31),
    timezones=st.just(timezone.utc),
)

# Generate legal_hold_until: None, past, or future
st_legal_hold = st.one_of(
    st.none(),
    st_past_hold,
    st_future_hold,
)

# Generate counts of audit entries (0-100)
st_audit_count = st.integers(min_value=0, max_value=100)

# Generate counts of non-audit records per table (0-100)
st_non_audit_counts = st.fixed_dictionaries({
    "sessions": st.integers(min_value=0, max_value=100),
    "research_jobs": st.integers(min_value=0, max_value=100),
    "metering_events": st.integers(min_value=0, max_value=100),
    "pipelines": st.integers(min_value=0, max_value=100),
    "citations": st.integers(min_value=0, max_value=100),
})


# ---------------------------------------------------------------------------
# Mock infrastructure for Property 41
# ---------------------------------------------------------------------------


class _DeletionMockConnection:
    """Mock asyncpg connection for deletion property tests.

    Tracks all queries issued and returns configurable results based on
    the legal_hold_until, audit_count, and delete_counts parameters.
    """

    def __init__(
        self,
        *,
        legal_hold_until: datetime | None = None,
        audit_count: int = 0,
        delete_counts: dict[str, int] | None = None,
    ):
        self._legal_hold_until = legal_hold_until
        self._audit_count = audit_count
        self._delete_counts = delete_counts or {}
        self.execute_calls: list[tuple[str, tuple]] = []
        self.fetchrow_calls: list[tuple[str, tuple]] = []
        self.fetch_calls: list[tuple[str, tuple]] = []

    async def execute(self, query: str, *args):
        self.execute_calls.append((query, args))
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
            return [{"audit_id": uuid.uuid4()} for _ in range(self._audit_count)]
        return []


class _DeletionMockPool:
    """Mock asyncpg pool for deletion property tests."""

    def __init__(self, conn: _DeletionMockConnection):
        self.conn = conn

    @asynccontextmanager
    async def acquire(self):
        yield self.conn


# ---------------------------------------------------------------------------
# Property Tests — Deletion partitions correctly under legal hold
# ---------------------------------------------------------------------------


class TestDeletionLegalHoldActive:
    """Property 41 (part 1): When legal hold is active, audit entries are preserved."""

    @given(
        audit_count=st_audit_count,
        non_audit_counts=st_non_audit_counts,
        legal_hold_until=st_future_hold,
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    async def test_active_hold_preserves_all_audit_entries(
        self, audit_count: int, non_audit_counts: dict[str, int], legal_hold_until: datetime
    ):
        """
        For any tenant with legal_hold_until in the future and any number of
        audit entries, records_preserved_legal_hold equals the total audit
        entry count.

        **Validates: Requirements 15.5**
        """
        tenant_id = str(uuid.uuid4())
        request_id = f"req-{uuid.uuid4().hex}"

        conn = _DeletionMockConnection(
            legal_hold_until=legal_hold_until,
            audit_count=audit_count,
            delete_counts=non_audit_counts,
        )
        pool = _DeletionMockPool(conn)
        emitter = InMemoryAuditEmitter()
        service = TenantDeletionService(db_pool=pool, audit_emitter=emitter)

        result = await service.execute_deletion(tenant_id, request_id)

        # All audit entries must be preserved
        assert result.records_preserved_legal_hold == audit_count

    @given(
        audit_count=st_audit_count,
        non_audit_counts=st_non_audit_counts,
        legal_hold_until=st_future_hold,
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    async def test_active_hold_records_deleted_excludes_audit(
        self, audit_count: int, non_audit_counts: dict[str, int], legal_hold_until: datetime
    ):
        """
        For any tenant with active legal hold, records_deleted includes only
        non-audit data (sessions, research_jobs, metering_events, pipelines,
        citations).

        **Validates: Requirements 15.3, 15.5**
        """
        tenant_id = str(uuid.uuid4())
        request_id = f"req-{uuid.uuid4().hex}"

        conn = _DeletionMockConnection(
            legal_hold_until=legal_hold_until,
            audit_count=audit_count,
            delete_counts=non_audit_counts,
        )
        pool = _DeletionMockPool(conn)
        emitter = InMemoryAuditEmitter()
        service = TenantDeletionService(db_pool=pool, audit_emitter=emitter)

        result = await service.execute_deletion(tenant_id, request_id)

        # records_deleted should be exactly the sum of non-audit deletions
        expected_deleted = sum(non_audit_counts.values())
        assert result.records_deleted == expected_deleted

    @given(
        audit_count=st_audit_count,
        non_audit_counts=st_non_audit_counts,
        legal_hold_until=st_future_hold,
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    async def test_active_hold_no_delete_on_audit_events(
        self, audit_count: int, non_audit_counts: dict[str, int], legal_hold_until: datetime
    ):
        """
        For any tenant with active legal hold, no DELETE query is issued
        against the audit_events table.

        **Validates: Requirements 15.5**
        """
        tenant_id = str(uuid.uuid4())
        request_id = f"req-{uuid.uuid4().hex}"

        conn = _DeletionMockConnection(
            legal_hold_until=legal_hold_until,
            audit_count=audit_count,
            delete_counts=non_audit_counts,
        )
        pool = _DeletionMockPool(conn)
        emitter = InMemoryAuditEmitter()
        service = TenantDeletionService(db_pool=pool, audit_emitter=emitter)

        await service.execute_deletion(tenant_id, request_id)

        # No DELETE FROM audit_events should have been issued
        delete_audit_calls = [
            c for c in conn.execute_calls
            if "DELETE" in c[0] and "audit_events" in c[0]
        ]
        assert len(delete_audit_calls) == 0


class TestDeletionNoLegalHold:
    """Property 41 (part 2): When no legal hold, all data including audit is deleted."""

    @given(
        audit_count=st.integers(min_value=0, max_value=100),
        non_audit_counts=st_non_audit_counts,
        legal_hold_until=st.one_of(st.none(), st_past_hold),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    async def test_no_hold_preserves_nothing(
        self, audit_count: int, non_audit_counts: dict[str, int], legal_hold_until: datetime | None
    ):
        """
        For any tenant with legal_hold_until None or in the past,
        records_preserved_legal_hold is always 0.

        **Validates: Requirements 15.3**
        """
        tenant_id = str(uuid.uuid4())
        request_id = f"req-{uuid.uuid4().hex}"

        delete_counts = {**non_audit_counts, "audit_events": audit_count}
        conn = _DeletionMockConnection(
            legal_hold_until=legal_hold_until,
            audit_count=audit_count,
            delete_counts=delete_counts,
        )
        pool = _DeletionMockPool(conn)
        emitter = InMemoryAuditEmitter()
        service = TenantDeletionService(db_pool=pool, audit_emitter=emitter)

        result = await service.execute_deletion(tenant_id, request_id)

        assert result.records_preserved_legal_hold == 0

    @given(
        audit_count=st.integers(min_value=0, max_value=100),
        non_audit_counts=st_non_audit_counts,
        legal_hold_until=st.one_of(st.none(), st_past_hold),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    async def test_no_hold_deletes_audit_and_non_audit(
        self, audit_count: int, non_audit_counts: dict[str, int], legal_hold_until: datetime | None
    ):
        """
        For any tenant without legal hold, records_deleted includes both
        audit and non-audit data.

        **Validates: Requirements 15.3**
        """
        tenant_id = str(uuid.uuid4())
        request_id = f"req-{uuid.uuid4().hex}"

        delete_counts = {**non_audit_counts, "audit_events": audit_count}
        conn = _DeletionMockConnection(
            legal_hold_until=legal_hold_until,
            audit_count=audit_count,
            delete_counts=delete_counts,
        )
        pool = _DeletionMockPool(conn)
        emitter = InMemoryAuditEmitter()
        service = TenantDeletionService(db_pool=pool, audit_emitter=emitter)

        result = await service.execute_deletion(tenant_id, request_id)

        expected_deleted = sum(non_audit_counts.values()) + audit_count
        assert result.records_deleted == expected_deleted

    @given(
        audit_count=st.integers(min_value=0, max_value=100),
        non_audit_counts=st_non_audit_counts,
        legal_hold_until=st.one_of(st.none(), st_past_hold),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    async def test_no_hold_issues_delete_on_audit_events(
        self, audit_count: int, non_audit_counts: dict[str, int], legal_hold_until: datetime | None
    ):
        """
        For any tenant without legal hold, a DELETE query IS issued
        against the audit_events table.

        **Validates: Requirements 15.3**
        """
        tenant_id = str(uuid.uuid4())
        request_id = f"req-{uuid.uuid4().hex}"

        delete_counts = {**non_audit_counts, "audit_events": audit_count}
        conn = _DeletionMockConnection(
            legal_hold_until=legal_hold_until,
            audit_count=audit_count,
            delete_counts=delete_counts,
        )
        pool = _DeletionMockPool(conn)
        emitter = InMemoryAuditEmitter()
        service = TenantDeletionService(db_pool=pool, audit_emitter=emitter)

        await service.execute_deletion(tenant_id, request_id)

        # DELETE FROM audit_events SHOULD have been issued
        delete_audit_calls = [
            c for c in conn.execute_calls
            if "DELETE" in c[0] and "audit_events" in c[0]
        ]
        assert len(delete_audit_calls) == 1


class TestDeletionNonAuditAlwaysDeleted:
    """Property 41 (part 3): Non-audit data is ALWAYS deleted regardless of legal hold."""

    @given(
        audit_count=st_audit_count,
        non_audit_counts=st_non_audit_counts,
        legal_hold_until=st_legal_hold,
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    async def test_non_audit_data_always_deleted(
        self, audit_count: int, non_audit_counts: dict[str, int], legal_hold_until: datetime | None
    ):
        """
        For any tenant regardless of legal hold status, non-audit data
        (sessions, research_jobs, metering_events, pipelines, citations)
        is always deleted.

        **Validates: Requirements 15.3**
        """
        tenant_id = str(uuid.uuid4())
        request_id = f"req-{uuid.uuid4().hex}"

        delete_counts = {**non_audit_counts}
        if legal_hold_until is None or legal_hold_until <= datetime.now(timezone.utc):
            delete_counts["audit_events"] = audit_count

        conn = _DeletionMockConnection(
            legal_hold_until=legal_hold_until,
            audit_count=audit_count,
            delete_counts=delete_counts,
        )
        pool = _DeletionMockPool(conn)
        emitter = InMemoryAuditEmitter()
        service = TenantDeletionService(db_pool=pool, audit_emitter=emitter)

        result = await service.execute_deletion(tenant_id, request_id)

        # Non-audit data should always be included in records_deleted
        non_audit_total = sum(non_audit_counts.values())
        assert result.records_deleted >= non_audit_total

        # Verify DELETE queries were issued for all non-audit tables
        non_audit_tables = ["sessions", "research_jobs", "metering_events", "pipelines", "citations"]
        delete_queries = [c[0] for c in conn.execute_calls if "DELETE" in c[0]]
        for table in non_audit_tables:
            assert any(table in q for q in delete_queries), (
                f"Expected DELETE on {table} but none found"
            )

    @given(
        audit_count=st_audit_count,
        non_audit_counts=st_non_audit_counts,
        legal_hold_until=st_legal_hold,
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    async def test_deletion_completed_audit_event_emitted(
        self, audit_count: int, non_audit_counts: dict[str, int], legal_hold_until: datetime | None
    ):
        """
        For any tenant deletion regardless of legal hold status, a
        deletion_completed audit event is emitted with correct counts.

        **Validates: Requirements 15.3**
        """
        tenant_id = str(uuid.uuid4())
        request_id = f"req-{uuid.uuid4().hex}"

        delete_counts = {**non_audit_counts}
        if legal_hold_until is None or legal_hold_until <= datetime.now(timezone.utc):
            delete_counts["audit_events"] = audit_count

        conn = _DeletionMockConnection(
            legal_hold_until=legal_hold_until,
            audit_count=audit_count,
            delete_counts=delete_counts,
        )
        pool = _DeletionMockPool(conn)
        emitter = InMemoryAuditEmitter()
        service = TenantDeletionService(db_pool=pool, audit_emitter=emitter)

        result = await service.execute_deletion(tenant_id, request_id)

        # A deletion_completed audit event must be emitted
        assert len(emitter.events) == 1
        event = emitter.events[0]
        assert event.action == "deletion_completed"
        assert event.tenant_id == tenant_id
        assert event.detail["records_deleted"] == result.records_deleted
        assert event.detail["records_preserved_legal_hold"] == result.records_preserved_legal_hold
