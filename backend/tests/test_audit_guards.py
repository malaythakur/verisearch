"""Unit tests for audit guards — privileged-action blocking (Task 5.3).

Validates:
- When audit succeeds, the privileged action executes
- When audit fails, the privileged action is NOT executed
- The error propagated is AuditLogUnavailableError with code `audit_log_unavailable`
- Works with both the decorator and the direct call pattern

**Validates: Requirements 15.6**
"""

from __future__ import annotations

import uuid

import pytest

from audit_log.guards import audit_or_block, require_audit
from audit_log.in_memory import InMemoryAuditEmitter
from audit_log.service import AuditLogUnavailableError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FailingAuditEmitter:
    """An audit emitter that always raises AuditLogUnavailableError."""

    def __init__(self, message: str = "DB connection refused"):
        self._message = message

    async def emit(self, **kwargs) -> None:
        raise AuditLogUnavailableError(self._message)


@pytest.fixture
def valid_request_id() -> str:
    """A valid request_id (16-64 code points)."""
    return f"req-{uuid.uuid4().hex}"


@pytest.fixture
def tenant_id() -> str:
    """A valid tenant UUID string."""
    return str(uuid.uuid4())


@pytest.fixture
def success_emitter() -> InMemoryAuditEmitter:
    """An audit emitter that always succeeds."""
    return InMemoryAuditEmitter()


@pytest.fixture
def failing_emitter() -> FailingAuditEmitter:
    """An audit emitter that always fails."""
    return FailingAuditEmitter()


# ---------------------------------------------------------------------------
# Tests: audit_or_block — direct call pattern
# ---------------------------------------------------------------------------


class TestAuditOrBlockSuccess:
    """When audit succeeds, audit_or_block returns normally and the action proceeds."""

    async def test_returns_normally_on_success(self, success_emitter, tenant_id, valid_request_id):
        """audit_or_block should return without raising when emit succeeds."""
        # Should not raise
        await audit_or_block(
            success_emitter,
            action="api_key_created",
            tenant_id=tenant_id,
            actor="admin",
            resource="api_key/new-key",
            request_id=valid_request_id,
            detail={"key_prefix": "sk_test"},
        )

        # Verify the audit entry was recorded
        assert len(success_emitter.events) == 1
        assert success_emitter.events[0].action == "api_key_created"

    async def test_privileged_action_executes_after_success(
        self, success_emitter, tenant_id, valid_request_id
    ):
        """A privileged action placed after audit_or_block should execute when audit succeeds."""
        action_executed = False

        await audit_or_block(
            success_emitter,
            action="pipeline_created",
            tenant_id=tenant_id,
            actor="user-1",
            resource="pipeline/new",
            request_id=valid_request_id,
        )
        # Privileged action proceeds
        action_executed = True

        assert action_executed is True

    async def test_passes_all_fields_to_emitter(self, success_emitter, tenant_id, valid_request_id):
        """audit_or_block should pass all fields correctly to the emitter."""
        detail = {"reason": "test"}

        await audit_or_block(
            success_emitter,
            action="session_created",
            tenant_id=tenant_id,
            actor="system",
            resource="session/abc",
            request_id=valid_request_id,
            detail=detail,
        )

        event = success_emitter.events[0]
        assert event.action == "session_created"
        assert event.tenant_id == tenant_id
        assert event.actor == "system"
        assert event.resource == "session/abc"
        assert event.request_id == valid_request_id
        assert event.detail == detail

    async def test_default_detail_is_empty_dict(self, success_emitter, tenant_id, valid_request_id):
        """When detail is not provided, it defaults to an empty dict."""
        await audit_or_block(
            success_emitter,
            action="test_action",
            tenant_id=tenant_id,
            actor="tester",
            resource="/test",
            request_id=valid_request_id,
        )

        assert success_emitter.events[0].detail == {}


class TestAuditOrBlockFailure:
    """When audit fails, audit_or_block raises and the privileged action is blocked."""

    async def test_raises_audit_log_unavailable_on_failure(
        self, failing_emitter, tenant_id, valid_request_id
    ):
        """audit_or_block should raise AuditLogUnavailableError when emit fails."""
        with pytest.raises(AuditLogUnavailableError) as exc_info:
            await audit_or_block(
                failing_emitter,
                action="api_key_created",
                tenant_id=tenant_id,
                actor="admin",
                resource="api_key/new-key",
                request_id=valid_request_id,
            )

        assert exc_info.value.code == "audit_log_unavailable"

    async def test_privileged_action_not_executed_on_failure(
        self, failing_emitter, tenant_id, valid_request_id
    ):
        """Code after audit_or_block should NOT execute when audit fails."""
        action_executed = False

        with pytest.raises(AuditLogUnavailableError):
            await audit_or_block(
                failing_emitter,
                action="api_key_created",
                tenant_id=tenant_id,
                actor="admin",
                resource="api_key/new-key",
                request_id=valid_request_id,
            )
            # This line should never be reached
            action_executed = True  # pragma: no cover

        assert action_executed is False

    async def test_error_message_propagates(self, tenant_id, valid_request_id):
        """The error message from the emitter should propagate."""
        emitter = FailingAuditEmitter(message="Connection timed out after 5000ms")

        with pytest.raises(AuditLogUnavailableError) as exc_info:
            await audit_or_block(
                emitter,
                action="test",
                tenant_id=tenant_id,
                actor="tester",
                resource="/test",
                request_id=valid_request_id,
            )

        assert "Connection timed out" in exc_info.value.message


# ---------------------------------------------------------------------------
# Tests: require_audit decorator
# ---------------------------------------------------------------------------


class TestRequireAuditDecoratorSuccess:
    """When audit succeeds, the decorated function executes normally."""

    async def test_decorated_function_executes(self, success_emitter, tenant_id, valid_request_id):
        """The decorated function body should run when audit succeeds."""

        @require_audit(action="api_key_created")
        async def create_api_key(*, audit_emitter, tenant_id, actor, resource, request_id, detail=None):
            return {"key_id": "new-key-123"}

        result = await create_api_key(
            audit_emitter=success_emitter,
            tenant_id=tenant_id,
            actor="admin",
            resource="api_key/new",
            request_id=valid_request_id,
            detail={"key_prefix": "sk_live"},
        )

        assert result == {"key_id": "new-key-123"}

    async def test_audit_entry_emitted_before_action(self, success_emitter, tenant_id, valid_request_id):
        """The audit entry should be emitted before the function body runs."""
        call_order = []

        @require_audit(action="pipeline_created")
        async def create_pipeline(*, audit_emitter, tenant_id, actor, resource, request_id, detail=None):
            # By the time we get here, audit should already be recorded
            call_order.append("action")
            return "pipeline-id"

        # Patch the emitter to track call order
        original_emit = success_emitter.emit

        async def tracking_emit(**kwargs):
            call_order.append("audit")
            await original_emit(**kwargs)

        success_emitter.emit = tracking_emit

        await create_pipeline(
            audit_emitter=success_emitter,
            tenant_id=tenant_id,
            actor="user-1",
            resource="pipeline/new",
            request_id=valid_request_id,
        )

        assert call_order == ["audit", "action"]

    async def test_return_value_preserved(self, success_emitter, tenant_id, valid_request_id):
        """The decorated function's return value should be preserved."""

        @require_audit(action="session_created")
        async def create_session(*, audit_emitter, tenant_id, actor, resource, request_id, detail=None):
            return {"session_id": "sess-abc", "retention_days": 14}

        result = await create_session(
            audit_emitter=success_emitter,
            tenant_id=tenant_id,
            actor="system",
            resource="session/new",
            request_id=valid_request_id,
        )

        assert result == {"session_id": "sess-abc", "retention_days": 14}

    async def test_audit_fields_correct(self, success_emitter, tenant_id, valid_request_id):
        """The decorator should pass the correct fields to the audit emitter."""

        @require_audit(action="deletion_requested")
        async def request_deletion(*, audit_emitter, tenant_id, actor, resource, request_id, detail=None):
            return "ok"

        await request_deletion(
            audit_emitter=success_emitter,
            tenant_id=tenant_id,
            actor="admin",
            resource="tenant/data",
            request_id=valid_request_id,
            detail={"scope": "full"},
        )

        event = success_emitter.events[0]
        assert event.action == "deletion_requested"
        assert event.tenant_id == tenant_id
        assert event.actor == "admin"
        assert event.resource == "tenant/data"
        assert event.request_id == valid_request_id
        assert event.detail == {"scope": "full"}


class TestRequireAuditDecoratorFailure:
    """When audit fails, the decorated function does NOT execute."""

    async def test_function_not_executed_on_audit_failure(
        self, failing_emitter, tenant_id, valid_request_id
    ):
        """The decorated function body should NOT run when audit fails."""
        function_called = False

        @require_audit(action="api_key_created")
        async def create_api_key(*, audit_emitter, tenant_id, actor, resource, request_id, detail=None):
            nonlocal function_called
            function_called = True  # pragma: no cover
            return {"key_id": "should-not-exist"}  # pragma: no cover

        with pytest.raises(AuditLogUnavailableError):
            await create_api_key(
                audit_emitter=failing_emitter,
                tenant_id=tenant_id,
                actor="admin",
                resource="api_key/new",
                request_id=valid_request_id,
            )

        assert function_called is False

    async def test_raises_audit_log_unavailable(self, failing_emitter, tenant_id, valid_request_id):
        """The decorator should propagate AuditLogUnavailableError with correct code."""

        @require_audit(action="api_key_revoked")
        async def revoke_api_key(*, audit_emitter, tenant_id, actor, resource, request_id, detail=None):
            return "revoked"  # pragma: no cover

        with pytest.raises(AuditLogUnavailableError) as exc_info:
            await revoke_api_key(
                audit_emitter=failing_emitter,
                tenant_id=tenant_id,
                actor="admin",
                resource="api_key/key-1",
                request_id=valid_request_id,
            )

        assert exc_info.value.code == "audit_log_unavailable"

    async def test_no_side_effects_on_failure(self, failing_emitter, tenant_id, valid_request_id):
        """No side effects should occur when audit fails."""
        side_effects = []

        @require_audit(action="data_exported")
        async def export_data(*, audit_emitter, tenant_id, actor, resource, request_id, detail=None):
            side_effects.append("exported")  # pragma: no cover
            return "data"  # pragma: no cover

        with pytest.raises(AuditLogUnavailableError):
            await export_data(
                audit_emitter=failing_emitter,
                tenant_id=tenant_id,
                actor="admin",
                resource="export/full",
                request_id=valid_request_id,
            )

        assert side_effects == []


# ---------------------------------------------------------------------------
# Tests: Integration pattern — full privileged action flow
# ---------------------------------------------------------------------------


class TestPrivilegedActionFlow:
    """End-to-end tests showing the full pattern of audit-before-action."""

    async def test_full_flow_success(self, success_emitter, tenant_id, valid_request_id):
        """Full flow: audit succeeds → action executes → result returned."""
        # Simulate a privileged action using the direct pattern
        await audit_or_block(
            success_emitter,
            action="api_key_created",
            tenant_id=tenant_id,
            actor="admin",
            resource="api_key/new",
            request_id=valid_request_id,
            detail={"key_prefix": "sk_live"},
        )

        # If we reach here, audit succeeded — perform the action
        new_key = {"key_id": str(uuid.uuid4()), "prefix": "sk_live"}

        assert new_key["prefix"] == "sk_live"
        assert len(success_emitter.events) == 1

    async def test_full_flow_failure_blocks_action(self, failing_emitter, tenant_id, valid_request_id):
        """Full flow: audit fails → action blocked → no key created."""
        new_key = None

        try:
            await audit_or_block(
                failing_emitter,
                action="api_key_created",
                tenant_id=tenant_id,
                actor="admin",
                resource="api_key/new",
                request_id=valid_request_id,
            )
            # This should not execute
            new_key = {"key_id": "should-not-exist"}  # pragma: no cover
        except AuditLogUnavailableError:
            pass  # Expected — action is blocked

        assert new_key is None

    async def test_decorator_preserves_function_name(self):
        """The require_audit decorator should preserve the function's __name__."""

        @require_audit(action="test_action")
        async def my_privileged_function(*, audit_emitter, tenant_id, actor, resource, request_id, detail=None):
            pass

        assert my_privileged_function.__name__ == "my_privileged_function"

    async def test_decorator_preserves_docstring(self):
        """The require_audit decorator should preserve the function's docstring."""

        @require_audit(action="test_action")
        async def documented_function(*, audit_emitter, tenant_id, actor, resource, request_id, detail=None):
            """This is a documented privileged action."""
            pass

        assert documented_function.__doc__ == "This is a documented privileged action."
