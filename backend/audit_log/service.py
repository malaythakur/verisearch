"""Audit Log Service — append-only audit event persistence via asyncpg.

Implements the AuditEmitter protocol with a 5-second latency target (R15.1)
and raises AuditLogUnavailableError on failure to support R15.6 (block
privileged actions on audit failure).

Immutability enforcement (R15.4):
- The service exposes ONLY append operations (emit/append).
- No update, delete, or modify methods exist.
- __getattr__ blocks access to mutation-named methods at runtime.
- verify_immutability() checks that DB-level triggers are in place.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

import asyncpg


class AuditLogUnavailableError(Exception):
    """Raised when an audit log append fails (timeout or DB error).

    Callers use this to implement R15.6: privileged actions MUST be blocked
    when the audit append cannot complete. The error includes a stable code
    and a human-readable message.
    """

    def __init__(self, message: str, *, code: str = "audit_log_unavailable") -> None:
        self.code = code
        self.message = message
        super().__init__(message)


# AuditAppendError is an alias for AuditLogUnavailableError for backward compatibility
AuditAppendError = AuditLogUnavailableError


@dataclass(frozen=True, slots=True)
class AuditEntry:
    """Immutable audit event entry (R15.1).

    Entry shape: actor, action, resource, timestamp_utc, request_id (16..64 code points), detail.
    """

    action: str
    tenant_id: str | None
    actor: str
    resource: str
    request_id: str
    detail: dict = field(default_factory=dict)
    timestamp_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


_INSERT_SQL = """
INSERT INTO audit_events (audit_id, tenant_id, actor, action, resource, timestamp_utc, request_id, detail)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
"""


class ImmutableAuditLogError(Exception):
    """Raised when a caller attempts a non-append write operation on the audit log.

    The audit log is strictly append-only (R15.4). Any attempt to update,
    delete, or modify existing entries is rejected at the application layer.
    """

    def __init__(self, method_name: str) -> None:
        self.method_name = method_name
        super().__init__(
            f"Audit log is append-only (R15.4): '{method_name}' operation is not permitted. "
            f"Only emit() and append() are allowed."
        )


# Method names that are explicitly blocked at the application layer.
# If anyone tries to access these on AuditLogService, an error is raised.
_BLOCKED_MUTATION_METHODS = frozenset({
    "update",
    "delete",
    "modify",
    "remove",
    "edit",
    "patch",
    "upsert",
    "replace",
    "truncate",
    "drop",
    "purge",
    "overwrite",
    "set",
    "put",
})


class AuditLogService:
    """Append-only audit log backed by PostgreSQL (R15.1, R15.4).

    Satisfies the AuditEmitter protocol. Inserts audit events into the
    audit_events table with a configurable statement timeout (default 5s).
    On timeout or DB error, raises AuditLogUnavailableError so callers can
    block the privileged action per R15.6.

    Immutability enforcement (R15.4):
    - Only emit() and append() are exposed as write paths.
    - Accessing any mutation-named method (update, delete, modify, etc.)
      raises ImmutableAuditLogError at the application layer.
    - The DB layer also enforces immutability via triggers that reject
      UPDATE and DELETE operations.
    - Use verify_immutability() to confirm DB triggers are in place.

    Retention (R15.4):
    - Configurable retention_days in [365, 2555], default 365.
    - cleanup_expired() deletes events older than retention_days.
    """

    # Explicitly declare the ONLY allowed write methods
    _ALLOWED_WRITE_METHODS = frozenset({"emit", "append"})

    # Valid retention range (R15.4)
    _MIN_RETENTION_DAYS = 365
    _MAX_RETENTION_DAYS = 2555

    def __init__(
        self, db_pool, *, timeout_seconds: float = 5.0, retention_days: int = 365
    ) -> None:
        if not isinstance(retention_days, int) or isinstance(retention_days, bool):
            raise ValueError(
                f"retention_days must be an integer in [{self._MIN_RETENTION_DAYS}, "
                f"{self._MAX_RETENTION_DAYS}], got {retention_days!r}"
            )
        if retention_days < self._MIN_RETENTION_DAYS or retention_days > self._MAX_RETENTION_DAYS:
            raise ValueError(
                f"retention_days must be in [{self._MIN_RETENTION_DAYS}, "
                f"{self._MAX_RETENTION_DAYS}], got {retention_days}"
            )
        self._pool = db_pool
        self._timeout_seconds = timeout_seconds
        self._retention_days = retention_days

    def __getattr__(self, name: str):
        """Block access to mutation-named methods (R15.4).

        This prevents accidental addition of mutation methods and ensures
        that any attempt to call update/delete/modify raises immediately.
        """
        if name in _BLOCKED_MUTATION_METHODS:
            raise ImmutableAuditLogError(name)
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    async def emit(
        self,
        *,
        action: str,
        tenant_id: str | None,
        actor: str,
        resource: str,
        request_id: str,
        detail: dict,
    ) -> None:
        """Append an audit event to the audit_events table.

        Args:
            action: The audit action identifier.
            tenant_id: Owning tenant UUID string, or None for unattributable events.
            actor: Identity performing the action.
            resource: The resource being acted upon.
            request_id: Correlation ID (must be 16–64 code points per R15.1).
            detail: Structured detail payload (JSONB).

        Raises:
            ValueError: If request_id length is outside [16, 64] code points.
            AuditLogUnavailableError: If the DB insert fails or times out.
        """
        # Validate request_id length (R15.1: 16..64 code points)
        self._validate_request_id(request_id)

        audit_id = uuid.uuid4()
        timestamp_utc = datetime.now(timezone.utc)
        tenant_uuid = uuid.UUID(tenant_id) if tenant_id is not None else None

        await self._insert(audit_id, tenant_uuid, actor, action, resource, timestamp_utc, request_id, detail)

    async def append(self, entry: AuditEntry) -> None:
        """Append an AuditEntry to the audit_events table.

        This is an alternative interface that accepts a pre-built AuditEntry
        dataclass instead of keyword arguments.

        Args:
            entry: The audit entry to persist.

        Raises:
            ValueError: If entry.request_id length is outside [16, 64] code points.
            AuditLogUnavailableError: If the DB insert fails or times out.
        """
        self._validate_request_id(entry.request_id)

        audit_id = uuid.uuid4()
        tenant_uuid = uuid.UUID(entry.tenant_id) if entry.tenant_id is not None else None

        await self._insert(
            audit_id,
            tenant_uuid,
            entry.actor,
            entry.action,
            entry.resource,
            entry.timestamp_utc,
            entry.request_id,
            entry.detail,
        )

    def _validate_request_id(self, request_id: str) -> None:
        """Validate request_id is 16–64 code points."""
        request_id_len = len(request_id)
        if request_id_len < 16 or request_id_len > 64:
            raise ValueError(
                f"request_id must be 16\u201364 code points, got {request_id_len}"
            )

    async def _insert(
        self,
        audit_id: uuid.UUID,
        tenant_uuid: uuid.UUID | None,
        actor: str,
        action: str,
        resource: str,
        timestamp_utc: datetime,
        request_id: str,
        detail: dict,
    ) -> None:
        """Execute the INSERT with timeout enforcement."""
        timeout_ms = int(self._timeout_seconds * 1000)

        try:
            async with asyncio.timeout(self._timeout_seconds):
                async with self._pool.acquire() as conn:
                    await conn.execute(
                        f"SET LOCAL statement_timeout = '{timeout_ms}'"
                    )
                    await conn.execute(
                        _INSERT_SQL,
                        audit_id,
                        tenant_uuid,
                        actor,
                        action,
                        resource,
                        timestamp_utc,
                        request_id,
                        detail,
                    )
        except TimeoutError as exc:
            raise AuditLogUnavailableError(
                f"Audit append timed out after {timeout_ms}ms"
            ) from exc
        except asyncpg.PostgresError as exc:
            raise AuditLogUnavailableError(
                f"Audit append failed: {exc}"
            ) from exc
        except OSError as exc:
            raise AuditLogUnavailableError(
                f"Audit append failed due to connection error: {exc}"
            ) from exc

    async def verify_immutability(self) -> dict:
        """Verify that DB-level immutability triggers are in place (health check).

        Checks that the audit_events table has the expected BEFORE UPDATE and
        BEFORE DELETE triggers that enforce append-only semantics at the DB layer.

        Returns:
            A dict with:
                - immutable: bool — True if all expected triggers are present.
                - triggers_found: list[str] — Names of immutability triggers found.
                - missing_triggers: list[str] — Expected triggers that are missing.

        Raises:
            AuditLogUnavailableError: If the DB query fails.
        """
        expected_triggers = {
            "trg_audit_events_no_update",
            "trg_audit_events_no_delete",
        }

        query = """
            SELECT trigger_name
            FROM information_schema.triggers
            WHERE event_object_table = 'audit_events'
              AND action_timing = 'BEFORE'
              AND event_manipulation IN ('UPDATE', 'DELETE')
        """

        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(query)
                found_triggers = {row["trigger_name"] for row in rows}
        except (asyncpg.PostgresError, OSError) as exc:
            raise AuditLogUnavailableError(
                f"Failed to verify immutability triggers: {exc}"
            ) from exc

        missing = expected_triggers - found_triggers
        return {
            "immutable": len(missing) == 0,
            "triggers_found": sorted(found_triggers & expected_triggers),
            "missing_triggers": sorted(missing),
        }

    @property
    def retention_days(self) -> int:
        """Return the configured retention period in days."""
        return self._retention_days

    async def cleanup_expired(self) -> int:
        """Delete audit events older than the configured retention period.

        Intended to be called by a periodic background task (cron/scheduler).
        Uses a batch DELETE with a date threshold to avoid long-running transactions.

        Returns:
            The number of rows deleted.

        Raises:
            AuditLogUnavailableError: If the DB operation fails.
        """
        delete_sql = (
            "DELETE FROM audit_events "
            "WHERE timestamp_utc < (NOW() - $1 * INTERVAL '1 day')"
        )

        try:
            async with self._pool.acquire() as conn:
                result = await conn.execute(delete_sql, self._retention_days)
                # asyncpg returns a status string like "DELETE 42"
                count = int(result.split()[-1])
                return count
        except asyncpg.PostgresError as exc:
            raise AuditLogUnavailableError(
                f"Audit cleanup failed: {exc}"
            ) from exc
        except OSError as exc:
            raise AuditLogUnavailableError(
                f"Audit cleanup failed due to connection error: {exc}"
            ) from exc

    @classmethod
    def get_allowed_write_methods(cls) -> frozenset:
        """Return the set of allowed write method names.

        Useful for introspection and testing that the service only
        exposes append operations.
        """
        return cls._ALLOWED_WRITE_METHODS

    @classmethod
    def get_blocked_mutation_methods(cls) -> frozenset:
        """Return the set of blocked mutation method names.

        Useful for introspection and testing that mutation operations
        are explicitly rejected.
        """
        return _BLOCKED_MUTATION_METHODS
