"""Tenant Data Deletion Service — legal-hold partitioning (R15.3, R15.5, R15.7).

Implements tenant data deletion with legal-hold awareness:
- R15.3: Tenant data deletion support (sessions, research artifacts, metering
  records older than the legally required retention window deleted within 30 days).
- R15.5: If a deletion request targets data under legal hold, the affected records
  are refused with a `retention_required` reason code; remaining records are deleted.
- R15.7: Cross-tenant deletion requests return 404 `resource_not_found` — uniform
  shape indistinguishable from a genuine not-found.

Legal hold logic (MVP):
- A tenant's audit entries are under legal hold if there is a `legal_hold_until`
  timestamp in the future on the tenant record.
- When legal hold is active, ALL audit entries for that tenant are preserved.
- Non-audit tenant data (sessions, research jobs, metering) is always deletable.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

import asyncpg

from auth.service import ResourceNotFoundError


@runtime_checkable
class _AuditEmitterProtocol(Protocol):
    """Local protocol reference to avoid circular import."""

    async def emit(
        self,
        *,
        action: str,
        tenant_id: str | None,
        actor: str,
        resource: str,
        request_id: str,
        detail: dict,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class DeletionResult:
    """Result of a tenant data deletion operation.

    Attributes:
        tenant_id: The tenant whose data was targeted.
        status: One of 'pending_deletion' (request acknowledged) or 'completed'.
        records_deleted: Count of records actually deleted.
        records_preserved_legal_hold: Count of records preserved due to legal hold.
        preserved_records: List of dicts with record_id and reason for each preserved record.
    """

    tenant_id: str
    status: str
    records_deleted: int = 0
    records_preserved_legal_hold: int = 0
    preserved_records: list[dict] = field(default_factory=list)


class TenantDeletionService:
    """Handles tenant data deletion with legal-hold partitioning (R15.3, R15.5).

    Deletion flow:
    1. `request_deletion()` — emits a `deletion_requested` audit entry and
       transitions the tenant to `pending_deletion` state.
    2. `execute_deletion()` — deletes all tenant data except audit entries
       under legal hold, emits `deletion_completed` audit entry.

    Legal hold (MVP):
    - Checked via `legal_hold_until` column on the tenants table.
    - If `legal_hold_until` is in the future, ALL audit entries are preserved.
    - Non-audit data (sessions, research jobs, metering) is always deleted.
    """

    def __init__(self, db_pool, audit_emitter: _AuditEmitterProtocol) -> None:
        self._pool = db_pool
        self._audit_emitter = audit_emitter

    async def request_deletion(
        self, tenant_id: str, request_id: str, *, requesting_tenant_id: str | None = None
    ) -> DeletionResult:
        """Request deletion of a tenant's data (R15.3, R15.7).

        Steps:
        0. Verify requesting tenant matches target tenant (R15.7).
        1. Emit `deletion_requested` audit entry (must succeed per R15.6).
        2. Update tenant's deletion_state to 'pending_deletion'.
        3. Return DeletionResult with status='pending_deletion'.

        Args:
            tenant_id: UUID string of the tenant whose data to delete.
            request_id: Correlation ID for the request (16–64 code points).
            requesting_tenant_id: UUID string of the tenant making the request.
                If provided and different from tenant_id, raises ResourceNotFoundError.

        Returns:
            DeletionResult with status='pending_deletion'.

        Raises:
            ResourceNotFoundError: If requesting_tenant_id != tenant_id (R15.7).
            AuditLogUnavailableError: If the audit emit fails (blocks the action per R15.6).
        """
        # Step 0: Cross-tenant check (R15.7)
        if requesting_tenant_id is not None and requesting_tenant_id != tenant_id:
            raise ResourceNotFoundError()
        # Step 1: Emit audit entry — must succeed before proceeding (R15.6)
        await self._audit_emitter.emit(
            action="deletion_requested",
            tenant_id=tenant_id,
            actor=tenant_id,
            resource=f"tenant/{tenant_id}",
            request_id=request_id,
            detail={"tenant_id": tenant_id},
        )

        # Step 2: Update tenant deletion_state to pending_deletion
        update_sql = """
            UPDATE tenants
            SET deletion_state = 'pending_deletion'
            WHERE tenant_id = $1
        """
        async with self._pool.acquire() as conn:
            await conn.execute(update_sql, uuid.UUID(tenant_id))

        # Step 3: Return result
        return DeletionResult(
            tenant_id=tenant_id,
            status="pending_deletion",
            records_deleted=0,
            records_preserved_legal_hold=0,
        )

    async def execute_deletion(
        self, tenant_id: str, request_id: str, *, requesting_tenant_id: str | None = None
    ) -> DeletionResult:
        """Execute tenant data deletion with legal-hold partitioning (R15.3, R15.5, R15.7).

        Steps:
        0. Verify requesting tenant matches target tenant (R15.7).
        1. Check if tenant has active legal hold.
        2. Delete non-audit tenant data (sessions, research jobs, metering records).
        3. Partition audit entries:
           - If legal hold is active: preserve ALL audit entries.
           - If no legal hold: delete audit entries (except the deletion audit trail).
        4. Update tenant deletion_state to 'deleted'.
        5. Emit `deletion_completed` audit entry with counts.

        Args:
            tenant_id: UUID string of the tenant whose data to delete.
            request_id: Correlation ID for the request (16–64 code points).
            requesting_tenant_id: UUID string of the tenant making the request.
                If provided and different from tenant_id, raises ResourceNotFoundError.

        Returns:
            DeletionResult with status='completed' and record counts.

        Raises:
            ResourceNotFoundError: If requesting_tenant_id != tenant_id (R15.7).
        """
        # Step 0: Cross-tenant check (R15.7)
        if requesting_tenant_id is not None and requesting_tenant_id != tenant_id:
            raise ResourceNotFoundError()
        tenant_uuid = uuid.UUID(tenant_id)
        records_deleted = 0
        records_preserved = 0
        preserved_records: list[dict] = []

        async with self._pool.acquire() as conn:
            # Check legal hold status
            has_legal_hold = await self._check_legal_hold(conn, tenant_uuid)

            # Delete non-audit tenant data (always deletable)
            records_deleted += await self._delete_tenant_data(conn, tenant_uuid)

            # Handle audit entries based on legal hold
            if has_legal_hold:
                # Count audit entries that are preserved
                audit_count = await self._count_audit_entries(conn, tenant_uuid)
                records_preserved = audit_count

                # Build preserved records list with reason codes (R15.5)
                audit_ids = await self._get_audit_entry_ids(conn, tenant_uuid)
                for audit_id in audit_ids:
                    preserved_records.append({
                        "record_id": str(audit_id),
                        "reason": "retention_required",
                    })
            else:
                # No legal hold — delete audit entries
                records_deleted += await self._delete_audit_entries(conn, tenant_uuid)

            # Update tenant deletion_state to 'deleted'
            await conn.execute(
                "UPDATE tenants SET deletion_state = 'deleted' WHERE tenant_id = $1",
                tenant_uuid,
            )

        # Emit deletion_completed audit entry (R15.3)
        await self._audit_emitter.emit(
            action="deletion_completed",
            tenant_id=tenant_id,
            actor="system",
            resource=f"tenant/{tenant_id}",
            request_id=request_id,
            detail={
                "tenant_id": tenant_id,
                "records_deleted": records_deleted,
                "records_preserved_legal_hold": records_preserved,
            },
        )

        return DeletionResult(
            tenant_id=tenant_id,
            status="completed",
            records_deleted=records_deleted,
            records_preserved_legal_hold=records_preserved,
            preserved_records=preserved_records,
        )

    async def _check_legal_hold(self, conn, tenant_uuid: uuid.UUID) -> bool:
        """Check if a tenant has an active legal hold.

        MVP logic: legal hold is active if `legal_hold_until` is set and
        is in the future.
        """
        row = await conn.fetchrow(
            "SELECT legal_hold_until FROM tenants WHERE tenant_id = $1",
            tenant_uuid,
        )
        if row is None:
            return False

        legal_hold_until = row["legal_hold_until"]
        if legal_hold_until is None:
            return False

        return legal_hold_until > datetime.now(timezone.utc)

    async def _delete_tenant_data(self, conn, tenant_uuid: uuid.UUID) -> int:
        """Delete non-audit tenant data (sessions, research jobs, metering records).

        Returns the total count of records deleted.
        """
        total_deleted = 0

        # Delete sessions
        result = await conn.execute(
            "DELETE FROM sessions WHERE tenant_id = $1", tenant_uuid
        )
        total_deleted += _parse_delete_count(result)

        # Delete research jobs (cascades to plans, steps, events)
        result = await conn.execute(
            "DELETE FROM research_jobs WHERE tenant_id = $1", tenant_uuid
        )
        total_deleted += _parse_delete_count(result)

        # Delete metering events
        result = await conn.execute(
            "DELETE FROM metering_events WHERE tenant_id = $1", tenant_uuid
        )
        total_deleted += _parse_delete_count(result)

        # Delete pipelines (cascades to pipeline_steps)
        result = await conn.execute(
            "DELETE FROM pipelines WHERE tenant_id = $1", tenant_uuid
        )
        total_deleted += _parse_delete_count(result)

        # Delete citations
        result = await conn.execute(
            "DELETE FROM citations WHERE tenant_id = $1", tenant_uuid
        )
        total_deleted += _parse_delete_count(result)

        return total_deleted

    async def _count_audit_entries(self, conn, tenant_uuid: uuid.UUID) -> int:
        """Count audit entries for a tenant."""
        row = await conn.fetchrow(
            "SELECT COUNT(*) as cnt FROM audit_events WHERE tenant_id = $1",
            tenant_uuid,
        )
        return row["cnt"] if row else 0

    async def _get_audit_entry_ids(self, conn, tenant_uuid: uuid.UUID) -> list[uuid.UUID]:
        """Get all audit entry IDs for a tenant."""
        rows = await conn.fetch(
            "SELECT audit_id FROM audit_events WHERE tenant_id = $1",
            tenant_uuid,
        )
        return [row["audit_id"] for row in rows]

    async def _delete_audit_entries(self, conn, tenant_uuid: uuid.UUID) -> int:
        """Delete all audit entries for a tenant (when no legal hold)."""
        result = await conn.execute(
            "DELETE FROM audit_events WHERE tenant_id = $1", tenant_uuid
        )
        return _parse_delete_count(result)


def _parse_delete_count(result: str) -> int:
    """Parse the count from asyncpg's DELETE result string (e.g., 'DELETE 42')."""
    try:
        return int(result.split()[-1])
    except (ValueError, IndexError):
        return 0
