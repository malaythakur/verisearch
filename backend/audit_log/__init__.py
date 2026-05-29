"""Audit Log - Append-only ledger of privileged actions and security events.

Provides the AuditEmitter protocol that subsystems use to emit audit entries
without coupling to the concrete audit log implementation.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from audit_log.deletion import DeletionResult, TenantDeletionService
from audit_log.guards import audit_or_block, require_audit
from audit_log.service import (
    AuditAppendError,
    AuditEntry,
    AuditLogService,
    AuditLogUnavailableError,
    ImmutableAuditLogError,
)


@runtime_checkable
class AuditEmitter(Protocol):
    """Protocol for emitting audit events (R15.1).

    Subsystems accept an AuditEmitter to decouple audit emission from the
    concrete append-only store implementation. This enables testing with
    in-memory collectors and swapping backends without changing callers.

    Entry shape per R15.1:
        actor, action, resource, timestamp_utc, request_id (16..64 code points), detail.
    """

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
        """Emit a single audit event.

        Args:
            action: The audit action identifier (e.g., "auth_failure", "session_expired").
            tenant_id: The tenant this event belongs to, or None for unattributable
                       events (e.g., auth_failure with unknown API key per R13.6).
            actor: The identity performing the action (e.g., "anonymous", a tenant_id).
            resource: The resource being acted upon (e.g., endpoint path).
            request_id: The request correlation ID (16–64 code points per R15.1).
            detail: Additional structured detail (must NOT contain sensitive values
                    like bearer tokens per R13.6).
        """
        ...  # pragma: no cover


__all__ = [
    "AuditEmitter",
    "AuditAppendError",
    "AuditEntry",
    "AuditLogService",
    "AuditLogUnavailableError",
    "DeletionResult",
    "ImmutableAuditLogError",
    "TenantDeletionService",
    "audit_or_block",
    "require_audit",
]
