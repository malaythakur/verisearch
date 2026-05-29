"""In-memory AuditEmitter implementation for testing purposes.

Collects emitted audit events in a list for assertion in unit tests.
Not intended for production use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True, slots=True)
class AuditEntry:
    """A captured audit event."""

    action: str
    tenant_id: str | None
    actor: str
    resource: str
    request_id: str
    detail: dict
    timestamp_utc: datetime


class InMemoryAuditEmitter:
    """In-memory audit emitter that collects events for test assertions.

    Satisfies the AuditEmitter protocol defined in backend.audit_log.

    Usage:
        emitter = InMemoryAuditEmitter()
        service = AuthService(db_pool=pool, audit_emitter=emitter)
        # ... perform operations ...
        assert len(emitter.events) == 1
        assert emitter.events[0].action == "auth_failure"
    """

    def __init__(self) -> None:
        self.events: list[AuditEntry] = []

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
        """Record an audit event in memory."""
        self.events.append(
            AuditEntry(
                action=action,
                tenant_id=tenant_id,
                actor=actor,
                resource=resource,
                request_id=request_id,
                detail=detail,
                timestamp_utc=datetime.now(timezone.utc),
            )
        )

    def clear(self) -> None:
        """Clear all recorded events."""
        self.events.clear()

    @property
    def last(self) -> AuditEntry | None:
        """Return the most recently emitted event, or None if empty."""
        return self.events[-1] if self.events else None
