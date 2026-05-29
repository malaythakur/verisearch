"""Audit guards — privileged-action blocking on audit failure (R15.6).

Provides two patterns for enforcing that privileged actions cannot complete
unless the audit append succeeds:

1. `audit_or_block(...)` — a direct helper that calls emit() and lets
   AuditLogUnavailableError propagate if it fails. Callers wrap their
   privileged action after this call.

2. `require_audit(...)` — a decorator that emits an audit entry before
   executing the decorated async function. If the audit fails, the
   function body is never executed.

Both patterns rely on AuditLogUnavailableError being raised by the
AuditEmitter implementation when the append cannot complete (timeout,
DB error, etc.). The caller (e.g., an HTTP handler) catches this
exception and returns HTTP 503 with error code `audit_log_unavailable`.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Protocol, TypeVar, runtime_checkable


@runtime_checkable
class _AuditEmitterProtocol(Protocol):
    """Local protocol reference to avoid circular import with audit_log.__init__."""

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

F = TypeVar("F", bound=Callable[..., Any])


async def audit_or_block(
    audit_emitter: _AuditEmitterProtocol,
    *,
    action: str,
    tenant_id: str | None,
    actor: str,
    resource: str,
    request_id: str,
    detail: dict | None = None,
) -> None:
    """Attempt to emit an audit entry; block the caller on failure.

    This is the simplest pattern for R15.6 enforcement: call this function
    before your privileged action. If it returns normally, the audit succeeded
    and you may proceed. If it raises AuditLogUnavailableError, the privileged
    action must NOT execute.

    Args:
        audit_emitter: The audit emitter to use (satisfies AuditEmitter protocol).
        action: The audit action identifier (e.g., "api_key_created").
        tenant_id: The tenant this event belongs to, or None.
        actor: The identity performing the action.
        resource: The resource being acted upon.
        request_id: The request correlation ID (16–64 code points).
        detail: Additional structured detail payload.

    Raises:
        AuditLogUnavailableError: If the audit append fails. The privileged
            action MUST NOT proceed when this is raised.
    """
    await audit_emitter.emit(
        action=action,
        tenant_id=tenant_id,
        actor=actor,
        resource=resource,
        request_id=request_id,
        detail=detail if detail is not None else {},
    )


def require_audit(
    *,
    action: str,
    audit_emitter_arg: str = "audit_emitter",
    tenant_id_arg: str = "tenant_id",
    actor_arg: str = "actor",
    resource_arg: str = "resource",
    request_id_arg: str = "request_id",
    detail_arg: str | None = "detail",
) -> Callable[[F], F]:
    """Decorator that emits an audit entry before executing a privileged action.

    The decorated function MUST be an async function. The decorator extracts
    audit parameters from the function's keyword arguments (or positional args
    by name) and calls emit() on the audit emitter before the function body
    executes.

    If the audit append fails (raises AuditLogUnavailableError), the decorated
    function is NOT executed and the exception propagates to the caller.

    Args:
        action: The audit action identifier to emit.
        audit_emitter_arg: Name of the kwarg holding the AuditEmitter instance.
        tenant_id_arg: Name of the kwarg holding the tenant_id.
        actor_arg: Name of the kwarg holding the actor.
        resource_arg: Name of the kwarg holding the resource.
        request_id_arg: Name of the kwarg holding the request_id.
        detail_arg: Name of the kwarg holding the detail dict, or None to use {}.

    Returns:
        A decorator that wraps the async function with audit-before-execute logic.

    Example:
        @require_audit(action="api_key_created")
        async def create_api_key(*, audit_emitter, tenant_id, actor, resource, request_id, detail=None):
            # This body only runs if audit succeeded
            ...
    """

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Extract audit parameters from kwargs
            emitter = kwargs[audit_emitter_arg]
            tenant_id = kwargs.get(tenant_id_arg)
            actor = kwargs.get(actor_arg, "unknown")
            resource = kwargs.get(resource_arg, "unknown")
            request_id = kwargs.get(request_id_arg, "unknown-request-id-pad")
            detail = kwargs.get(detail_arg) if detail_arg else None

            # Emit audit entry — if this raises, the function body never runs
            await audit_or_block(
                emitter,
                action=action,
                tenant_id=tenant_id,
                actor=actor,
                resource=resource,
                request_id=request_id,
                detail=detail,
            )

            # Audit succeeded — proceed with the privileged action
            return await fn(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
