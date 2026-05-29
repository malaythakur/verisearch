"""Session Store service — create, read, expire sessions (R8.1-R8.5).

Implements:
- create(): Create session with retention_days [1, 90] default 14 (R8.1).
- get_memory(): Get session memory for incorporation (R8.2, R8.3).
- add_to_memory(): Add citations and doc_ids to session memory.
- expire_sweep(): Delete expired sessions within 24h, emit audit (R8.4).
- Cross-tenant/expired/missing → uniform 404 session_not_found (R8.5).
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from typing import Any

from backend.session_store.models import (
    DEFAULT_RETENTION_DAYS,
    MAX_RETENTION_DAYS,
    MIN_RETENTION_DAYS,
    Session,
    SessionCitation,
    SessionMemory,
    SessionState,
)


class SessionNotFoundError(Exception):
    """Raised for cross-tenant, expired, or missing sessions (R8.5).

    Returns uniform 404 without disclosing whether the session exists
    in another tenant.
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(f"Session not found: {session_id}")


class InvalidSessionRequestError(Exception):
    """Raised when session creation parameters are invalid."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class SessionService:
    """Service for managing research sessions (R8).

    Provides:
    - create: Create a new session with retention policy (R8.1).
    - get_memory: Get session memory for research/answer context (R8.2).
    - add_to_memory: Add citations and doc_ids after research.
    - expire_sweep: Background sweep to delete expired sessions (R8.4).
    """

    def __init__(self, audit_log: Any | None = None) -> None:
        """Initialize the session service.

        Args:
            audit_log: Audit log service for session_expired events.
        """
        self._sessions: dict[str, Session] = {}
        self._audit_log = audit_log
        self._lock = threading.Lock()

    def create(
        self,
        tenant_id: str,
        retention_days: int | None = None,
    ) -> Session:
        """Create a new session (R8.1).

        Args:
            tenant_id: The owning tenant's ID.
            retention_days: Days to retain the session [1, 90], default 14.

        Returns:
            The created Session with a unique session_id.

        Raises:
            InvalidSessionRequestError: If retention_days is out of range.
        """
        # Apply default and validate retention_days
        days = retention_days if retention_days is not None else DEFAULT_RETENTION_DAYS

        if not isinstance(days, int) or days < MIN_RETENTION_DAYS or days > MAX_RETENTION_DAYS:
            raise InvalidSessionRequestError(
                f"retention_days must be an integer in [{MIN_RETENTION_DAYS}, {MAX_RETENTION_DAYS}]"
            )

        session = Session(
            session_id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            retention_days=days,
            state=SessionState.ACTIVE,
        )

        with self._lock:
            self._sessions[session.session_id] = session

        return session

    def get_memory(self, session_id: str, tenant_id: str) -> dict[str, Any]:
        """Get session memory for incorporation into research/answer (R8.2, R8.3).

        Returns up to 50 most recent citations and 20 most recent doc_ids.

        Args:
            session_id: The session to get memory from.
            tenant_id: The requesting tenant's ID.

        Returns:
            Dict with 'citations' and 'doc_ids' lists.

        Raises:
            SessionNotFoundError: If session doesn't exist, is expired,
                or belongs to another tenant (R8.5).
        """
        session = self._get_session(session_id, tenant_id)
        return session.memory.get_context()

    def add_to_memory(
        self,
        session_id: str,
        tenant_id: str,
        citations: list[dict[str, Any]] | None = None,
        doc_ids: list[str] | None = None,
    ) -> None:
        """Add citations and doc_ids to session memory.

        Args:
            session_id: The session to update.
            tenant_id: The requesting tenant's ID.
            citations: Citations to add (most recent last).
            doc_ids: Document IDs to add (most recent last).

        Raises:
            SessionNotFoundError: If session doesn't exist, is expired,
                or belongs to another tenant.
        """
        session = self._get_session(session_id, tenant_id)

        if citations:
            for cit in citations:
                session.memory.add_citation(
                    SessionCitation(
                        document_id=cit.get("document_id", ""),
                        version=cit.get("version", 1),
                        answer_start=cit.get("answer_start", 0),
                        answer_end=cit.get("answer_end", 0),
                        source_start=cit.get("source_start", 0),
                        source_end=cit.get("source_end", 0),
                    )
                )

        if doc_ids:
            for doc_id in doc_ids:
                session.memory.add_doc_id(doc_id)

    def expire_sweep(self, now: datetime | None = None) -> list[str]:
        """Sweep and delete expired sessions (R8.4).

        Deletes sessions whose expiry time has passed. Emits session_expired
        audit events for each deleted session.

        Args:
            now: Current time for expiry check. Defaults to UTC now.

        Returns:
            List of session_ids that were expired and deleted.
        """
        current = now or datetime.now(timezone.utc)
        expired_ids: list[str] = []

        with self._lock:
            for session_id, session in list(self._sessions.items()):
                if session.state == SessionState.ACTIVE and session.is_expired(current):
                    # Mark as expired and clear memory
                    session.state = SessionState.EXPIRED
                    session.memory.clear()
                    expired_ids.append(session_id)

                    # Emit audit event (R8.4)
                    if self._audit_log is not None:
                        try:
                            self._audit_log.append({
                                "action": "session_expired",
                                "resource": session_id,
                                "tenant_id": session.tenant_id,
                                "timestamp_utc": current.isoformat(),
                                "detail": {
                                    "session_id": session_id,
                                    "tenant_id": session.tenant_id,
                                    "deletion_timestamp": current.isoformat(),
                                },
                            })
                        except Exception:
                            # Audit failure should not prevent expiry
                            pass

        return expired_ids

    def delete_session(self, session_id: str, tenant_id: str) -> None:
        """Explicitly delete a session.

        Args:
            session_id: The session to delete.
            tenant_id: The requesting tenant's ID.

        Raises:
            SessionNotFoundError: If session doesn't exist or belongs to another tenant.
        """
        session = self._get_session(session_id, tenant_id)
        session.state = SessionState.DELETED
        session.memory.clear()

    def _get_session(self, session_id: str, tenant_id: str) -> Session:
        """Get a session with tenant isolation and expiry check (R8.5).

        Returns uniform 404 for:
        - Non-existent sessions
        - Sessions belonging to another tenant
        - Expired/deleted sessions
        """
        with self._lock:
            session = self._sessions.get(session_id)

        if session is None:
            raise SessionNotFoundError(session_id)

        # Cross-tenant check (R8.5)
        if session.tenant_id != tenant_id:
            raise SessionNotFoundError(session_id)

        # Expired/deleted check (R8.5)
        if session.state in (SessionState.EXPIRED, SessionState.DELETED):
            raise SessionNotFoundError(session_id)

        # Check if expired by time but not yet swept
        if session.is_expired():
            raise SessionNotFoundError(session_id)

        return session

    def session_exists(self, session_id: str, tenant_id: str) -> bool:
        """Check if a session exists and is accessible.

        Args:
            session_id: The session to check.
            tenant_id: The requesting tenant's ID.

        Returns:
            True if the session exists, is active, and belongs to the tenant.
        """
        try:
            self._get_session(session_id, tenant_id)
            return True
        except SessionNotFoundError:
            return False
