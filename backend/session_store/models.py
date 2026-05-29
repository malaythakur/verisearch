"""Session Store data models (R8.1, R8.2).

Core types:
- Session: A research session with tenant scope, retention, and memory.
- SessionMemory: Bounded memory with ≤50 citations and ≤20 doc_ids.
"""

from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any


# Memory bounds (R8.2)
MAX_CITATIONS = 50
MAX_DOC_IDS = 20

# Retention bounds (R8.1)
MIN_RETENTION_DAYS = 1
MAX_RETENTION_DAYS = 90
DEFAULT_RETENTION_DAYS = 14


class SessionState(str, Enum):
    """Session lifecycle states."""

    ACTIVE = "active"
    EXPIRED = "expired"
    DELETED = "deleted"


@dataclass
class SessionCitation:
    """A citation stored in session memory."""

    citation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str = ""
    version: int = 1
    answer_start: int = 0
    answer_end: int = 0
    source_start: int = 0
    source_end: int = 0
    added_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class SessionMemory:
    """Bounded session memory (R8.2).

    Maintains:
    - ≤50 most recent citations (ring buffer behavior)
    - ≤20 most recent unique doc_ids (ring buffer behavior)
    """

    _citations: deque[SessionCitation] = field(
        default_factory=lambda: deque(maxlen=MAX_CITATIONS)
    )
    _doc_ids: deque[str] = field(
        default_factory=lambda: deque(maxlen=MAX_DOC_IDS)
    )

    @property
    def citations(self) -> list[SessionCitation]:
        """Get all citations in recency order (most recent last)."""
        return list(self._citations)

    @property
    def doc_ids(self) -> list[str]:
        """Get all doc_ids in recency order (most recent last)."""
        return list(self._doc_ids)

    @property
    def citation_count(self) -> int:
        """Number of citations currently stored."""
        return len(self._citations)

    @property
    def doc_id_count(self) -> int:
        """Number of doc_ids currently stored."""
        return len(self._doc_ids)

    def add_citation(self, citation: SessionCitation) -> None:
        """Add a citation to memory. Evicts oldest if at capacity (R8.2)."""
        self._citations.append(citation)

    def add_doc_id(self, doc_id: str) -> None:
        """Add a doc_id to memory. Evicts oldest if at capacity (R8.2).

        Only adds if not already present (unique constraint).
        If already present, moves it to the most recent position.
        """
        # Remove if already present (to move to end)
        try:
            self._doc_ids.remove(doc_id)
        except ValueError:
            pass
        self._doc_ids.append(doc_id)

    def clear(self) -> None:
        """Clear all memory."""
        self._citations.clear()
        self._doc_ids.clear()

    def get_context(self) -> dict[str, Any]:
        """Get memory as context for research/answer requests (R8.2).

        Returns the most recent citations and doc_ids for incorporation.
        """
        return {
            "citations": [
                {
                    "document_id": c.document_id,
                    "version": c.version,
                    "answer_start": c.answer_start,
                    "answer_end": c.answer_end,
                    "source_start": c.source_start,
                    "source_end": c.source_end,
                }
                for c in self._citations
            ],
            "doc_ids": list(self._doc_ids),
        }


@dataclass
class Session:
    """A research session with tenant scope and bounded memory (R8.1, R8.2).

    Sessions persist research context across calls within a tenant.
    Memory is bounded to ≤50 citations and ≤20 doc_ids.
    Sessions expire after retention_days and are swept within 24h.
    """

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    retention_days: int = DEFAULT_RETENTION_DAYS
    state: SessionState = SessionState.ACTIVE
    memory: SessionMemory = field(default_factory=SessionMemory)

    @property
    def expires_at(self) -> datetime:
        """Calculate expiry time from creation + retention_days."""
        return self.created_at + timedelta(days=self.retention_days)

    def is_expired(self, now: datetime | None = None) -> bool:
        """Check if the session has expired.

        Args:
            now: Current time. Defaults to UTC now.

        Returns:
            True if the session has passed its expiry time.
        """
        if self.state in (SessionState.EXPIRED, SessionState.DELETED):
            return True
        current = now or datetime.now(timezone.utc)
        return current >= self.expires_at
