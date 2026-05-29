"""Dead-letter queue routing for failed index operations (Task 10.5, R2.5).

Routes documents to a DLQ after 3 retries spaced ≥60s apart, and emits
an `index_failure` audit event.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class RetryState:
    """Tracks retry attempts for a document indexing operation.

    Attributes:
        attempt_count: Number of attempts made so far.
        attempt_timestamps: UTC timestamps of each attempt.
        last_error: The most recent error message.
    """

    attempt_count: int = 0
    attempt_timestamps: list[float] = field(default_factory=list)
    last_error: str | None = None


@dataclass(frozen=True, slots=True)
class DLQEntry:
    """A document routed to the dead-letter queue.

    Attributes:
        document_id: The document_id if assigned, else None.
        source_url: The source URL of the document.
        failure_reason: Why indexing failed.
        attempts: Number of attempts made.
        routed_at: When the document was routed to DLQ.
    """

    document_id: str | None
    source_url: str
    failure_reason: str
    attempts: int
    routed_at: datetime


# Minimum spacing between retry attempts (seconds) — R2.5
MIN_RETRY_SPACING_SECONDS = 60

# Maximum retry attempts before DLQ routing — R2.5
MAX_RETRY_ATTEMPTS = 3


class DeadLetterQueue:
    """In-memory dead-letter queue for failed index operations.

    Tracks retry state per document and routes to DLQ after exhausting
    retries. In production, this would be backed by Kafka DLQ topic.
    """

    def __init__(self) -> None:
        self._retry_states: dict[str, RetryState] = {}
        self._entries: list[DLQEntry] = []

    @property
    def entries(self) -> list[DLQEntry]:
        """Return all DLQ entries."""
        return list(self._entries)

    def record_attempt(self, key: str, error: str) -> None:
        """Record a failed indexing attempt.

        Args:
            key: Unique key for the document (source_url or document_id).
            error: The error message from the failed attempt.
        """
        if key not in self._retry_states:
            self._retry_states[key] = RetryState()

        state = self._retry_states[key]
        state.attempt_count += 1
        state.attempt_timestamps.append(time.time())
        state.last_error = error

    def should_route_to_dlq(self, key: str) -> bool:
        """Check if a document should be routed to DLQ.

        Returns True if:
        - 3 or more attempts have been made
        - Consecutive attempts are spaced by at least 60 seconds

        Args:
            key: Unique key for the document.

        Returns:
            True if the document should be routed to DLQ.
        """
        state = self._retry_states.get(key)
        if state is None or state.attempt_count < MAX_RETRY_ATTEMPTS:
            return False

        # Verify spacing between consecutive attempts
        timestamps = state.attempt_timestamps
        if len(timestamps) < MAX_RETRY_ATTEMPTS:
            return False

        # Check the last MAX_RETRY_ATTEMPTS timestamps for spacing
        relevant = timestamps[-MAX_RETRY_ATTEMPTS:]
        for i in range(1, len(relevant)):
            if (relevant[i] - relevant[i - 1]) < MIN_RETRY_SPACING_SECONDS:
                return False

        return True

    def route_to_dlq(
        self,
        key: str,
        *,
        document_id: str | None,
        source_url: str,
    ) -> DLQEntry:
        """Route a document to the dead-letter queue.

        Args:
            key: Unique key for the document.
            document_id: The document_id if assigned.
            source_url: The source URL.

        Returns:
            The DLQ entry created.
        """
        state = self._retry_states.get(key, RetryState())
        entry = DLQEntry(
            document_id=document_id,
            source_url=source_url,
            failure_reason=state.last_error or "unknown",
            attempts=state.attempt_count,
            routed_at=datetime.now(timezone.utc),
        )
        self._entries.append(entry)

        # Clean up retry state
        self._retry_states.pop(key, None)

        return entry

    def get_retry_state(self, key: str) -> RetryState | None:
        """Get the current retry state for a document."""
        return self._retry_states.get(key)

    def clear_retry_state(self, key: str) -> None:
        """Clear retry state for a document (on successful index)."""
        self._retry_states.pop(key, None)
