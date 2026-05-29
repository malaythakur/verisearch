"""Research Agent event stream — monotonic event_id and replay buffer (R7.3).

Provides:
- EventBuffer: Per-job durable buffer with ≥24h retention for Last-Event-ID replay.
- EventEmitter: Emits events with strictly monotonic event_id per job.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from datetime import datetime, timezone

from backend.research_agent.models import EventType, ResearchEvent


# Minimum retention for replay buffer (24 hours in seconds)
REPLAY_BUFFER_RETENTION_SECONDS = 24 * 60 * 60


class EventBuffer:
    """Per-job durable event buffer supporting Last-Event-ID replay (R7.3).

    Stores events per job_id with strictly monotonic event_id values.
    Supports replay from a given event_id for SSE reconnection.
    Retains events for at least 24 hours after the terminal event.
    """

    def __init__(self, retention_seconds: int = REPLAY_BUFFER_RETENTION_SECONDS) -> None:
        self._retention_seconds = retention_seconds
        self._buffers: dict[str, list[ResearchEvent]] = defaultdict(list)
        self._terminal_times: dict[str, float] = {}
        self._lock = threading.Lock()

    def append(self, event: ResearchEvent) -> None:
        """Append an event to the job's buffer.

        Args:
            event: The event to store. Must have a valid job_id and event_id.
        """
        with self._lock:
            buffer = self._buffers[event.job_id]
            # Verify strict monotonicity
            if buffer and event.event_id <= buffer[-1].event_id:
                raise ValueError(
                    f"Event ID {event.event_id} is not strictly greater than "
                    f"last event ID {buffer[-1].event_id} for job {event.job_id}"
                )
            buffer.append(event)

            # Track terminal event time for retention
            if event.type in (EventType.DONE, EventType.ERROR):
                self._terminal_times[event.job_id] = time.time()

    def replay(self, job_id: str, last_event_id: int | None = None) -> list[ResearchEvent]:
        """Replay events for a job from after the given event_id (R7.3).

        Args:
            job_id: The job to replay events for.
            last_event_id: If provided, only events with event_id > last_event_id
                are returned. If None, all events are returned.

        Returns:
            List of events in original order with event_id > last_event_id.
        """
        with self._lock:
            buffer = self._buffers.get(job_id, [])
            if last_event_id is None:
                return list(buffer)
            return [e for e in buffer if e.event_id > last_event_id]

    def get_all_events(self, job_id: str) -> list[ResearchEvent]:
        """Get all events for a job.

        Args:
            job_id: The job to get events for.

        Returns:
            All events for the job in order.
        """
        with self._lock:
            return list(self._buffers.get(job_id, []))

    def cleanup_expired(self) -> list[str]:
        """Remove buffers that have exceeded the retention period.

        Returns:
            List of job_ids whose buffers were cleaned up.
        """
        now = time.time()
        expired: list[str] = []

        with self._lock:
            for job_id, terminal_time in list(self._terminal_times.items()):
                if now - terminal_time > self._retention_seconds:
                    del self._buffers[job_id]
                    del self._terminal_times[job_id]
                    expired.append(job_id)

        return expired

    def has_job(self, job_id: str) -> bool:
        """Check if a job has any events in the buffer."""
        with self._lock:
            return job_id in self._buffers and len(self._buffers[job_id]) > 0


class EventEmitter:
    """Emits research events with strictly monotonic event_id per job (R7.3).

    Manages the event_id counter per job and writes to the EventBuffer.
    """

    def __init__(self, buffer: EventBuffer | None = None) -> None:
        self._buffer = buffer or EventBuffer()
        self._counters: dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    @property
    def buffer(self) -> EventBuffer:
        """Access the underlying event buffer."""
        return self._buffer

    def emit(self, job_id: str, event_type: EventType, payload: dict | None = None) -> ResearchEvent:
        """Emit an event with the next monotonic event_id for the job.

        Args:
            job_id: The job this event belongs to.
            event_type: The type of event.
            payload: Optional event payload data.

        Returns:
            The created ResearchEvent with assigned event_id.
        """
        with self._lock:
            self._counters[job_id] += 1
            event_id = self._counters[job_id]

        event = ResearchEvent(
            event_id=event_id,
            job_id=job_id,
            type=event_type,
            payload=payload or {},
            emitted_at=datetime.now(timezone.utc),
        )

        self._buffer.append(event)
        return event

    def get_last_event_id(self, job_id: str) -> int:
        """Get the last emitted event_id for a job."""
        with self._lock:
            return self._counters.get(job_id, 0)
