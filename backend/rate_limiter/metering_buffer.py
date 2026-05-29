"""Durable local buffer for metering events during pipeline outages.

Implements R14.5: When the metering pipeline (DB) is unreachable, events
are buffered locally. When the buffer reaches 80% capacity, a
`metering_delivery_degraded` audit event is emitted (once per threshold
crossing). The buffer uses a bounded deque — oldest events are dropped
when full.

The flush mechanism is called externally (e.g., by a background task with
bounded backoff).
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Callable, Coroutine, Protocol

import asyncpg

from backend.rate_limiter.metering import MeteringEvent

logger = logging.getLogger(__name__)

# Threshold at which the audit event is emitted
_DEGRADED_THRESHOLD = 0.8


class AuditEmitter(Protocol):
    """Protocol for emitting audit events."""

    async def __call__(self, action: str, resource: str, detail: dict[str, Any]) -> None: ...


class DurableMeteringBuffer:
    """Bounded local buffer for metering events during pipeline outages.

    When the metering pipeline (DB) is unavailable, events are stored in
    a deque. When the buffer reaches 80% of max_size, a
    `metering_delivery_degraded` audit event is emitted once per threshold
    crossing.

    Args:
        max_size: Maximum number of events the buffer can hold. When full,
            oldest events are dropped (deque maxlen behavior).
        audit_emitter: Optional async callable to emit audit events. Should
            accept (action, resource, detail) parameters.
    """

    def __init__(
        self,
        max_size: int = 10000,
        audit_emitter: AuditEmitter | None = None,
    ) -> None:
        self._max_size = max_size
        self._buffer: deque[MeteringEvent] = deque(maxlen=max_size)
        self._audit_emitter = audit_emitter
        self._degraded_audit_emitted = False

    @property
    def fill_ratio(self) -> float:
        """Current fill level as a ratio [0.0, 1.0]."""
        if self._max_size == 0:
            return 1.0
        return len(self._buffer) / self._max_size

    @property
    def size(self) -> int:
        """Current number of buffered events."""
        return len(self._buffer)

    @property
    def max_size(self) -> int:
        """Maximum buffer capacity."""
        return self._max_size

    async def buffer_event(self, event: MeteringEvent) -> None:
        """Add an event to the local buffer.

        If the buffer is at max capacity, the oldest event is dropped
        (deque maxlen behavior). When the buffer crosses the 80% fill
        threshold, emits a `metering_delivery_degraded` audit event
        (once per crossing).

        Args:
            event: The metering event to buffer.
        """
        self._buffer.append(event)

        # Check if we crossed the 80% threshold
        if not self._degraded_audit_emitted and self.fill_ratio >= _DEGRADED_THRESHOLD:
            self._degraded_audit_emitted = True
            await self._emit_degraded_audit()

    async def flush(self, db_pool: asyncpg.Pool) -> int:
        """Attempt to flush buffered events to the database.

        Processes events in FIFO order. Stops on the first DB error
        and returns the count of successfully flushed events.

        After a successful full flush, resets the degraded-audit flag
        so it can fire again on the next threshold crossing.

        Args:
            db_pool: An asyncpg connection pool to write events to.

        Returns:
            The number of events successfully flushed.
        """
        flushed = 0

        while self._buffer:
            event = self._buffer[0]  # Peek at the front
            try:
                async with db_pool.acquire() as conn:
                    await conn.execute(
                        """
                        INSERT INTO metering_events (
                            metering_event_id, request_id, tenant_id, endpoint,
                            timestamp_utc, response_status, tokens_consumed, dedup_key
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                        ON CONFLICT (dedup_key) DO NOTHING
                        """,
                        __import__("uuid").uuid4(),
                        event.request_id,
                        __import__("uuid").UUID(event.tenant_id),
                        event.endpoint,
                        event.timestamp_utc,
                        event.response_status,
                        event.tokens_consumed,
                        event.dedup_key,
                    )
                # Successfully written — remove from buffer
                self._buffer.popleft()
                flushed += 1
            except Exception:
                logger.warning(
                    "Failed to flush metering event during buffer drain, "
                    "stopping flush. %d events flushed so far.",
                    flushed,
                )
                break

        # If buffer is now empty, reset the degraded flag so it can
        # fire again on the next fill-up cycle
        if not self._buffer:
            self._degraded_audit_emitted = False

        return flushed

    async def _emit_degraded_audit(self) -> None:
        """Emit a metering_delivery_degraded audit event."""
        if self._audit_emitter is None:
            logger.warning(
                "Metering buffer at %.0f%% capacity (%d/%d) but no audit emitter configured.",
                self.fill_ratio * 100,
                self.size,
                self._max_size,
            )
            return

        try:
            await self._audit_emitter(
                action="metering_delivery_degraded",
                resource="metering_pipeline",
                detail={
                    "buffer_size": self.size,
                    "buffer_max_size": self._max_size,
                    "fill_ratio": round(self.fill_ratio, 4),
                },
            )
        except Exception:
            logger.exception("Failed to emit metering_delivery_degraded audit event.")
