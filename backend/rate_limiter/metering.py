"""Metering event emission for billable API responses.

Implements R14.2 (one metering event per billable 2xx response) and
R14.3 (at-least-once delivery with deduplication via dedup_key).

The MeteringService is called after a successful response — it is NOT
middleware. It uses INSERT ... ON CONFLICT DO NOTHING on the dedup_key
column to guarantee at-most-once persistence even when retried.

DB errors are logged but never block the response (fire-and-forget).
On DB failure, events are buffered locally via DurableMeteringBuffer (R14.5).
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import asyncpg

if TYPE_CHECKING:
    from backend.rate_limiter.metering_buffer import DurableMeteringBuffer

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MeteringEvent:
    """Represents a single metering event for a billable API response.

    Attributes:
        request_id: The unique request identifier.
        tenant_id: The tenant that made the request.
        endpoint: The API endpoint that was called.
        timestamp_utc: When the event was recorded (UTC).
        response_status: The HTTP response status code.
        tokens_consumed: Optional token count for LLM-backed endpoints.
        dedup_key: Deterministic key derived from request_id for idempotency.
    """

    request_id: str
    tenant_id: str
    endpoint: str
    timestamp_utc: datetime
    response_status: int
    tokens_consumed: int | None
    dedup_key: str


class MeteringService:
    """Emits metering events for billable 2xx API responses.

    Uses asyncpg to insert into the metering_events table with
    ON CONFLICT DO NOTHING on dedup_key for at-least-once delivery
    with deduplication (R14.3).

    On DB failure, events are buffered locally via DurableMeteringBuffer
    for later retry (R14.5).

    Args:
        db_pool: An asyncpg connection pool.
        buffer: Optional DurableMeteringBuffer for local buffering on DB errors.
    """

    def __init__(
        self,
        db_pool: asyncpg.Pool,
        buffer: DurableMeteringBuffer | None = None,
    ) -> None:
        self._db_pool = db_pool
        self._buffer = buffer

    @staticmethod
    def _make_dedup_key(request_id: str) -> str:
        """Derive a deterministic dedup_key from request_id.

        Simple and deterministic: ensures at-most-once per request
        even with retries.
        """
        return f"meter:{request_id}"

    @staticmethod
    def _is_billable_status(response_status: int) -> bool:
        """Return True if the response status is a billable 2xx."""
        return 200 <= response_status <= 299

    async def emit_metering_event(
        self,
        *,
        request_id: str,
        tenant_id: str,
        endpoint: str,
        response_status: int,
        tokens_consumed: int | None = None,
    ) -> None:
        """Emit a metering event for a billable response.

        Only emits for 2xx responses (R14.2). Uses ON CONFLICT DO NOTHING
        on dedup_key for idempotent at-least-once delivery (R14.3).

        On DB error, logs the failure but does NOT raise — fire-and-forget
        semantics to avoid blocking the API response.

        Args:
            request_id: The unique request identifier.
            tenant_id: The tenant that made the request.
            endpoint: The API endpoint that was called.
            response_status: The HTTP response status code.
            tokens_consumed: Optional token count for LLM-backed endpoints.
        """
        if not self._is_billable_status(response_status):
            return

        dedup_key = self._make_dedup_key(request_id)
        timestamp_utc = datetime.now(timezone.utc)

        event = MeteringEvent(
            request_id=request_id,
            tenant_id=tenant_id,
            endpoint=endpoint,
            timestamp_utc=timestamp_utc,
            response_status=response_status,
            tokens_consumed=tokens_consumed,
            dedup_key=dedup_key,
        )

        try:
            async with self._db_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO metering_events (
                        metering_event_id, request_id, tenant_id, endpoint,
                        timestamp_utc, response_status, tokens_consumed, dedup_key
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (dedup_key) DO NOTHING
                    """,
                    uuid.uuid4(),
                    event.request_id,
                    uuid.UUID(event.tenant_id),
                    event.endpoint,
                    event.timestamp_utc,
                    event.response_status,
                    event.tokens_consumed,
                    event.dedup_key,
                )
        except Exception:
            # Fire-and-forget: log but never block the response.
            # Buffer locally for later retry (R14.5).
            logger.exception(
                "Failed to emit metering event for request_id=%s tenant_id=%s endpoint=%s",
                request_id,
                tenant_id,
                endpoint,
            )
            if self._buffer is not None:
                await self._buffer.buffer_event(event)
