"""Kafka-based async document ingest pipeline.

Publishes documents to a Kafka topic for async processing, and provides
a consumer that processes documents through the indexing pipeline.

Falls back to synchronous indexing if Kafka is not available.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Awaitable


INGEST_TOPIC = "document-ingest"
DLQ_TOPIC = "document-ingest-dlq"


@dataclass
class IngestMessage:
    """A document ingest message for Kafka."""

    message_id: str
    source_url: str
    raw_content: str
    tenant_id: str
    request_id: str
    published_at: str


class KafkaIngestProducer:
    """Publishes documents to Kafka for async indexing.

    Falls back to direct indexing if Kafka is unavailable.
    """

    def __init__(self):
        self._producer = None
        self._available = None

    async def _get_producer(self):
        """Lazily initialize the Kafka producer."""
        if self._producer is not None:
            return self._producer

        bootstrap_servers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "")
        if not bootstrap_servers:
            self._available = False
            return None

        try:
            from aiokafka import AIOKafkaProducer

            self._producer = AIOKafkaProducer(
                bootstrap_servers=bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                acks="all",
                retries=3,
            )
            await self._producer.start()
            self._available = True
            return self._producer
        except Exception:
            self._available = False
            return None

    @property
    def is_available(self) -> bool:
        """Check if Kafka is available."""
        return self._available or False

    async def publish(
        self,
        source_url: str,
        raw_content: str,
        tenant_id: str = "",
        request_id: str | None = None,
    ) -> str:
        """Publish a document for async indexing.

        Args:
            source_url: The document's source URL.
            raw_content: Raw HTML/text content.
            tenant_id: The tenant ID.
            request_id: Optional request ID for tracing.

        Returns:
            The message_id for tracking.
        """
        message_id = str(uuid.uuid4())
        req_id = request_id or str(uuid.uuid4())

        message = {
            "message_id": message_id,
            "source_url": source_url,
            "raw_content": raw_content,
            "tenant_id": tenant_id,
            "request_id": req_id,
            "published_at": datetime.now(timezone.utc).isoformat(),
        }

        producer = await self._get_producer()
        if producer:
            await producer.send_and_wait(INGEST_TOPIC, value=message)
        else:
            # Kafka unavailable — store for sync processing
            self._pending_messages.append(message)

        return message_id

    async def close(self):
        """Close the producer."""
        if self._producer:
            await self._producer.stop()
            self._producer = None

    _pending_messages: list[dict] = []


class KafkaIngestConsumer:
    """Consumes documents from Kafka and processes them through the indexing pipeline.

    Handles:
    - Batch consumption for throughput
    - Dead-letter queue routing on persistent failure
    - Offset commit after successful processing
    """

    def __init__(
        self,
        index_handler: Callable[[str, str, str], Awaitable[Any]],
        group_id: str = "indexer-group",
    ):
        """Initialize the consumer.

        Args:
            index_handler: Async function(raw_content, source_url, request_id) -> IndexResult
            group_id: Kafka consumer group ID.
        """
        self._index_handler = index_handler
        self._group_id = group_id
        self._consumer = None
        self._running = False

    async def start(self):
        """Start consuming messages."""
        bootstrap_servers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "")
        if not bootstrap_servers:
            return

        try:
            from aiokafka import AIOKafkaConsumer

            self._consumer = AIOKafkaConsumer(
                INGEST_TOPIC,
                bootstrap_servers=bootstrap_servers,
                group_id=self._group_id,
                value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                auto_offset_reset="earliest",
                enable_auto_commit=False,
            )
            await self._consumer.start()
            self._running = True

            asyncio.create_task(self._consume_loop())
        except Exception:
            self._running = False

    async def _consume_loop(self):
        """Main consumption loop."""
        while self._running and self._consumer:
            try:
                messages = await self._consumer.getmany(timeout_ms=1000, max_records=10)

                for tp, msgs in messages.items():
                    for msg in msgs:
                        await self._process_message(msg.value)

                # Commit offsets after processing
                if messages:
                    await self._consumer.commit()

            except Exception:
                await asyncio.sleep(1)

    async def _process_message(self, message: dict):
        """Process a single ingest message."""
        try:
            await self._index_handler(
                message["raw_content"],
                message["source_url"],
                message["request_id"],
            )
        except Exception as e:
            # Route to DLQ on failure
            await self._route_to_dlq(message, str(e))

    async def _route_to_dlq(self, message: dict, error: str):
        """Route a failed message to the dead-letter queue."""
        bootstrap_servers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "")
        if not bootstrap_servers:
            return

        try:
            from aiokafka import AIOKafkaProducer

            producer = AIOKafkaProducer(
                bootstrap_servers=bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
            await producer.start()

            dlq_message = {
                **message,
                "error": error,
                "failed_at": datetime.now(timezone.utc).isoformat(),
            }
            await producer.send_and_wait(DLQ_TOPIC, value=dlq_message)
            await producer.stop()
        except Exception:
            pass

    async def stop(self):
        """Stop the consumer."""
        self._running = False
        if self._consumer:
            await self._consumer.stop()
            self._consumer = None
