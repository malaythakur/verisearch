"""Tests for the SSE endpoint infrastructure.

Validates:
- SSE response has correct content-type and headers
- Events are formatted correctly (event/data/id/retry fields)
- Keepalive comments are emitted after 15s of silence
- Last-Event-ID is propagated to the generator
- Client disconnect stops the generator
"""

from __future__ import annotations

import asyncio
import json
from typing import AsyncGenerator
from unittest.mock import MagicMock

import pytest
from starlette.requests import Request
from starlette.testclient import TestClient
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from backend.api_gateway.sse import (
    SSEEvent,
    SSEResponse,
    SSEStream,
    create_sse_response,
)
from backend.config.constants import Constants


# ---------------------------------------------------------------------------
# SSEEvent formatting tests
# ---------------------------------------------------------------------------


class TestSSEEventFormat:
    """Tests for SSEEvent wire format serialization."""

    def test_basic_event_format(self) -> None:
        """Event with type and data is formatted correctly."""
        event = SSEEvent(event="token", data={"text": "hello"})
        formatted = event.format()

        assert "event: token\n" in formatted
        assert 'data: {"text": "hello"}\n' in formatted
        assert formatted.endswith("\n\n")

    def test_event_with_id(self) -> None:
        """Event with id field includes id line before event type."""
        event = SSEEvent(event="citation", data={"doc_id": "abc"}, id="42")
        formatted = event.format()

        assert "id: 42\n" in formatted
        assert "event: citation\n" in formatted
        # id should come before event
        id_pos = formatted.index("id: 42")
        event_pos = formatted.index("event: citation")
        assert id_pos < event_pos

    def test_event_with_retry(self) -> None:
        """Event with retry field includes retry line."""
        event = SSEEvent(event="error", data={"code": "timeout"}, retry=3000)
        formatted = event.format()

        assert "retry: 3000\n" in formatted
        assert "event: error\n" in formatted

    def test_event_with_all_fields(self) -> None:
        """Event with all fields formats them in correct order: id, retry, event, data."""
        event = SSEEvent(event="done", data={"answer": "result"}, id="99", retry=5000)
        formatted = event.format()

        id_pos = formatted.index("id: 99")
        retry_pos = formatted.index("retry: 5000")
        event_pos = formatted.index("event: done")
        data_pos = formatted.index("data: ")

        assert id_pos < retry_pos < event_pos < data_pos

    def test_event_data_is_json_serialized(self) -> None:
        """Data field is JSON-serialized."""
        event = SSEEvent(event="token", data={"text": "hello world", "count": 42})
        formatted = event.format()

        # Extract the data line
        lines = formatted.strip().split("\n")
        data_lines = [l for l in lines if l.startswith("data: ")]
        assert len(data_lines) == 1
        data_json = data_lines[0][len("data: "):]
        parsed = json.loads(data_json)
        assert parsed == {"text": "hello world", "count": 42}

    def test_event_string_data(self) -> None:
        """String data is JSON-serialized (quoted)."""
        event = SSEEvent(event="token", data="hello")
        formatted = event.format()

        lines = formatted.strip().split("\n")
        data_lines = [l for l in lines if l.startswith("data: ")]
        assert len(data_lines) == 1
        assert data_lines[0] == 'data: "hello"'

    def test_event_null_data(self) -> None:
        """Null data is serialized as JSON null."""
        event = SSEEvent(event="done", data=None)
        formatted = event.format()

        assert "data: null\n" in formatted

    def test_event_numeric_data(self) -> None:
        """Numeric data is serialized correctly."""
        event = SSEEvent(event="progress", data=0.75)
        formatted = event.format()

        assert "data: 0.75\n" in formatted

    def test_event_double_newline_terminator(self) -> None:
        """Every formatted event ends with exactly \\n\\n."""
        event = SSEEvent(event="token", data="x")
        formatted = event.format()

        assert formatted.endswith("\n\n")
        # Should not end with triple newline
        assert not formatted.endswith("\n\n\n")


# ---------------------------------------------------------------------------
# SSEStream tests
# ---------------------------------------------------------------------------


class TestSSEStream:
    """Tests for SSEStream keepalive and streaming behavior."""

    async def test_stream_yields_formatted_events(self) -> None:
        """Stream yields properly formatted event bytes."""

        async def gen(last_event_id: str | None = None) -> AsyncGenerator[SSEEvent, None]:
            yield SSEEvent(event="token", data={"text": "hi"})
            yield SSEEvent(event="done", data=None)

        stream = SSEStream(gen())
        chunks: list[bytes] = []
        async for chunk in stream.stream():
            chunks.append(chunk)

        assert len(chunks) == 2
        assert b"event: token\n" in chunks[0]
        assert b"event: done\n" in chunks[1]

    async def test_keepalive_emitted_on_silence(self) -> None:
        """Keepalive comment is emitted when no events arrive within the interval."""

        async def slow_gen() -> AsyncGenerator[SSEEvent, None]:
            # Wait significantly longer than keepalive interval before yielding
            await asyncio.sleep(0.6)
            yield SSEEvent(event="done", data=None)

        # Use a short keepalive for testing (0.2s) with generous delay
        stream = SSEStream(slow_gen(), keepalive_seconds=0.2)
        chunks: list[bytes] = []
        async for chunk in stream.stream():
            chunks.append(chunk)

        # Should have at least one keepalive comment before the event
        keepalive_chunks = [c for c in chunks if c == b": keepalive\n\n"]
        event_chunks = [c for c in chunks if b"event: done" in c]

        assert len(keepalive_chunks) >= 1
        assert len(event_chunks) == 1

    async def test_multiple_keepalives_on_extended_silence(self) -> None:
        """Multiple keepalive comments are emitted during extended silence."""

        async def very_slow_gen() -> AsyncGenerator[SSEEvent, None]:
            await asyncio.sleep(0.8)
            yield SSEEvent(event="done", data=None)

        stream = SSEStream(very_slow_gen(), keepalive_seconds=0.2)
        chunks: list[bytes] = []
        async for chunk in stream.stream():
            chunks.append(chunk)

        keepalive_chunks = [c for c in chunks if c == b": keepalive\n\n"]
        # Should have at least 2 keepalives (0.8s / 0.2s = ~4 intervals)
        assert len(keepalive_chunks) >= 2

    async def test_stream_stops_on_generator_exhaustion(self) -> None:
        """Stream ends when the generator is exhausted."""

        async def finite_gen() -> AsyncGenerator[SSEEvent, None]:
            yield SSEEvent(event="token", data="a")
            yield SSEEvent(event="token", data="b")

        stream = SSEStream(finite_gen(), keepalive_seconds=10)
        chunks: list[bytes] = []
        async for chunk in stream.stream():
            chunks.append(chunk)

        assert len(chunks) == 2

    async def test_close_stops_stream(self) -> None:
        """Calling close() stops the stream and cleans up the generator."""
        generator_closed = False

        async def infinite_gen() -> AsyncGenerator[SSEEvent, None]:
            nonlocal generator_closed
            try:
                while True:
                    yield SSEEvent(event="token", data="x")
                    await asyncio.sleep(0.01)
            finally:
                generator_closed = True

        stream = SSEStream(infinite_gen(), keepalive_seconds=10)

        # Consume a few events then close
        count = 0
        async for chunk in stream.stream():
            count += 1
            if count >= 3:
                await stream.close()
                break

        assert count >= 3
        assert generator_closed

    async def test_stream_handles_generator_exception(self) -> None:
        """Stream handles exceptions from the generator gracefully."""

        async def failing_gen() -> AsyncGenerator[SSEEvent, None]:
            yield SSEEvent(event="token", data="ok")
            raise RuntimeError("LLM provider error")

        stream = SSEStream(failing_gen(), keepalive_seconds=10)
        chunks: list[bytes] = []

        with pytest.raises(RuntimeError, match="LLM provider error"):
            async for chunk in stream.stream():
                chunks.append(chunk)

        # Should have gotten the first event before the error
        assert len(chunks) == 1


# ---------------------------------------------------------------------------
# SSEResponse tests
# ---------------------------------------------------------------------------


class TestSSEResponse:
    """Tests for SSEResponse headers and content type."""

    async def test_content_type_is_event_stream(self) -> None:
        """SSEResponse has Content-Type: text/event-stream."""

        async def gen() -> AsyncGenerator[SSEEvent, None]:
            yield SSEEvent(event="done", data=None)

        stream = SSEStream(gen())
        response = SSEResponse(stream)

        assert response.media_type == "text/event-stream"

    async def test_cache_control_header(self) -> None:
        """SSEResponse includes Cache-Control: no-cache."""

        async def gen() -> AsyncGenerator[SSEEvent, None]:
            yield SSEEvent(event="done", data=None)

        stream = SSEStream(gen())
        response = SSEResponse(stream)

        assert response.headers["cache-control"] == "no-cache"

    async def test_connection_header(self) -> None:
        """SSEResponse includes Connection: keep-alive."""

        async def gen() -> AsyncGenerator[SSEEvent, None]:
            yield SSEEvent(event="done", data=None)

        stream = SSEStream(gen())
        response = SSEResponse(stream)

        assert response.headers["connection"] == "keep-alive"

    async def test_x_accel_buffering_header(self) -> None:
        """SSEResponse includes X-Accel-Buffering: no for nginx compatibility."""

        async def gen() -> AsyncGenerator[SSEEvent, None]:
            yield SSEEvent(event="done", data=None)

        stream = SSEStream(gen())
        response = SSEResponse(stream)

        assert response.headers["x-accel-buffering"] == "no"

    async def test_custom_headers_merged(self) -> None:
        """Custom headers are merged with SSE defaults."""

        async def gen() -> AsyncGenerator[SSEEvent, None]:
            yield SSEEvent(event="done", data=None)

        stream = SSEStream(gen())
        response = SSEResponse(stream, headers={"X-Custom": "value"})

        assert response.headers["x-custom"] == "value"
        # SSE headers still present
        assert response.headers["cache-control"] == "no-cache"


# ---------------------------------------------------------------------------
# create_sse_response integration tests
# ---------------------------------------------------------------------------


class TestCreateSSEResponse:
    """Tests for the create_sse_response helper function."""

    async def test_last_event_id_propagated(self) -> None:
        """Last-Event-ID header is passed to the generator factory."""
        received_last_id: list[str | None] = []

        async def factory(last_event_id: str | None) -> AsyncGenerator[SSEEvent, None]:
            received_last_id.append(last_event_id)
            yield SSEEvent(event="done", data=None, id="10")

        app = FastAPI()

        @app.get("/v1/test-sse")
        async def sse_endpoint(request: Request):
            return create_sse_response(request, factory)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(
                "/v1/test-sse",
                headers={"Last-Event-ID": "5"},
            )

        assert response.status_code == 200
        assert received_last_id == ["5"]

    async def test_last_event_id_none_when_absent(self) -> None:
        """Generator receives None when no Last-Event-ID header is present."""
        received_last_id: list[str | None] = []

        async def factory(last_event_id: str | None) -> AsyncGenerator[SSEEvent, None]:
            received_last_id.append(last_event_id)
            yield SSEEvent(event="done", data=None)

        app = FastAPI()

        @app.get("/v1/test-sse-no-id")
        async def sse_endpoint(request: Request):
            return create_sse_response(request, factory)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/v1/test-sse-no-id")

        assert response.status_code == 200
        assert received_last_id == [None]

    async def test_full_sse_stream_content(self) -> None:
        """Full integration: SSE stream delivers correctly formatted events."""

        async def factory(last_event_id: str | None) -> AsyncGenerator[SSEEvent, None]:
            yield SSEEvent(event="token", data={"text": "Hello"}, id="1")
            yield SSEEvent(event="token", data={"text": " world"}, id="2")
            yield SSEEvent(event="done", data={"answer": "Hello world"}, id="3")

        app = FastAPI()

        @app.get("/v1/test-sse-full")
        async def sse_endpoint(request: Request):
            return create_sse_response(request, factory)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/v1/test-sse-full")

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
        assert response.headers["cache-control"] == "no-cache"

        # Parse the response body as SSE events
        body = response.text
        events = body.strip().split("\n\n")
        assert len(events) == 3

        # Verify first event
        assert "id: 1" in events[0]
        assert "event: token" in events[0]
        assert 'data: {"text": "Hello"}' in events[0]

        # Verify last event
        assert "id: 3" in events[2]
        assert "event: done" in events[2]

    async def test_keepalive_in_response(self) -> None:
        """Keepalive comments appear in the response during silence."""

        async def factory(last_event_id: str | None) -> AsyncGenerator[SSEEvent, None]:
            await asyncio.sleep(0.6)
            yield SSEEvent(event="done", data=None)

        app = FastAPI()

        @app.get("/v1/test-sse-keepalive")
        async def sse_endpoint(request: Request):
            return create_sse_response(request, factory, keepalive_seconds=0.2)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/v1/test-sse-keepalive")

        assert response.status_code == 200
        body = response.text
        # Should contain keepalive comments
        assert ": keepalive" in body
        # Should also contain the final event
        assert "event: done" in body

    async def test_client_disconnect_stops_generator(self) -> None:
        """When the stream is closed (simulating client disconnect), the generator is cleaned up."""
        generator_cleaned_up = False

        async def factory(last_event_id: str | None) -> AsyncGenerator[SSEEvent, None]:
            nonlocal generator_cleaned_up
            try:
                while True:
                    yield SSEEvent(event="token", data="x")
                    await asyncio.sleep(0.01)
            finally:
                generator_cleaned_up = True

        # Test via SSEStream.close() which is what happens on client disconnect
        stream = SSEStream(factory(None), keepalive_seconds=10)

        count = 0
        async for chunk in stream.stream():
            count += 1
            if count >= 3:
                await stream.close()
                break

        assert count >= 3
        assert generator_cleaned_up

    async def test_sse_response_status_code(self) -> None:
        """SSE response returns 200 status code."""

        async def factory(last_event_id: str | None) -> AsyncGenerator[SSEEvent, None]:
            yield SSEEvent(event="done", data=None)

        app = FastAPI()

        @app.get("/v1/test-sse-status")
        async def sse_endpoint(request: Request):
            return create_sse_response(request, factory)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/v1/test-sse-status")

        assert response.status_code == 200

    async def test_uses_default_keepalive_from_constants(self) -> None:
        """Default keepalive interval comes from Constants.SSE_KEEPALIVE_SECONDS."""
        assert Constants.SSE_KEEPALIVE_SECONDS == 15

        async def factory(last_event_id: str | None) -> AsyncGenerator[SSEEvent, None]:
            yield SSEEvent(event="done", data=None)

        # SSEStream default should use the constant
        stream = SSEStream(factory(None))
        assert stream._keepalive_seconds == 15
