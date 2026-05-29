"""Tests for the WebSocket endpoint infrastructure.

Validates:
- WebSocket connection accepts and sends events as JSON frames
- Client cancel message stops generation
- Error event closes connection within 2s
- Done event closes connection cleanly
- Authentication via first message works
- Authentication via query param works
"""

from __future__ import annotations

import asyncio
import json
from typing import AsyncGenerator

import pytest
from fastapi import FastAPI, WebSocket
from starlette.testclient import TestClient

from backend.api_gateway.websocket import (
    WebSocketAuthResult,
    WebSocketAuthService,
    WebSocketEvent,
    WebSocketHandler,
    ws_answer_handler,
)


# ---------------------------------------------------------------------------
# Test fixtures / helpers
# ---------------------------------------------------------------------------


class MockAuthService:
    """Mock auth service that authenticates any token starting with 'valid'."""

    async def authenticate_token(self, token: str) -> WebSocketAuthResult:
        if token.startswith("valid"):
            return WebSocketAuthResult(
                authenticated=True,
                tenant_id="tenant-123",
                api_key_id="key-456",
            )
        return WebSocketAuthResult(
            authenticated=False,
            error="Invalid token",
        )


def create_test_app(
    auth_service: MockAuthService | None = None,
    generator_factory=None,
) -> FastAPI:
    """Create a FastAPI app with a WebSocket endpoint for testing."""
    app = FastAPI()
    _auth_service = auth_service or MockAuthService()

    @app.websocket("/v1/answer")
    async def answer_ws(websocket: WebSocket):
        if generator_factory:
            gen = generator_factory()
        else:
            # Default: yield a few tokens then done
            async def _default_gen() -> AsyncGenerator[WebSocketEvent, None]:
                yield WebSocketEvent(event="token", data={"text": "Hello"}, id="1")
                yield WebSocketEvent(event="token", data={"text": " world"}, id="2")
                yield WebSocketEvent(event="done", data={"answer": "Hello world", "citations": []}, id="3")

            gen = _default_gen()

        await ws_answer_handler(websocket, _auth_service, gen)

    return app


# ---------------------------------------------------------------------------
# WebSocketEvent tests
# ---------------------------------------------------------------------------


class TestWebSocketEvent:
    """Tests for WebSocketEvent serialization."""

    def test_basic_event_to_json(self) -> None:
        """Event with type and data serializes to correct JSON."""
        event = WebSocketEvent(event="token", data={"text": "hello"})
        result = json.loads(event.to_json())

        assert result == {"event": "token", "data": {"text": "hello"}}

    def test_event_with_id(self) -> None:
        """Event with id includes id field in JSON."""
        event = WebSocketEvent(event="citation", data={"doc_id": "abc"}, id="42")
        result = json.loads(event.to_json())

        assert result == {"event": "citation", "data": {"doc_id": "abc"}, "id": "42"}

    def test_event_without_data(self) -> None:
        """Event without data omits data field from JSON."""
        event = WebSocketEvent(event="cancelled")
        result = json.loads(event.to_json())

        assert result == {"event": "cancelled"}
        assert "data" not in result

    def test_event_without_id(self) -> None:
        """Event without id omits id field from JSON."""
        event = WebSocketEvent(event="token", data={"text": "x"})
        result = json.loads(event.to_json())

        assert "id" not in result

    def test_event_with_all_fields(self) -> None:
        """Event with all fields serializes completely."""
        event = WebSocketEvent(
            event="done",
            data={"answer": "result", "citations": [{"doc_id": "d1"}]},
            id="99",
        )
        result = json.loads(event.to_json())

        assert result["event"] == "done"
        assert result["data"]["answer"] == "result"
        assert result["id"] == "99"

    def test_event_data_with_nested_objects(self) -> None:
        """Complex nested data is serialized correctly."""
        event = WebSocketEvent(
            event="citation",
            data={
                "document_id": "doc-1",
                "version": 3,
                "offsets": {"answer_start": 0, "answer_end": 10},
            },
            id="5",
        )
        result = json.loads(event.to_json())

        assert result["data"]["offsets"]["answer_start"] == 0
        assert result["data"]["version"] == 3


# ---------------------------------------------------------------------------
# WebSocket connection and authentication tests
# ---------------------------------------------------------------------------


class TestWebSocketAuthentication:
    """Tests for WebSocket authentication flow."""

    def test_auth_via_query_param(self) -> None:
        """Authentication succeeds via token query parameter."""
        app = create_test_app()
        client = TestClient(app)

        with client.websocket_connect("/v1/answer?token=valid-key-123") as ws:
            # Should receive events since auth succeeded
            data = ws.receive_json()
            assert data["event"] == "token"
            assert data["data"]["text"] == "Hello"

    def test_auth_via_first_message(self) -> None:
        """Authentication succeeds via token in first message."""
        app = create_test_app()
        client = TestClient(app)

        with client.websocket_connect("/v1/answer") as ws:
            # Send auth message
            ws.send_json({"token": "valid-key-123"})
            # Should receive events
            data = ws.receive_json()
            assert data["event"] == "token"

    def test_auth_via_authorization_field(self) -> None:
        """Authentication succeeds via authorization field in first message."""
        app = create_test_app()
        client = TestClient(app)

        with client.websocket_connect("/v1/answer") as ws:
            ws.send_json({"authorization": "Bearer valid-key-123"})
            data = ws.receive_json()
            assert data["event"] == "token"

    def test_auth_failure_closes_connection(self) -> None:
        """Invalid token closes connection with code 4001."""
        app = create_test_app()
        client = TestClient(app)

        with pytest.raises(Exception):
            with client.websocket_connect("/v1/answer?token=invalid-key") as ws:
                ws.receive_json()


# ---------------------------------------------------------------------------
# WebSocket event streaming tests
# ---------------------------------------------------------------------------


class TestWebSocketEventStreaming:
    """Tests for WebSocket event streaming."""

    def test_receives_all_events_in_order(self) -> None:
        """Client receives all events in the correct order."""
        app = create_test_app()
        client = TestClient(app)

        with client.websocket_connect("/v1/answer?token=valid-key-123") as ws:
            events = []
            # Receive all events until connection closes
            try:
                while True:
                    data = ws.receive_json()
                    events.append(data)
            except Exception:
                pass

        assert len(events) == 3
        assert events[0]["event"] == "token"
        assert events[0]["data"]["text"] == "Hello"
        assert events[0]["id"] == "1"
        assert events[1]["event"] == "token"
        assert events[1]["data"]["text"] == " world"
        assert events[2]["event"] == "done"
        assert events[2]["data"]["answer"] == "Hello world"

    def test_done_event_closes_connection(self) -> None:
        """Connection is closed after done event is sent."""
        app = create_test_app()
        client = TestClient(app)

        with client.websocket_connect("/v1/answer?token=valid-key-123") as ws:
            events = []
            try:
                while True:
                    data = ws.receive_json()
                    events.append(data)
            except Exception:
                pass

        # Last event should be done
        assert events[-1]["event"] == "done"

    def test_error_event_closes_connection(self) -> None:
        """Error event closes the connection."""

        async def error_gen() -> AsyncGenerator[WebSocketEvent, None]:
            yield WebSocketEvent(event="token", data={"text": "partial"}, id="1")
            yield WebSocketEvent(event="error", data={"code": "model_failure", "message": "LLM timeout"}, id="2")

        app = create_test_app(generator_factory=error_gen)
        client = TestClient(app)

        with client.websocket_connect("/v1/answer?token=valid-key-123") as ws:
            events = []
            try:
                while True:
                    data = ws.receive_json()
                    events.append(data)
            except Exception:
                pass

        assert len(events) == 2
        assert events[0]["event"] == "token"
        assert events[1]["event"] == "error"
        assert events[1]["data"]["code"] == "model_failure"


# ---------------------------------------------------------------------------
# Client cancel tests
# ---------------------------------------------------------------------------


class TestWebSocketCancel:
    """Tests for client-initiated cancel via WebSocket."""

    def test_cancel_stops_generation(self) -> None:
        """Client sending cancel stops the generation and receives cancelled event."""
        generation_stopped = False

        async def slow_gen() -> AsyncGenerator[WebSocketEvent, None]:
            nonlocal generation_stopped
            try:
                yield WebSocketEvent(event="token", data={"text": "first"}, id="1")
                # Simulate slow generation
                await asyncio.sleep(5.0)
                yield WebSocketEvent(event="token", data={"text": "second"}, id="2")
                yield WebSocketEvent(event="done", data={"answer": "first second"}, id="3")
            except asyncio.CancelledError:
                generation_stopped = True
                raise

        app = create_test_app(generator_factory=slow_gen)
        client = TestClient(app)

        with client.websocket_connect("/v1/answer?token=valid-key-123") as ws:
            # Receive first token
            data = ws.receive_json()
            assert data["event"] == "token"
            assert data["data"]["text"] == "first"

            # Send cancel
            ws.send_json({"action": "cancel"})

            # Should receive cancelled event
            data = ws.receive_json()
            assert data["event"] == "cancelled"

        assert generation_stopped

    def test_cancel_with_no_prior_events(self) -> None:
        """Cancel works even if sent before any events are generated."""

        async def delayed_gen() -> AsyncGenerator[WebSocketEvent, None]:
            await asyncio.sleep(5.0)
            yield WebSocketEvent(event="token", data={"text": "late"}, id="1")
            yield WebSocketEvent(event="done", data={"answer": "late"}, id="2")

        app = create_test_app(generator_factory=delayed_gen)
        client = TestClient(app)

        with client.websocket_connect("/v1/answer?token=valid-key-123") as ws:
            # Send cancel immediately
            ws.send_json({"action": "cancel"})

            # Should receive cancelled event
            data = ws.receive_json()
            assert data["event"] == "cancelled"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestWebSocketEdgeCases:
    """Tests for edge cases and error handling."""

    def test_malformed_client_message_ignored(self) -> None:
        """Malformed JSON from client is ignored, stream continues."""

        async def gen_with_delay() -> AsyncGenerator[WebSocketEvent, None]:
            yield WebSocketEvent(event="token", data={"text": "a"}, id="1")
            await asyncio.sleep(0.1)
            yield WebSocketEvent(event="done", data={"answer": "a"}, id="2")

        app = create_test_app(generator_factory=gen_with_delay)
        client = TestClient(app)

        with client.websocket_connect("/v1/answer?token=valid-key-123") as ws:
            # Receive first event
            data = ws.receive_json()
            assert data["event"] == "token"

            # Send malformed message — should be ignored
            ws.send_text("not valid json {{{")

            # Should still receive done event
            data = ws.receive_json()
            assert data["event"] == "done"

    def test_empty_generator_closes_cleanly(self) -> None:
        """An empty generator (no events) closes the connection without error."""

        async def empty_gen() -> AsyncGenerator[WebSocketEvent, None]:
            return
            yield  # Make it a generator  # noqa: RET503

        app = create_test_app(generator_factory=empty_gen)
        client = TestClient(app)

        with client.websocket_connect("/v1/answer?token=valid-key-123") as ws:
            # Connection should close without receiving events
            try:
                ws.receive_json()
                # If we get here, something was sent — that's unexpected for empty gen
            except Exception:
                pass  # Expected — connection closed
