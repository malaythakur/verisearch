"""WebSocket endpoint infrastructure for the API Gateway.

Provides bidirectional WebSocket support for `/v1/answer` with client-initiated
cancel support. This is an alternative to SSE for scenarios requiring bidirectional
control (e.g., the client can send a `cancel` message to stop generation mid-stream).

Design references:
- R6.1: First token event within 3s p95
- R6.5: Error event closes stream within 2s
- WebSocket is used where bidirectional control is required (cancel support)

Event framing (JSON):
    Server → Client: {"event": "<type>", "data": {...}, "id": "<event_id>"}
    Client → Server: {"action": "cancel"}

Event types: token, citation, done, error, cancelled
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Protocol

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from backend.config.constants import Constants


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WebSocketEvent:
    """A single event to be sent over a WebSocket connection.

    Attributes:
        event: Event type string (token, citation, done, error, cancelled).
        data: JSON-serializable payload, or None for events without data.
        id: Optional event ID for ordering and deduplication.
    """

    event: str
    data: dict[str, Any] | None = None
    id: str | None = None

    def to_json(self) -> str:
        """Serialize this event to a JSON string for WebSocket transmission.

        Returns:
            A JSON string representing the event frame.
        """
        frame: dict[str, Any] = {"event": self.event}
        if self.data is not None:
            frame["data"] = self.data
        if self.id is not None:
            frame["id"] = self.id
        return json.dumps(frame, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Auth protocol
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WebSocketAuthResult:
    """Result of WebSocket authentication."""

    authenticated: bool
    tenant_id: str | None = None
    api_key_id: str | None = None
    error: str | None = None


class WebSocketAuthService(Protocol):
    """Protocol for WebSocket authentication.

    Implementations should authenticate a bearer token and return
    a WebSocketAuthResult.
    """

    async def authenticate_token(self, token: str) -> WebSocketAuthResult: ...


# ---------------------------------------------------------------------------
# WebSocket Handler
# ---------------------------------------------------------------------------


class WebSocketHandler:
    """Manages a single WebSocket connection lifecycle.

    Responsibilities:
    - Accept the WebSocket connection
    - Authenticate via bearer token (query param or first message)
    - Send events as JSON frames
    - Listen for client messages (cancel action)
    - Handle graceful close on done/error/cancel

    Usage:
        handler = WebSocketHandler(websocket, auth_service)
        auth_result = await handler.accept_and_authenticate()
        if not auth_result.authenticated:
            return  # connection already closed with 4001

        # Send events
        await handler.send_event(WebSocketEvent(event="token", data={"text": "hi"}))

        # Or use the full answer handler for concurrent send/receive
    """

    def __init__(
        self,
        websocket: WebSocket,
        auth_service: WebSocketAuthService,
    ) -> None:
        self._websocket = websocket
        self._auth_service = auth_service
        self._authenticated = False
        self._tenant_id: str | None = None
        self._closed = False
        self._cancel_event = asyncio.Event()

    @property
    def authenticated(self) -> bool:
        """Whether the connection has been successfully authenticated."""
        return self._authenticated

    @property
    def tenant_id(self) -> str | None:
        """The authenticated tenant ID, or None if not yet authenticated."""
        return self._tenant_id

    @property
    def is_cancelled(self) -> bool:
        """Whether the client has sent a cancel message."""
        return self._cancel_event.is_set()

    @property
    def cancel_event(self) -> asyncio.Event:
        """The asyncio.Event that is set when the client sends cancel."""
        return self._cancel_event

    async def accept_and_authenticate(self) -> WebSocketAuthResult:
        """Accept the WebSocket connection and authenticate the client.

        Authentication is attempted in this order:
        1. Bearer token in the `token` query parameter
        2. Bearer token in the first text message (must be sent within 10s)

        If authentication fails, the connection is closed with code 4001
        and a JSON close reason.

        Returns:
            WebSocketAuthResult indicating success or failure.
        """
        await self._websocket.accept()

        # Try query param first
        token = self._websocket.query_params.get("token")

        if not token:
            # Wait for first message with token
            try:
                raw = await asyncio.wait_for(
                    self._websocket.receive_text(),
                    timeout=10.0,
                )
                msg = json.loads(raw)
                token = msg.get("token") or msg.get("authorization")
            except (asyncio.TimeoutError, json.JSONDecodeError, WebSocketDisconnect):
                result = WebSocketAuthResult(
                    authenticated=False,
                    error="Authentication timeout or invalid message",
                )
                await self._close_with_error(4001, result.error or "Auth failed")
                return result

        if not token:
            result = WebSocketAuthResult(
                authenticated=False,
                error="No token provided",
            )
            await self._close_with_error(4001, result.error)
            return result

        # Strip "Bearer " prefix if present
        if token.lower().startswith("bearer "):
            token = token[7:]

        auth_result = await self._auth_service.authenticate_token(token)

        if not auth_result.authenticated:
            await self._close_with_error(4001, auth_result.error or "Authentication failed")
            return auth_result

        self._authenticated = True
        self._tenant_id = auth_result.tenant_id
        return auth_result

    async def send_event(self, event: WebSocketEvent) -> bool:
        """Send an event to the client as a JSON text frame.

        Args:
            event: The WebSocketEvent to send.

        Returns:
            True if the event was sent successfully, False if the connection
            is closed or an error occurred.
        """
        if self._closed:
            return False

        try:
            await self._websocket.send_text(event.to_json())
            return True
        except (WebSocketDisconnect, RuntimeError):
            self._closed = True
            return False

    async def send_cancelled(self) -> None:
        """Send a cancelled event and close the connection."""
        await self.send_event(WebSocketEvent(event="cancelled"))
        await self._close_gracefully()

    async def send_error(self, data: dict[str, Any]) -> None:
        """Send an error event and close the connection within 2s (R6.5).

        Args:
            data: Error payload (should include at minimum a `code` field).
        """
        await self.send_event(WebSocketEvent(event="error", data=data))
        # Close within 2s per R6.5
        await self._close_gracefully()

    async def send_done(self, data: dict[str, Any]) -> None:
        """Send a done event and close the connection.

        Args:
            data: Done payload (full answer text + citations).
        """
        await self.send_event(WebSocketEvent(event="done", data=data))
        await self._close_gracefully()

    async def listen_for_cancel(self) -> None:
        """Listen for client messages, specifically cancel actions.

        This should be run concurrently with the event generation task.
        Sets the cancel_event when a cancel message is received.
        Exits on disconnect or after cancel is received.
        """
        try:
            while not self._closed:
                try:
                    raw = await asyncio.wait_for(
                        self._websocket.receive_text(),
                        timeout=1.0,
                    )
                except asyncio.TimeoutError:
                    # Check if we should stop
                    continue
                try:
                    msg = json.loads(raw)
                    if msg.get("action") == "cancel":
                        self._cancel_event.set()
                        return
                except json.JSONDecodeError:
                    # Ignore malformed messages
                    continue
        except WebSocketDisconnect:
            self._closed = True
        except RuntimeError:
            # Connection already closed
            self._closed = True

    async def _close_with_error(self, code: int, reason: str) -> None:
        """Close the WebSocket with an error code and reason."""
        self._closed = True
        try:
            await self._websocket.close(code=code, reason=reason)
        except RuntimeError:
            pass

    async def _close_gracefully(self) -> None:
        """Close the WebSocket connection gracefully with normal closure."""
        self._closed = True
        try:
            await self._websocket.close(code=1000)
        except (WebSocketDisconnect, RuntimeError):
            pass


# ---------------------------------------------------------------------------
# Answer handler helper
# ---------------------------------------------------------------------------


async def ws_answer_handler(
    websocket: WebSocket,
    auth_service: WebSocketAuthService,
    answer_generator: AsyncGenerator[WebSocketEvent, None],
) -> None:
    """Handle a WebSocket `/v1/answer` connection end-to-end.

    This is the primary entry point for WebSocket answer streaming. It:
    1. Accepts the WebSocket connection
    2. Authenticates via first message or query param
    3. Runs the answer generator in a task
    4. Concurrently listens for cancel messages
    5. Cancels the generator task on cancel
    6. Sends events as they arrive from the generator
    7. Closes the connection on done/error/cancel

    Args:
        websocket: The FastAPI WebSocket connection.
        auth_service: Service implementing WebSocketAuthService protocol.
        answer_generator: An async generator yielding WebSocketEvent objects.
            The generator should yield token, citation events, and end with
            a done or error event. It should check for cancellation via
            asyncio.current_task().cancelled() or similar mechanism.
    """
    handler = WebSocketHandler(websocket, auth_service)

    # Step 1: Accept and authenticate
    auth_result = await handler.accept_and_authenticate()
    if not auth_result.authenticated:
        # Connection already closed by accept_and_authenticate
        await answer_generator.aclose()
        return

    # Step 2: Run generator and cancel listener concurrently
    generator_task: asyncio.Task | None = None
    listener_task: asyncio.Task | None = None

    async def _run_generator() -> None:
        """Consume the answer generator and send events."""
        try:
            async for event in answer_generator:
                if handler.is_cancelled:
                    break
                sent = await handler.send_event(event)
                if not sent:
                    break

                # If this is a terminal event, close the connection and stop
                if event.event == "done":
                    await handler._close_gracefully()
                    return
                if event.event == "error":
                    await handler._close_gracefully()
                    return
        except asyncio.CancelledError:
            pass
        finally:
            await answer_generator.aclose()

    try:
        generator_task = asyncio.create_task(_run_generator())
        listener_task = asyncio.create_task(handler.listen_for_cancel())

        # Wait for either the generator to finish or cancel to be received
        done, pending = await asyncio.wait(
            {generator_task, listener_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        if handler.is_cancelled:
            # Client requested cancel — stop the generator
            if generator_task and not generator_task.done():
                generator_task.cancel()
                try:
                    await generator_task
                except asyncio.CancelledError:
                    pass
            await handler.send_cancelled()
        elif generator_task in done:
            # Generator finished (done or error event already sent)
            # Close the connection gracefully
            await handler._close_gracefully()

    except WebSocketDisconnect:
        pass
    except Exception:
        # Unexpected error — try to send error event
        try:
            await handler.send_error({"code": "internal_error", "message": "Unexpected server error"})
        except Exception:
            pass
    finally:
        # Clean up any pending tasks
        for task in [generator_task, listener_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
