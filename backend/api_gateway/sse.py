"""Server-Sent Events (SSE) infrastructure for the API Gateway.

Provides a lightweight, reusable SSE implementation for streaming endpoints
(/v1/answer, /v1/research/{job_id}/events). Features:

- Correct text/event-stream formatting with event/data/id/retry fields
- Keepalive comments every 15 seconds of silence (Constants.SSE_KEEPALIVE_SECONDS)
- Last-Event-ID header support for reconnection replay
- Graceful client disconnect handling
- Cache-Control and X-Accel-Buffering headers for proxy compatibility

Design references: R6.1, R7.3 (SSE streaming with Last-Event-ID reconnect)
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable, Awaitable

from starlette.requests import Request
from starlette.responses import StreamingResponse

from backend.config.constants import Constants


@dataclass
class SSEEvent:
    """A single Server-Sent Event.

    Attributes:
        event: Event type string (e.g., "token", "citation", "done", "error").
        data: JSON-serializable payload.
        id: Optional event ID for Last-Event-ID reconnect support.
        retry: Optional retry interval in milliseconds for client reconnection.
    """

    event: str
    data: Any
    id: str | None = None
    retry: int | None = None

    def format(self) -> str:
        """Format this event as SSE wire format.

        Returns:
            A string in SSE wire format ending with a double newline.
        """
        lines: list[str] = []

        if self.id is not None:
            lines.append(f"id: {self.id}")

        if self.retry is not None:
            lines.append(f"retry: {self.retry}")

        lines.append(f"event: {self.event}")

        # Serialize data as JSON; handle multi-line by splitting
        data_str = json.dumps(self.data, ensure_ascii=False)
        for line in data_str.split("\n"):
            lines.append(f"data: {line}")

        # Double newline terminates the event
        return "\n".join(lines) + "\n\n"


# Type alias for an async generator factory that produces SSE events.
# The factory receives the optional last_event_id for replay support.
SSEGeneratorFactory = Callable[[str | None], AsyncGenerator[SSEEvent, None]]


class SSEStream:
    """Wraps an async generator of SSEEvent objects into a byte stream.

    Handles:
    - Formatting events as SSE wire format
    - Emitting keepalive comments (`: keepalive\\n\\n`) every 15s of silence
    - Propagating Last-Event-ID to the generator for replay
    - Graceful shutdown on client disconnect
    """

    def __init__(
        self,
        generator: AsyncGenerator[SSEEvent, None],
        *,
        keepalive_seconds: int = Constants.SSE_KEEPALIVE_SECONDS,
    ) -> None:
        self._generator = generator
        self._keepalive_seconds = keepalive_seconds
        self._closed = False

    async def stream(self) -> AsyncGenerator[bytes, None]:
        """Yield formatted SSE bytes, interleaving keepalive comments on silence.

        Yields:
            Encoded SSE event bytes or keepalive comment bytes.
        """
        next_event_task: asyncio.Task | None = None
        try:
            # Create a persistent task for fetching the next event so that
            # keepalive timeouts don't cancel the generator's internal awaits.
            while not self._closed:
                if next_event_task is None:
                    next_event_task = asyncio.ensure_future(
                        self._generator.__anext__()
                    )

                done, _ = await asyncio.wait(
                    {next_event_task},
                    timeout=self._keepalive_seconds,
                )

                if done:
                    # Task completed — get the result
                    task = done.pop()
                    next_event_task = None
                    try:
                        event = task.result()
                        yield event.format().encode("utf-8")
                    except StopAsyncIteration:
                        break
                    # Let other exceptions (e.g. RuntimeError) propagate
                else:
                    # Timeout — emit keepalive, task is still running
                    yield b": keepalive\n\n"
        except asyncio.CancelledError:
            # Client disconnected
            pass
        finally:
            self._closed = True
            if next_event_task is not None and not next_event_task.done():
                next_event_task.cancel()
                try:
                    await next_event_task
                except (asyncio.CancelledError, StopAsyncIteration, Exception):
                    pass
            await self._generator.aclose()

    async def close(self) -> None:
        """Signal the stream to stop and clean up the generator."""
        self._closed = True
        await self._generator.aclose()


class SSEResponse(StreamingResponse):
    """A Starlette StreamingResponse configured for Server-Sent Events.

    Sets appropriate headers:
    - Content-Type: text/event-stream
    - Cache-Control: no-cache
    - Connection: keep-alive
    - X-Accel-Buffering: no (for nginx proxy compatibility)
    """

    def __init__(
        self,
        stream: SSEStream,
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        sse_headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
        if headers:
            sse_headers.update(headers)

        super().__init__(
            content=stream.stream(),
            status_code=status_code,
            media_type="text/event-stream",
            headers=sse_headers,
        )
        self._sse_stream = stream


def create_sse_response(
    request: Request,
    generator_factory: SSEGeneratorFactory,
    *,
    keepalive_seconds: int = Constants.SSE_KEEPALIVE_SECONDS,
    headers: dict[str, str] | None = None,
) -> SSEResponse:
    """Create an SSE response from an async generator factory.

    This is the primary entry point for SSE endpoints. It:
    1. Extracts Last-Event-ID from the request headers
    2. Creates the generator with the last_event_id for replay support
    3. Wraps it in an SSEStream with keepalive timer
    4. Returns a properly configured SSEResponse

    Args:
        request: The incoming Starlette/FastAPI request.
        generator_factory: An async callable that accepts an optional last_event_id
            string and returns an async generator yielding SSEEvent objects.
        keepalive_seconds: Interval for keepalive comments (default from Constants).
        headers: Additional headers to include in the response.

    Returns:
        An SSEResponse ready to be returned from a route handler.
    """
    last_event_id = request.headers.get("last-event-id")

    generator = generator_factory(last_event_id)
    stream = SSEStream(generator, keepalive_seconds=keepalive_seconds)

    return SSEResponse(stream, headers=headers)
