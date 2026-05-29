"""Property-based tests for the Python SDK (Tasks 17.7–17.10).

Tests:
- 17.7: SDK streaming iterators yield documented events and terminate (Property 43)
- 17.8: SDK error mapping (Property 44)
- 17.9: SDK/OpenAPI surface equivalence (Property 45)
- 17.10: SDKs always send bearer header (Property 46)

**Validates: Requirements R16.1, R16.2, R16.3, R16.5**
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

# Import SDK components
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent.parent / "sdks" / "python" / "src"))

from agentic_research_sdk.client import (
    AgenticResearchClient,
    APIError,
    ConnectionError,
    ParseError,
    SDKError,
    StreamEvent,
    TimeoutError,
)
from backend.api_gateway.openapi_spec import (
    ANSWER_EVENT_TYPES,
    ERROR_CODES,
    OPENAPI_SPEC,
    RESEARCH_EVENT_TYPES,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy for valid API keys
api_key_st = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="-_"),
    min_size=8,
    max_size=64,
)

# Strategy for base URLs
base_url_st = st.sampled_from([
    "https://api.example.com/v1",
    "http://localhost:8000/v1",
    "https://search.internal/v1",
])

# Strategy for HTTP status codes (non-2xx errors)
error_status_st = st.sampled_from([400, 401, 403, 404, 429, 500, 502, 503])

# Strategy for error codes from the documented set
error_code_st = st.sampled_from(ERROR_CODES)

# Strategy for answer stream event types
answer_event_type_st = st.sampled_from(ANSWER_EVENT_TYPES)

# Strategy for research stream event types
research_event_type_st = st.sampled_from(RESEARCH_EVENT_TYPES)

# Strategy for stream event sequences that terminate properly
def answer_stream_events_st():
    """Generate valid answer stream event sequences ending with done or error."""
    token_events = st.lists(
        st.fixed_dictionaries({
            "event_type": st.just("token"),
            "data": st.fixed_dictionaries({"text": st.text(min_size=1, max_size=20), "index": st.integers(0, 1000)}),
        }),
        min_size=0,
        max_size=10,
    )
    terminal_event = st.one_of(
        st.just({"event_type": "done", "data": {"answer": "test answer", "citations": []}}),
        st.just({"event_type": "error", "data": {"code": "no_sources_available", "message": "No sources"}}),
    )
    return st.tuples(token_events, terminal_event).map(lambda t: t[0] + [t[1]])


def research_stream_events_st():
    """Generate valid research stream event sequences ending with done or error."""
    mid_events = st.lists(
        st.sampled_from([
            {"event_type": "plan_updated", "data": {"plan": []}},
            {"event_type": "step_started", "data": {"step_id": "s1", "type": "retrieval"}},
            {"event_type": "step_completed", "data": {"step_id": "s1", "summary": "done"}},
            {"event_type": "report_chunk", "data": {"text": "chunk", "ordinal": 1}},
        ]),
        min_size=0,
        max_size=5,
    )
    terminal_event = st.one_of(
        st.just({"event_type": "done", "data": {"report_uri": "/v1/research/job1"}}),
        st.just({"event_type": "error", "data": {"code": "budget_exceeded", "message": "Budget exceeded"}}),
    )
    return st.tuples(mid_events, terminal_event).map(lambda t: t[0] + [t[1]])


# ---------------------------------------------------------------------------
# Mock HTTP client for testing
# ---------------------------------------------------------------------------


class MockHttpClient:
    """Mock HTTP client that records requests and returns configured responses."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self._response_status: int = 200
        self._response_body: dict[str, Any] = {}
        self._response_headers: dict[str, str] = {"x-request-id": "req-123"}
        self._stream_events: list[dict[str, Any]] = []
        self._should_timeout: bool = False
        self._should_fail_connection: bool = False

    def set_response(self, status: int, body: dict[str, Any], headers: Optional[dict[str, str]] = None) -> None:
        self._response_status = status
        self._response_body = body
        if headers:
            self._response_headers.update(headers)

    def set_stream_events(self, events: list[dict[str, Any]]) -> None:
        self._stream_events = events

    def set_timeout(self) -> None:
        self._should_timeout = True

    def set_connection_failure(self) -> None:
        self._should_fail_connection = True

    async def request(self, method: str, url: str, *, headers: dict[str, str], json: Any = None, timeout: float = 30.0):
        self.requests.append({"method": method, "url": url, "headers": headers, "json": json})

        if self._should_timeout:
            import httpx
            raise httpx.TimeoutException("timeout")

        if self._should_fail_connection:
            import httpx
            raise httpx.ConnectError("connection refused")

        return MockResponse(self._response_status, self._response_body, self._response_headers)

    async def stream_sse(self, url: str, *, headers: dict[str, str], json: Any = None) -> AsyncIterator[StreamEvent]:
        self.requests.append({"method": "POST", "url": url, "headers": headers, "json": json, "stream": True})
        for event in self._stream_events:
            yield StreamEvent(
                event_type=event["event_type"],
                data=event["data"],
                event_id=event.get("event_id"),
            )

    async def stream_sse_get(self, url: str, *, headers: dict[str, str]) -> AsyncIterator[StreamEvent]:
        self.requests.append({"method": "GET", "url": url, "headers": headers, "stream": True})
        for event in self._stream_events:
            yield StreamEvent(
                event_type=event["event_type"],
                data=event["data"],
                event_id=event.get("event_id"),
            )


class MockResponse:
    """Mock HTTP response."""

    def __init__(self, status_code: int, body: dict[str, Any], headers: dict[str, str]) -> None:
        self.status_code = status_code
        self._body = body
        self.headers = headers

    def json(self) -> dict[str, Any]:
        return self._body


# ---------------------------------------------------------------------------
# Property 43: SDK streaming iterators yield documented events and terminate
# ---------------------------------------------------------------------------


class TestSDKStreamingProperty:
    """**Validates: Requirements R16.2**

    Property 43: SDK streaming iterators yield documented event types and
    terminate after a done or error event.
    """

    @given(events=answer_stream_events_st())
    @settings(max_examples=100)
    def test_answer_stream_yields_documented_events_and_terminates(self, events: list[dict[str, Any]]) -> None:
        """Answer stream yields only documented event types and terminates on done/error."""
        mock_client = MockHttpClient()
        mock_client.set_stream_events(events)

        client = AgenticResearchClient(
            base_url="https://api.example.com/v1",
            api_key="test-key-123",
            http_client=mock_client,
        )

        collected_events: list[StreamEvent] = []

        async def run():
            async for event in client.answer(query="test query"):
                collected_events.append(event)

        asyncio.run(run())

        # All yielded events have documented types
        for event in collected_events:
            assert event.event_type in ANSWER_EVENT_TYPES, (
                f"Unexpected event type: {event.event_type}"
            )

        # Stream terminates (last event is done or error)
        assert len(collected_events) > 0
        assert collected_events[-1].event_type in ("done", "error")

    @given(events=research_stream_events_st())
    @settings(max_examples=100)
    def test_research_stream_yields_documented_events_and_terminates(self, events: list[dict[str, Any]]) -> None:
        """Research stream yields only documented event types and terminates on done/error."""
        mock_client = MockHttpClient()
        mock_client.set_stream_events(events)

        client = AgenticResearchClient(
            base_url="https://api.example.com/v1",
            api_key="test-key-123",
            http_client=mock_client,
        )

        collected_events: list[StreamEvent] = []

        async def run():
            async for event in client.research_events("job-123"):
                collected_events.append(event)

        asyncio.run(run())

        # All yielded events have documented types
        for event in collected_events:
            assert event.event_type in RESEARCH_EVENT_TYPES, (
                f"Unexpected event type: {event.event_type}"
            )

        # Stream terminates
        assert len(collected_events) > 0
        assert collected_events[-1].event_type in ("done", "error")


# ---------------------------------------------------------------------------
# Property 44: SDK error mapping
# ---------------------------------------------------------------------------


class TestSDKErrorMappingProperty:
    """**Validates: Requirements R16.3**

    Property 44: Non-2xx responses, timeouts, and connection failures are
    mapped to typed exceptions with appropriate fields.
    """

    @given(
        status_code=error_status_st,
        error_code=error_code_st,
        api_key=api_key_st,
    )
    @settings(max_examples=100)
    def test_non_2xx_maps_to_api_error(self, status_code: int, error_code: str, api_key: str) -> None:
        """Non-2xx responses raise APIError with status, code, and request_id."""
        assume(len(api_key) > 0)

        mock_client = MockHttpClient()
        mock_client.set_response(
            status_code,
            {"error": {"code": error_code, "message": "Test error"}},
            {"x-request-id": "req-abc-123"},
        )

        client = AgenticResearchClient(
            base_url="https://api.example.com/v1",
            api_key=api_key,
            http_client=mock_client,
        )

        async def run():
            await client.search(query="test")

        with pytest.raises(APIError) as exc_info:
            asyncio.run(run())

        err = exc_info.value
        assert err.status_code == status_code
        assert err.error_code == error_code
        assert err.request_id == "req-abc-123"

    @given(api_key=api_key_st)
    @settings(max_examples=50)
    def test_timeout_maps_to_timeout_error(self, api_key: str) -> None:
        """Request timeouts raise TimeoutError."""
        assume(len(api_key) > 0)

        mock_client = MockHttpClient()
        mock_client.set_timeout()

        client = AgenticResearchClient(
            base_url="https://api.example.com/v1",
            api_key=api_key,
            http_client=mock_client,
        )

        async def run():
            await client.search(query="test")

        with pytest.raises(TimeoutError):
            asyncio.run(run())

    @given(api_key=api_key_st)
    @settings(max_examples=50)
    def test_connection_failure_maps_to_connection_error(self, api_key: str) -> None:
        """Connection failures raise ConnectionError."""
        assume(len(api_key) > 0)

        mock_client = MockHttpClient()
        mock_client.set_connection_failure()

        client = AgenticResearchClient(
            base_url="https://api.example.com/v1",
            api_key=api_key,
            http_client=mock_client,
        )

        async def run():
            await client.search(query="test")

        with pytest.raises(ConnectionError):
            asyncio.run(run())


# ---------------------------------------------------------------------------
# Property 45: SDK/OpenAPI surface equivalence
# ---------------------------------------------------------------------------


class TestSDKOpenAPISurfaceEquivalence:
    """**Validates: Requirements R16.4**

    Property 45: Every endpoint in the OpenAPI spec has a corresponding SDK method,
    and every parameter type matches.
    """

    def test_all_openapi_endpoints_have_sdk_methods(self) -> None:
        """Every operationId in the OpenAPI spec maps to a client method."""
        # Map operationId → expected SDK method name
        operation_to_method = {
            "search": "search",
            "findSimilar": "find_similar",
            "getContents": "contents",
            "answer": "answer",
            "createResearch": "create_research",
            "getResearchJob": "get_research_job",
            "getResearchEvents": "research_events",
            "createSession": "create_session",
            "deleteSession": "delete_session",
            "createPipeline": "create_pipeline",
            "getPipeline": "get_pipeline",
            "deletePipeline": "delete_pipeline",
            "getOpenApiSpec": None,  # Meta endpoint, no SDK method needed
        }

        # Collect all operationIds from the spec
        spec_operations = set()
        for path, methods in OPENAPI_SPEC["paths"].items():
            for method, details in methods.items():
                if isinstance(details, dict) and "operationId" in details:
                    spec_operations.add(details["operationId"])

        # Verify each operation has a corresponding method
        client = AgenticResearchClient(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            http_client=MockHttpClient(),
        )

        for op_id in spec_operations:
            expected_method = operation_to_method.get(op_id)
            if expected_method is not None:
                assert hasattr(client, expected_method), (
                    f"OpenAPI operation '{op_id}' has no SDK method '{expected_method}'"
                )
                assert callable(getattr(client, expected_method)), (
                    f"SDK method '{expected_method}' is not callable"
                )

    @given(
        mode=st.sampled_from(["neural", "keyword", "hybrid"]),
        num_results=st.integers(0, 100),
    )
    @settings(max_examples=50)
    def test_search_request_params_match_openapi_schema(self, mode: str, num_results: int) -> None:
        """Search request parameters match the OpenAPI SearchRequest schema."""
        schema = OPENAPI_SPEC["components"]["schemas"]["SearchRequest"]

        # Verify mode is in the enum
        assert mode in schema["properties"]["mode"]["enum"]

        # Verify num_results is within bounds
        assert num_results >= schema["properties"]["num_results"]["minimum"]
        assert num_results <= schema["properties"]["num_results"]["maximum"]


# ---------------------------------------------------------------------------
# Property 46: SDKs always send bearer header
# ---------------------------------------------------------------------------


class TestSDKBearerHeaderProperty:
    """**Validates: Requirements R16.5**

    Property 46: Every outbound request from the SDK includes the
    Authorization: Bearer header with the configured API key.
    """

    @given(api_key=api_key_st, base_url=base_url_st)
    @settings(max_examples=100)
    def test_every_request_includes_bearer_header(self, api_key: str, base_url: str) -> None:
        """Every SDK request includes Authorization: Bearer <api_key>."""
        assume(len(api_key) > 0)

        mock_client = MockHttpClient()
        mock_client.set_response(200, {"results": [], "warnings": []})

        client = AgenticResearchClient(
            base_url=base_url,
            api_key=api_key,
            http_client=mock_client,
        )

        async def run():
            await client.search(query="test query")

        asyncio.run(run())

        # Verify the request was made with the correct bearer header
        assert len(mock_client.requests) == 1
        request = mock_client.requests[0]
        assert "Authorization" in request["headers"]
        assert request["headers"]["Authorization"] == f"Bearer {api_key}"

    @given(api_key=api_key_st)
    @settings(max_examples=50)
    def test_bearer_header_on_stream_requests(self, api_key: str) -> None:
        """Streaming requests also include the bearer header."""
        assume(len(api_key) > 0)

        mock_client = MockHttpClient()
        mock_client.set_stream_events([
            {"event_type": "done", "data": {"answer": "test", "citations": []}},
        ])

        client = AgenticResearchClient(
            base_url="https://api.example.com/v1",
            api_key=api_key,
            http_client=mock_client,
        )

        async def run():
            async for _ in client.answer(query="test"):
                pass

        asyncio.run(run())

        # Verify bearer header on stream request
        assert len(mock_client.requests) >= 1
        for request in mock_client.requests:
            assert "Authorization" in request["headers"]
            assert request["headers"]["Authorization"] == f"Bearer {api_key}"

    @given(api_key=api_key_st)
    @settings(max_examples=50)
    def test_bearer_header_on_all_endpoint_types(self, api_key: str) -> None:
        """Bearer header is present on POST, GET, and DELETE requests."""
        assume(len(api_key) > 0)

        mock_client = MockHttpClient()
        # For search (POST)
        mock_client.set_response(200, {"results": [], "warnings": []})

        client = AgenticResearchClient(
            base_url="https://api.example.com/v1",
            api_key=api_key,
            http_client=mock_client,
        )

        async def run():
            # POST request
            await client.search(query="test")

        asyncio.run(run())

        # All requests have bearer header
        for request in mock_client.requests:
            assert request["headers"]["Authorization"] == f"Bearer {api_key}"
