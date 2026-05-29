"""Agentic Research Search Engine - Python SDK Client.

Task 17.2: Generated Python SDK from OpenAPI with typed methods for all endpoints
and async iterators for streams.

Task 17.4: SDK error mapping — non-2xx, timeout, connection failure → typed exception.
Task 17.5: SDK bearer header injection from configured key.

Validates: Requirements R16.1, R16.2, R16.3, R16.5
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator, Optional


# ---------------------------------------------------------------------------
# Error types (Task 17.4)
# ---------------------------------------------------------------------------


class SDKError(Exception):
    """Base exception for all SDK errors."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        error_code: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.request_id = request_id


class APIError(SDKError):
    """Raised when the API returns a non-2xx response."""

    pass


class TimeoutError(SDKError):
    """Raised when a request times out."""

    pass


class ConnectionError(SDKError):
    """Raised when a network connection fails."""

    pass


class ParseError(SDKError):
    """Raised when the SDK fails to parse the response payload."""

    pass


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class SearchMode(str, Enum):
    """Search retrieval mode."""

    NEURAL = "neural"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


@dataclass
class ProvenanceInfo:
    """Provenance scoring information for a document."""

    credibility_score: float
    ai_generated_likelihood: float
    scored_at: str


@dataclass
class SearchResult:
    """A single search result."""

    document_id: str
    url: str
    title: str
    score: float
    published_at: Optional[str]
    provenance: ProvenanceInfo


@dataclass
class SearchResponse:
    """Response from a search request."""

    results: list[SearchResult]
    warnings: list[dict[str, str]] = field(default_factory=list)
    index_version: Optional[int] = None


@dataclass
class Citation:
    """A citation linking answer text to a source document."""

    document_id: str
    version: int
    answer_start: int
    answer_end: int
    source_start: int
    source_end: int


@dataclass
class ContentEntry:
    """A single content retrieval entry."""

    document_id: str
    version: Optional[int] = None
    cleaned_text: Optional[str] = None
    highlights: Optional[list[dict[str, int]]] = None
    summary: Optional[str] = None
    error: Optional[dict[str, str]] = None


@dataclass
class ContentsResponse:
    """Response from a contents request."""

    results: list[ContentEntry]


@dataclass
class Session:
    """A research session."""

    session_id: str
    created_at: str
    retention_days: int
    expires_at: Optional[str] = None


@dataclass
class Pipeline:
    """A retrieval pipeline."""

    pipeline_id: str
    name: str
    steps: list[dict[str, Any]]
    created_at: str


@dataclass
class ResearchJob:
    """A research job."""

    job_id: str
    state: str
    created_at: str
    report: Optional[dict[str, Any]] = None
    citations: Optional[list[Citation]] = None


# ---------------------------------------------------------------------------
# Stream event types
# ---------------------------------------------------------------------------


@dataclass
class StreamEvent:
    """A single event from an SSE stream."""

    event_type: str
    data: dict[str, Any]
    event_id: Optional[int] = None


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class AgenticResearchClient:
    """Python SDK client for the Agentic Research Search Engine API.

    Provides typed methods for all endpoints with async iterators for streams.
    Automatically injects the bearer token on every request (R16.5).

    Args:
        base_url: The base URL of the API (e.g., "https://api.example.com/v1").
        api_key: The tenant-scoped API key for authentication.
        timeout: Request timeout in seconds (default 30).
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout: float = 30.0,
        http_client: Any = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._http_client = http_client

    @property
    def _headers(self) -> dict[str, str]:
        """Build request headers with bearer token injection (R16.5)."""
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _build_url(self, path: str) -> str:
        """Build full URL from path."""
        return f"{self._base_url}{path}"

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Optional[dict[str, Any]] = None,
        expected_status: int = 200,
    ) -> dict[str, Any]:
        """Make an HTTP request with error mapping (R16.3).

        Maps non-2xx responses, timeouts, and connection failures to typed exceptions.
        """
        url = self._build_url(path)

        if self._http_client is not None:
            # Use injected client (for testing)
            try:
                response = await self._http_client.request(
                    method, url, headers=self._headers, json=json_body, timeout=self._timeout
                )
            except Exception as e:
                # Map exceptions from injected client the same way
                err_name = type(e).__name__
                if "Timeout" in err_name or "timeout" in str(e).lower():
                    raise TimeoutError(
                        f"Request timed out after {self._timeout}s",
                        status_code=None,
                        error_code=None,
                        request_id=None,
                    ) from e
                elif "Connect" in err_name or "connection" in str(e).lower():
                    raise ConnectionError(
                        f"Connection failed: {e}",
                        status_code=None,
                        error_code=None,
                        request_id=None,
                    ) from e
                raise
        else:
            # Use httpx
            import httpx

            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.request(
                        method, url, headers=self._headers, json=json_body
                    )
            except httpx.TimeoutException as e:
                raise TimeoutError(
                    f"Request timed out after {self._timeout}s",
                    status_code=None,
                    error_code=None,
                    request_id=None,
                ) from e
            except httpx.ConnectError as e:
                raise ConnectionError(
                    f"Connection failed: {e}",
                    status_code=None,
                    error_code=None,
                    request_id=None,
                ) from e

        # Extract request_id from response headers
        request_id = None
        if hasattr(response, "headers"):
            request_id = response.headers.get("X-Request-Id") or response.headers.get("x-request-id")

        status_code = response.status_code if hasattr(response, "status_code") else getattr(response, "status", None)

        if status_code != expected_status and (status_code < 200 or status_code >= 300):
            # Parse error body
            error_code = None
            message = f"HTTP {status_code}"
            try:
                body = response.json() if hasattr(response, "json") else {}
                if callable(body):
                    body = body()
                if "error" in body:
                    error_code = body["error"].get("code")
                    message = body["error"].get("message", message)
            except Exception:
                pass

            raise APIError(
                message,
                status_code=status_code,
                error_code=error_code,
                request_id=request_id,
            )

        try:
            result = response.json() if hasattr(response, "json") else {}
            if callable(result):
                result = result()
            return result
        except Exception as e:
            raise ParseError(
                f"Failed to parse response: {e}",
                status_code=status_code,
                error_code=None,
                request_id=request_id,
            ) from e

    # -----------------------------------------------------------------------
    # Search endpoints
    # -----------------------------------------------------------------------

    async def search(
        self,
        query: str,
        mode: SearchMode = SearchMode.HYBRID,
        *,
        num_results: int = 10,
        filters: Optional[str] = None,
        pipeline_id: Optional[str] = None,
        min_credibility: Optional[float] = None,
        max_ai_generated_likelihood: Optional[float] = None,
    ) -> SearchResponse:
        """Execute a search query (POST /v1/search).

        Args:
            query: Search query (1-2048 code points).
            mode: Retrieval mode (neural, keyword, hybrid).
            num_results: Number of results (0-100, default 10).
            filters: Optional Query_Filter_DSL string.
            pipeline_id: Optional pipeline ID.
            min_credibility: Minimum credibility threshold.
            max_ai_generated_likelihood: Maximum AI-generated likelihood threshold.

        Returns:
            SearchResponse with ranked results.
        """
        body: dict[str, Any] = {"query": query, "mode": mode.value, "num_results": num_results}
        if filters is not None:
            body["filters"] = filters
        if pipeline_id is not None:
            body["pipeline_id"] = pipeline_id
        if min_credibility is not None:
            body["min_credibility"] = min_credibility
        if max_ai_generated_likelihood is not None:
            body["max_ai_generated_likelihood"] = max_ai_generated_likelihood

        data = await self._request("POST", "/search", json_body=body)
        return self._parse_search_response(data)

    async def find_similar(
        self,
        url: str,
        *,
        num_results: int = 10,
        filters: Optional[str] = None,
        min_credibility: Optional[float] = None,
        max_ai_generated_likelihood: Optional[float] = None,
    ) -> SearchResponse:
        """Find semantically similar documents (POST /v1/find_similar).

        Args:
            url: URL to find similar documents for.
            num_results: Number of results (0-100, default 10).
            filters: Optional Query_Filter_DSL string.
            min_credibility: Minimum credibility threshold.
            max_ai_generated_likelihood: Maximum AI-generated likelihood threshold.

        Returns:
            SearchResponse with similar results.
        """
        body: dict[str, Any] = {"url": url, "num_results": num_results}
        if filters is not None:
            body["filters"] = filters
        if min_credibility is not None:
            body["min_credibility"] = min_credibility
        if max_ai_generated_likelihood is not None:
            body["max_ai_generated_likelihood"] = max_ai_generated_likelihood

        data = await self._request("POST", "/find_similar", json_body=body)
        return self._parse_search_response(data)

    async def contents(
        self,
        document_ids: list[str],
        *,
        highlights: bool = False,
        query: Optional[str] = None,
        summary: bool = False,
    ) -> ContentsResponse:
        """Retrieve cleaned text, highlights, summaries (POST /v1/contents).

        Args:
            document_ids: List of document IDs (1-100).
            highlights: Whether to include highlight spans.
            query: Query for highlights (required if highlights=True).
            summary: Whether to include summaries.

        Returns:
            ContentsResponse with content entries.
        """
        body: dict[str, Any] = {"document_ids": document_ids, "highlights": highlights, "summary": summary}
        if query is not None:
            body["query"] = query

        data = await self._request("POST", "/contents", json_body=body)
        results = []
        for entry in data.get("results", []):
            results.append(
                ContentEntry(
                    document_id=entry.get("document_id", ""),
                    version=entry.get("version"),
                    cleaned_text=entry.get("cleaned_text"),
                    highlights=entry.get("highlights"),
                    summary=entry.get("summary"),
                    error=entry.get("error"),
                )
            )
        return ContentsResponse(results=results)

    # -----------------------------------------------------------------------
    # Answer endpoint with streaming (R16.2)
    # -----------------------------------------------------------------------

    async def answer(
        self,
        query: str,
        *,
        mode: SearchMode = SearchMode.HYBRID,
        num_results: int = 10,
        stream: bool = True,
        session_id: Optional[str] = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream an answer with citations (POST /v1/answer).

        Yields StreamEvent objects for token, citation, done, and error events.
        Terminates after yielding a done or error event (R16.2).

        Args:
            query: The question to answer.
            mode: Retrieval mode.
            num_results: Number of source documents.
            stream: Whether to stream (default True).
            session_id: Optional session for context.

        Yields:
            StreamEvent objects.
        """
        body: dict[str, Any] = {
            "query": query,
            "mode": mode.value,
            "num_results": num_results,
            "stream": stream,
        }
        if session_id is not None:
            body["session_id"] = session_id

        async for event in self._stream_sse("/answer", body):
            yield event
            if event.event_type in ("done", "error"):
                return

    # -----------------------------------------------------------------------
    # Research endpoints
    # -----------------------------------------------------------------------

    async def create_research(
        self,
        research_goal: str,
        *,
        output_schema: Optional[dict[str, Any]] = None,
        session_id: Optional[str] = None,
        max_steps: Optional[int] = None,
        max_duration_ms: Optional[int] = None,
        max_tool_calls: Optional[int] = None,
    ) -> str:
        """Launch a research job (POST /v1/research).

        Args:
            research_goal: The research goal (1-4096 chars).
            output_schema: Optional JSON Schema for structured output.
            session_id: Optional session for context.
            max_steps: Maximum steps budget.
            max_duration_ms: Maximum duration budget.
            max_tool_calls: Maximum tool calls budget.

        Returns:
            The job_id string.
        """
        body: dict[str, Any] = {"research_goal": research_goal}
        if output_schema is not None:
            body["output_schema"] = output_schema
        if session_id is not None:
            body["session_id"] = session_id
        if max_steps is not None:
            body["max_steps"] = max_steps
        if max_duration_ms is not None:
            body["max_duration_ms"] = max_duration_ms
        if max_tool_calls is not None:
            body["max_tool_calls"] = max_tool_calls

        data = await self._request("POST", "/research", json_body=body, expected_status=201)
        return data["job_id"]

    async def get_research_job(self, job_id: str) -> ResearchJob:
        """Get research job report (GET /v1/research/{job_id}).

        Args:
            job_id: The research job ID.

        Returns:
            ResearchJob with report and citations.
        """
        data = await self._request("GET", f"/research/{job_id}")
        citations = None
        if data.get("citations"):
            citations = [
                Citation(
                    document_id=c["document_id"],
                    version=c["version"],
                    answer_start=c["answer_start"],
                    answer_end=c["answer_end"],
                    source_start=c["source_start"],
                    source_end=c["source_end"],
                )
                for c in data["citations"]
            ]
        return ResearchJob(
            job_id=data["job_id"],
            state=data["state"],
            created_at=data["created_at"],
            report=data.get("report"),
            citations=citations,
        )

    async def research_events(
        self, job_id: str, *, last_event_id: Optional[int] = None
    ) -> AsyncIterator[StreamEvent]:
        """Stream research job events (GET /v1/research/{job_id}/events).

        Yields StreamEvent objects. Terminates after done or error event (R16.2).

        Args:
            job_id: The research job ID.
            last_event_id: Resume from this event ID.

        Yields:
            StreamEvent objects.
        """
        path = f"/research/{job_id}/events"
        extra_headers = {}
        if last_event_id is not None:
            extra_headers["Last-Event-ID"] = str(last_event_id)

        async for event in self._stream_sse_get(path, extra_headers=extra_headers):
            yield event
            if event.event_type in ("done", "error"):
                return

    # -----------------------------------------------------------------------
    # Session endpoints
    # -----------------------------------------------------------------------

    async def create_session(self, *, retention_days: int = 14) -> Session:
        """Create a research session (POST /v1/sessions).

        Args:
            retention_days: Session retention in days (1-90, default 14).

        Returns:
            Session object.
        """
        body = {"retention_days": retention_days}
        data = await self._request("POST", "/sessions", json_body=body, expected_status=201)
        return Session(
            session_id=data["session_id"],
            created_at=data["created_at"],
            retention_days=data["retention_days"],
            expires_at=data.get("expires_at"),
        )

    async def delete_session(self, session_id: str) -> None:
        """Delete a session (DELETE /v1/sessions/{session_id}).

        Args:
            session_id: The session ID to delete.
        """
        url = self._build_url(f"/sessions/{session_id}")
        if self._http_client is not None:
            response = await self._http_client.request("DELETE", url, headers=self._headers, timeout=self._timeout)
        else:
            import httpx

            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.request("DELETE", url, headers=self._headers)
            except httpx.TimeoutException as e:
                raise TimeoutError(f"Request timed out", status_code=None, error_code=None, request_id=None) from e
            except httpx.ConnectError as e:
                raise ConnectionError(f"Connection failed: {e}", status_code=None, error_code=None, request_id=None) from e

        status_code = response.status_code if hasattr(response, "status_code") else getattr(response, "status", None)
        if status_code not in (204, 200):
            request_id = None
            if hasattr(response, "headers"):
                request_id = response.headers.get("X-Request-Id")
            raise APIError(f"HTTP {status_code}", status_code=status_code, error_code=None, request_id=request_id)

    # -----------------------------------------------------------------------
    # Pipeline endpoints
    # -----------------------------------------------------------------------

    async def create_pipeline(self, name: str, steps: list[dict[str, Any]]) -> Pipeline:
        """Create a retrieval pipeline (POST /v1/pipelines).

        Args:
            name: Pipeline name.
            steps: List of pipeline step definitions.

        Returns:
            Pipeline object.
        """
        body = {"name": name, "steps": steps}
        data = await self._request("POST", "/pipelines", json_body=body, expected_status=201)
        return Pipeline(
            pipeline_id=data["pipeline_id"],
            name=data["name"],
            steps=data["steps"],
            created_at=data["created_at"],
        )

    async def get_pipeline(self, pipeline_id: str) -> Pipeline:
        """Get pipeline definition (GET /v1/pipelines/{pipeline_id}).

        Args:
            pipeline_id: The pipeline ID.

        Returns:
            Pipeline object.
        """
        data = await self._request("GET", f"/pipelines/{pipeline_id}")
        return Pipeline(
            pipeline_id=data["pipeline_id"],
            name=data["name"],
            steps=data["steps"],
            created_at=data["created_at"],
        )

    async def delete_pipeline(self, pipeline_id: str) -> None:
        """Delete a pipeline (DELETE /v1/pipelines/{pipeline_id}).

        Args:
            pipeline_id: The pipeline ID to delete.
        """
        url = self._build_url(f"/pipelines/{pipeline_id}")
        if self._http_client is not None:
            response = await self._http_client.request("DELETE", url, headers=self._headers, timeout=self._timeout)
        else:
            import httpx

            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.request("DELETE", url, headers=self._headers)
            except httpx.TimeoutException as e:
                raise TimeoutError(f"Request timed out", status_code=None, error_code=None, request_id=None) from e
            except httpx.ConnectError as e:
                raise ConnectionError(f"Connection failed: {e}", status_code=None, error_code=None, request_id=None) from e

        status_code = response.status_code if hasattr(response, "status_code") else getattr(response, "status", None)
        if status_code not in (204, 200):
            request_id = None
            if hasattr(response, "headers"):
                request_id = response.headers.get("X-Request-Id")
            raise APIError(f"HTTP {status_code}", status_code=status_code, error_code=None, request_id=request_id)

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _parse_search_response(self, data: dict[str, Any]) -> SearchResponse:
        """Parse a search response dict into SearchResponse."""
        results = []
        for r in data.get("results", []):
            prov = r.get("provenance", {})
            results.append(
                SearchResult(
                    document_id=r["document_id"],
                    url=r["url"],
                    title=r["title"],
                    score=r["score"],
                    published_at=r.get("published_at"),
                    provenance=ProvenanceInfo(
                        credibility_score=prov.get("credibility_score", 0.0),
                        ai_generated_likelihood=prov.get("ai_generated_likelihood", 0.0),
                        scored_at=prov.get("scored_at", ""),
                    ),
                )
            )
        return SearchResponse(
            results=results,
            warnings=data.get("warnings", []),
            index_version=data.get("index_version"),
        )

    async def _stream_sse(self, path: str, body: dict[str, Any]) -> AsyncIterator[StreamEvent]:
        """Stream SSE events from a POST endpoint."""
        url = self._build_url(path)
        headers = {**self._headers, "Accept": "text/event-stream"}

        if self._http_client is not None:
            # For testing: use mock client's stream method
            async for event in self._http_client.stream_sse(url, headers=headers, json=body):
                yield event
        else:
            import httpx

            try:
                async with httpx.AsyncClient(timeout=None) as client:
                    async with client.stream("POST", url, headers=headers, json=body) as response:
                        async for event in self._parse_sse_stream(response.aiter_lines()):
                            yield event
            except httpx.TimeoutException as e:
                raise TimeoutError(f"Stream timed out", status_code=None, error_code=None, request_id=None) from e
            except httpx.ConnectError as e:
                raise ConnectionError(f"Connection failed: {e}", status_code=None, error_code=None, request_id=None) from e

    async def _stream_sse_get(
        self, path: str, *, extra_headers: Optional[dict[str, str]] = None
    ) -> AsyncIterator[StreamEvent]:
        """Stream SSE events from a GET endpoint."""
        url = self._build_url(path)
        headers = {**self._headers, "Accept": "text/event-stream"}
        if extra_headers:
            headers.update(extra_headers)

        if self._http_client is not None:
            async for event in self._http_client.stream_sse_get(url, headers=headers):
                yield event
        else:
            import httpx

            try:
                async with httpx.AsyncClient(timeout=None) as client:
                    async with client.stream("GET", url, headers=headers) as response:
                        async for event in self._parse_sse_stream(response.aiter_lines()):
                            yield event
            except httpx.TimeoutException as e:
                raise TimeoutError(f"Stream timed out", status_code=None, error_code=None, request_id=None) from e
            except httpx.ConnectError as e:
                raise ConnectionError(f"Connection failed: {e}", status_code=None, error_code=None, request_id=None) from e

    async def _parse_sse_stream(self, lines: AsyncIterator[str]) -> AsyncIterator[StreamEvent]:
        """Parse SSE text/event-stream lines into StreamEvent objects."""
        event_type = ""
        event_id: Optional[int] = None
        data_lines: list[str] = []

        async for line in lines:
            if line.startswith("event:"):
                event_type = line[6:].strip()
            elif line.startswith("id:"):
                try:
                    event_id = int(line[3:].strip())
                except ValueError:
                    event_id = None
            elif line.startswith("data:"):
                data_lines.append(line[5:].strip())
            elif line.strip() == "" and (event_type or data_lines):
                # End of event
                data_str = "\n".join(data_lines)
                try:
                    data = json.loads(data_str) if data_str else {}
                except json.JSONDecodeError:
                    data = {"raw": data_str}

                yield StreamEvent(event_type=event_type, data=data, event_id=event_id)

                event_type = ""
                event_id = None
                data_lines = []
