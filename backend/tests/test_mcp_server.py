"""Unit tests for the MCP Server (Task 16, R12).

Tests cover:
- Tool definitions with JSON Schema (R12.1).
- Tool dispatch to backing subsystems (R12.2).
- Input schema validation → MCP-standard validation error (R12.3).
- Shared auth + rate limits (R12.4–R12.6).
- Output schema validation → MCP-standard tool-execution error (R12.7).
"""

from __future__ import annotations

from typing import Any

import pytest

from backend.mcp_server.schemas import TOOL_DEFINITIONS
from backend.mcp_server.server import (
    MCPError,
    MCPErrorCode,
    MCPServer,
    MCPToolCall,
    MCPToolResult,
)


# ---------------------------------------------------------------------------
# Mock subsystems
# ---------------------------------------------------------------------------


class MockAuthChecker:
    """Mock auth checker for testing."""

    def __init__(self, *, authenticated: bool = True, tenant_id: str = "tenant-1"):
        self._authenticated = authenticated
        self._tenant_id = tenant_id
        self._error_message = "Invalid API key"

    async def authenticate(self, api_key: str) -> tuple[bool, str | None, str | None]:
        if self._authenticated:
            return (True, self._tenant_id, None)
        return (False, None, self._error_message)


class MockRateLimiter:
    """Mock rate limiter for testing."""

    def __init__(self, *, allowed: bool = True, retry_after: int = 60):
        self._allowed = allowed
        self._retry_after = retry_after

    async def check_rate_limit(self, tenant_id: str, endpoint: str) -> tuple[bool, int | None]:
        if self._allowed:
            return (True, None)
        return (False, self._retry_after)


class MockSearchSubsystem:
    """Mock search subsystem."""

    def __init__(self, *, result: dict[str, Any] | None = None, error: Exception | None = None):
        self._result = result or {"results": [], "warnings": []}
        self._error = error
        self.last_call: tuple[dict, str] | None = None

    async def search(self, arguments: dict[str, Any], tenant_id: str) -> dict[str, Any]:
        self.last_call = (arguments, tenant_id)
        if self._error:
            raise self._error
        return self._result


class MockFindSimilarSubsystem:
    """Mock find_similar subsystem."""

    def __init__(self, *, result: dict[str, Any] | None = None, error: Exception | None = None):
        self._result = result or {"results": [], "warnings": []}
        self._error = error
        self.last_call: tuple[dict, str] | None = None

    async def find_similar(self, arguments: dict[str, Any], tenant_id: str) -> dict[str, Any]:
        self.last_call = (arguments, tenant_id)
        if self._error:
            raise self._error
        return self._result


class MockContentsSubsystem:
    """Mock contents subsystem."""

    def __init__(self, *, result: dict[str, Any] | None = None, error: Exception | None = None):
        self._result = result or {"results": []}
        self._error = error
        self.last_call: tuple[dict, str] | None = None

    async def get_contents(self, arguments: dict[str, Any], tenant_id: str) -> dict[str, Any]:
        self.last_call = (arguments, tenant_id)
        if self._error:
            raise self._error
        return self._result


class MockAnswerSubsystem:
    """Mock answer subsystem."""

    def __init__(self, *, result: dict[str, Any] | None = None, error: Exception | None = None):
        self._result = result or {"answer": "Test answer", "citations": []}
        self._error = error
        self.last_call: tuple[dict, str] | None = None

    async def generate_answer(self, arguments: dict[str, Any], tenant_id: str) -> dict[str, Any]:
        self.last_call = (arguments, tenant_id)
        if self._error:
            raise self._error
        return self._result


class MockResearchSubsystem:
    """Mock research subsystem."""

    def __init__(self, *, result: dict[str, Any] | None = None, error: Exception | None = None):
        self._result = result or {"job_id": "job-123", "status": "queued"}
        self._error = error
        self.last_call: tuple[dict, str] | None = None

    async def start_research(self, arguments: dict[str, Any], tenant_id: str) -> dict[str, Any]:
        self.last_call = (arguments, tenant_id)
        if self._error:
            raise self._error
        return self._result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def search_subsystem():
    return MockSearchSubsystem()


@pytest.fixture
def find_similar_subsystem():
    return MockFindSimilarSubsystem()


@pytest.fixture
def contents_subsystem():
    return MockContentsSubsystem()


@pytest.fixture
def answer_subsystem():
    return MockAnswerSubsystem()


@pytest.fixture
def research_subsystem():
    return MockResearchSubsystem()


@pytest.fixture
def mcp_server(search_subsystem, find_similar_subsystem, contents_subsystem, answer_subsystem, research_subsystem):
    """Create an MCP server with all mock subsystems."""
    return MCPServer(
        auth_checker=MockAuthChecker(),
        rate_limiter=MockRateLimiter(),
        search_subsystem=search_subsystem,
        find_similar_subsystem=find_similar_subsystem,
        contents_subsystem=contents_subsystem,
        answer_subsystem=answer_subsystem,
        research_subsystem=research_subsystem,
    )


# ---------------------------------------------------------------------------
# Task 16.1: Tool definitions with JSON Schema
# ---------------------------------------------------------------------------


class TestToolDefinitions:
    """Tests for R12.1: MCP tool definitions with JSON Schema."""

    def test_all_five_tools_defined(self):
        """All five tools are defined: search, find_similar, contents, answer, research."""
        expected_tools = {"search", "find_similar", "contents", "answer", "research"}
        assert set(TOOL_DEFINITIONS.keys()) == expected_tools

    def test_each_tool_has_input_schema(self):
        """Each tool has an input_schema."""
        for name, defn in TOOL_DEFINITIONS.items():
            assert "input_schema" in defn, f"Tool {name} missing input_schema"
            assert isinstance(defn["input_schema"], dict)
            assert defn["input_schema"].get("type") == "object"

    def test_each_tool_has_output_schema(self):
        """Each tool has an output_schema."""
        for name, defn in TOOL_DEFINITIONS.items():
            assert "output_schema" in defn, f"Tool {name} missing output_schema"
            assert isinstance(defn["output_schema"], dict)

    def test_each_tool_has_description(self):
        """Each tool has a description."""
        for name, defn in TOOL_DEFINITIONS.items():
            assert "description" in defn, f"Tool {name} missing description"
            assert len(defn["description"]) > 0

    def test_list_tools_returns_all(self, mcp_server):
        """list_tools returns all five tools with schemas."""
        tools = mcp_server.list_tools()
        assert len(tools) == 5
        names = {t["name"] for t in tools}
        assert names == {"search", "find_similar", "contents", "answer", "research"}
        for tool in tools:
            assert "inputSchema" in tool
            assert "description" in tool

    def test_search_input_schema_requires_query_and_mode(self):
        """Search input schema requires query and mode."""
        schema = TOOL_DEFINITIONS["search"]["input_schema"]
        assert "query" in schema["required"]
        assert "mode" in schema["required"]

    def test_contents_input_schema_requires_document_ids(self):
        """Contents input schema requires document_ids."""
        schema = TOOL_DEFINITIONS["contents"]["input_schema"]
        assert "document_ids" in schema["required"]

    def test_research_input_schema_requires_research_goal(self):
        """Research input schema requires research_goal."""
        schema = TOOL_DEFINITIONS["research"]["input_schema"]
        assert "research_goal" in schema["required"]


# ---------------------------------------------------------------------------
# Task 16.2: Tool dispatch to backing subsystems
# ---------------------------------------------------------------------------


class TestToolDispatch:
    """Tests for R12.2: tool dispatch to backing subsystems."""

    @pytest.mark.asyncio
    async def test_search_dispatches_to_retriever(self, mcp_server, search_subsystem):
        """Search tool dispatches to the search subsystem."""
        call = MCPToolCall(
            tool_name="search",
            arguments={"query": "test query", "mode": "neural"},
            api_key="test-key",
        )
        result = await mcp_server.call_tool(call)
        assert result.success is True
        assert search_subsystem.last_call is not None
        assert search_subsystem.last_call[0] == {"query": "test query", "mode": "neural"}
        assert search_subsystem.last_call[1] == "tenant-1"

    @pytest.mark.asyncio
    async def test_find_similar_dispatches_to_retriever(self, mcp_server, find_similar_subsystem):
        """find_similar tool dispatches to the find_similar subsystem."""
        call = MCPToolCall(
            tool_name="find_similar",
            arguments={"url": "https://example.com/page"},
            api_key="test-key",
        )
        result = await mcp_server.call_tool(call)
        assert result.success is True
        assert find_similar_subsystem.last_call is not None

    @pytest.mark.asyncio
    async def test_contents_dispatches_to_search_engine(self, mcp_server, contents_subsystem):
        """Contents tool dispatches to the contents subsystem."""
        call = MCPToolCall(
            tool_name="contents",
            arguments={"document_ids": ["doc-1", "doc-2"]},
            api_key="test-key",
        )
        result = await mcp_server.call_tool(call)
        assert result.success is True
        assert contents_subsystem.last_call is not None

    @pytest.mark.asyncio
    async def test_answer_dispatches_to_answer_engine(self, mcp_server, answer_subsystem):
        """Answer tool dispatches to the answer subsystem."""
        call = MCPToolCall(
            tool_name="answer",
            arguments={"query": "What is ML?"},
            api_key="test-key",
        )
        result = await mcp_server.call_tool(call)
        assert result.success is True
        assert answer_subsystem.last_call is not None

    @pytest.mark.asyncio
    async def test_research_dispatches_to_research_agent(self, mcp_server, research_subsystem):
        """Research tool dispatches to the research subsystem."""
        call = MCPToolCall(
            tool_name="research",
            arguments={"research_goal": "Investigate quantum computing"},
            api_key="test-key",
        )
        result = await mcp_server.call_tool(call)
        assert result.success is True
        assert research_subsystem.last_call is not None

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self, mcp_server):
        """Unknown tool name returns validation error."""
        call = MCPToolCall(
            tool_name="nonexistent_tool",
            arguments={},
            api_key="test-key",
        )
        result = await mcp_server.call_tool(call)
        assert result.success is False
        assert result.error.code == MCPErrorCode.VALIDATION_ERROR

    @pytest.mark.asyncio
    async def test_subsystem_error_returns_tool_execution_error(self, mcp_server):
        """Subsystem exception returns tool_execution_error."""
        # Replace search subsystem with one that raises
        mcp_server._search = MockSearchSubsystem(error=RuntimeError("DB connection failed"))
        call = MCPToolCall(
            tool_name="search",
            arguments={"query": "test", "mode": "neural"},
            api_key="test-key",
        )
        result = await mcp_server.call_tool(call)
        assert result.success is False
        assert result.error.code == MCPErrorCode.TOOL_EXECUTION_ERROR
        assert "DB connection failed" in result.error.message


# ---------------------------------------------------------------------------
# Task 16.3: Input schema validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Tests for R12.3: input schema validation → MCP-standard validation error."""

    @pytest.mark.asyncio
    async def test_missing_required_field_rejected(self, mcp_server):
        """Missing required field returns validation error."""
        call = MCPToolCall(
            tool_name="search",
            arguments={"query": "test"},  # Missing 'mode'
            api_key="test-key",
        )
        result = await mcp_server.call_tool(call)
        assert result.success is False
        assert result.error.code == MCPErrorCode.VALIDATION_ERROR
        assert "mode" in result.error.message.lower() or "required" in result.error.message.lower()

    @pytest.mark.asyncio
    async def test_invalid_mode_value_rejected(self, mcp_server):
        """Invalid enum value returns validation error."""
        call = MCPToolCall(
            tool_name="search",
            arguments={"query": "test", "mode": "invalid_mode"},
            api_key="test-key",
        )
        result = await mcp_server.call_tool(call)
        assert result.success is False
        assert result.error.code == MCPErrorCode.VALIDATION_ERROR

    @pytest.mark.asyncio
    async def test_num_results_out_of_range_rejected(self, mcp_server):
        """num_results > 100 returns validation error."""
        call = MCPToolCall(
            tool_name="search",
            arguments={"query": "test", "mode": "neural", "num_results": 101},
            api_key="test-key",
        )
        result = await mcp_server.call_tool(call)
        assert result.success is False
        assert result.error.code == MCPErrorCode.VALIDATION_ERROR

    @pytest.mark.asyncio
    async def test_empty_document_ids_rejected(self, mcp_server):
        """Empty document_ids array returns validation error."""
        call = MCPToolCall(
            tool_name="contents",
            arguments={"document_ids": []},
            api_key="test-key",
        )
        result = await mcp_server.call_tool(call)
        assert result.success is False
        assert result.error.code == MCPErrorCode.VALIDATION_ERROR

    @pytest.mark.asyncio
    async def test_additional_properties_rejected(self, mcp_server):
        """Additional properties not in schema are rejected."""
        call = MCPToolCall(
            tool_name="search",
            arguments={"query": "test", "mode": "neural", "unknown_field": "value"},
            api_key="test-key",
        )
        result = await mcp_server.call_tool(call)
        assert result.success is False
        assert result.error.code == MCPErrorCode.VALIDATION_ERROR

    @pytest.mark.asyncio
    async def test_valid_input_passes_validation(self, mcp_server):
        """Valid input passes validation and reaches the subsystem."""
        call = MCPToolCall(
            tool_name="search",
            arguments={"query": "test query", "mode": "hybrid", "num_results": 10},
            api_key="test-key",
        )
        result = await mcp_server.call_tool(call)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_validation_error_includes_path(self, mcp_server):
        """Validation error includes the offending argument path."""
        call = MCPToolCall(
            tool_name="search",
            arguments={"query": "test", "mode": "neural", "num_results": "not_a_number"},
            api_key="test-key",
        )
        result = await mcp_server.call_tool(call)
        assert result.success is False
        assert result.error.code == MCPErrorCode.VALIDATION_ERROR
        assert result.error.details is not None

    @pytest.mark.asyncio
    async def test_subsystem_not_invoked_on_validation_failure(self, mcp_server, search_subsystem):
        """Subsystem is NOT invoked when input validation fails."""
        call = MCPToolCall(
            tool_name="search",
            arguments={"query": "test"},  # Missing mode
            api_key="test-key",
        )
        await mcp_server.call_tool(call)
        assert search_subsystem.last_call is None


# ---------------------------------------------------------------------------
# Task 16.4: Shared auth + rate limits
# ---------------------------------------------------------------------------


class TestAuthAndRateLimits:
    """Tests for R12.4–R12.6: shared auth + rate limits with REST gateway."""

    @pytest.mark.asyncio
    async def test_missing_api_key_returns_auth_error(self):
        """Missing API key returns authentication error."""
        server = MCPServer(
            auth_checker=MockAuthChecker(),
            search_subsystem=MockSearchSubsystem(),
        )
        call = MCPToolCall(
            tool_name="search",
            arguments={"query": "test", "mode": "neural"},
            api_key=None,
        )
        result = await server.call_tool(call)
        assert result.success is False
        assert result.error.code == MCPErrorCode.AUTHENTICATION_ERROR

    @pytest.mark.asyncio
    async def test_invalid_api_key_returns_auth_error(self):
        """Invalid API key returns authentication error."""
        server = MCPServer(
            auth_checker=MockAuthChecker(authenticated=False),
            search_subsystem=MockSearchSubsystem(),
        )
        call = MCPToolCall(
            tool_name="search",
            arguments={"query": "test", "mode": "neural"},
            api_key="invalid-key",
        )
        result = await server.call_tool(call)
        assert result.success is False
        assert result.error.code == MCPErrorCode.AUTHENTICATION_ERROR

    @pytest.mark.asyncio
    async def test_rate_limited_returns_rate_limit_error(self):
        """Rate-limited request returns rate_limit_error."""
        server = MCPServer(
            auth_checker=MockAuthChecker(),
            rate_limiter=MockRateLimiter(allowed=False, retry_after=30),
            search_subsystem=MockSearchSubsystem(),
        )
        call = MCPToolCall(
            tool_name="search",
            arguments={"query": "test", "mode": "neural"},
            api_key="valid-key",
        )
        result = await server.call_tool(call)
        assert result.success is False
        assert result.error.code == MCPErrorCode.RATE_LIMIT_ERROR
        assert result.error.details["retry_after"] == 30

    @pytest.mark.asyncio
    async def test_auth_checked_before_rate_limit(self):
        """Auth is checked before rate limits."""
        server = MCPServer(
            auth_checker=MockAuthChecker(authenticated=False),
            rate_limiter=MockRateLimiter(allowed=False),
            search_subsystem=MockSearchSubsystem(),
        )
        call = MCPToolCall(
            tool_name="search",
            arguments={"query": "test", "mode": "neural"},
            api_key="invalid-key",
        )
        result = await server.call_tool(call)
        # Should get auth error, not rate limit error
        assert result.error.code == MCPErrorCode.AUTHENTICATION_ERROR

    @pytest.mark.asyncio
    async def test_subsystem_not_invoked_on_auth_failure(self):
        """Subsystem is NOT invoked when auth fails."""
        search = MockSearchSubsystem()
        server = MCPServer(
            auth_checker=MockAuthChecker(authenticated=False),
            search_subsystem=search,
        )
        call = MCPToolCall(
            tool_name="search",
            arguments={"query": "test", "mode": "neural"},
            api_key="invalid-key",
        )
        await server.call_tool(call)
        assert search.last_call is None

    @pytest.mark.asyncio
    async def test_subsystem_not_invoked_on_rate_limit(self):
        """Subsystem is NOT invoked when rate limited."""
        search = MockSearchSubsystem()
        server = MCPServer(
            auth_checker=MockAuthChecker(),
            rate_limiter=MockRateLimiter(allowed=False),
            search_subsystem=search,
        )
        call = MCPToolCall(
            tool_name="search",
            arguments={"query": "test", "mode": "neural"},
            api_key="valid-key",
        )
        await server.call_tool(call)
        assert search.last_call is None


# ---------------------------------------------------------------------------
# Task 16.5: Output schema validation
# ---------------------------------------------------------------------------


class TestOutputValidation:
    """Tests for R12.7: output schema validation → tool-execution error."""

    @pytest.mark.asyncio
    async def test_valid_output_passes(self):
        """Valid output passes schema validation."""
        search = MockSearchSubsystem(result={"results": [], "warnings": []})
        server = MCPServer(
            auth_checker=MockAuthChecker(),
            search_subsystem=search,
        )
        call = MCPToolCall(
            tool_name="search",
            arguments={"query": "test", "mode": "neural"},
            api_key="valid-key",
        )
        result = await server.call_tool(call)
        assert result.success is True
        assert result.data == {"results": [], "warnings": []}

    @pytest.mark.asyncio
    async def test_invalid_output_returns_tool_execution_error(self):
        """Invalid output (fails schema) returns tool_execution_error."""
        # Return output that doesn't match the schema (missing 'results')
        search = MockSearchSubsystem(result={"invalid_field": "bad data"})
        server = MCPServer(
            auth_checker=MockAuthChecker(),
            search_subsystem=search,
        )
        call = MCPToolCall(
            tool_name="search",
            arguments={"query": "test", "mode": "neural"},
            api_key="valid-key",
        )
        result = await server.call_tool(call)
        assert result.success is False
        assert result.error.code == MCPErrorCode.TOOL_EXECUTION_ERROR
        assert "output" in result.error.message.lower() or "validation" in result.error.message.lower()

    @pytest.mark.asyncio
    async def test_partial_output_not_returned(self):
        """Partial/malformed output is never returned to the client."""
        # Return output with wrong type for results
        search = MockSearchSubsystem(result={"results": "not_an_array"})
        server = MCPServer(
            auth_checker=MockAuthChecker(),
            search_subsystem=search,
        )
        call = MCPToolCall(
            tool_name="search",
            arguments={"query": "test", "mode": "neural"},
            api_key="valid-key",
        )
        result = await server.call_tool(call)
        assert result.success is False
        assert result.data is None  # No partial data returned
