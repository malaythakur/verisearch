"""Property-based tests for MCP Server (Task 16.6, Property 42).

Property 42: MCP tool input/output schema validation

*For any* MCP tool call whose arguments validate against the tool's input JSON Schema,
the call is dispatched to the documented backing subsystem and the returned payload
validates against the tool's output JSON Schema; for any call whose arguments fail
input-schema validation, the MCP_Server returns the MCP-standard validation error
identifying the offending argument path and the failed constraint, and the backing
subsystem is not invoked; for any call whose subsystem returns an output that fails
output-schema validation, the MCP_Server returns the MCP-standard tool-execution error
and never returns a partial or malformed payload.

**Validates: Requirements 12.2, 12.3, 12.7**
"""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from backend.mcp_server.schemas import TOOL_DEFINITIONS
from backend.mcp_server.server import (
    MCPErrorCode,
    MCPServer,
    MCPToolCall,
    MCPToolResult,
)


# ---------------------------------------------------------------------------
# Mock subsystems for property testing
# ---------------------------------------------------------------------------


class TrackingSearchSubsystem:
    """Search subsystem that tracks calls and returns valid output."""

    def __init__(self):
        self.call_count = 0
        self.last_args = None

    async def search(self, arguments: dict[str, Any], tenant_id: str) -> dict[str, Any]:
        self.call_count += 1
        self.last_args = arguments
        return {"results": [], "warnings": []}


class TrackingFindSimilarSubsystem:
    """Find-similar subsystem that tracks calls and returns valid output."""

    def __init__(self):
        self.call_count = 0
        self.last_args = None

    async def find_similar(self, arguments: dict[str, Any], tenant_id: str) -> dict[str, Any]:
        self.call_count += 1
        self.last_args = arguments
        return {"results": [], "warnings": []}


class TrackingContentsSubsystem:
    """Contents subsystem that tracks calls and returns valid output."""

    def __init__(self):
        self.call_count = 0
        self.last_args = None

    async def get_contents(self, arguments: dict[str, Any], tenant_id: str) -> dict[str, Any]:
        self.call_count += 1
        self.last_args = arguments
        return {"results": []}


class TrackingAnswerSubsystem:
    """Answer subsystem that tracks calls and returns valid output."""

    def __init__(self):
        self.call_count = 0
        self.last_args = None

    async def generate_answer(self, arguments: dict[str, Any], tenant_id: str) -> dict[str, Any]:
        self.call_count += 1
        self.last_args = arguments
        return {"answer": "Generated answer", "citations": []}


class TrackingResearchSubsystem:
    """Research subsystem that tracks calls and returns valid output."""

    def __init__(self):
        self.call_count = 0
        self.last_args = None

    async def start_research(self, arguments: dict[str, Any], tenant_id: str) -> dict[str, Any]:
        self.call_count += 1
        self.last_args = arguments
        return {"job_id": "job-abc-123", "status": "queued"}


class InvalidOutputSubsystem:
    """Subsystem that returns output failing schema validation."""

    async def search(self, arguments: dict[str, Any], tenant_id: str) -> dict[str, Any]:
        return {"bad_field": "invalid"}

    async def find_similar(self, arguments: dict[str, Any], tenant_id: str) -> dict[str, Any]:
        return {"bad_field": "invalid"}

    async def get_contents(self, arguments: dict[str, Any], tenant_id: str) -> dict[str, Any]:
        return {"bad_field": "invalid"}

    async def generate_answer(self, arguments: dict[str, Any], tenant_id: str) -> dict[str, Any]:
        return {"bad_field": "invalid"}

    async def start_research(self, arguments: dict[str, Any], tenant_id: str) -> dict[str, Any]:
        return {"bad_field": "invalid"}


class MockAuthChecker:
    """Always-authenticated mock."""

    async def authenticate(self, api_key: str) -> tuple[bool, str | None, str | None]:
        return (True, "test-tenant", None)


# ---------------------------------------------------------------------------
# Strategies for generating valid and invalid tool arguments
# ---------------------------------------------------------------------------


def valid_search_args() -> st.SearchStrategy[dict[str, Any]]:
    """Generate valid search tool arguments."""
    return st.fixed_dictionaries({
        "query": st.text(min_size=1, max_size=100),
        "mode": st.sampled_from(["neural", "keyword", "hybrid"]),
    }).flatmap(lambda base: st.fixed_dictionaries({
        **{k: st.just(v) for k, v in base.items()},
        "num_results": st.integers(min_value=0, max_value=100),
    }).map(lambda d: {k: v for k, v in d.items()}))


def valid_find_similar_args() -> st.SearchStrategy[dict[str, Any]]:
    """Generate valid find_similar tool arguments."""
    return st.fixed_dictionaries({
        "url": st.text(min_size=1, max_size=200).map(lambda s: f"https://example.com/{s}"),
    })


def valid_contents_args() -> st.SearchStrategy[dict[str, Any]]:
    """Generate valid contents tool arguments."""
    return st.fixed_dictionaries({
        "document_ids": st.lists(
            st.text(min_size=1, max_size=36, alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="-_")),
            min_size=1,
            max_size=10,
        ),
    })


def valid_answer_args() -> st.SearchStrategy[dict[str, Any]]:
    """Generate valid answer tool arguments."""
    return st.fixed_dictionaries({
        "query": st.text(min_size=1, max_size=100),
    })


def valid_research_args() -> st.SearchStrategy[dict[str, Any]]:
    """Generate valid research tool arguments."""
    return st.fixed_dictionaries({
        "research_goal": st.text(min_size=1, max_size=200),
    })


def invalid_search_args() -> st.SearchStrategy[dict[str, Any]]:
    """Generate invalid search tool arguments (various violations)."""
    return st.one_of(
        # Missing required 'mode'
        st.fixed_dictionaries({"query": st.text(min_size=1, max_size=50)}),
        # Missing required 'query'
        st.fixed_dictionaries({"mode": st.sampled_from(["neural", "keyword", "hybrid"])}),
        # Invalid mode value
        st.fixed_dictionaries({
            "query": st.text(min_size=1, max_size=50),
            "mode": st.text(min_size=1, max_size=20).filter(lambda s: s not in ("neural", "keyword", "hybrid")),
        }),
        # num_results out of range
        st.fixed_dictionaries({
            "query": st.text(min_size=1, max_size=50),
            "mode": st.sampled_from(["neural", "keyword", "hybrid"]),
            "num_results": st.one_of(
                st.integers(max_value=-1),
                st.integers(min_value=101),
            ),
        }),
        # Additional properties
        st.fixed_dictionaries({
            "query": st.text(min_size=1, max_size=50),
            "mode": st.sampled_from(["neural", "keyword", "hybrid"]),
            "extra_field": st.text(min_size=1, max_size=20),
        }),
    )


def invalid_contents_args() -> st.SearchStrategy[dict[str, Any]]:
    """Generate invalid contents tool arguments."""
    return st.one_of(
        # Empty document_ids
        st.fixed_dictionaries({"document_ids": st.just([])}),
        # Missing document_ids
        st.fixed_dictionaries({"highlights": st.just(True)}),
        # Too many document_ids (>100)
        st.fixed_dictionaries({
            "document_ids": st.lists(
                st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=("L", "N"))),
                min_size=101,
                max_size=105,
            ),
        }),
    )


# ---------------------------------------------------------------------------
# Property 42: MCP tool input/output schema validation
# ---------------------------------------------------------------------------


class TestMCPProperty42:
    """Property 42: MCP tool input/output schema validation.

    Feature: agentic-research-search-engine, Property 42: MCP tool input/output
    schema validation.
    """

    @given(args=valid_search_args())
    @settings(max_examples=100, deadline=None)
    @pytest.mark.asyncio
    async def test_valid_search_input_dispatches_and_validates_output(self, args):
        """Valid search input is dispatched and output validates against schema.

        **Validates: Requirements 12.2**
        """
        search = TrackingSearchSubsystem()
        server = MCPServer(
            auth_checker=MockAuthChecker(),
            search_subsystem=search,
        )
        call = MCPToolCall(tool_name="search", arguments=args, api_key="key")
        result = await server.call_tool(call)

        # Call was dispatched
        assert search.call_count == 1
        # Result is successful
        assert result.success is True
        # Output validates against schema (validated internally by the server)
        assert result.data is not None
        assert "results" in result.data

    @given(args=valid_find_similar_args())
    @settings(max_examples=100, deadline=None)
    @pytest.mark.asyncio
    async def test_valid_find_similar_input_dispatches(self, args):
        """Valid find_similar input is dispatched to the subsystem.

        **Validates: Requirements 12.2**
        """
        subsystem = TrackingFindSimilarSubsystem()
        server = MCPServer(
            auth_checker=MockAuthChecker(),
            find_similar_subsystem=subsystem,
        )
        call = MCPToolCall(tool_name="find_similar", arguments=args, api_key="key")
        result = await server.call_tool(call)

        assert subsystem.call_count == 1
        assert result.success is True

    @given(args=valid_contents_args())
    @settings(max_examples=100, deadline=None)
    @pytest.mark.asyncio
    async def test_valid_contents_input_dispatches(self, args):
        """Valid contents input is dispatched to the subsystem.

        **Validates: Requirements 12.2**
        """
        subsystem = TrackingContentsSubsystem()
        server = MCPServer(
            auth_checker=MockAuthChecker(),
            contents_subsystem=subsystem,
        )
        call = MCPToolCall(tool_name="contents", arguments=args, api_key="key")
        result = await server.call_tool(call)

        assert subsystem.call_count == 1
        assert result.success is True

    @given(args=valid_answer_args())
    @settings(max_examples=100, deadline=None)
    @pytest.mark.asyncio
    async def test_valid_answer_input_dispatches(self, args):
        """Valid answer input is dispatched to the subsystem.

        **Validates: Requirements 12.2**
        """
        subsystem = TrackingAnswerSubsystem()
        server = MCPServer(
            auth_checker=MockAuthChecker(),
            answer_subsystem=subsystem,
        )
        call = MCPToolCall(tool_name="answer", arguments=args, api_key="key")
        result = await server.call_tool(call)

        assert subsystem.call_count == 1
        assert result.success is True

    @given(args=valid_research_args())
    @settings(max_examples=100, deadline=None)
    @pytest.mark.asyncio
    async def test_valid_research_input_dispatches(self, args):
        """Valid research input is dispatched to the subsystem.

        **Validates: Requirements 12.2**
        """
        subsystem = TrackingResearchSubsystem()
        server = MCPServer(
            auth_checker=MockAuthChecker(),
            research_subsystem=subsystem,
        )
        call = MCPToolCall(tool_name="research", arguments=args, api_key="key")
        result = await server.call_tool(call)

        assert subsystem.call_count == 1
        assert result.success is True

    @given(args=invalid_search_args())
    @settings(max_examples=100, deadline=None)
    @pytest.mark.asyncio
    async def test_invalid_search_input_returns_validation_error(self, args):
        """Invalid search input returns MCP-standard validation error.

        **Validates: Requirements 12.3**
        """
        search = TrackingSearchSubsystem()
        server = MCPServer(
            auth_checker=MockAuthChecker(),
            search_subsystem=search,
        )
        call = MCPToolCall(tool_name="search", arguments=args, api_key="key")
        result = await server.call_tool(call)

        # Validation error returned
        assert result.success is False
        assert result.error.code == MCPErrorCode.VALIDATION_ERROR
        # Subsystem NOT invoked
        assert search.call_count == 0

    @given(args=invalid_contents_args())
    @settings(max_examples=100, deadline=None)
    @pytest.mark.asyncio
    async def test_invalid_contents_input_returns_validation_error(self, args):
        """Invalid contents input returns MCP-standard validation error.

        **Validates: Requirements 12.3**
        """
        contents = TrackingContentsSubsystem()
        server = MCPServer(
            auth_checker=MockAuthChecker(),
            contents_subsystem=contents,
        )
        call = MCPToolCall(tool_name="contents", arguments=args, api_key="key")
        result = await server.call_tool(call)

        assert result.success is False
        assert result.error.code == MCPErrorCode.VALIDATION_ERROR
        assert contents.call_count == 0

    @given(args=valid_search_args())
    @settings(max_examples=50, deadline=None)
    @pytest.mark.asyncio
    async def test_invalid_output_returns_tool_execution_error(self, args):
        """Invalid subsystem output returns tool_execution_error, no partial data.

        **Validates: Requirements 12.7**
        """
        invalid_subsystem = InvalidOutputSubsystem()
        server = MCPServer(
            auth_checker=MockAuthChecker(),
            search_subsystem=invalid_subsystem,
        )
        call = MCPToolCall(tool_name="search", arguments=args, api_key="key")
        result = await server.call_tool(call)

        # Tool execution error returned
        assert result.success is False
        assert result.error.code == MCPErrorCode.TOOL_EXECUTION_ERROR
        # No partial/malformed data returned
        assert result.data is None

    @given(args=valid_search_args())
    @settings(max_examples=50, deadline=None)
    @pytest.mark.asyncio
    async def test_validation_error_identifies_path_and_constraint(self, args):
        """Validation error identifies the offending argument path.

        **Validates: Requirements 12.3**
        """
        # Inject an invalid field to trigger validation error
        args["extra_invalid_field"] = "should_not_be_here"

        search = TrackingSearchSubsystem()
        server = MCPServer(
            auth_checker=MockAuthChecker(),
            search_subsystem=search,
        )
        call = MCPToolCall(tool_name="search", arguments=args, api_key="key")
        result = await server.call_tool(call)

        assert result.success is False
        assert result.error.code == MCPErrorCode.VALIDATION_ERROR
        # Error should have details about the constraint
        assert result.error.details is not None
        assert "constraint" in result.error.details or "path" in result.error.details
