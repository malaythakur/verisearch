"""MCP Server implementation (Task 16, R12).

Implements:
- Tool definitions with JSON Schema for input/output (R12.1).
- Tool dispatch to backing subsystems (R12.2).
- Input schema validation → MCP-standard validation error (R12.3).
- Shared auth + rate limits with REST gateway (R12.4–R12.6).
- Output schema validation → MCP-standard tool-execution error (R12.7).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

import jsonschema
from jsonschema import ValidationError as JsonSchemaValidationError

from backend.mcp_server.schemas import TOOL_DEFINITIONS


# ---------------------------------------------------------------------------
# MCP standard error types
# ---------------------------------------------------------------------------


class MCPErrorCode(str, Enum):
    """MCP-standard error codes."""

    VALIDATION_ERROR = "validation_error"
    AUTHENTICATION_ERROR = "authentication_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    TOOL_EXECUTION_ERROR = "tool_execution_error"


@dataclass(frozen=True)
class MCPError:
    """MCP-standard error response."""

    code: MCPErrorCode
    message: str
    details: dict[str, Any] | None = None


@dataclass(frozen=True)
class MCPToolResult:
    """Result of an MCP tool call — either success or error."""

    success: bool
    data: dict[str, Any] | None = None
    error: MCPError | None = None


@dataclass(frozen=True)
class MCPToolCall:
    """An incoming MCP tool call request."""

    tool_name: str
    arguments: dict[str, Any]
    api_key: str | None = None


# ---------------------------------------------------------------------------
# Protocols for backing subsystems
# ---------------------------------------------------------------------------


class AuthChecker(Protocol):
    """Protocol for authentication checking."""

    async def authenticate(self, api_key: str) -> tuple[bool, str | None, str | None]:
        """Authenticate an API key.

        Returns:
            (is_authenticated, tenant_id, error_message)
        """
        ...


class RateLimitChecker(Protocol):
    """Protocol for rate limit checking."""

    async def check_rate_limit(self, tenant_id: str, endpoint: str) -> tuple[bool, int | None]:
        """Check if a request is within rate limits.

        Returns:
            (is_allowed, retry_after_seconds)
        """
        ...


class SearchSubsystem(Protocol):
    """Protocol for the search subsystem (Retriever)."""

    async def search(self, arguments: dict[str, Any], tenant_id: str) -> dict[str, Any]:
        """Execute a search query."""
        ...


class FindSimilarSubsystem(Protocol):
    """Protocol for the find_similar subsystem (Retriever)."""

    async def find_similar(self, arguments: dict[str, Any], tenant_id: str) -> dict[str, Any]:
        """Execute a find_similar query."""
        ...


class ContentsSubsystem(Protocol):
    """Protocol for the contents subsystem (Search_Engine)."""

    async def get_contents(self, arguments: dict[str, Any], tenant_id: str) -> dict[str, Any]:
        """Fetch document contents."""
        ...


class AnswerSubsystem(Protocol):
    """Protocol for the answer subsystem (Answer_Engine)."""

    async def generate_answer(self, arguments: dict[str, Any], tenant_id: str) -> dict[str, Any]:
        """Generate an answer."""
        ...


class ResearchSubsystem(Protocol):
    """Protocol for the research subsystem (Research_Agent)."""

    async def start_research(self, arguments: dict[str, Any], tenant_id: str) -> dict[str, Any]:
        """Start a research job."""
        ...


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------


class MCPServer:
    """Model Context Protocol server (R12).

    Exposes search, find_similar, contents, answer, and research as MCP tools.
    Validates input/output against JSON Schema, shares auth and rate limits
    with the REST gateway.
    """

    def __init__(
        self,
        *,
        auth_checker: AuthChecker | None = None,
        rate_limiter: RateLimitChecker | None = None,
        search_subsystem: SearchSubsystem | None = None,
        find_similar_subsystem: FindSimilarSubsystem | None = None,
        contents_subsystem: ContentsSubsystem | None = None,
        answer_subsystem: AnswerSubsystem | None = None,
        research_subsystem: ResearchSubsystem | None = None,
    ) -> None:
        """Initialize the MCP Server.

        Args:
            auth_checker: Authentication checker (shared with REST gateway).
            rate_limiter: Rate limit checker (shared with REST gateway).
            search_subsystem: Retriever for search tool.
            find_similar_subsystem: Retriever for find_similar tool.
            contents_subsystem: Search_Engine for contents tool.
            answer_subsystem: Answer_Engine for answer tool.
            research_subsystem: Research_Agent for research tool.
        """
        self._auth_checker = auth_checker
        self._rate_limiter = rate_limiter
        self._search = search_subsystem
        self._find_similar = find_similar_subsystem
        self._contents = contents_subsystem
        self._answer = answer_subsystem
        self._research = research_subsystem

        # Map tool names to their dispatch handlers
        self._dispatch_map: dict[str, Any] = {
            "search": self._dispatch_search,
            "find_similar": self._dispatch_find_similar,
            "contents": self._dispatch_contents,
            "answer": self._dispatch_answer,
            "research": self._dispatch_research,
        }

    def list_tools(self) -> list[dict[str, Any]]:
        """List all available MCP tools with their schemas (R12.1).

        Returns:
            List of tool definitions with name, description, and schemas.
        """
        return [
            {
                "name": defn["name"],
                "description": defn["description"],
                "inputSchema": defn["input_schema"],
            }
            for defn in TOOL_DEFINITIONS.values()
        ]

    async def call_tool(self, tool_call: MCPToolCall) -> MCPToolResult:
        """Execute an MCP tool call (R12.2–R12.7).

        Steps:
        1. Authenticate the request (R12.4, R12.5).
        2. Check rate limits (R12.6).
        3. Validate input against the tool's input schema (R12.3).
        4. Dispatch to the backing subsystem (R12.2).
        5. Validate output against the tool's output schema (R12.7).

        Args:
            tool_call: The incoming tool call request.

        Returns:
            MCPToolResult with either success data or an error.
        """
        # Step 0: Check tool exists
        tool_name = tool_call.tool_name
        if tool_name not in TOOL_DEFINITIONS:
            return MCPToolResult(
                success=False,
                error=MCPError(
                    code=MCPErrorCode.VALIDATION_ERROR,
                    message=f"Unknown tool: {tool_name}",
                ),
            )

        # Step 1: Authenticate (R12.4, R12.5)
        tenant_id = await self._authenticate(tool_call.api_key)
        if isinstance(tenant_id, MCPToolResult):
            return tenant_id  # Authentication error

        # Step 2: Check rate limits (R12.6)
        rate_limit_result = await self._check_rate_limit(tenant_id, tool_name)
        if rate_limit_result is not None:
            return rate_limit_result

        # Step 3: Validate input schema (R12.3)
        validation_error = self._validate_input(tool_name, tool_call.arguments)
        if validation_error is not None:
            return validation_error

        # Step 4: Dispatch to backing subsystem (R12.2)
        try:
            handler = self._dispatch_map.get(tool_name)
            if handler is None:
                return MCPToolResult(
                    success=False,
                    error=MCPError(
                        code=MCPErrorCode.TOOL_EXECUTION_ERROR,
                        message=f"No handler registered for tool: {tool_name}",
                    ),
                )

            result = await handler(tool_call.arguments, tenant_id)
        except Exception as e:
            return MCPToolResult(
                success=False,
                error=MCPError(
                    code=MCPErrorCode.TOOL_EXECUTION_ERROR,
                    message=f"Subsystem error: {str(e)}",
                    details={"tool": tool_name},
                ),
            )

        # Step 5: Validate output schema (R12.7)
        output_validation_error = self._validate_output(tool_name, result)
        if output_validation_error is not None:
            return output_validation_error

        return MCPToolResult(success=True, data=result)

    async def _authenticate(self, api_key: str | None) -> str | MCPToolResult:
        """Authenticate the API key (R12.4, R12.5).

        Returns:
            tenant_id on success, or MCPToolResult error on failure.
        """
        if self._auth_checker is None:
            # No auth configured — pass through (for testing)
            return "test-tenant"

        if api_key is None:
            return MCPToolResult(
                success=False,
                error=MCPError(
                    code=MCPErrorCode.AUTHENTICATION_ERROR,
                    message="API key is required",
                ),
            )

        is_authenticated, tenant_id, error_message = await self._auth_checker.authenticate(api_key)
        if not is_authenticated:
            return MCPToolResult(
                success=False,
                error=MCPError(
                    code=MCPErrorCode.AUTHENTICATION_ERROR,
                    message=error_message or "Authentication failed",
                ),
            )

        return tenant_id  # type: ignore[return-value]

    async def _check_rate_limit(self, tenant_id: str, tool_name: str) -> MCPToolResult | None:
        """Check rate limits (R12.6).

        Returns:
            MCPToolResult error if rate limited, None if allowed.
        """
        if self._rate_limiter is None:
            return None

        # Map tool names to REST endpoint equivalents for shared rate limits
        endpoint_map = {
            "search": "/v1/search",
            "find_similar": "/v1/find_similar",
            "contents": "/v1/contents",
            "answer": "/v1/answer",
            "research": "/v1/research",
        }
        endpoint = endpoint_map.get(tool_name, f"/v1/{tool_name}")

        is_allowed, retry_after = await self._rate_limiter.check_rate_limit(tenant_id, endpoint)
        if not is_allowed:
            return MCPToolResult(
                success=False,
                error=MCPError(
                    code=MCPErrorCode.RATE_LIMIT_ERROR,
                    message="Rate limit exceeded",
                    details={"retry_after": retry_after},
                ),
            )

        return None

    def _validate_input(self, tool_name: str, arguments: dict[str, Any]) -> MCPToolResult | None:
        """Validate input arguments against the tool's input schema (R12.3).

        Returns:
            MCPToolResult error if validation fails, None if valid.
        """
        tool_def = TOOL_DEFINITIONS[tool_name]
        input_schema = tool_def["input_schema"]

        try:
            jsonschema.validate(instance=arguments, schema=input_schema)
        except JsonSchemaValidationError as e:
            # Build the argument path from the validation error
            path = ".".join(str(p) for p in e.absolute_path) if e.absolute_path else ""
            return MCPToolResult(
                success=False,
                error=MCPError(
                    code=MCPErrorCode.VALIDATION_ERROR,
                    message=f"Input validation failed: {e.message}",
                    details={
                        "path": path,
                        "constraint": e.validator,
                        "schema_path": list(e.absolute_schema_path),
                    },
                ),
            )

        return None

    def _validate_output(self, tool_name: str, result: dict[str, Any]) -> MCPToolResult | None:
        """Validate output against the tool's output schema (R12.7).

        Returns:
            MCPToolResult error if validation fails, None if valid.
        """
        tool_def = TOOL_DEFINITIONS[tool_name]
        output_schema = tool_def["output_schema"]

        try:
            jsonschema.validate(instance=result, schema=output_schema)
        except JsonSchemaValidationError as e:
            return MCPToolResult(
                success=False,
                error=MCPError(
                    code=MCPErrorCode.TOOL_EXECUTION_ERROR,
                    message=f"Output validation failed: {e.message}",
                    details={
                        "tool": tool_name,
                        "path": ".".join(str(p) for p in e.absolute_path) if e.absolute_path else "",
                    },
                ),
            )

        return None

    # ---------------------------------------------------------------------------
    # Dispatch handlers
    # ---------------------------------------------------------------------------

    async def _dispatch_search(self, arguments: dict[str, Any], tenant_id: str) -> dict[str, Any]:
        """Dispatch to the search subsystem (Retriever)."""
        if self._search is None:
            raise RuntimeError("Search subsystem not configured")
        return await self._search.search(arguments, tenant_id)

    async def _dispatch_find_similar(self, arguments: dict[str, Any], tenant_id: str) -> dict[str, Any]:
        """Dispatch to the find_similar subsystem (Retriever)."""
        if self._find_similar is None:
            raise RuntimeError("Find-similar subsystem not configured")
        return await self._find_similar.find_similar(arguments, tenant_id)

    async def _dispatch_contents(self, arguments: dict[str, Any], tenant_id: str) -> dict[str, Any]:
        """Dispatch to the contents subsystem (Search_Engine)."""
        if self._contents is None:
            raise RuntimeError("Contents subsystem not configured")
        return await self._contents.get_contents(arguments, tenant_id)

    async def _dispatch_answer(self, arguments: dict[str, Any], tenant_id: str) -> dict[str, Any]:
        """Dispatch to the answer subsystem (Answer_Engine)."""
        if self._answer is None:
            raise RuntimeError("Answer subsystem not configured")
        return await self._answer.generate_answer(arguments, tenant_id)

    async def _dispatch_research(self, arguments: dict[str, Any], tenant_id: str) -> dict[str, Any]:
        """Dispatch to the research subsystem (Research_Agent)."""
        if self._research is None:
            raise RuntimeError("Research subsystem not configured")
        return await self._research.start_research(arguments, tenant_id)
