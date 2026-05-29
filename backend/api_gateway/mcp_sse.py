"""MCP Server SSE endpoint for AI agent integration.

Implements the Model Context Protocol over Server-Sent Events (SSE transport).
Exposes tools: search, find_similar, contents, answer, research.

Compatible with Kiro, Claude Desktop, and other MCP clients.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from starlette.responses import StreamingResponse

mcp_router = APIRouter()

# Tool definitions for MCP
MCP_TOOLS = [
    {
        "name": "search",
        "description": "Search documents using neural, keyword, or hybrid retrieval. Returns ranked results with provenance scores.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "mode": {"type": "string", "enum": ["neural", "keyword", "hybrid"], "default": "hybrid"},
                "num_results": {"type": "integer", "default": 10, "description": "Max results (0-100)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "find_similar",
        "description": "Find documents similar to a given URL.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to find similar documents for"},
                "num_results": {"type": "integer", "default": 10},
            },
            "required": ["url"],
        },
    },
    {
        "name": "contents",
        "description": "Fetch cleaned text content for documents by ID.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "document_ids": {"type": "array", "items": {"type": "string"}, "description": "Document IDs to fetch"},
            },
            "required": ["document_ids"],
        },
    },
    {
        "name": "answer",
        "description": "Get an AI-generated answer with citations from indexed documents.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Question to answer"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "research",
        "description": "Launch a multi-hop research job that searches, reads, and synthesizes information.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "research_goal": {"type": "string", "description": "Research goal (1-4096 chars)"},
            },
            "required": ["research_goal"],
        },
    },
]


async def _execute_tool(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Execute an MCP tool by calling the corresponding API logic."""
    from backend.api_gateway.routes import _get_services, _get_tenant_id

    services = _get_services()

    if tool_name == "search":
        from backend.retriever.models import SearchRequest, SearchMode

        retriever = services["retriever"]
        request = SearchRequest(
            query=arguments.get("query", ""),
            mode=SearchMode(arguments.get("mode", "hybrid")),
            num_results=arguments.get("num_results", 10),
            tenant_id="mcp-agent",
        )
        response = retriever.search(request)
        return {
            "results": [
                {"document_id": r.document_id, "url": r.url, "title": r.title, "score": r.score}
                for r in response.results
            ]
        }

    elif tool_name == "find_similar":
        from backend.retriever.models import FindSimilarRequest
        from backend.retriever.service import DocumentNotFoundError

        retriever = services["retriever"]
        try:
            request = FindSimilarRequest(url=arguments.get("url", ""), num_results=arguments.get("num_results", 10), tenant_id="mcp-agent")
            response = retriever.find_similar(request)
            return {"results": [{"document_id": r.document_id, "url": r.url, "score": r.score} for r in response.results]}
        except DocumentNotFoundError:
            return {"error": "URL not found in index"}

    elif tool_name == "contents":
        contents_service = services["contents_service"]
        response = contents_service.fetch_contents(arguments.get("document_ids", []))
        return {"results": contents_service.to_response_dict(response)}

    elif tool_name == "answer":
        from backend.retriever.models import SearchRequest, SearchMode
        from backend.answer_engine.models import RetrievalResult, TokenEvent, CitationEvent, DoneEvent, ErrorEvent

        retriever = services["retriever"]
        answer_engine = services["answer_engine"]
        indexer = services["indexer"]

        search_request = SearchRequest(query=arguments.get("query", ""), mode=SearchMode.HYBRID, num_results=5, tenant_id="mcp-agent")
        search_response = retriever.search(search_request)

        retrieval_results = []
        for r in search_response.results:
            doc = indexer.get_latest_version(r.document_id)
            if doc:
                retrieval_results.append(RetrievalResult(document_id=r.document_id, version=r.version, url=r.url, title=r.title, score=r.score, cleaned_text=doc.cleaned_text))

        full_answer = ""
        citations = []
        async for event in answer_engine.generate_answer(arguments.get("query", ""), retrieval_results):
            if isinstance(event, TokenEvent):
                full_answer += event.text
            elif isinstance(event, DoneEvent):
                full_answer = event.answer
                citations = [{"document_id": c.document_id, "version": c.version} for c in event.citations]
            elif isinstance(event, ErrorEvent):
                return {"error": event.message}

        return {"answer": full_answer, "citations": citations}

    elif tool_name == "research":
        from backend.research_agent.models import BudgetConfig

        research_agent = services["research_agent"]
        job_id = research_agent.start_job(tenant_id="mcp-agent", research_goal=arguments.get("research_goal", ""))
        report = research_agent.get_report(job_id, "mcp-agent")
        return {"job_id": job_id, "text": report.text, "citations": [{"document_id": c.document_id} for c in report.citations]}

    return {"error": f"Unknown tool: {tool_name}"}


# SSE endpoint for MCP
@mcp_router.get("/sse")
async def mcp_sse(request: Request):
    """MCP SSE endpoint — establishes connection and sends endpoint info."""
    session_id = str(uuid.uuid4())

    async def event_stream():
        # Send the endpoint event telling the client where to POST messages
        # Use full URL for compatibility with all MCP clients
        host = request.headers.get("host", "verisearch-production.up.railway.app")
        scheme = "https" if "railway" in host or "https" in str(request.url) else "http"
        endpoint_url = f"{scheme}://{host}/mcp/messages?session_id={session_id}"
        event_data = json.dumps(endpoint_url)
        yield f"event: endpoint\ndata: {event_data}\n\n"

        # Keep connection alive
        import asyncio
        try:
            while True:
                await asyncio.sleep(15)
                yield ": keepalive\n\n"
        except asyncio.CancelledError:
            pass

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


# Message endpoint for MCP tool calls
@mcp_router.post("/messages")
async def mcp_messages(request: Request):
    """Handle MCP JSON-RPC messages."""
    body = await request.json()
    method = body.get("method", "")
    msg_id = body.get("id")
    params = body.get("params", {})

    if method == "initialize":
        return JSONResponse(content={
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "verisearch", "version": "1.0.0"},
            },
        })

    elif method == "notifications/initialized":
        return JSONResponse(content={"jsonrpc": "2.0", "id": msg_id, "result": {}})

    elif method == "tools/list":
        return JSONResponse(content={
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {"tools": MCP_TOOLS},
        })

    elif method == "tools/call":
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        try:
            result = await _execute_tool(tool_name, arguments)
            return JSONResponse(content={
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(result, indent=2)}],
                },
            })
        except Exception as e:
            return JSONResponse(content={
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps({"error": str(e)})}],
                    "isError": True,
                },
            })

    return JSONResponse(content={
        "jsonrpc": "2.0",
        "id": msg_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    })
