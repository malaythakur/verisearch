"""MCP tool JSON Schema definitions for input and output (Task 16.1, R12.1).

Each MCP tool declares:
- input_schema: JSON Schema for the tool's input arguments.
- output_schema: JSON Schema for the tool's response payload.

Tools: search, find_similar, contents, answer, research.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# search tool schemas
# ---------------------------------------------------------------------------

SEARCH_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "minLength": 1,
            "maxLength": 2048,
            "description": "Search query (1–2048 Unicode code points after trim).",
        },
        "mode": {
            "type": "string",
            "enum": ["neural", "keyword", "hybrid"],
            "description": "Retrieval mode.",
        },
        "num_results": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
            "default": 10,
            "description": "Maximum number of results (0–100).",
        },
        "filters": {
            "type": "string",
            "description": "Optional Query_Filter_DSL expression.",
        },
        "pipeline_id": {
            "type": "string",
            "description": "Optional pipeline ID to apply.",
        },
        "min_credibility": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Minimum credibility threshold.",
        },
        "max_ai_generated_likelihood": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Maximum AI-generation likelihood threshold.",
        },
    },
    "required": ["query", "mode"],
    "additionalProperties": False,
}

SEARCH_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "document_id": {"type": "string"},
                    "url": {"type": "string"},
                    "title": {"type": "string"},
                    "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "published_at": {"type": ["string", "null"]},
                    "provenance": {
                        "type": "object",
                        "properties": {
                            "credibility_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            "ai_generated_likelihood": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            "scored_at": {"type": "string"},
                        },
                        "required": ["credibility_score", "ai_generated_likelihood", "scored_at"],
                    },
                },
                "required": ["document_id", "url", "title", "score", "provenance"],
            },
        },
        "warnings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "step": {"type": "string"},
                },
            },
        },
    },
    "required": ["results"],
}

# ---------------------------------------------------------------------------
# find_similar tool schemas
# ---------------------------------------------------------------------------

FIND_SIMILAR_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "url": {
            "type": "string",
            "minLength": 1,
            "maxLength": 2048,
            "description": "URL to find similar documents for.",
        },
        "num_results": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
            "default": 10,
            "description": "Maximum number of results (0–100).",
        },
        "filters": {
            "type": "string",
            "description": "Optional Query_Filter_DSL expression.",
        },
    },
    "required": ["url"],
    "additionalProperties": False,
}

FIND_SIMILAR_OUTPUT_SCHEMA: dict[str, Any] = SEARCH_OUTPUT_SCHEMA  # Same shape

# ---------------------------------------------------------------------------
# contents tool schemas
# ---------------------------------------------------------------------------

CONTENTS_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "document_ids": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 100,
            "description": "List of 1–100 document IDs to fetch.",
        },
        "highlights": {
            "type": "boolean",
            "description": "Whether to include highlight spans (requires query).",
        },
        "query": {
            "type": "string",
            "description": "Query for highlight extraction.",
        },
        "summary": {
            "type": "boolean",
            "description": "Whether to include document summaries.",
        },
    },
    "required": ["document_ids"],
    "additionalProperties": False,
}

CONTENTS_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "document_id": {"type": "string"},
                    "version": {"type": "integer"},
                    "url": {"type": "string"},
                    "cleaned_text": {"type": "string"},
                    "highlights": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "start": {"type": "integer", "minimum": 0},
                                "end": {"type": "integer", "minimum": 1},
                            },
                            "required": ["start", "end"],
                        },
                        "maxItems": 5,
                    },
                    "summary": {"type": "string"},
                    "provenance": {
                        "type": "object",
                        "properties": {
                            "credibility_score": {"type": "number"},
                            "ai_generated_likelihood": {"type": "number"},
                            "scored_at": {"type": "string"},
                        },
                    },
                    "error": {
                        "type": "object",
                        "properties": {
                            "code": {"type": "string"},
                            "message": {"type": "string"},
                        },
                        "required": ["code", "message"],
                    },
                },
                "required": ["document_id"],
            },
        },
    },
    "required": ["results"],
}

# ---------------------------------------------------------------------------
# answer tool schemas
# ---------------------------------------------------------------------------

ANSWER_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "minLength": 1,
            "maxLength": 2048,
            "description": "The question to answer.",
        },
        "num_results": {
            "type": "integer",
            "minimum": 1,
            "maximum": 100,
            "default": 10,
            "description": "Number of documents to retrieve for context.",
        },
        "mode": {
            "type": "string",
            "enum": ["neural", "keyword", "hybrid"],
            "default": "hybrid",
            "description": "Retrieval mode for source documents.",
        },
        "session_id": {
            "type": "string",
            "description": "Optional session ID for context.",
        },
    },
    "required": ["query"],
    "additionalProperties": False,
}

ANSWER_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "document_id": {"type": "string"},
                    "version": {"type": "integer"},
                    "answer_start": {"type": "integer", "minimum": 0},
                    "answer_end": {"type": "integer", "minimum": 1},
                    "source_start": {"type": "integer", "minimum": 0},
                    "source_end": {"type": "integer", "minimum": 1},
                },
                "required": ["document_id", "version", "answer_start", "answer_end", "source_start", "source_end"],
            },
        },
    },
    "required": ["answer", "citations"],
}

# ---------------------------------------------------------------------------
# research tool schemas
# ---------------------------------------------------------------------------

RESEARCH_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "research_goal": {
            "type": "string",
            "minLength": 1,
            "maxLength": 4096,
            "description": "The research goal (1–4096 code points).",
        },
        "output_schema": {
            "type": "object",
            "description": "Optional JSON Schema for the final report.",
        },
        "session_id": {
            "type": "string",
            "description": "Optional session ID for context.",
        },
        "max_steps": {
            "type": "integer",
            "minimum": 1,
            "description": "Maximum number of research steps.",
        },
        "max_duration_ms": {
            "type": "integer",
            "minimum": 1,
            "description": "Maximum duration in milliseconds.",
        },
        "max_tool_calls": {
            "type": "integer",
            "minimum": 1,
            "description": "Maximum number of tool calls.",
        },
    },
    "required": ["research_goal"],
    "additionalProperties": False,
}

RESEARCH_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "job_id": {"type": "string"},
        "status": {
            "type": "string",
            "enum": ["queued", "planning", "running", "succeeded", "failed", "budget_exceeded"],
        },
    },
    "required": ["job_id"],
}

# ---------------------------------------------------------------------------
# Tool definitions registry
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS: dict[str, dict[str, Any]] = {
    "search": {
        "name": "search",
        "description": "Search the web index using neural, keyword, or hybrid retrieval.",
        "input_schema": SEARCH_INPUT_SCHEMA,
        "output_schema": SEARCH_OUTPUT_SCHEMA,
    },
    "find_similar": {
        "name": "find_similar",
        "description": "Find documents semantically similar to a given URL.",
        "input_schema": FIND_SIMILAR_INPUT_SCHEMA,
        "output_schema": FIND_SIMILAR_OUTPUT_SCHEMA,
    },
    "contents": {
        "name": "contents",
        "description": "Fetch cleaned page text, highlights, and summaries for document IDs.",
        "input_schema": CONTENTS_INPUT_SCHEMA,
        "output_schema": CONTENTS_OUTPUT_SCHEMA,
    },
    "answer": {
        "name": "answer",
        "description": "Generate a streaming answer with citations from retrieved documents.",
        "input_schema": ANSWER_INPUT_SCHEMA,
        "output_schema": ANSWER_OUTPUT_SCHEMA,
    },
    "research": {
        "name": "research",
        "description": "Launch a multi-hop research job that plans, searches, reads, and synthesizes.",
        "input_schema": RESEARCH_INPUT_SCHEMA,
        "output_schema": RESEARCH_OUTPUT_SCHEMA,
    },
}
