"""OpenAPI 3.1 specification for the Agentic Research Search Engine API.

Covers all endpoints, event shapes, and error codes per R16.4.
The spec is served at GET /v1/openapi.json.

Task 17.1: Author OpenAPI 3.1 spec covering all endpoints, event shapes, error codes.
"""

from __future__ import annotations

OPENAPI_SPEC: dict = {
    "openapi": "3.1.0",
    "info": {
        "title": "Agentic Research Search Engine",
        "version": "0.1.0",
        "description": "API-first multi-tenant SaaS for agentic deep research with neural, keyword, and hybrid search.",
    },
    "servers": [{"url": "/v1", "description": "API v1"}],
    "security": [{"BearerAuth": []}],
    "components": {
        "securitySchemes": {
            "BearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "description": "Tenant-scoped API key (R13)",
            }
        },
        "schemas": {
            "ErrorDetail": {
                "type": "object",
                "required": ["code", "message"],
                "properties": {
                    "code": {"type": "string", "description": "Stable error code"},
                    "message": {"type": "string", "description": "Human-readable description"},
                },
            },
            "ErrorResponse": {
                "type": "object",
                "required": ["error"],
                "properties": {"error": {"$ref": "#/components/schemas/ErrorDetail"}},
            },
            "ProvenanceInfo": {
                "type": "object",
                "required": ["credibility_score", "ai_generated_likelihood", "scored_at"],
                "properties": {
                    "credibility_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "ai_generated_likelihood": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "scored_at": {"type": "string", "format": "date-time"},
                },
            },
            "SearchResult": {
                "type": "object",
                "required": ["document_id", "url", "title", "score", "provenance"],
                "properties": {
                    "document_id": {"type": "string", "format": "uuid"},
                    "url": {"type": "string", "format": "uri"},
                    "title": {"type": "string"},
                    "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "published_at": {"type": ["string", "null"], "format": "date-time"},
                    "provenance": {"$ref": "#/components/schemas/ProvenanceInfo"},
                },
            },
            "SearchRequest": {
                "type": "object",
                "required": ["query", "mode"],
                "properties": {
                    "query": {"type": "string", "minLength": 1, "maxLength": 2048},
                    "mode": {"type": "string", "enum": ["neural", "keyword", "hybrid"]},
                    "num_results": {"type": "integer", "minimum": 0, "maximum": 100, "default": 10},
                    "filters": {"type": "string", "description": "Query_Filter_DSL string"},
                    "pipeline_id": {"type": "string", "format": "uuid"},
                    "min_credibility": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "max_ai_generated_likelihood": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                },
            },
            "SearchResponse": {
                "type": "object",
                "required": ["results"],
                "properties": {
                    "results": {"type": "array", "items": {"$ref": "#/components/schemas/SearchResult"}},
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
            },
            "FindSimilarRequest": {
                "type": "object",
                "required": ["url"],
                "properties": {
                    "url": {"type": "string", "format": "uri", "maxLength": 2048},
                    "num_results": {"type": "integer", "minimum": 0, "maximum": 100, "default": 10},
                    "filters": {"type": "string"},
                    "min_credibility": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "max_ai_generated_likelihood": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                },
            },
            "ContentsRequest": {
                "type": "object",
                "required": ["document_ids"],
                "properties": {
                    "document_ids": {
                        "type": "array",
                        "items": {"type": "string", "format": "uuid"},
                        "minItems": 1,
                        "maxItems": 100,
                    },
                    "highlights": {"type": "boolean", "default": False},
                    "query": {"type": "string"},
                    "summary": {"type": "boolean", "default": False},
                },
            },
            "ContentEntry": {
                "type": "object",
                "properties": {
                    "document_id": {"type": "string", "format": "uuid"},
                    "version": {"type": "integer"},
                    "cleaned_text": {"type": "string"},
                    "highlights": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "start": {"type": "integer", "minimum": 0},
                                "end": {"type": "integer", "minimum": 0},
                            },
                        },
                        "maxItems": 5,
                    },
                    "summary": {"type": "string"},
                    "error": {"$ref": "#/components/schemas/ErrorDetail"},
                },
            },
            "ContentsResponse": {
                "type": "object",
                "required": ["results"],
                "properties": {
                    "results": {"type": "array", "items": {"$ref": "#/components/schemas/ContentEntry"}},
                },
            },
            "AnswerRequest": {
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {"type": "string", "minLength": 1, "maxLength": 2048},
                    "mode": {"type": "string", "enum": ["neural", "keyword", "hybrid"], "default": "hybrid"},
                    "num_results": {"type": "integer", "minimum": 1, "maximum": 100, "default": 10},
                    "stream": {"type": "boolean", "default": True},
                    "session_id": {"type": "string", "format": "uuid"},
                },
            },
            "Citation": {
                "type": "object",
                "required": ["document_id", "version", "answer_start", "answer_end", "source_start", "source_end"],
                "properties": {
                    "document_id": {"type": "string", "format": "uuid"},
                    "version": {"type": "integer"},
                    "answer_start": {"type": "integer", "minimum": 0},
                    "answer_end": {"type": "integer", "minimum": 0},
                    "source_start": {"type": "integer", "minimum": 0},
                    "source_end": {"type": "integer", "minimum": 0},
                },
            },
            "TokenEvent": {
                "type": "object",
                "required": ["text", "index"],
                "properties": {
                    "text": {"type": "string"},
                    "index": {"type": "integer"},
                },
            },
            "DoneEvent": {
                "type": "object",
                "required": ["answer", "citations"],
                "properties": {
                    "answer": {"type": "string"},
                    "citations": {"type": "array", "items": {"$ref": "#/components/schemas/Citation"}},
                },
            },
            "StreamErrorEvent": {
                "type": "object",
                "required": ["code", "message"],
                "properties": {
                    "code": {
                        "type": "string",
                        "enum": [
                            "no_sources_available",
                            "stream_timeout",
                            "model_error",
                            "internal_error",
                            "client_cancelled",
                        ],
                    },
                    "message": {"type": "string"},
                },
            },
            "ResearchRequest": {
                "type": "object",
                "required": ["research_goal"],
                "properties": {
                    "research_goal": {"type": "string", "minLength": 1, "maxLength": 4096},
                    "output_schema": {"type": "object", "description": "JSON Schema for structured output"},
                    "session_id": {"type": "string", "format": "uuid"},
                    "max_steps": {"type": "integer", "minimum": 1},
                    "max_duration_ms": {"type": "integer", "minimum": 1000},
                    "max_tool_calls": {"type": "integer", "minimum": 1},
                },
            },
            "ResearchJob": {
                "type": "object",
                "required": ["job_id", "state", "created_at"],
                "properties": {
                    "job_id": {"type": "string", "format": "uuid"},
                    "state": {
                        "type": "string",
                        "enum": ["queued", "planning", "running", "succeeded", "failed", "budget_exceeded"],
                    },
                    "created_at": {"type": "string", "format": "date-time"},
                    "report": {"type": "object"},
                    "citations": {"type": "array", "items": {"$ref": "#/components/schemas/Citation"}},
                },
            },
            "PlanStep": {
                "type": "object",
                "required": ["step_id", "type"],
                "properties": {
                    "step_id": {"type": "string", "format": "uuid"},
                    "type": {"type": "string", "enum": ["sub_query", "retrieval", "read", "synthesis"]},
                },
            },
            "ResearchEvent": {
                "type": "object",
                "required": ["event_id", "type", "data"],
                "properties": {
                    "event_id": {"type": "integer"},
                    "type": {
                        "type": "string",
                        "enum": [
                            "plan_updated",
                            "step_started",
                            "step_completed",
                            "citation",
                            "report_chunk",
                            "done",
                            "error",
                        ],
                    },
                    "data": {"type": "object"},
                },
            },
            "SessionCreateRequest": {
                "type": "object",
                "properties": {
                    "retention_days": {"type": "integer", "minimum": 1, "maximum": 90, "default": 14},
                },
            },
            "Session": {
                "type": "object",
                "required": ["session_id", "created_at", "retention_days"],
                "properties": {
                    "session_id": {"type": "string", "format": "uuid"},
                    "created_at": {"type": "string", "format": "date-time"},
                    "retention_days": {"type": "integer"},
                    "expires_at": {"type": "string", "format": "date-time"},
                },
            },
            "PipelineStepDef": {
                "type": "object",
                "required": ["type", "registry_name"],
                "properties": {
                    "type": {"type": "string", "enum": ["filter", "reranker", "transform"]},
                    "registry_name": {"type": "string"},
                    "config": {"type": "object"},
                    "timeout_ms": {"type": "integer", "minimum": 100, "maximum": 30000, "default": 2000},
                },
            },
            "PipelineCreateRequest": {
                "type": "object",
                "required": ["name", "steps"],
                "properties": {
                    "name": {"type": "string"},
                    "steps": {
                        "type": "array",
                        "items": {"$ref": "#/components/schemas/PipelineStepDef"},
                        "minItems": 1,
                        "maxItems": 20,
                    },
                },
            },
            "Pipeline": {
                "type": "object",
                "required": ["pipeline_id", "name", "steps", "created_at"],
                "properties": {
                    "pipeline_id": {"type": "string", "format": "uuid"},
                    "name": {"type": "string"},
                    "steps": {"type": "array", "items": {"$ref": "#/components/schemas/PipelineStepDef"}},
                    "created_at": {"type": "string", "format": "date-time"},
                },
            },
        },
        "responses": {
            "BadRequest": {
                "description": "Validation error",
                "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}},
            },
            "Unauthorized": {
                "description": "Authentication failed",
                "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}},
            },
            "NotFound": {
                "description": "Resource not found",
                "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}},
            },
            "RateLimited": {
                "description": "Rate limit exceeded",
                "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}},
            },
        },
    },
    "paths": {
        "/search": {
            "post": {
                "operationId": "search",
                "summary": "Neural, keyword, or hybrid search (R3)",
                "tags": ["Search"],
                "requestBody": {
                    "required": True,
                    "content": {"application/json": {"schema": {"$ref": "#/components/schemas/SearchRequest"}}},
                },
                "responses": {
                    "200": {
                        "description": "Search results",
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/SearchResponse"}}},
                        "headers": {
                            "X-Index-Version": {"schema": {"type": "integer"}, "description": "Monotonic index version"},
                            "X-Request-Id": {"schema": {"type": "string"}},
                            "X-RateLimit-Limit": {"schema": {"type": "integer"}},
                            "X-RateLimit-Remaining": {"schema": {"type": "integer"}},
                            "X-RateLimit-Reset": {"schema": {"type": "integer"}},
                        },
                    },
                    "400": {"$ref": "#/components/responses/BadRequest"},
                    "401": {"$ref": "#/components/responses/Unauthorized"},
                    "429": {"$ref": "#/components/responses/RateLimited"},
                },
            }
        },
        "/find_similar": {
            "post": {
                "operationId": "findSimilar",
                "summary": "Find semantically similar documents (R4)",
                "tags": ["Search"],
                "requestBody": {
                    "required": True,
                    "content": {"application/json": {"schema": {"$ref": "#/components/schemas/FindSimilarRequest"}}},
                },
                "responses": {
                    "200": {
                        "description": "Similar results",
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/SearchResponse"}}},
                    },
                    "400": {"$ref": "#/components/responses/BadRequest"},
                    "401": {"$ref": "#/components/responses/Unauthorized"},
                    "404": {"$ref": "#/components/responses/NotFound"},
                    "429": {"$ref": "#/components/responses/RateLimited"},
                },
            }
        },
        "/contents": {
            "post": {
                "operationId": "getContents",
                "summary": "Retrieve cleaned text, highlights, summaries (R5)",
                "tags": ["Content"],
                "requestBody": {
                    "required": True,
                    "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ContentsRequest"}}},
                },
                "responses": {
                    "200": {
                        "description": "Content entries",
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ContentsResponse"}}},
                    },
                    "400": {"$ref": "#/components/responses/BadRequest"},
                    "401": {"$ref": "#/components/responses/Unauthorized"},
                    "429": {"$ref": "#/components/responses/RateLimited"},
                },
            }
        },
        "/answer": {
            "post": {
                "operationId": "answer",
                "summary": "Streaming answer with citations (R6)",
                "tags": ["Answer"],
                "requestBody": {
                    "required": True,
                    "content": {"application/json": {"schema": {"$ref": "#/components/schemas/AnswerRequest"}}},
                },
                "responses": {
                    "200": {
                        "description": "SSE stream of token/citation/done/error events",
                        "content": {
                            "text/event-stream": {
                                "schema": {
                                    "type": "string",
                                    "description": "SSE stream with event types: token, citation, done, error",
                                }
                            }
                        },
                    },
                    "400": {"$ref": "#/components/responses/BadRequest"},
                    "401": {"$ref": "#/components/responses/Unauthorized"},
                    "429": {"$ref": "#/components/responses/RateLimited"},
                },
            }
        },
        "/research": {
            "post": {
                "operationId": "createResearch",
                "summary": "Launch a multi-hop research job (R7)",
                "tags": ["Research"],
                "requestBody": {
                    "required": True,
                    "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ResearchRequest"}}},
                },
                "responses": {
                    "201": {
                        "description": "Job created",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": ["job_id"],
                                    "properties": {"job_id": {"type": "string", "format": "uuid"}},
                                }
                            }
                        },
                    },
                    "400": {"$ref": "#/components/responses/BadRequest"},
                    "401": {"$ref": "#/components/responses/Unauthorized"},
                    "429": {"$ref": "#/components/responses/RateLimited"},
                },
            }
        },
        "/research/{job_id}": {
            "get": {
                "operationId": "getResearchJob",
                "summary": "Get research job report (R7.4)",
                "tags": ["Research"],
                "parameters": [
                    {"name": "job_id", "in": "path", "required": True, "schema": {"type": "string", "format": "uuid"}}
                ],
                "responses": {
                    "200": {
                        "description": "Research job with report",
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ResearchJob"}}},
                    },
                    "401": {"$ref": "#/components/responses/Unauthorized"},
                    "404": {"$ref": "#/components/responses/NotFound"},
                    "429": {"$ref": "#/components/responses/RateLimited"},
                },
            }
        },
        "/research/{job_id}/events": {
            "get": {
                "operationId": "getResearchEvents",
                "summary": "SSE event stream for research job (R7.3)",
                "tags": ["Research"],
                "parameters": [
                    {"name": "job_id", "in": "path", "required": True, "schema": {"type": "string", "format": "uuid"}},
                    {"name": "Last-Event-ID", "in": "header", "schema": {"type": "integer"}},
                ],
                "responses": {
                    "200": {
                        "description": "SSE event stream",
                        "content": {"text/event-stream": {"schema": {"type": "string"}}},
                    },
                    "401": {"$ref": "#/components/responses/Unauthorized"},
                    "404": {"$ref": "#/components/responses/NotFound"},
                    "429": {"$ref": "#/components/responses/RateLimited"},
                },
            }
        },
        "/sessions": {
            "post": {
                "operationId": "createSession",
                "summary": "Create a research session (R8)",
                "tags": ["Sessions"],
                "requestBody": {
                    "required": True,
                    "content": {"application/json": {"schema": {"$ref": "#/components/schemas/SessionCreateRequest"}}},
                },
                "responses": {
                    "201": {
                        "description": "Session created",
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Session"}}},
                    },
                    "400": {"$ref": "#/components/responses/BadRequest"},
                    "401": {"$ref": "#/components/responses/Unauthorized"},
                    "429": {"$ref": "#/components/responses/RateLimited"},
                },
            }
        },
        "/sessions/{session_id}": {
            "delete": {
                "operationId": "deleteSession",
                "summary": "Delete a session (R8)",
                "tags": ["Sessions"],
                "parameters": [
                    {
                        "name": "session_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string", "format": "uuid"},
                    }
                ],
                "responses": {
                    "204": {"description": "Session deleted"},
                    "401": {"$ref": "#/components/responses/Unauthorized"},
                    "404": {"$ref": "#/components/responses/NotFound"},
                    "429": {"$ref": "#/components/responses/RateLimited"},
                },
            }
        },
        "/pipelines": {
            "post": {
                "operationId": "createPipeline",
                "summary": "Create a retrieval pipeline (R9)",
                "tags": ["Pipelines"],
                "requestBody": {
                    "required": True,
                    "content": {"application/json": {"schema": {"$ref": "#/components/schemas/PipelineCreateRequest"}}},
                },
                "responses": {
                    "201": {
                        "description": "Pipeline created",
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Pipeline"}}},
                    },
                    "400": {"$ref": "#/components/responses/BadRequest"},
                    "401": {"$ref": "#/components/responses/Unauthorized"},
                    "429": {"$ref": "#/components/responses/RateLimited"},
                },
            }
        },
        "/pipelines/{pipeline_id}": {
            "get": {
                "operationId": "getPipeline",
                "summary": "Get pipeline definition",
                "tags": ["Pipelines"],
                "parameters": [
                    {
                        "name": "pipeline_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string", "format": "uuid"},
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Pipeline definition",
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Pipeline"}}},
                    },
                    "401": {"$ref": "#/components/responses/Unauthorized"},
                    "404": {"$ref": "#/components/responses/NotFound"},
                    "429": {"$ref": "#/components/responses/RateLimited"},
                },
            },
            "delete": {
                "operationId": "deletePipeline",
                "summary": "Delete a pipeline",
                "tags": ["Pipelines"],
                "parameters": [
                    {
                        "name": "pipeline_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string", "format": "uuid"},
                    }
                ],
                "responses": {
                    "204": {"description": "Pipeline deleted"},
                    "401": {"$ref": "#/components/responses/Unauthorized"},
                    "404": {"$ref": "#/components/responses/NotFound"},
                    "429": {"$ref": "#/components/responses/RateLimited"},
                },
            },
        },
        "/openapi.json": {
            "get": {
                "operationId": "getOpenApiSpec",
                "summary": "Get OpenAPI specification (R16.4)",
                "tags": ["Meta"],
                "security": [],
                "responses": {
                    "200": {
                        "description": "OpenAPI 3.1 specification",
                        "content": {"application/json": {"schema": {"type": "object"}}},
                    }
                },
            }
        },
    },
}


# Error codes enumeration for SDK mapping
ERROR_CODES = [
    "missing_token",
    "invalid_token",
    "expired_token",
    "revoked_token",
    "invalid_num_results",
    "invalid_mode",
    "invalid_query",
    "invalid_url",
    "invalid_document_id_count",
    "missing_highlight_query",
    "invalid_research_request",
    "invalid_threshold",
    "resource_not_found",
    "pipeline_not_found",
    "job_not_found",
    "session_not_found",
    "unknown_url",
    "document_not_found",
    "rate_limited",
    "audit_log_unavailable",
    "no_sources_available",
    "budget_exceeded",
    "internal_error",
    "empty_input",
    "filter_too_large",
    "unknown_pipeline_step",
    "step_timeout",
]

# SSE event types for answer streaming
ANSWER_EVENT_TYPES = ["token", "citation", "done", "error"]

# SSE event types for research streaming
RESEARCH_EVENT_TYPES = [
    "plan_updated",
    "step_started",
    "step_completed",
    "citation",
    "report_chunk",
    "done",
    "error",
]
