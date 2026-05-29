"""Request validation models and error mapping for the API Gateway.

Implements Pydantic v2 models with Field validators for all endpoint request bodies,
enforcing bounds specified in the requirements:
  - R3.5–R3.7: POST /v1/search
  - R4.5–R4.6: POST /v1/find_similar
  - R5.5–R5.6: POST /v1/contents
  - R7.8: POST /v1/research
  - R10.5: search thresholds (min_credibility, max_ai_generated_likelihood)

Each validation failure maps to a stable error code string matching the spec.
"""

from __future__ import annotations

import json
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator, model_validator

from backend.config.constants import Constants


# ---------------------------------------------------------------------------
# Stable error codes (spec-defined)
# ---------------------------------------------------------------------------

ERROR_INVALID_NUM_RESULTS = "invalid_num_results"
ERROR_INVALID_MODE = "invalid_mode"
ERROR_INVALID_QUERY = "invalid_query"
ERROR_INVALID_URL = "invalid_url"
ERROR_INVALID_DOCUMENT_ID_COUNT = "invalid_document_id_count"
ERROR_MISSING_HIGHLIGHT_QUERY = "missing_highlight_query"
ERROR_INVALID_RESEARCH_REQUEST = "invalid_research_request"
ERROR_INVALID_THRESHOLD = "invalid_threshold"


# ---------------------------------------------------------------------------
# Custom validation error
# ---------------------------------------------------------------------------


class ValidationErrorDetail:
    """Represents a single validation error with a stable error code."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}


class RequestValidationError(Exception):
    """Raised when request validation fails with a spec-defined error code."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)

    def to_response_body(self) -> dict[str, Any]:
        """Return the uniform error response shape."""
        return {"error": {"code": self.code, "message": self.message}}


# ---------------------------------------------------------------------------
# Pydantic request models
# ---------------------------------------------------------------------------

VALID_SEARCH_MODES = {"neural", "keyword", "hybrid"}


class SearchRequest(BaseModel):
    """Validates POST /v1/search request body.

    Enforces:
      - R3.7: query must be 1–2048 code points after trimming
      - R3.6: mode must be one of neural, keyword, hybrid
      - R3.5: num_results must be 0–100
      - R10.5: min_credibility and max_ai_generated_likelihood must be in [0.0, 1.0]
    """

    query: str
    mode: str
    num_results: int = Field(default=10)
    filters: str | None = None
    pipeline_id: str | None = None
    min_credibility: float | None = None
    max_ai_generated_likelihood: float | None = None

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        """R3.7: query must be non-empty after trimming and ≤2048 code points."""
        if v is None:
            raise RequestValidationError(
                ERROR_INVALID_QUERY,
                "query is required",
            )
        trimmed = v.strip()
        if len(trimmed) == 0:
            raise RequestValidationError(
                ERROR_INVALID_QUERY,
                "query must not be empty after trimming",
            )
        if len(v) > Constants.QUERY_MAX_CODE_POINTS:
            raise RequestValidationError(
                ERROR_INVALID_QUERY,
                f"query must not exceed {Constants.QUERY_MAX_CODE_POINTS} code points",
            )
        return v

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        """R3.6: mode must be one of neural, keyword, hybrid."""
        if v not in VALID_SEARCH_MODES:
            raise RequestValidationError(
                ERROR_INVALID_MODE,
                f"mode must be one of: {', '.join(sorted(VALID_SEARCH_MODES))}",
            )
        return v

    @field_validator("num_results")
    @classmethod
    def validate_num_results(cls, v: int) -> int:
        """R3.5: num_results must be in [0, 100]."""
        if v < 0 or v > 100:
            raise RequestValidationError(
                ERROR_INVALID_NUM_RESULTS,
                "num_results must be between 0 and 100 inclusive",
            )
        return v

    @field_validator("min_credibility")
    @classmethod
    def validate_min_credibility(cls, v: float | None) -> float | None:
        """R10.5: min_credibility must be in [0.0, 1.0] if provided."""
        if v is None:
            return v
        if not isinstance(v, (int, float)):
            raise RequestValidationError(
                ERROR_INVALID_THRESHOLD,
                "min_credibility must be a number between 0.0 and 1.0",
            )
        if v < 0.0 or v > 1.0:
            raise RequestValidationError(
                ERROR_INVALID_THRESHOLD,
                "min_credibility must be between 0.0 and 1.0 inclusive",
            )
        return v

    @field_validator("max_ai_generated_likelihood")
    @classmethod
    def validate_max_ai_generated_likelihood(cls, v: float | None) -> float | None:
        """R10.5: max_ai_generated_likelihood must be in [0.0, 1.0] if provided."""
        if v is None:
            return v
        if not isinstance(v, (int, float)):
            raise RequestValidationError(
                ERROR_INVALID_THRESHOLD,
                "max_ai_generated_likelihood must be a number between 0.0 and 1.0",
            )
        if v < 0.0 or v > 1.0:
            raise RequestValidationError(
                ERROR_INVALID_THRESHOLD,
                "max_ai_generated_likelihood must be between 0.0 and 1.0 inclusive",
            )
        return v


class FindSimilarRequest(BaseModel):
    """Validates POST /v1/find_similar request body.

    Enforces:
      - R4.5: url must be present, ≤2048 code points, and syntactically valid
      - R4.6: num_results must be 0–100
    """

    url: str
    num_results: int = Field(default=10)
    filters: str | None = None

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """R4.5: url must be present, ≤2048 code points, and syntactically valid."""
        if v is None or len(v.strip()) == 0:
            raise RequestValidationError(
                ERROR_INVALID_URL,
                "url is required",
            )
        if len(v) > Constants.URL_MAX_CODE_POINTS:
            raise RequestValidationError(
                ERROR_INVALID_URL,
                f"url must not exceed {Constants.URL_MAX_CODE_POINTS} code points",
            )
        # Syntactic URL validation
        try:
            parsed = urlparse(v)
            if not parsed.scheme or not parsed.netloc:
                raise RequestValidationError(
                    ERROR_INVALID_URL,
                    "url must be a syntactically valid URL with scheme and host",
                )
        except Exception as e:
            if isinstance(e, RequestValidationError):
                raise
            raise RequestValidationError(
                ERROR_INVALID_URL,
                "url must be a syntactically valid URL",
            ) from e
        return v

    @field_validator("num_results")
    @classmethod
    def validate_num_results(cls, v: int) -> int:
        """R4.6: num_results must be in [0, 100]."""
        if v < 0 or v > 100:
            raise RequestValidationError(
                ERROR_INVALID_NUM_RESULTS,
                "num_results must be between 0 and 100 inclusive",
            )
        return v


class ContentsRequest(BaseModel):
    """Validates POST /v1/contents request body.

    Enforces:
      - R5.5: document_ids count must be 1–100
      - R5.6: highlights=true requires non-empty query (cross-field)
    """

    document_ids: list[str]
    highlights: bool | None = None
    query: str | None = None
    summary: bool | None = None

    @field_validator("document_ids")
    @classmethod
    def validate_document_ids(cls, v: list[str]) -> list[str]:
        """R5.5: document_ids count must be between 1 and 100."""
        if len(v) < 1 or len(v) > 100:
            raise RequestValidationError(
                ERROR_INVALID_DOCUMENT_ID_COUNT,
                "document_ids must contain between 1 and 100 items",
            )
        return v


class ResearchRequest(BaseModel):
    """Validates POST /v1/research request body.

    Enforces:
      - R7.8: research_goal must be 1–4096 code points
      - R7.8: output_schema must be a syntactically valid JSON Schema if provided
    """

    research_goal: str
    output_schema: dict[str, Any] | None = None
    session_id: str | None = None
    max_steps: int | None = None
    max_duration_ms: int | None = None
    max_tool_calls: int | None = None

    @field_validator("research_goal")
    @classmethod
    def validate_research_goal(cls, v: str) -> str:
        """R7.8: research_goal must be 1–4096 code points."""
        if v is None or len(v.strip()) == 0:
            raise RequestValidationError(
                ERROR_INVALID_RESEARCH_REQUEST,
                "research_goal is required and must not be empty",
            )
        if len(v) > Constants.RESEARCH_GOAL_MAX_CODE_POINTS:
            raise RequestValidationError(
                ERROR_INVALID_RESEARCH_REQUEST,
                f"research_goal must not exceed {Constants.RESEARCH_GOAL_MAX_CODE_POINTS} code points",
            )
        return v

    @field_validator("output_schema")
    @classmethod
    def validate_output_schema(cls, v: dict[str, Any] | None) -> dict[str, Any] | None:
        """R7.8: output_schema must be a syntactically valid JSON Schema if provided."""
        if v is None:
            return v
        # A valid JSON Schema must be a JSON object (dict)
        if not isinstance(v, dict):
            raise RequestValidationError(
                ERROR_INVALID_RESEARCH_REQUEST,
                "output_schema must be a valid JSON Schema object",
            )
        # Minimal structural check: JSON Schema documents should be objects.
        # We verify it can be serialized (no circular refs, etc.)
        try:
            json.dumps(v)
        except (TypeError, ValueError) as e:
            raise RequestValidationError(
                ERROR_INVALID_RESEARCH_REQUEST,
                "output_schema must be a syntactically valid JSON Schema document",
            ) from e
        return v


# ---------------------------------------------------------------------------
# Cross-field validation functions
# ---------------------------------------------------------------------------


def validate_contents_request(req: ContentsRequest) -> None:
    """Validate cross-field constraints for ContentsRequest.

    R5.6: highlights=true requires a non-empty query field.

    Args:
        req: A validated ContentsRequest instance.

    Raises:
        RequestValidationError: If highlights=true but query is missing or empty.
    """
    if req.highlights is True:
        if req.query is None or len(req.query.strip()) == 0:
            raise RequestValidationError(
                ERROR_MISSING_HIGHLIGHT_QUERY,
                "highlights requires a non-empty query field",
            )


# ---------------------------------------------------------------------------
# Pydantic error → spec error code mapping
# ---------------------------------------------------------------------------

# Maps Pydantic field location to the appropriate spec error code.
# Used when Pydantic itself raises a ValidationError (e.g., type coercion failures).
_FIELD_TO_ERROR_CODE: dict[str, str] = {
    "query": ERROR_INVALID_QUERY,
    "mode": ERROR_INVALID_MODE,
    "num_results": ERROR_INVALID_NUM_RESULTS,
    "url": ERROR_INVALID_URL,
    "document_ids": ERROR_INVALID_DOCUMENT_ID_COUNT,
    "research_goal": ERROR_INVALID_RESEARCH_REQUEST,
    "output_schema": ERROR_INVALID_RESEARCH_REQUEST,
    "min_credibility": ERROR_INVALID_THRESHOLD,
    "max_ai_generated_likelihood": ERROR_INVALID_THRESHOLD,
}


def map_pydantic_error_to_code(errors: list[dict[str, Any]]) -> RequestValidationError:
    """Map a Pydantic ValidationError's error list to a spec-defined error code.

    Examines the first error's location to determine the appropriate stable
    error code. Falls back to the first field's code or a generic message.

    Args:
        errors: The list of error dicts from pydantic's ValidationError.errors().

    Returns:
        A RequestValidationError with the appropriate stable error code.
    """
    if not errors:
        return RequestValidationError("validation_error", "Request validation failed")

    first_error = errors[0]
    loc = first_error.get("loc", ())

    # Find the field name from the location tuple
    # Location is typically ("body", "field_name") or just ("field_name",)
    field_name = None
    for part in loc:
        if isinstance(part, str) and part in _FIELD_TO_ERROR_CODE:
            field_name = part
            break

    if field_name:
        code = _FIELD_TO_ERROR_CODE[field_name]
        msg = first_error.get("msg", f"Invalid value for {field_name}")
        return RequestValidationError(code, msg)

    # Fallback: use the first error's message
    msg = first_error.get("msg", "Request validation failed")
    return RequestValidationError("validation_error", msg)
