"""Uniform error response format with stable error codes.

Implements the uniform error response shape used across all API Gateway responses:

    {
        "error": {
            "code": "<stable_error_code>",
            "message": "<human_readable_description>"
        }
    }

Provides:
- ErrorResponse Pydantic model for the error shape
- All stable error codes as constants (grouped by category)
- FastAPI exception handlers for RequestValidationError, ResourceNotFoundError,
  Pydantic ValidationError, and generic Exception
- ErrorHandlingMiddleware that catches exceptions in the middleware chain
- Helper function error_response() for building uniform error JSONResponses
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from backend.api_gateway.validation import RequestValidationError, map_pydantic_error_to_code
from backend.auth.service import ResourceNotFoundError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Error response model
# ---------------------------------------------------------------------------


class ErrorDetail(BaseModel):
    """The inner error object containing code and message."""

    code: str
    message: str


class ErrorResponse(BaseModel):
    """Uniform error response shape for all API errors.

    Shape:
        {
            "error": {
                "code": "<stable_error_code>",
                "message": "<human_readable_description>"
            }
        }
    """

    error: ErrorDetail


# ---------------------------------------------------------------------------
# Stable error codes — Auth
# ---------------------------------------------------------------------------

ERROR_MISSING_TOKEN = "missing_token"
ERROR_INVALID_TOKEN = "invalid_token"
ERROR_EXPIRED_TOKEN = "expired_token"
ERROR_REVOKED_TOKEN = "revoked_token"

# ---------------------------------------------------------------------------
# Stable error codes — Validation
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
# Stable error codes — Resource
# ---------------------------------------------------------------------------

ERROR_RESOURCE_NOT_FOUND = "resource_not_found"
ERROR_PIPELINE_NOT_FOUND = "pipeline_not_found"
ERROR_JOB_NOT_FOUND = "job_not_found"
ERROR_SESSION_NOT_FOUND = "session_not_found"
ERROR_UNKNOWN_URL = "unknown_url"
ERROR_DOCUMENT_NOT_FOUND = "document_not_found"

# ---------------------------------------------------------------------------
# Stable error codes — Rate limit
# ---------------------------------------------------------------------------

ERROR_RATE_LIMITED = "rate_limited"

# ---------------------------------------------------------------------------
# Stable error codes — System
# ---------------------------------------------------------------------------

ERROR_AUDIT_LOG_UNAVAILABLE = "audit_log_unavailable"
ERROR_NO_SOURCES_AVAILABLE = "no_sources_available"
ERROR_BUDGET_EXCEEDED = "budget_exceeded"
ERROR_INTERNAL = "internal_error"

# ---------------------------------------------------------------------------
# Stable error codes — Filter
# ---------------------------------------------------------------------------

ERROR_EMPTY_INPUT = "empty_input"
ERROR_FILTER_TOO_LARGE = "filter_too_large"

# ---------------------------------------------------------------------------
# Stable error codes — Pipeline
# ---------------------------------------------------------------------------

ERROR_UNKNOWN_PIPELINE_STEP = "unknown_pipeline_step"
ERROR_STEP_TIMEOUT = "step_timeout"


# ---------------------------------------------------------------------------
# All error codes (for validation/testing)
# ---------------------------------------------------------------------------

ALL_ERROR_CODES: frozenset[str] = frozenset(
    [
        # Auth
        ERROR_MISSING_TOKEN,
        ERROR_INVALID_TOKEN,
        ERROR_EXPIRED_TOKEN,
        ERROR_REVOKED_TOKEN,
        # Validation
        ERROR_INVALID_NUM_RESULTS,
        ERROR_INVALID_MODE,
        ERROR_INVALID_QUERY,
        ERROR_INVALID_URL,
        ERROR_INVALID_DOCUMENT_ID_COUNT,
        ERROR_MISSING_HIGHLIGHT_QUERY,
        ERROR_INVALID_RESEARCH_REQUEST,
        ERROR_INVALID_THRESHOLD,
        # Resource
        ERROR_RESOURCE_NOT_FOUND,
        ERROR_PIPELINE_NOT_FOUND,
        ERROR_JOB_NOT_FOUND,
        ERROR_SESSION_NOT_FOUND,
        ERROR_UNKNOWN_URL,
        ERROR_DOCUMENT_NOT_FOUND,
        # Rate limit
        ERROR_RATE_LIMITED,
        # System
        ERROR_AUDIT_LOG_UNAVAILABLE,
        ERROR_NO_SOURCES_AVAILABLE,
        ERROR_BUDGET_EXCEEDED,
        ERROR_INTERNAL,
        # Filter
        ERROR_EMPTY_INPUT,
        ERROR_FILTER_TOO_LARGE,
        # Pipeline
        ERROR_UNKNOWN_PIPELINE_STEP,
        ERROR_STEP_TIMEOUT,
    ]
)


# ---------------------------------------------------------------------------
# Helper function
# ---------------------------------------------------------------------------


def error_response(status_code: int, code: str, message: str) -> JSONResponse:
    """Create a JSONResponse with the uniform error shape.

    Args:
        status_code: HTTP status code for the response.
        code: Stable error code string.
        message: Human-readable error description.

    Returns:
        A JSONResponse with the uniform error body shape.
    """
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
    )


# ---------------------------------------------------------------------------
# Exception handlers (used by ErrorHandlingMiddleware)
# ---------------------------------------------------------------------------


def _handle_request_validation_error(exc: RequestValidationError) -> JSONResponse:
    """Handle RequestValidationError → 400 with the appropriate error code."""
    return error_response(400, exc.code, exc.message)


def _handle_resource_not_found_error(exc: ResourceNotFoundError) -> JSONResponse:
    """Handle ResourceNotFoundError → 404 with resource_not_found."""
    return error_response(404, exc.code, exc.message)


def _handle_pydantic_validation_error(exc: ValidationError) -> JSONResponse:
    """Handle Pydantic ValidationError → 400 with mapped error code."""
    mapped = map_pydantic_error_to_code(exc.errors())
    return error_response(400, mapped.code, mapped.message)


def _handle_generic_exception(request: Request, exc: Exception) -> JSONResponse:
    """Handle generic Exception → 500 with internal_error.

    No stack trace or internal details are leaked in the response body.
    The actual exception is logged server-side for debugging.
    """
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return error_response(
        500,
        ERROR_INTERNAL,
        "An internal error occurred. Please try again later.",
    )


# ---------------------------------------------------------------------------
# Error handling middleware
# ---------------------------------------------------------------------------


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware that catches exceptions and returns uniform error responses.

    This middleware sits in the middleware chain and catches any exceptions
    that propagate from route handlers or inner middleware, converting them
    to the uniform error response format. This ensures that:
    - RequestValidationError → 400 with the appropriate error code
    - ResourceNotFoundError → 404 with resource_not_found
    - Pydantic ValidationError → 400 with mapped error code
    - Generic Exception → 500 with internal_error (no stack trace leaked)
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        try:
            return await call_next(request)
        except RequestValidationError as exc:
            return _handle_request_validation_error(exc)
        except ResourceNotFoundError as exc:
            return _handle_resource_not_found_error(exc)
        except ValidationError as exc:
            return _handle_pydantic_validation_error(exc)
        except Exception as exc:
            return _handle_generic_exception(request, exc)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_exception_handlers(app: FastAPI) -> None:
    """Register exception handlers on the FastAPI application.

    Registers both:
    1. FastAPI-level exception handlers (for exceptions raised in route handlers
       that don't propagate through middleware)
    2. The ErrorHandlingMiddleware (catches exceptions that propagate through
       the BaseHTTPMiddleware chain)

    The ErrorHandlingMiddleware is added as the innermost middleware (closest
    to route handlers) to catch exceptions before they propagate through
    the outer middleware stack.
    """
    # Register FastAPI-level exception handlers as a fallback
    app.add_exception_handler(RequestValidationError, _async_handle_request_validation_error)  # type: ignore[arg-type]
    app.add_exception_handler(ResourceNotFoundError, _async_handle_resource_not_found_error)  # type: ignore[arg-type]
    app.add_exception_handler(ValidationError, _async_handle_pydantic_validation_error)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, _async_handle_generic_exception)  # type: ignore[arg-type]


# Async wrappers for FastAPI exception handler registration
async def _async_handle_request_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    return _handle_request_validation_error(exc)


async def _async_handle_resource_not_found_error(request: Request, exc: ResourceNotFoundError) -> JSONResponse:
    return _handle_resource_not_found_error(exc)


async def _async_handle_pydantic_validation_error(request: Request, exc: ValidationError) -> JSONResponse:
    return _handle_pydantic_validation_error(exc)


async def _async_handle_generic_exception(request: Request, exc: Exception) -> JSONResponse:
    return _handle_generic_exception(request, exc)
