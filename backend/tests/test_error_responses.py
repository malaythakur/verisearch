"""Tests for uniform error response format with stable error codes.

Validates:
- All error responses have the uniform shape {"error": {"code": ..., "message": ...}}
- Error codes are stable strings (from the defined set)
- No stack traces leak in production error responses
- 404 responses are indistinguishable between cross-tenant and genuine not-found
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient
from starlette.requests import Request

from backend.api_gateway.app import create_app
from backend.api_gateway.errors import (
    ALL_ERROR_CODES,
    ERROR_INTERNAL,
    ERROR_RESOURCE_NOT_FOUND,
    ErrorDetail,
    ErrorResponse,
    error_response,
)
from backend.api_gateway.validation import RequestValidationError
from backend.auth.service import AuthError, AuthResult, ResourceNotFoundError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class FakeAuthService:
    """A fake AuthService for testing."""

    def __init__(self, *, should_fail: bool = False, error: AuthError | None = None) -> None:
        self._should_fail = should_fail
        self._error = error or AuthError(code="missing_token", message="Authorization header is required")

    async def authenticate(self, headers: dict[str, str], **kwargs) -> AuthResult | AuthError:  # noqa: ANN003
        if self._should_fail:
            return self._error
        return AuthResult(tenant_id="tenant-123", api_key_id="key-456")


@pytest.fixture
def auth_success() -> FakeAuthService:
    """Auth service that always succeeds."""
    return FakeAuthService(should_fail=False)


@pytest.fixture
def auth_failure() -> FakeAuthService:
    """Auth service that always fails."""
    return FakeAuthService(should_fail=True)


# ---------------------------------------------------------------------------
# Helper to validate uniform error shape
# ---------------------------------------------------------------------------


def assert_uniform_error_shape(body: dict) -> None:
    """Assert that a response body matches the uniform error shape."""
    assert "error" in body, "Response must have 'error' key"
    error = body["error"]
    assert "code" in error, "Error must have 'code' field"
    assert "message" in error, "Error must have 'message' field"
    assert isinstance(error["code"], str), "Error code must be a string"
    assert isinstance(error["message"], str), "Error message must be a string"
    assert len(error["code"]) > 0, "Error code must not be empty"
    assert len(error["message"]) > 0, "Error message must not be empty"
    # Ensure no extra fields leak
    assert set(body.keys()) == {"error"}, "Response must only contain 'error' key"
    assert set(error.keys()) == {"code", "message"}, "Error must only contain 'code' and 'message'"


# ---------------------------------------------------------------------------
# ErrorResponse model tests
# ---------------------------------------------------------------------------


class TestErrorResponseModel:
    """Tests for the ErrorResponse Pydantic model."""

    def test_valid_error_response(self) -> None:
        """ErrorResponse model validates correct shape."""
        resp = ErrorResponse(error=ErrorDetail(code="invalid_query", message="query is required"))
        assert resp.error.code == "invalid_query"
        assert resp.error.message == "query is required"

    def test_error_response_serialization(self) -> None:
        """ErrorResponse serializes to the expected JSON shape."""
        resp = ErrorResponse(error=ErrorDetail(code="missing_token", message="Token required"))
        data = resp.model_dump()
        assert data == {"error": {"code": "missing_token", "message": "Token required"}}


# ---------------------------------------------------------------------------
# error_response helper tests
# ---------------------------------------------------------------------------


class TestErrorResponseHelper:
    """Tests for the error_response() helper function."""

    def test_returns_json_response_with_correct_status(self) -> None:
        """error_response returns a JSONResponse with the given status code."""
        resp = error_response(400, "invalid_query", "query is required")
        assert resp.status_code == 400

    def test_returns_uniform_shape(self) -> None:
        """error_response returns the uniform error body shape."""
        resp = error_response(404, "resource_not_found", "Not found")
        # JSONResponse stores body as bytes
        import json

        body = json.loads(resp.body)
        assert_uniform_error_shape(body)
        assert body["error"]["code"] == "resource_not_found"
        assert body["error"]["message"] == "Not found"

    def test_various_status_codes(self) -> None:
        """error_response works with various HTTP status codes."""
        for status in (400, 401, 403, 404, 429, 500, 503):
            resp = error_response(status, "test_code", "test message")
            assert resp.status_code == status


# ---------------------------------------------------------------------------
# Stable error codes tests
# ---------------------------------------------------------------------------


class TestStableErrorCodes:
    """Tests that error codes are stable strings from the defined set."""

    def test_all_error_codes_are_strings(self) -> None:
        """All defined error codes are non-empty strings."""
        for code in ALL_ERROR_CODES:
            assert isinstance(code, str)
            assert len(code) > 0

    def test_all_error_codes_are_snake_case(self) -> None:
        """All error codes follow snake_case convention."""
        for code in ALL_ERROR_CODES:
            assert code == code.lower(), f"Error code '{code}' is not lowercase"
            assert " " not in code, f"Error code '{code}' contains spaces"
            # Only alphanumeric and underscores
            assert all(
                c.isalnum() or c == "_" for c in code
            ), f"Error code '{code}' contains invalid characters"

    def test_error_codes_are_unique(self) -> None:
        """All error codes are unique (no duplicates)."""
        codes_list = list(ALL_ERROR_CODES)
        assert len(codes_list) == len(set(codes_list))

    def test_expected_auth_codes_present(self) -> None:
        """Auth error codes are in the defined set."""
        auth_codes = {"missing_token", "invalid_token", "expired_token", "revoked_token"}
        assert auth_codes.issubset(ALL_ERROR_CODES)

    def test_expected_validation_codes_present(self) -> None:
        """Validation error codes are in the defined set."""
        validation_codes = {
            "invalid_num_results",
            "invalid_mode",
            "invalid_query",
            "invalid_url",
            "invalid_document_id_count",
            "missing_highlight_query",
            "invalid_research_request",
            "invalid_threshold",
        }
        assert validation_codes.issubset(ALL_ERROR_CODES)

    def test_expected_resource_codes_present(self) -> None:
        """Resource error codes are in the defined set."""
        resource_codes = {
            "resource_not_found",
            "pipeline_not_found",
            "job_not_found",
            "session_not_found",
            "unknown_url",
            "document_not_found",
        }
        assert resource_codes.issubset(ALL_ERROR_CODES)

    def test_expected_system_codes_present(self) -> None:
        """System error codes are in the defined set."""
        system_codes = {
            "audit_log_unavailable",
            "no_sources_available",
            "budget_exceeded",
            "internal_error",
        }
        assert system_codes.issubset(ALL_ERROR_CODES)

    def test_expected_filter_codes_present(self) -> None:
        """Filter error codes are in the defined set."""
        filter_codes = {"empty_input", "filter_too_large"}
        assert filter_codes.issubset(ALL_ERROR_CODES)

    def test_expected_pipeline_codes_present(self) -> None:
        """Pipeline error codes are in the defined set."""
        pipeline_codes = {"unknown_pipeline_step", "step_timeout"}
        assert pipeline_codes.issubset(ALL_ERROR_CODES)

    def test_rate_limit_code_present(self) -> None:
        """Rate limit error code is in the defined set."""
        assert "rate_limited" in ALL_ERROR_CODES


# ---------------------------------------------------------------------------
# Exception handler integration tests
# ---------------------------------------------------------------------------


class TestRequestValidationErrorHandler:
    """Tests that RequestValidationError is caught and returns uniform 400."""

    async def test_validation_error_returns_400(self, auth_success: FakeAuthService) -> None:
        """RequestValidationError from a route returns 400 with uniform shape."""
        app = create_app(auth_service=auth_success)

        @app.post("/v1/test-validation")
        async def validation_route(request: Request):  # noqa: ANN202
            raise RequestValidationError(
                code="invalid_query",
                message="query must not be empty after trimming",
            )

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/v1/test-validation",
                headers={"Authorization": "Bearer testprefix_secretvalue"},
            )

        assert response.status_code == 400
        body = response.json()
        assert_uniform_error_shape(body)
        assert body["error"]["code"] == "invalid_query"
        assert body["error"]["message"] == "query must not be empty after trimming"

    async def test_various_validation_codes(self, auth_success: FakeAuthService) -> None:
        """Different validation error codes are preserved in the response."""
        app = create_app(auth_service=auth_success)

        codes_to_test = [
            ("invalid_num_results", "num_results must be between 0 and 100"),
            ("invalid_mode", "mode must be one of: neural, keyword, hybrid"),
            ("invalid_url", "url is required"),
            ("invalid_threshold", "min_credibility must be between 0.0 and 1.0"),
        ]

        for code, message in codes_to_test:

            @app.post(f"/v1/test-{code}")
            async def route(request: Request, _code=code, _msg=message):  # noqa: ANN202
                raise RequestValidationError(code=_code, message=_msg)

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.post(
                    f"/v1/test-{code}",
                    headers={"Authorization": "Bearer testprefix_secretvalue"},
                )

            assert response.status_code == 400
            body = response.json()
            assert_uniform_error_shape(body)
            assert body["error"]["code"] == code


class TestResourceNotFoundErrorHandler:
    """Tests that ResourceNotFoundError is caught and returns uniform 404."""

    async def test_resource_not_found_returns_404(self, auth_success: FakeAuthService) -> None:
        """ResourceNotFoundError returns 404 with uniform shape."""
        app = create_app(auth_service=auth_success)

        @app.get("/v1/test-not-found")
        async def not_found_route(request: Request):  # noqa: ANN202
            raise ResourceNotFoundError()

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(
                "/v1/test-not-found",
                headers={"Authorization": "Bearer testprefix_secretvalue"},
            )

        assert response.status_code == 404
        body = response.json()
        assert_uniform_error_shape(body)
        assert body["error"]["code"] == "resource_not_found"

    async def test_custom_not_found_code(self, auth_success: FakeAuthService) -> None:
        """ResourceNotFoundError with custom code preserves the code."""
        app = create_app(auth_service=auth_success)

        @app.get("/v1/test-pipeline-not-found")
        async def pipeline_not_found_route(request: Request):  # noqa: ANN202
            raise ResourceNotFoundError(
                code="pipeline_not_found",
                message="The requested pipeline was not found",
            )

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(
                "/v1/test-pipeline-not-found",
                headers={"Authorization": "Bearer testprefix_secretvalue"},
            )

        assert response.status_code == 404
        body = response.json()
        assert_uniform_error_shape(body)
        assert body["error"]["code"] == "pipeline_not_found"


class TestGenericExceptionHandler:
    """Tests that unhandled exceptions return 500 with no stack trace."""

    async def test_generic_exception_returns_500(self, auth_success: FakeAuthService) -> None:
        """Unhandled exception returns 500 with uniform shape."""
        app = create_app(auth_service=auth_success)

        @app.get("/v1/test-crash")
        async def crash_route(request: Request):  # noqa: ANN202
            raise RuntimeError("Something went terribly wrong internally")

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(
                "/v1/test-crash",
                headers={"Authorization": "Bearer testprefix_secretvalue"},
            )

        assert response.status_code == 500
        body = response.json()
        assert_uniform_error_shape(body)
        assert body["error"]["code"] == "internal_error"

    async def test_no_stack_trace_in_500_response(self, auth_success: FakeAuthService) -> None:
        """500 responses do not leak stack traces or internal details."""
        app = create_app(auth_service=auth_success)

        @app.get("/v1/test-leak")
        async def leak_route(request: Request):  # noqa: ANN202
            raise ValueError("secret_database_password=hunter2")

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(
                "/v1/test-leak",
                headers={"Authorization": "Bearer testprefix_secretvalue"},
            )

        assert response.status_code == 500
        body = response.json()
        assert_uniform_error_shape(body)
        # Ensure no internal details leak
        response_text = response.text
        assert "secret_database_password" not in response_text
        assert "hunter2" not in response_text
        assert "Traceback" not in response_text
        assert "ValueError" not in response_text
        assert "File" not in response_text

    async def test_no_exception_class_name_in_response(self, auth_success: FakeAuthService) -> None:
        """500 responses do not include the exception class name."""
        app = create_app(auth_service=auth_success)

        @app.get("/v1/test-class-leak")
        async def class_leak_route(request: Request):  # noqa: ANN202
            raise ZeroDivisionError("division by zero")

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(
                "/v1/test-class-leak",
                headers={"Authorization": "Bearer testprefix_secretvalue"},
            )

        assert response.status_code == 500
        body = response.json()
        assert_uniform_error_shape(body)
        assert "ZeroDivisionError" not in response.text
        assert "division by zero" not in response.text


# ---------------------------------------------------------------------------
# Cross-tenant indistinguishability tests (R13.3)
# ---------------------------------------------------------------------------


class TestCrossTenantIndistinguishability:
    """Tests that 404 responses are indistinguishable between cross-tenant and genuine not-found.

    Per R13.3, R7.7, R8.5, R9.7: cross-tenant access must return the same
    response shape as a genuine not-found, making it impossible for an attacker
    to distinguish between "resource belongs to another tenant" and "resource
    does not exist".
    """

    async def test_cross_tenant_same_shape_as_not_found(self, auth_success: FakeAuthService) -> None:
        """Cross-tenant 404 has identical shape to genuine not-found 404."""
        app = create_app(auth_service=auth_success)

        @app.get("/v1/test-genuine-not-found")
        async def genuine_not_found(request: Request):  # noqa: ANN202
            raise ResourceNotFoundError(
                code="resource_not_found",
                message="The requested resource was not found",
            )

        @app.get("/v1/test-cross-tenant")
        async def cross_tenant(request: Request):  # noqa: ANN202
            # Simulates cross-tenant access — same error raised
            raise ResourceNotFoundError(
                code="resource_not_found",
                message="The requested resource was not found",
            )

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            genuine_response = await client.get(
                "/v1/test-genuine-not-found",
                headers={"Authorization": "Bearer testprefix_secretvalue"},
            )
            cross_tenant_response = await client.get(
                "/v1/test-cross-tenant",
                headers={"Authorization": "Bearer testprefix_secretvalue"},
            )

        # Status codes must be identical
        assert genuine_response.status_code == cross_tenant_response.status_code == 404

        # Response bodies must be identical
        genuine_body = genuine_response.json()
        cross_tenant_body = cross_tenant_response.json()
        assert genuine_body == cross_tenant_body

        # Both must have uniform shape
        assert_uniform_error_shape(genuine_body)
        assert_uniform_error_shape(cross_tenant_body)

        # Code must be the same generic "resource_not_found"
        assert genuine_body["error"]["code"] == "resource_not_found"
        assert cross_tenant_body["error"]["code"] == "resource_not_found"

    async def test_no_tenant_info_in_404(self, auth_success: FakeAuthService) -> None:
        """404 responses never include tenant identifiers or ownership hints."""
        app = create_app(auth_service=auth_success)

        @app.get("/v1/test-no-tenant-leak")
        async def no_tenant_leak(request: Request):  # noqa: ANN202
            raise ResourceNotFoundError()

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(
                "/v1/test-no-tenant-leak",
                headers={"Authorization": "Bearer testprefix_secretvalue"},
            )

        assert response.status_code == 404
        response_text = response.text
        # Must not contain any tenant-related information
        assert "tenant" not in response_text.lower()
        assert "forbidden" not in response_text.lower()
        assert "access_denied" not in response_text.lower()
        assert "unauthorized" not in response_text.lower()

    async def test_specific_resource_not_found_codes(self, auth_success: FakeAuthService) -> None:
        """Resource-specific not-found codes (job_not_found, session_not_found, etc.)
        still use the same uniform shape."""
        app = create_app(auth_service=auth_success)

        resource_codes = [
            ("job_not_found", "The requested job was not found"),
            ("session_not_found", "The requested session was not found"),
            ("pipeline_not_found", "The requested pipeline was not found"),
        ]

        for code, message in resource_codes:

            @app.get(f"/v1/test-{code}")
            async def resource_route(request: Request, _code=code, _msg=message):  # noqa: ANN202
                raise ResourceNotFoundError(code=_code, message=_msg)

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.get(
                    f"/v1/test-{code}",
                    headers={"Authorization": "Bearer testprefix_secretvalue"},
                )

            assert response.status_code == 404
            body = response.json()
            assert_uniform_error_shape(body)
            assert body["error"]["code"] == code


# ---------------------------------------------------------------------------
# Auth error uniform shape tests
# ---------------------------------------------------------------------------


class TestAuthErrorUniformShape:
    """Tests that auth errors (401) also follow the uniform shape."""

    async def test_missing_token_uniform_shape(self) -> None:
        """Missing token 401 has uniform error shape."""
        auth = FakeAuthService(
            should_fail=True,
            error=AuthError(code="missing_token", message="Authorization header is required"),
        )
        app = create_app(auth_service=auth)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/v1/search")

        assert response.status_code == 401
        body = response.json()
        assert_uniform_error_shape(body)
        assert body["error"]["code"] == "missing_token"

    async def test_invalid_token_uniform_shape(self) -> None:
        """Invalid token 401 has uniform error shape."""
        auth = FakeAuthService(
            should_fail=True,
            error=AuthError(code="invalid_token", message="API key is not valid"),
        )
        app = create_app(auth_service=auth)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/v1/search")

        assert response.status_code == 401
        body = response.json()
        assert_uniform_error_shape(body)
        assert body["error"]["code"] == "invalid_token"

    async def test_expired_token_uniform_shape(self) -> None:
        """Expired token 401 has uniform error shape."""
        auth = FakeAuthService(
            should_fail=True,
            error=AuthError(code="expired_token", message="API key has expired"),
        )
        app = create_app(auth_service=auth)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/v1/search")

        assert response.status_code == 401
        body = response.json()
        assert_uniform_error_shape(body)
        assert body["error"]["code"] == "expired_token"

    async def test_revoked_token_uniform_shape(self) -> None:
        """Revoked token 401 has uniform error shape."""
        auth = FakeAuthService(
            should_fail=True,
            error=AuthError(code="revoked_token", message="API key has been revoked"),
        )
        app = create_app(auth_service=auth)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/v1/search")

        assert response.status_code == 401
        body = response.json()
        assert_uniform_error_shape(body)
        assert body["error"]["code"] == "revoked_token"
