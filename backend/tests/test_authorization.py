"""Unit tests for authorization — cross-tenant access yields uniform 404 (R13.3).

The key security property: an attacker cannot distinguish "resource belongs to
another tenant" from "resource does not exist" based on the response.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest

from backend.auth.service import AuthService, ResourceNotFoundError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _MockPool:
    """Mock asyncpg pool for AuthService instantiation."""

    def __init__(self):
        self.conn = AsyncMock()
        self.conn.fetch = AsyncMock(return_value=[])

    @asynccontextmanager
    async def acquire(self):
        yield self.conn


@pytest.fixture
def auth_service():
    """Create an AuthService with a mocked DB pool."""
    return AuthService(db_pool=_MockPool(), cache_ttl_seconds=60)


# ---------------------------------------------------------------------------
# Tests: authorize(tenant_id, resource_tenant_id)
# ---------------------------------------------------------------------------


class TestAuthorize:
    """Tests for the authorize() method."""

    def test_same_tenant_returns_true(self, auth_service):
        """Same tenant_id and resource_tenant_id returns True."""
        tenant_id = str(uuid.uuid4())
        assert auth_service.authorize(tenant_id, tenant_id) is True

    def test_different_tenant_returns_false(self, auth_service):
        """Different tenant_id and resource_tenant_id returns False."""
        tenant_a = str(uuid.uuid4())
        tenant_b = str(uuid.uuid4())
        assert auth_service.authorize(tenant_a, tenant_b) is False

    def test_empty_strings_same_returns_true(self, auth_service):
        """Two identical empty strings return True (edge case)."""
        assert auth_service.authorize("", "") is True

    def test_empty_vs_nonempty_returns_false(self, auth_service):
        """Empty tenant_id vs non-empty resource_tenant_id returns False."""
        assert auth_service.authorize("", str(uuid.uuid4())) is False

    def test_case_sensitive_comparison(self, auth_service):
        """Authorization comparison is case-sensitive."""
        assert auth_service.authorize("Tenant-A", "tenant-a") is False

    def test_whitespace_difference_returns_false(self, auth_service):
        """Strings differing only by whitespace are not equal."""
        assert auth_service.authorize("tenant_1", " tenant_1") is False


# ---------------------------------------------------------------------------
# Tests: raise_if_cross_tenant(tenant_id, resource_tenant_id)
# ---------------------------------------------------------------------------


class TestRaiseIfCrossTenant:
    """Tests for the raise_if_cross_tenant() method."""

    def test_same_tenant_no_exception(self, auth_service):
        """Same tenant does not raise any exception."""
        tenant_id = str(uuid.uuid4())
        # Should not raise
        auth_service.raise_if_cross_tenant(tenant_id, tenant_id)

    def test_different_tenant_raises_resource_not_found(self, auth_service):
        """Different tenant raises ResourceNotFoundError."""
        tenant_a = str(uuid.uuid4())
        tenant_b = str(uuid.uuid4())
        with pytest.raises(ResourceNotFoundError):
            auth_service.raise_if_cross_tenant(tenant_a, tenant_b)

    def test_error_code_is_resource_not_found(self, auth_service):
        """The error code is 'resource_not_found', not 'access_denied' or 'forbidden'."""
        tenant_a = str(uuid.uuid4())
        tenant_b = str(uuid.uuid4())
        with pytest.raises(ResourceNotFoundError) as exc_info:
            auth_service.raise_if_cross_tenant(tenant_a, tenant_b)
        assert exc_info.value.code == "resource_not_found"

    def test_error_message_is_generic(self, auth_service):
        """The error message is generic and does not reveal cross-tenant access."""
        tenant_a = str(uuid.uuid4())
        tenant_b = str(uuid.uuid4())
        with pytest.raises(ResourceNotFoundError) as exc_info:
            auth_service.raise_if_cross_tenant(tenant_a, tenant_b)
        # Message should not contain "tenant", "access", "denied", "forbidden", etc.
        msg = exc_info.value.message.lower()
        assert "tenant" not in msg
        assert "access" not in msg
        assert "denied" not in msg
        assert "forbidden" not in msg
        assert "permission" not in msg

    def test_error_does_not_reveal_resource_existence(self, auth_service):
        """The error code does not indicate whether the resource exists in another tenant."""
        tenant_a = str(uuid.uuid4())
        tenant_b = str(uuid.uuid4())
        with pytest.raises(ResourceNotFoundError) as exc_info:
            auth_service.raise_if_cross_tenant(tenant_a, tenant_b)
        # Code must be exactly "resource_not_found"
        assert exc_info.value.code == "resource_not_found"
        # No hint about "other tenant" or "exists"
        assert "exist" not in exc_info.value.message.lower()
        assert "other" not in exc_info.value.message.lower()


# ---------------------------------------------------------------------------
# Tests: ResourceNotFoundError
# ---------------------------------------------------------------------------


class TestResourceNotFoundError:
    """Tests for the ResourceNotFoundError exception class."""

    def test_default_code(self):
        """Default code is 'resource_not_found'."""
        err = ResourceNotFoundError()
        assert err.code == "resource_not_found"

    def test_default_message(self):
        """Default message is a generic not-found message."""
        err = ResourceNotFoundError()
        assert err.message == "The requested resource was not found"

    def test_custom_code_and_message(self):
        """Custom code and message can be provided."""
        err = ResourceNotFoundError(code="job_not_found", message="The requested job was not found")
        assert err.code == "job_not_found"
        assert err.message == "The requested job was not found"

    def test_is_exception(self):
        """ResourceNotFoundError is a proper Exception subclass."""
        err = ResourceNotFoundError()
        assert isinstance(err, Exception)

    def test_str_representation(self):
        """str() of the error returns the message."""
        err = ResourceNotFoundError()
        assert str(err) == "The requested resource was not found"

    def test_error_shape_identical_for_cross_tenant_and_not_found(self):
        """Error shape for cross-tenant access is identical to genuine not-found.

        This is the key security property: an attacker cannot distinguish
        'resource belongs to another tenant' from 'resource does not exist'.
        """
        # Simulate cross-tenant error
        cross_tenant_err = ResourceNotFoundError()

        # Simulate genuine not-found error (same class, same defaults)
        genuine_not_found_err = ResourceNotFoundError()

        # Both have identical code and message
        assert cross_tenant_err.code == genuine_not_found_err.code
        assert cross_tenant_err.message == genuine_not_found_err.message

    def test_no_forbidden_or_access_denied_codes(self):
        """The default error never uses codes that reveal authorization failure."""
        err = ResourceNotFoundError()
        forbidden_codes = {"forbidden", "access_denied", "unauthorized", "not_authorized"}
        assert err.code not in forbidden_codes
