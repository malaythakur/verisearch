"""Unit tests for auth_failure audit emission (R13.6).

Validates:
- auth_failure emitted on invalid token (tenant_id is None)
- auth_failure emitted on expired token (tenant_id is the key's tenant)
- auth_failure emitted on revoked token (tenant_id is the key's tenant)
- Token value is NEVER present in the audit detail
- request_id is propagated to the audit entry
"""

from __future__ import annotations

import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest
from argon2 import PasswordHasher

from backend.audit_log.in_memory import InMemoryAuditEmitter
from backend.auth.service import AuthError, AuthResult, AuthService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_hasher = PasswordHasher()


def _make_token(prefix: str = "testprefix") -> str:
    """Create a token in the expected format: {prefix}_{random_part}."""
    return f"{prefix}_{uuid.uuid4().hex}"


def _make_key_row(
    token: str,
    tenant_id: str | None = None,
    api_key_id: str | None = None,
    expires_at: datetime | None = None,
    revoked_at: datetime | None = None,
    rotation_grace_seconds: int = 3600,
) -> dict:
    """Create a mock api_keys row with the token hashed."""
    return {
        "api_key_id": api_key_id or str(uuid.uuid4()),
        "tenant_id": tenant_id or str(uuid.uuid4()),
        "key_hash": _hasher.hash(token),
        "expires_at": expires_at,
        "revoked_at": revoked_at,
        "rotation_grace_seconds": rotation_grace_seconds,
    }


class _MockPool:
    """Mock asyncpg pool that supports `async with pool.acquire() as conn`."""

    def __init__(self):
        self.conn = AsyncMock()
        self.conn.fetch = AsyncMock(return_value=[])

    @asynccontextmanager
    async def acquire(self):
        yield self.conn


@pytest.fixture
def mock_db_pool():
    """Create a mock asyncpg pool."""
    return _MockPool()


@pytest.fixture
def audit_emitter():
    """Create an in-memory audit emitter for capturing events."""
    return InMemoryAuditEmitter()


@pytest.fixture
def auth_service(mock_db_pool, audit_emitter):
    """Create an AuthService with a mocked DB pool and audit emitter."""
    return AuthService(db_pool=mock_db_pool, cache_ttl_seconds=60, audit_emitter=audit_emitter)


# ---------------------------------------------------------------------------
# Tests: auth_failure emitted on invalid token (tenant_id is None)
# ---------------------------------------------------------------------------


class TestAuthFailureInvalidToken:
    """auth_failure audit events for invalid/unknown tokens have null tenant_id."""

    async def test_missing_auth_header_emits_auth_failure_with_null_tenant(
        self, auth_service, audit_emitter
    ):
        """Missing Authorization header emits auth_failure with tenant_id=None."""
        result = await auth_service.authenticate(
            {}, request_id="req-001", resource="/v1/search"
        )
        assert isinstance(result, AuthError)
        assert result.code == "missing_token"

        assert len(audit_emitter.events) == 1
        event = audit_emitter.events[0]
        assert event.action == "auth_failure"
        assert event.tenant_id is None
        assert event.actor == "anonymous"

    async def test_malformed_token_emits_auth_failure_with_null_tenant(
        self, auth_service, audit_emitter
    ):
        """Token with invalid format emits auth_failure with tenant_id=None."""
        result = await auth_service.authenticate(
            {"authorization": "Bearer nounderscore"},
            request_id="req-002",
            resource="/v1/search",
        )
        assert isinstance(result, AuthError)
        assert result.code == "invalid_token"

        assert len(audit_emitter.events) == 1
        event = audit_emitter.events[0]
        assert event.action == "auth_failure"
        assert event.tenant_id is None

    async def test_unknown_key_emits_auth_failure_with_null_tenant(
        self, auth_service, mock_db_pool, audit_emitter
    ):
        """A token that doesn't match any DB row emits auth_failure with tenant_id=None."""
        token = _make_token("unknownk")
        # DB returns no matching rows
        mock_db_pool.conn.fetch = AsyncMock(return_value=[])

        result = await auth_service.authenticate(
            {"authorization": f"Bearer {token}"},
            request_id="req-003",
            resource="/v1/search",
        )
        assert isinstance(result, AuthError)
        assert result.code == "invalid_token"

        assert len(audit_emitter.events) == 1
        event = audit_emitter.events[0]
        assert event.action == "auth_failure"
        assert event.tenant_id is None

    async def test_no_hash_match_emits_auth_failure_with_null_tenant(
        self, auth_service, mock_db_pool, audit_emitter
    ):
        """Token with valid prefix but wrong hash emits auth_failure with tenant_id=None."""
        token = _make_token("validpfx")
        other_token = _make_token("validpfx")
        row = _make_key_row(other_token)

        mock_db_pool.conn.fetch = AsyncMock(return_value=[row])

        result = await auth_service.authenticate(
            {"authorization": f"Bearer {token}"},
            request_id="req-004",
            resource="/v1/answer",
        )
        assert isinstance(result, AuthError)
        assert result.code == "invalid_token"

        assert len(audit_emitter.events) == 1
        event = audit_emitter.events[0]
        assert event.action == "auth_failure"
        assert event.tenant_id is None


# ---------------------------------------------------------------------------
# Tests: auth_failure emitted on expired token (tenant_id is the key's tenant)
# ---------------------------------------------------------------------------


class TestAuthFailureExpiredToken:
    """auth_failure audit events for expired tokens include the resolved tenant_id."""

    async def test_expired_key_emits_auth_failure_with_tenant_id(
        self, auth_service, mock_db_pool, audit_emitter
    ):
        """An expired key emits auth_failure with the key's tenant_id."""
        token = _make_token("expiredp")
        tenant_id = str(uuid.uuid4())
        past = datetime.now(timezone.utc) - timedelta(hours=1)
        row = _make_key_row(token, tenant_id=tenant_id, expires_at=past)

        mock_db_pool.conn.fetch = AsyncMock(return_value=[row])

        result = await auth_service.authenticate(
            {"authorization": f"Bearer {token}"},
            request_id="req-005",
            resource="/v1/research",
        )
        assert isinstance(result, AuthError)
        assert result.code == "expired_token"

        assert len(audit_emitter.events) == 1
        event = audit_emitter.events[0]
        assert event.action == "auth_failure"
        assert event.tenant_id == tenant_id
        assert event.detail["error_code"] == "expired_token"


# ---------------------------------------------------------------------------
# Tests: auth_failure emitted on revoked token (tenant_id is the key's tenant)
# ---------------------------------------------------------------------------


class TestAuthFailureRevokedToken:
    """auth_failure audit events for revoked tokens include the resolved tenant_id."""

    async def test_revoked_key_past_grace_emits_auth_failure_with_tenant_id(
        self, auth_service, mock_db_pool, audit_emitter
    ):
        """A revoked key past grace period emits auth_failure with the key's tenant_id."""
        token = _make_token("revokedp")
        tenant_id = str(uuid.uuid4())
        # Revoked 2 hours ago with 1 hour grace → past grace
        revoked_at = datetime.now(timezone.utc) - timedelta(hours=2)
        row = _make_key_row(
            token, tenant_id=tenant_id, revoked_at=revoked_at, rotation_grace_seconds=3600
        )

        mock_db_pool.conn.fetch = AsyncMock(return_value=[row])

        result = await auth_service.authenticate(
            {"authorization": f"Bearer {token}"},
            request_id="req-006",
            resource="/v1/sessions",
        )
        assert isinstance(result, AuthError)
        assert result.code == "revoked_token"

        assert len(audit_emitter.events) == 1
        event = audit_emitter.events[0]
        assert event.action == "auth_failure"
        assert event.tenant_id == tenant_id
        assert event.detail["error_code"] == "revoked_token"


# ---------------------------------------------------------------------------
# Tests: Token value is NEVER present in the audit detail
# ---------------------------------------------------------------------------


class TestTokenNeverLogged:
    """Bearer token value must NEVER appear in audit detail (R13.6)."""

    async def test_invalid_token_detail_does_not_contain_token_value(
        self, auth_service, mock_db_pool, audit_emitter
    ):
        """The actual token value is never in the audit detail for invalid tokens."""
        token = _make_token("leaktest")
        mock_db_pool.conn.fetch = AsyncMock(return_value=[])

        await auth_service.authenticate(
            {"authorization": f"Bearer {token}"},
            request_id="req-007",
            resource="/v1/search",
        )

        assert len(audit_emitter.events) == 1
        event = audit_emitter.events[0]
        # Serialize the entire detail to check for token leakage
        detail_str = json.dumps(event.detail)
        assert token not in detail_str
        # Also check individual fields
        assert token not in event.resource
        assert token not in event.actor
        assert token not in event.request_id

    async def test_expired_token_detail_does_not_contain_token_value(
        self, auth_service, mock_db_pool, audit_emitter
    ):
        """The actual token value is never in the audit detail for expired tokens."""
        token = _make_token("leakexp1")
        tenant_id = str(uuid.uuid4())
        past = datetime.now(timezone.utc) - timedelta(hours=1)
        row = _make_key_row(token, tenant_id=tenant_id, expires_at=past)

        mock_db_pool.conn.fetch = AsyncMock(return_value=[row])

        await auth_service.authenticate(
            {"authorization": f"Bearer {token}"},
            request_id="req-008",
            resource="/v1/search",
        )

        assert len(audit_emitter.events) == 1
        event = audit_emitter.events[0]
        detail_str = json.dumps(event.detail)
        assert token not in detail_str

    async def test_revoked_token_detail_does_not_contain_token_value(
        self, auth_service, mock_db_pool, audit_emitter
    ):
        """The actual token value is never in the audit detail for revoked tokens."""
        token = _make_token("leakrev1")
        tenant_id = str(uuid.uuid4())
        revoked_at = datetime.now(timezone.utc) - timedelta(hours=2)
        row = _make_key_row(
            token, tenant_id=tenant_id, revoked_at=revoked_at, rotation_grace_seconds=3600
        )

        mock_db_pool.conn.fetch = AsyncMock(return_value=[row])

        await auth_service.authenticate(
            {"authorization": f"Bearer {token}"},
            request_id="req-009",
            resource="/v1/search",
        )

        assert len(audit_emitter.events) == 1
        event = audit_emitter.events[0]
        detail_str = json.dumps(event.detail)
        assert token not in detail_str

    async def test_detail_only_contains_error_code(self, auth_service, mock_db_pool, audit_emitter):
        """The audit detail contains only the error_code key, nothing else sensitive."""
        token = _make_token("onlycode")
        mock_db_pool.conn.fetch = AsyncMock(return_value=[])

        await auth_service.authenticate(
            {"authorization": f"Bearer {token}"},
            request_id="req-010",
            resource="/v1/search",
        )

        assert len(audit_emitter.events) == 1
        event = audit_emitter.events[0]
        # Detail should only have error_code
        assert set(event.detail.keys()) == {"error_code"}
        assert event.detail["error_code"] == "invalid_token"


# ---------------------------------------------------------------------------
# Tests: request_id is propagated to the audit entry
# ---------------------------------------------------------------------------


class TestRequestIdPropagation:
    """request_id from the authenticate call is propagated to the audit entry."""

    async def test_request_id_propagated_on_missing_token(
        self, auth_service, audit_emitter
    ):
        """request_id is propagated when auth fails due to missing token."""
        request_id = "req-propagate-001"
        await auth_service.authenticate(
            {}, request_id=request_id, resource="/v1/search"
        )

        assert len(audit_emitter.events) == 1
        assert audit_emitter.events[0].request_id == request_id

    async def test_request_id_propagated_on_invalid_token(
        self, auth_service, mock_db_pool, audit_emitter
    ):
        """request_id is propagated when auth fails due to invalid token."""
        token = _make_token("proptest")
        mock_db_pool.conn.fetch = AsyncMock(return_value=[])
        request_id = "req-propagate-002"

        await auth_service.authenticate(
            {"authorization": f"Bearer {token}"},
            request_id=request_id,
            resource="/v1/search",
        )

        assert len(audit_emitter.events) == 1
        assert audit_emitter.events[0].request_id == request_id

    async def test_request_id_propagated_on_expired_token(
        self, auth_service, mock_db_pool, audit_emitter
    ):
        """request_id is propagated when auth fails due to expired token."""
        token = _make_token("propexp1")
        tenant_id = str(uuid.uuid4())
        past = datetime.now(timezone.utc) - timedelta(hours=1)
        row = _make_key_row(token, tenant_id=tenant_id, expires_at=past)
        mock_db_pool.conn.fetch = AsyncMock(return_value=[row])
        request_id = "req-propagate-003"

        await auth_service.authenticate(
            {"authorization": f"Bearer {token}"},
            request_id=request_id,
            resource="/v1/research",
        )

        assert len(audit_emitter.events) == 1
        assert audit_emitter.events[0].request_id == request_id

    async def test_request_id_propagated_on_revoked_token(
        self, auth_service, mock_db_pool, audit_emitter
    ):
        """request_id is propagated when auth fails due to revoked token."""
        token = _make_token("proprev1")
        tenant_id = str(uuid.uuid4())
        revoked_at = datetime.now(timezone.utc) - timedelta(hours=2)
        row = _make_key_row(
            token, tenant_id=tenant_id, revoked_at=revoked_at, rotation_grace_seconds=3600
        )
        mock_db_pool.conn.fetch = AsyncMock(return_value=[row])
        request_id = "req-propagate-004"

        await auth_service.authenticate(
            {"authorization": f"Bearer {token}"},
            request_id=request_id,
            resource="/v1/sessions",
        )

        assert len(audit_emitter.events) == 1
        assert audit_emitter.events[0].request_id == request_id

    async def test_resource_propagated_to_audit_entry(
        self, auth_service, mock_db_pool, audit_emitter
    ):
        """The resource parameter is propagated to the audit entry."""
        token = _make_token("restest1")
        mock_db_pool.conn.fetch = AsyncMock(return_value=[])

        await auth_service.authenticate(
            {"authorization": f"Bearer {token}"},
            request_id="req-011",
            resource="/v1/find_similar",
        )

        assert len(audit_emitter.events) == 1
        assert audit_emitter.events[0].resource == "/v1/find_similar"


# ---------------------------------------------------------------------------
# Tests: No audit emitted on success
# ---------------------------------------------------------------------------


class TestNoAuditOnSuccess:
    """No auth_failure audit event is emitted on successful authentication."""

    async def test_successful_auth_does_not_emit_audit(
        self, auth_service, mock_db_pool, audit_emitter
    ):
        """Successful authentication does not emit any audit event."""
        token = _make_token("successpf")
        tenant_id = str(uuid.uuid4())
        row = _make_key_row(token, tenant_id=tenant_id)
        mock_db_pool.conn.fetch = AsyncMock(return_value=[row])

        result = await auth_service.authenticate(
            {"authorization": f"Bearer {token}"},
            request_id="req-012",
            resource="/v1/search",
        )
        assert isinstance(result, AuthResult)
        assert len(audit_emitter.events) == 0


# ---------------------------------------------------------------------------
# Tests: No audit emitter configured (graceful no-op)
# ---------------------------------------------------------------------------


class TestNoAuditEmitterConfigured:
    """When no audit_emitter is provided, auth failures still work without error."""

    async def test_auth_failure_without_emitter_does_not_raise(self, mock_db_pool):
        """AuthService without audit_emitter handles failures gracefully."""
        service = AuthService(db_pool=mock_db_pool, cache_ttl_seconds=60)

        result = await service.authenticate(
            {}, request_id="req-013", resource="/v1/search"
        )
        assert isinstance(result, AuthError)
        assert result.code == "missing_token"
        # No exception raised — graceful no-op
