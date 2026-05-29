"""Unit tests for key rotation grace period logic (R13.5).

Tests verify:
- During grace window [T, T+G]: both old and new keys authenticate successfully
- After grace window (T+G): old key is rejected, new key still works
- rotation_grace_seconds validation at application layer [1, 86400]
- rotate_key() correctly revokes old key and creates new key
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from argon2 import PasswordHasher

from backend.auth.key_management import (
    ROTATION_GRACE_DEFAULT,
    ROTATION_GRACE_MAX,
    ROTATION_GRACE_MIN,
    KeyManagementService,
    KeyNotFoundError,
    RotationResult,
    ValidationError,
    _generate_api_key,
    validate_rotation_grace_seconds,
)
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
        self.conn.execute = AsyncMock(return_value="UPDATE 1")
        self.conn.transaction = MagicMock()
        # Make transaction() return an async context manager
        self.conn.transaction.return_value = _MockTransaction()

    @asynccontextmanager
    async def acquire(self):
        yield self.conn


class _MockTransaction:
    """Mock asyncpg transaction context manager."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


@pytest.fixture
def mock_db_pool():
    """Create a mock asyncpg pool."""
    return _MockPool()


@pytest.fixture
def auth_service(mock_db_pool):
    """Create an AuthService with a mocked DB pool."""
    return AuthService(db_pool=mock_db_pool, cache_ttl_seconds=60)


@pytest.fixture
def key_mgmt_service(mock_db_pool, auth_service):
    """Create a KeyManagementService with mocked DB pool and auth service."""
    return KeyManagementService(db_pool=mock_db_pool, auth_service=auth_service)


# ---------------------------------------------------------------------------
# Tests: rotation_grace_seconds validation
# ---------------------------------------------------------------------------


class TestRotationGraceValidation:
    """Tests for rotation_grace_seconds validation at application layer."""

    def test_valid_minimum_value(self):
        """Minimum value (1) is accepted."""
        assert validate_rotation_grace_seconds(1) == 1

    def test_valid_maximum_value(self):
        """Maximum value (86400) is accepted."""
        assert validate_rotation_grace_seconds(86400) == 86400

    def test_valid_default_value(self):
        """Default value (3600) is accepted."""
        assert validate_rotation_grace_seconds(3600) == 3600

    def test_valid_mid_range(self):
        """Mid-range values are accepted."""
        assert validate_rotation_grace_seconds(43200) == 43200

    def test_zero_rejected(self):
        """Zero is below minimum and rejected."""
        with pytest.raises(ValidationError) as exc_info:
            validate_rotation_grace_seconds(0)
        assert exc_info.value.code == "invalid_rotation_grace_seconds"

    def test_negative_rejected(self):
        """Negative values are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            validate_rotation_grace_seconds(-1)
        assert exc_info.value.code == "invalid_rotation_grace_seconds"

    def test_above_maximum_rejected(self):
        """Values above 86400 are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            validate_rotation_grace_seconds(86401)
        assert exc_info.value.code == "invalid_rotation_grace_seconds"

    def test_large_value_rejected(self):
        """Very large values are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            validate_rotation_grace_seconds(1_000_000)
        assert exc_info.value.code == "invalid_rotation_grace_seconds"

    def test_non_integer_rejected(self):
        """Non-integer types are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            validate_rotation_grace_seconds(3600.5)  # type: ignore
        assert exc_info.value.code == "invalid_rotation_grace_seconds"

    def test_constants_are_correct(self):
        """Module constants match R13.5 specification."""
        assert ROTATION_GRACE_MIN == 1
        assert ROTATION_GRACE_MAX == 86400
        assert ROTATION_GRACE_DEFAULT == 3600


# ---------------------------------------------------------------------------
# Tests: Grace window — both old and new keys authenticate during [T, T+G]
# ---------------------------------------------------------------------------


class TestGraceWindowBothKeysWork:
    """During grace window [T, T+G], both old and new keys authenticate successfully."""

    async def test_old_key_authenticates_within_grace_window(self):
        """Old key (revoked 10 minutes ago, grace=3600s) still authenticates."""
        old_token = _make_token("oldkeypr")
        tenant_id = str(uuid.uuid4())
        # Revoked 10 minutes ago with 1 hour grace → within grace window
        revoked_at = datetime.now(timezone.utc) - timedelta(minutes=10)
        old_row = _make_key_row(
            old_token,
            tenant_id=tenant_id,
            revoked_at=revoked_at,
            rotation_grace_seconds=3600,
        )

        pool = _MockPool()
        pool.conn.fetch = AsyncMock(return_value=[old_row])

        service = AuthService(db_pool=pool, cache_ttl_seconds=60)
        result = await service.authenticate({"authorization": f"Bearer {old_token}"})

        assert isinstance(result, AuthResult)
        assert result.tenant_id == tenant_id

    async def test_new_key_authenticates_immediately_after_rotation(self):
        """New key (not revoked) authenticates immediately after creation."""
        new_token = _make_token("newkeypr")
        tenant_id = str(uuid.uuid4())
        # New key has no revoked_at
        new_row = _make_key_row(
            new_token,
            tenant_id=tenant_id,
            revoked_at=None,
            rotation_grace_seconds=3600,
        )

        pool = _MockPool()
        pool.conn.fetch = AsyncMock(return_value=[new_row])

        service = AuthService(db_pool=pool, cache_ttl_seconds=60)
        result = await service.authenticate({"authorization": f"Bearer {new_token}"})

        assert isinstance(result, AuthResult)
        assert result.tenant_id == tenant_id

    async def test_both_keys_work_simultaneously_during_grace(self):
        """Both old (revoked within grace) and new keys authenticate for the same tenant."""
        tenant_id = str(uuid.uuid4())

        # Old key — revoked 5 minutes ago, grace = 3600s
        old_token = _make_token("oldrotpf")
        revoked_at = datetime.now(timezone.utc) - timedelta(minutes=5)
        old_row = _make_key_row(
            old_token,
            tenant_id=tenant_id,
            revoked_at=revoked_at,
            rotation_grace_seconds=3600,
        )

        # New key — active, not revoked
        new_token = _make_token("newrotpf")
        new_row = _make_key_row(
            new_token,
            tenant_id=tenant_id,
            revoked_at=None,
            rotation_grace_seconds=3600,
        )

        pool = _MockPool()

        # Simulate: old key prefix returns old_row, new key prefix returns new_row
        service = AuthService(db_pool=pool, cache_ttl_seconds=60)

        # Authenticate with old key
        pool.conn.fetch = AsyncMock(return_value=[old_row])
        result_old = await service.authenticate({"authorization": f"Bearer {old_token}"})
        assert isinstance(result_old, AuthResult)
        assert result_old.tenant_id == tenant_id

        # Clear cache to simulate fresh lookup for new key
        service._cache.clear()

        # Authenticate with new key
        pool.conn.fetch = AsyncMock(return_value=[new_row])
        result_new = await service.authenticate({"authorization": f"Bearer {new_token}"})
        assert isinstance(result_new, AuthResult)
        assert result_new.tenant_id == tenant_id

    async def test_old_key_works_at_grace_boundary(self):
        """Old key authenticates when exactly at the grace boundary (revoked_at + grace - 1s)."""
        old_token = _make_token("boundary")
        tenant_id = str(uuid.uuid4())
        # Revoked exactly 3599 seconds ago with 3600s grace → just within window
        revoked_at = datetime.now(timezone.utc) - timedelta(seconds=3599)
        old_row = _make_key_row(
            old_token,
            tenant_id=tenant_id,
            revoked_at=revoked_at,
            rotation_grace_seconds=3600,
        )

        pool = _MockPool()
        pool.conn.fetch = AsyncMock(return_value=[old_row])

        service = AuthService(db_pool=pool, cache_ttl_seconds=60)
        result = await service.authenticate({"authorization": f"Bearer {old_token}"})

        assert isinstance(result, AuthResult)
        assert result.tenant_id == tenant_id

    async def test_grace_window_with_minimum_grace_period(self):
        """Old key works within a 1-second grace window (minimum allowed)."""
        old_token = _make_token("mingrace")
        tenant_id = str(uuid.uuid4())
        # Revoked just now with 1s grace → within window
        revoked_at = datetime.now(timezone.utc)
        old_row = _make_key_row(
            old_token,
            tenant_id=tenant_id,
            revoked_at=revoked_at,
            rotation_grace_seconds=1,
        )

        pool = _MockPool()
        pool.conn.fetch = AsyncMock(return_value=[old_row])

        service = AuthService(db_pool=pool, cache_ttl_seconds=60)
        result = await service.authenticate({"authorization": f"Bearer {old_token}"})

        assert isinstance(result, AuthResult)
        assert result.tenant_id == tenant_id

    async def test_grace_window_with_maximum_grace_period(self):
        """Old key works within a 86400-second (24h) grace window (maximum allowed)."""
        old_token = _make_token("maxgrace")
        tenant_id = str(uuid.uuid4())
        # Revoked 12 hours ago with 24h grace → within window
        revoked_at = datetime.now(timezone.utc) - timedelta(hours=12)
        old_row = _make_key_row(
            old_token,
            tenant_id=tenant_id,
            revoked_at=revoked_at,
            rotation_grace_seconds=86400,
        )

        pool = _MockPool()
        pool.conn.fetch = AsyncMock(return_value=[old_row])

        service = AuthService(db_pool=pool, cache_ttl_seconds=60)
        result = await service.authenticate({"authorization": f"Bearer {old_token}"})

        assert isinstance(result, AuthResult)
        assert result.tenant_id == tenant_id


# ---------------------------------------------------------------------------
# Tests: After grace window — old key rejected, new key works
# ---------------------------------------------------------------------------


class TestAfterGraceWindow:
    """After grace window (T+G): old key is rejected, new key still works."""

    async def test_old_key_rejected_after_grace_window(self):
        """Old key is rejected when now > revoked_at + rotation_grace_seconds."""
        old_token = _make_token("expired1")
        tenant_id = str(uuid.uuid4())
        # Revoked 2 hours ago with 1 hour grace → past grace window
        revoked_at = datetime.now(timezone.utc) - timedelta(hours=2)
        old_row = _make_key_row(
            old_token,
            tenant_id=tenant_id,
            revoked_at=revoked_at,
            rotation_grace_seconds=3600,
        )

        pool = _MockPool()
        pool.conn.fetch = AsyncMock(return_value=[old_row])

        service = AuthService(db_pool=pool, cache_ttl_seconds=60)
        result = await service.authenticate({"authorization": f"Bearer {old_token}"})

        assert isinstance(result, AuthError)
        assert result.code == "revoked_token"

    async def test_new_key_still_works_after_old_grace_expires(self):
        """New key (not revoked) continues to work after old key's grace expires."""
        new_token = _make_token("newstill")
        tenant_id = str(uuid.uuid4())
        new_row = _make_key_row(
            new_token,
            tenant_id=tenant_id,
            revoked_at=None,
            rotation_grace_seconds=3600,
        )

        pool = _MockPool()
        pool.conn.fetch = AsyncMock(return_value=[new_row])

        service = AuthService(db_pool=pool, cache_ttl_seconds=60)
        result = await service.authenticate({"authorization": f"Bearer {new_token}"})

        assert isinstance(result, AuthResult)
        assert result.tenant_id == tenant_id

    async def test_old_key_rejected_just_past_grace_boundary(self):
        """Old key is rejected when just past the grace boundary (revoked_at + grace + 1s)."""
        old_token = _make_token("justpast")
        tenant_id = str(uuid.uuid4())
        # Revoked 3601 seconds ago with 3600s grace → just past window
        revoked_at = datetime.now(timezone.utc) - timedelta(seconds=3601)
        old_row = _make_key_row(
            old_token,
            tenant_id=tenant_id,
            revoked_at=revoked_at,
            rotation_grace_seconds=3600,
        )

        pool = _MockPool()
        pool.conn.fetch = AsyncMock(return_value=[old_row])

        service = AuthService(db_pool=pool, cache_ttl_seconds=60)
        result = await service.authenticate({"authorization": f"Bearer {old_token}"})

        assert isinstance(result, AuthError)
        assert result.code == "revoked_token"

    async def test_old_key_rejected_with_minimum_grace_expired(self):
        """Old key rejected when 1-second grace period has passed."""
        old_token = _make_token("minexpir")
        tenant_id = str(uuid.uuid4())
        # Revoked 2 seconds ago with 1s grace → past window
        revoked_at = datetime.now(timezone.utc) - timedelta(seconds=2)
        old_row = _make_key_row(
            old_token,
            tenant_id=tenant_id,
            revoked_at=revoked_at,
            rotation_grace_seconds=1,
        )

        pool = _MockPool()
        pool.conn.fetch = AsyncMock(return_value=[old_row])

        service = AuthService(db_pool=pool, cache_ttl_seconds=60)
        result = await service.authenticate({"authorization": f"Bearer {old_token}"})

        assert isinstance(result, AuthError)
        assert result.code == "revoked_token"

    async def test_old_key_rejected_with_maximum_grace_expired(self):
        """Old key rejected when 86400-second (24h) grace period has passed."""
        old_token = _make_token("maxexpir")
        tenant_id = str(uuid.uuid4())
        # Revoked 25 hours ago with 24h grace → past window
        revoked_at = datetime.now(timezone.utc) - timedelta(hours=25)
        old_row = _make_key_row(
            old_token,
            tenant_id=tenant_id,
            revoked_at=revoked_at,
            rotation_grace_seconds=86400,
        )

        pool = _MockPool()
        pool.conn.fetch = AsyncMock(return_value=[old_row])

        service = AuthService(db_pool=pool, cache_ttl_seconds=60)
        result = await service.authenticate({"authorization": f"Bearer {old_token}"})

        assert isinstance(result, AuthError)
        assert result.code == "revoked_token"

    async def test_full_rotation_scenario(self):
        """Full scenario: rotate, both work during grace, only new works after."""
        tenant_id = str(uuid.uuid4())

        # Old key — revoked 30 minutes ago, grace = 3600s (within window)
        old_token = _make_token("fullold1")
        revoked_at_within = datetime.now(timezone.utc) - timedelta(minutes=30)
        old_row_within_grace = _make_key_row(
            old_token,
            tenant_id=tenant_id,
            revoked_at=revoked_at_within,
            rotation_grace_seconds=3600,
        )

        # New key — active
        new_token = _make_token("fullnew1")
        new_row = _make_key_row(
            new_token,
            tenant_id=tenant_id,
            revoked_at=None,
            rotation_grace_seconds=3600,
        )

        pool = _MockPool()
        service = AuthService(db_pool=pool, cache_ttl_seconds=60)

        # Phase 1: During grace window — both keys work
        pool.conn.fetch = AsyncMock(return_value=[old_row_within_grace])
        result_old = await service.authenticate({"authorization": f"Bearer {old_token}"})
        assert isinstance(result_old, AuthResult), f"Expected AuthResult, got {result_old}"
        assert result_old.tenant_id == tenant_id

        service._cache.clear()

        pool.conn.fetch = AsyncMock(return_value=[new_row])
        result_new = await service.authenticate({"authorization": f"Bearer {new_token}"})
        assert isinstance(result_new, AuthResult)
        assert result_new.tenant_id == tenant_id

        service._cache.clear()

        # Phase 2: After grace window — old key rejected, new key works
        revoked_at_expired = datetime.now(timezone.utc) - timedelta(hours=2)
        old_row_past_grace = _make_key_row(
            old_token,
            tenant_id=tenant_id,
            revoked_at=revoked_at_expired,
            rotation_grace_seconds=3600,
        )

        pool.conn.fetch = AsyncMock(return_value=[old_row_past_grace])
        result_old_expired = await service.authenticate({"authorization": f"Bearer {old_token}"})
        assert isinstance(result_old_expired, AuthError)
        assert result_old_expired.code == "revoked_token"

        service._cache.clear()

        pool.conn.fetch = AsyncMock(return_value=[new_row])
        result_new_still = await service.authenticate({"authorization": f"Bearer {new_token}"})
        assert isinstance(result_new_still, AuthResult)
        assert result_new_still.tenant_id == tenant_id


# ---------------------------------------------------------------------------
# Tests: rotate_key() method
# ---------------------------------------------------------------------------


class TestRotateKey:
    """Tests for the KeyManagementService.rotate_key() method."""

    async def test_rotate_key_returns_rotation_result(self, key_mgmt_service, mock_db_pool):
        """rotate_key() returns a RotationResult with new key details."""
        tenant_id = str(uuid.uuid4())
        old_prefix = "oldprefix"

        result = await key_mgmt_service.rotate_key(
            tenant_id=tenant_id,
            old_key_prefix=old_prefix,
            rotation_grace_seconds=3600,
        )

        assert isinstance(result, RotationResult)
        assert result.new_api_key_id is not None
        assert result.new_key_plaintext is not None
        assert result.new_key_prefix is not None
        assert result.rotation_grace_seconds == 3600
        assert result.old_key_revoked_at is not None

    async def test_rotate_key_new_key_format(self, key_mgmt_service, mock_db_pool):
        """New key follows the {prefix}_{random_part} format."""
        tenant_id = str(uuid.uuid4())

        result = await key_mgmt_service.rotate_key(
            tenant_id=tenant_id,
            old_key_prefix="oldprefix",
        )

        # Key should have format prefix_randompart
        parts = result.new_key_plaintext.split("_", 1)
        assert len(parts) == 2
        assert parts[0] == result.new_key_prefix
        assert len(parts[1]) > 0

    async def test_rotate_key_invalidates_old_cache(self, mock_db_pool, auth_service):
        """rotate_key() invalidates the cache for the old key's prefix."""
        key_mgmt = KeyManagementService(db_pool=mock_db_pool, auth_service=auth_service)
        tenant_id = str(uuid.uuid4())
        old_prefix = "cacheold"

        # Pre-populate cache
        auth_service._cache.put(old_prefix, [{"some": "data"}])
        assert auth_service._cache.get(old_prefix) is not None

        await key_mgmt.rotate_key(
            tenant_id=tenant_id,
            old_key_prefix=old_prefix,
        )

        # Cache should be invalidated
        assert auth_service._cache.get(old_prefix) is None

    async def test_rotate_key_invalid_grace_seconds(self, key_mgmt_service):
        """rotate_key() raises ValidationError for invalid grace seconds."""
        tenant_id = str(uuid.uuid4())

        with pytest.raises(ValidationError) as exc_info:
            await key_mgmt_service.rotate_key(
                tenant_id=tenant_id,
                old_key_prefix="someprefix",
                rotation_grace_seconds=0,
            )
        assert exc_info.value.code == "invalid_rotation_grace_seconds"

    async def test_rotate_key_not_found(self, mock_db_pool, auth_service):
        """rotate_key() raises KeyNotFoundError when no active key matches."""
        mock_db_pool.conn.execute = AsyncMock(return_value="UPDATE 0")
        key_mgmt = KeyManagementService(db_pool=mock_db_pool, auth_service=auth_service)

        with pytest.raises(KeyNotFoundError):
            await key_mgmt.rotate_key(
                tenant_id=str(uuid.uuid4()),
                old_key_prefix="nonexist",
            )

    async def test_rotate_key_uses_default_grace(self, key_mgmt_service):
        """rotate_key() uses default grace period of 3600 when not specified."""
        tenant_id = str(uuid.uuid4())

        result = await key_mgmt_service.rotate_key(
            tenant_id=tenant_id,
            old_key_prefix="defgrace",
        )

        assert result.rotation_grace_seconds == ROTATION_GRACE_DEFAULT

    async def test_rotate_key_with_custom_grace(self, key_mgmt_service):
        """rotate_key() accepts custom grace period within valid range."""
        tenant_id = str(uuid.uuid4())

        result = await key_mgmt_service.rotate_key(
            tenant_id=tenant_id,
            old_key_prefix="custgrac",
            rotation_grace_seconds=7200,
        )

        assert result.rotation_grace_seconds == 7200


# ---------------------------------------------------------------------------
# Tests: _generate_api_key helper
# ---------------------------------------------------------------------------


class TestGenerateApiKey:
    """Tests for the API key generation helper."""

    def test_key_format(self):
        """Generated key follows {prefix}_{random_part} format."""
        key, prefix = _generate_api_key()
        assert "_" in key
        parts = key.split("_", 1)
        assert parts[0] == prefix
        assert len(parts[1]) > 0

    def test_prefix_length(self):
        """Generated prefix is 8 characters."""
        _, prefix = _generate_api_key()
        assert len(prefix) == 8

    def test_keys_are_unique(self):
        """Each generated key is unique."""
        keys = {_generate_api_key()[0] for _ in range(100)}
        assert len(keys) == 100

    def test_key_is_hashable_with_argon2(self):
        """Generated key can be hashed and verified with Argon2id."""
        key, _ = _generate_api_key()
        hasher = PasswordHasher()
        hashed = hasher.hash(key)
        assert hasher.verify(hashed, key) is True
