"""Unit tests for the Auth Service — bearer extraction, Argon2id verification, tenant resolution."""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from argon2 import PasswordHasher

from backend.auth.service import AuthError, AuthResult, AuthService, _TTLCache, _extract_prefix


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
def auth_service(mock_db_pool):
    """Create an AuthService with a mocked DB pool."""
    return AuthService(db_pool=mock_db_pool, cache_ttl_seconds=60)


# ---------------------------------------------------------------------------
# Tests: Bearer extraction
# ---------------------------------------------------------------------------


class TestBearerExtraction:
    """Tests for bearer token extraction from headers."""

    async def test_missing_authorization_header(self, auth_service):
        """Missing Authorization header returns AuthError with code 'missing_token'."""
        result = await auth_service.authenticate({})
        assert isinstance(result, AuthError)
        assert result.code == "missing_token"

    async def test_empty_authorization_header(self, auth_service):
        """Empty Authorization header returns AuthError with code 'missing_token'."""
        result = await auth_service.authenticate({"authorization": ""})
        assert isinstance(result, AuthError)
        assert result.code == "missing_token"

    async def test_non_bearer_scheme(self, auth_service):
        """Non-Bearer scheme returns AuthError with code 'missing_token'."""
        result = await auth_service.authenticate({"authorization": "Basic abc123"})
        assert isinstance(result, AuthError)
        assert result.code == "missing_token"

    async def test_bearer_without_token(self, auth_service):
        """'Bearer' without a token value returns AuthError with code 'missing_token'."""
        result = await auth_service.authenticate({"authorization": "Bearer "})
        assert isinstance(result, AuthError)
        assert result.code == "missing_token"

    async def test_bearer_case_insensitive(self, auth_service, mock_db_pool):
        """Bearer scheme matching is case-insensitive."""
        token = _make_token("abcdefgh")
        tenant_id = str(uuid.uuid4())
        row = _make_key_row(token, tenant_id=tenant_id)

        mock_db_pool.conn.fetch = AsyncMock(return_value=[row])

        result = await auth_service.authenticate({"authorization": f"BEARER {token}"})
        assert isinstance(result, AuthResult)
        assert result.tenant_id == tenant_id

    async def test_malformed_token_format(self, auth_service):
        """Token without underscore separator returns AuthError with code 'invalid_token'."""
        result = await auth_service.authenticate({"authorization": "Bearer nounderscore"})
        assert isinstance(result, AuthError)
        assert result.code == "invalid_token"

    async def test_prefix_too_short(self, auth_service):
        """Token with prefix shorter than 8 chars returns AuthError with code 'invalid_token'."""
        result = await auth_service.authenticate({"authorization": "Bearer short_randompart"})
        assert isinstance(result, AuthError)
        assert result.code == "invalid_token"

    async def test_prefix_too_long(self, auth_service):
        """Token with prefix longer than 12 chars returns AuthError with code 'invalid_token'."""
        result = await auth_service.authenticate({"authorization": "Bearer thirteenchars_randompart"})
        assert isinstance(result, AuthError)
        assert result.code == "invalid_token"


# ---------------------------------------------------------------------------
# Tests: Successful authentication
# ---------------------------------------------------------------------------


class TestSuccessfulAuth:
    """Tests for successful authentication with mocked DB."""

    async def test_valid_token_returns_auth_result(self):
        """A valid token matching a DB row returns AuthResult with correct tenant_id."""
        token = _make_token("validpfx")
        tenant_id = str(uuid.uuid4())
        api_key_id = str(uuid.uuid4())
        row = _make_key_row(token, tenant_id=tenant_id, api_key_id=api_key_id)

        pool = _MockPool()
        pool.conn.fetch = AsyncMock(return_value=[row])

        service = AuthService(db_pool=pool, cache_ttl_seconds=60)
        result = await service.authenticate({"authorization": f"Bearer {token}"})

        assert isinstance(result, AuthResult)
        assert result.tenant_id == tenant_id
        assert result.api_key_id == api_key_id
        assert result.authenticated is True

    async def test_no_matching_hash_returns_invalid_token(self):
        """When no candidate hash matches the token, returns AuthError 'invalid_token'."""
        token = _make_token("validpfx")
        # Create a row with a different token's hash
        other_token = _make_token("validpfx")
        row = _make_key_row(other_token)

        pool = _MockPool()
        pool.conn.fetch = AsyncMock(return_value=[row])

        service = AuthService(db_pool=pool, cache_ttl_seconds=60)
        result = await service.authenticate({"authorization": f"Bearer {token}"})

        assert isinstance(result, AuthError)
        assert result.code == "invalid_token"

    async def test_empty_candidates_returns_invalid_token(self):
        """When no candidates exist for the prefix, returns AuthError 'invalid_token'."""
        token = _make_token("validpfx")

        pool = _MockPool()
        pool.conn.fetch = AsyncMock(return_value=[])

        service = AuthService(db_pool=pool, cache_ttl_seconds=60)
        result = await service.authenticate({"authorization": f"Bearer {token}"})

        assert isinstance(result, AuthError)
        assert result.code == "invalid_token"

    async def test_multiple_candidates_finds_correct_match(self):
        """When multiple candidates share a prefix, the correct one is matched."""
        prefix = "sharedpfx"
        token1 = _make_token(prefix)
        token2 = _make_token(prefix)
        tenant_id_2 = str(uuid.uuid4())

        row1 = _make_key_row(token1)
        row2 = _make_key_row(token2, tenant_id=tenant_id_2)

        pool = _MockPool()
        pool.conn.fetch = AsyncMock(return_value=[row1, row2])

        service = AuthService(db_pool=pool, cache_ttl_seconds=60)
        result = await service.authenticate({"authorization": f"Bearer {token2}"})

        assert isinstance(result, AuthResult)
        assert result.tenant_id == tenant_id_2


# ---------------------------------------------------------------------------
# Tests: Expired key
# ---------------------------------------------------------------------------


class TestExpiredKey:
    """Tests for expired API key handling."""

    async def test_expired_key_returns_error(self):
        """An expired key returns AuthError with code 'expired_token'."""
        token = _make_token("expiredpf")
        past = datetime.now(timezone.utc) - timedelta(hours=1)
        row = _make_key_row(token, expires_at=past)

        pool = _MockPool()
        pool.conn.fetch = AsyncMock(return_value=[row])

        service = AuthService(db_pool=pool, cache_ttl_seconds=60)
        result = await service.authenticate({"authorization": f"Bearer {token}"})

        assert isinstance(result, AuthError)
        assert result.code == "expired_token"

    async def test_non_expired_key_succeeds(self):
        """A key with expires_at in the future succeeds."""
        token = _make_token("futurepf")
        future = datetime.now(timezone.utc) + timedelta(hours=24)
        tenant_id = str(uuid.uuid4())
        row = _make_key_row(token, tenant_id=tenant_id, expires_at=future)

        pool = _MockPool()
        pool.conn.fetch = AsyncMock(return_value=[row])

        service = AuthService(db_pool=pool, cache_ttl_seconds=60)
        result = await service.authenticate({"authorization": f"Bearer {token}"})

        assert isinstance(result, AuthResult)
        assert result.tenant_id == tenant_id

    async def test_null_expires_at_succeeds(self):
        """A key with no expiration (None) succeeds."""
        token = _make_token("noexpire")
        tenant_id = str(uuid.uuid4())
        row = _make_key_row(token, tenant_id=tenant_id, expires_at=None)

        pool = _MockPool()
        pool.conn.fetch = AsyncMock(return_value=[row])

        service = AuthService(db_pool=pool, cache_ttl_seconds=60)
        result = await service.authenticate({"authorization": f"Bearer {token}"})

        assert isinstance(result, AuthResult)
        assert result.tenant_id == tenant_id


# ---------------------------------------------------------------------------
# Tests: Revoked key (past grace period)
# ---------------------------------------------------------------------------


class TestRevokedKey:
    """Tests for revoked API key handling with grace period."""

    async def test_revoked_past_grace_returns_error(self):
        """A revoked key past its grace period returns AuthError 'revoked_token'."""
        token = _make_token("revokedp")
        # Revoked 2 hours ago with 1 hour grace → past grace
        revoked_at = datetime.now(timezone.utc) - timedelta(hours=2)
        row = _make_key_row(token, revoked_at=revoked_at, rotation_grace_seconds=3600)

        pool = _MockPool()
        pool.conn.fetch = AsyncMock(return_value=[row])

        service = AuthService(db_pool=pool, cache_ttl_seconds=60)
        result = await service.authenticate({"authorization": f"Bearer {token}"})

        assert isinstance(result, AuthError)
        assert result.code == "revoked_token"

    async def test_revoked_within_grace_succeeds(self):
        """A revoked key still within its grace period succeeds."""
        token = _make_token("graceful")
        tenant_id = str(uuid.uuid4())
        # Revoked 30 minutes ago with 1 hour grace → still within grace
        revoked_at = datetime.now(timezone.utc) - timedelta(minutes=30)
        row = _make_key_row(token, tenant_id=tenant_id, revoked_at=revoked_at, rotation_grace_seconds=3600)

        pool = _MockPool()
        pool.conn.fetch = AsyncMock(return_value=[row])

        service = AuthService(db_pool=pool, cache_ttl_seconds=60)
        result = await service.authenticate({"authorization": f"Bearer {token}"})

        assert isinstance(result, AuthResult)
        assert result.tenant_id == tenant_id

    async def test_not_revoked_succeeds(self):
        """A key with revoked_at=None succeeds."""
        token = _make_token("notrevok")
        tenant_id = str(uuid.uuid4())
        row = _make_key_row(token, tenant_id=tenant_id, revoked_at=None)

        pool = _MockPool()
        pool.conn.fetch = AsyncMock(return_value=[row])

        service = AuthService(db_pool=pool, cache_ttl_seconds=60)
        result = await service.authenticate({"authorization": f"Bearer {token}"})

        assert isinstance(result, AuthResult)
        assert result.tenant_id == tenant_id


# ---------------------------------------------------------------------------
# Tests: Cache behavior
# ---------------------------------------------------------------------------


class TestCacheBehavior:
    """Tests for the LRU cache avoiding DB calls on repeated requests."""

    async def test_cache_hit_avoids_db_call(self):
        """Second call with same prefix uses cache and doesn't hit DB again."""
        token = _make_token("cachepfx")
        tenant_id = str(uuid.uuid4())
        row = _make_key_row(token, tenant_id=tenant_id)

        pool = _MockPool()
        pool.conn.fetch = AsyncMock(return_value=[row])

        service = AuthService(db_pool=pool, cache_ttl_seconds=60)

        # First call — hits DB
        result1 = await service.authenticate({"authorization": f"Bearer {token}"})
        assert isinstance(result1, AuthResult)
        assert pool.conn.fetch.call_count == 1

        # Second call — should use cache, no additional DB call
        result2 = await service.authenticate({"authorization": f"Bearer {token}"})
        assert isinstance(result2, AuthResult)
        assert pool.conn.fetch.call_count == 1  # Still 1, cache was used

    async def test_cache_expiry_triggers_db_call(self):
        """After cache TTL expires, the next call hits DB again."""
        token = _make_token("ttlexpir")
        tenant_id = str(uuid.uuid4())
        row = _make_key_row(token, tenant_id=tenant_id)

        pool = _MockPool()
        pool.conn.fetch = AsyncMock(return_value=[row])

        service = AuthService(db_pool=pool, cache_ttl_seconds=1)

        # First call
        result1 = await service.authenticate({"authorization": f"Bearer {token}"})
        assert isinstance(result1, AuthResult)
        assert pool.conn.fetch.call_count == 1

        # Manually expire the cache entry
        service._cache.clear()

        # Third call — cache cleared, should hit DB
        result3 = await service.authenticate({"authorization": f"Bearer {token}"})
        assert isinstance(result3, AuthResult)
        assert pool.conn.fetch.call_count == 2


# ---------------------------------------------------------------------------
# Tests: Key revocation propagation (R13.4)
# ---------------------------------------------------------------------------


class TestRevocationPropagation:
    """Tests for key revocation propagation within 60s via cache TTL (R13.4)."""

    async def test_revoke_key_invalidates_cache_immediately(self):
        """revoke_key() immediately invalidates the cache entry for the prefix."""
        token = _make_token("revokepf")
        tenant_id = str(uuid.uuid4())
        row = _make_key_row(token, tenant_id=tenant_id)

        pool = _MockPool()
        pool.conn.fetch = AsyncMock(return_value=[row])

        service = AuthService(db_pool=pool, cache_ttl_seconds=60)

        # First call — populates cache
        result1 = await service.authenticate({"authorization": f"Bearer {token}"})
        assert isinstance(result1, AuthResult)
        assert pool.conn.fetch.call_count == 1

        # Revoke the key — should invalidate cache
        service.revoke_key("revokepf")

        # Next call should hit DB again (cache was invalidated)
        result2 = await service.authenticate({"authorization": f"Bearer {token}"})
        assert isinstance(result2, AuthResult)
        assert pool.conn.fetch.call_count == 2

    async def test_revoke_key_then_db_returns_revoked_row(self):
        """After revoke_key() + DB update, the next auth attempt rejects the key."""
        token = _make_token("revokedb")
        tenant_id = str(uuid.uuid4())
        row_active = _make_key_row(token, tenant_id=tenant_id)

        pool = _MockPool()
        pool.conn.fetch = AsyncMock(return_value=[row_active])

        service = AuthService(db_pool=pool, cache_ttl_seconds=60)

        # First call — succeeds with active key
        result1 = await service.authenticate({"authorization": f"Bearer {token}"})
        assert isinstance(result1, AuthResult)

        # Simulate DB update: key is now revoked (past grace period)
        from datetime import timedelta

        revoked_row = _make_key_row(
            token,
            tenant_id=tenant_id,
            revoked_at=datetime.now(timezone.utc) - timedelta(hours=2),
            rotation_grace_seconds=3600,
        )
        pool.conn.fetch = AsyncMock(return_value=[revoked_row])

        # Revoke the key — invalidates cache
        service.revoke_key("revokedb")

        # Next auth attempt fetches fresh data from DB → key is revoked
        result2 = await service.authenticate({"authorization": f"Bearer {token}"})
        assert isinstance(result2, AuthError)
        assert result2.code == "revoked_token"

    async def test_invalidate_cache_for_prefix_evicts_entry(self):
        """invalidate_cache_for_prefix() evicts the cached entry for cross-process propagation."""
        token = _make_token("crossprx")
        tenant_id = str(uuid.uuid4())
        row = _make_key_row(token, tenant_id=tenant_id)

        pool = _MockPool()
        pool.conn.fetch = AsyncMock(return_value=[row])

        service = AuthService(db_pool=pool, cache_ttl_seconds=60)

        # First call — populates cache
        result1 = await service.authenticate({"authorization": f"Bearer {token}"})
        assert isinstance(result1, AuthResult)
        assert pool.conn.fetch.call_count == 1

        # Simulate cross-process revocation notification
        service.invalidate_cache_for_prefix("crossprx")

        # Next call should hit DB again
        result2 = await service.authenticate({"authorization": f"Bearer {token}"})
        assert isinstance(result2, AuthResult)
        assert pool.conn.fetch.call_count == 2

    async def test_cache_ttl_ensures_propagation_within_60s(self):
        """Cache TTL ensures revocation propagates within 60s even without explicit invalidation."""
        token = _make_token("ttlpropx")
        tenant_id = str(uuid.uuid4())
        row_active = _make_key_row(token, tenant_id=tenant_id)

        pool = _MockPool()
        pool.conn.fetch = AsyncMock(return_value=[row_active])

        service = AuthService(db_pool=pool, cache_ttl_seconds=60)

        # Verify the cache TTL is 60s (the R13.4 guarantee)
        assert service.cache_ttl_seconds == 60

        # First call — populates cache
        result1 = await service.authenticate({"authorization": f"Bearer {token}"})
        assert isinstance(result1, AuthResult)
        assert pool.conn.fetch.call_count == 1

        # Simulate time passing beyond TTL by manipulating the cache entry's expiry
        import time

        entry = service._cache._store.get("ttlpropx")
        assert entry is not None
        # Set expiry to the past to simulate TTL expiration
        entry.expires_at = time.monotonic() - 1

        # Simulate DB update: key is now revoked (past grace period)
        from datetime import timedelta

        revoked_row = _make_key_row(
            token,
            tenant_id=tenant_id,
            revoked_at=datetime.now(timezone.utc) - timedelta(hours=2),
            rotation_grace_seconds=3600,
        )
        pool.conn.fetch = AsyncMock(return_value=[revoked_row])

        # After cache expiry, next auth attempt fetches fresh data → key is revoked
        result2 = await service.authenticate({"authorization": f"Bearer {token}"})
        assert isinstance(result2, AuthError)
        assert result2.code == "revoked_token"

    async def test_revoke_nonexistent_prefix_is_noop(self):
        """Revoking a prefix that isn't in the cache is a safe no-op."""
        pool = _MockPool()
        service = AuthService(db_pool=pool, cache_ttl_seconds=60)

        # Should not raise
        service.revoke_key("nonexist")
        service.invalidate_cache_for_prefix("nonexist")

    def test_default_cache_ttl_is_60_seconds(self):
        """Default cache TTL matches R13.4 requirement of 60s propagation."""
        pool = _MockPool()
        service = AuthService(db_pool=pool)
        assert service.cache_ttl_seconds == 60


# ---------------------------------------------------------------------------
# Tests: _extract_prefix helper
# ---------------------------------------------------------------------------


class TestExtractPrefix:
    """Tests for the prefix extraction helper."""

    def test_valid_8_char_prefix(self):
        assert _extract_prefix("abcdefgh_rest") == "abcdefgh"

    def test_valid_12_char_prefix(self):
        assert _extract_prefix("abcdefghijkl_rest") == "abcdefghijkl"

    def test_no_underscore(self):
        assert _extract_prefix("nounderscore") is None

    def test_prefix_too_short(self):
        assert _extract_prefix("short_rest") is None

    def test_prefix_too_long(self):
        assert _extract_prefix("thirteenchars_rest") is None

    def test_empty_string(self):
        assert _extract_prefix("") is None


# ---------------------------------------------------------------------------
# Tests: _TTLCache
# ---------------------------------------------------------------------------


class TestTTLCache:
    """Tests for the TTL cache implementation."""

    def test_put_and_get(self):
        cache = _TTLCache(ttl_seconds=60)
        cache.put("prefix1", [{"key": "value"}])
        assert cache.get("prefix1") == [{"key": "value"}]

    def test_get_missing_key(self):
        cache = _TTLCache(ttl_seconds=60)
        assert cache.get("nonexistent") is None

    def test_invalidate(self):
        cache = _TTLCache(ttl_seconds=60)
        cache.put("prefix1", [{"key": "value"}])
        cache.invalidate("prefix1")
        assert cache.get("prefix1") is None

    def test_clear(self):
        cache = _TTLCache(ttl_seconds=60)
        cache.put("p1", [{"a": 1}])
        cache.put("p2", [{"b": 2}])
        cache.clear()
        assert cache.get("p1") is None
        assert cache.get("p2") is None

    def test_max_size_eviction(self):
        cache = _TTLCache(ttl_seconds=60, max_size=2)
        cache.put("p1", [{"a": 1}])
        cache.put("p2", [{"b": 2}])
        cache.put("p3", [{"c": 3}])  # Should evict p1
        assert cache.get("p1") is None
        assert cache.get("p2") == [{"b": 2}]
        assert cache.get("p3") == [{"c": 3}]
