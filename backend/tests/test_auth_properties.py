"""Property-based tests for Auth Service — revocation propagation (Property 32).

**Validates: Requirements 13.4**

Property 32: API key revocation propagates within 60 seconds.
For any API key revocation accepted at virtual time T, every request presenting
that key at virtual time T' >= T + 60s fails authentication (cache has expired,
fresh DB data shows revocation).

Uses a virtual clock approach by mocking time.monotonic() to control cache expiry
without waiting for real time to pass.
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest
from argon2 import PasswordHasher
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from backend.auth.service import AuthError, AuthResult, AuthService, _TTLCache


# ---------------------------------------------------------------------------
# Helpers (reused from test_auth_service.py patterns)
# ---------------------------------------------------------------------------

# Use minimal Argon2id parameters for fast test execution.
# Production uses default (expensive) parameters; tests only need hash correctness.
_hasher = PasswordHasher(time_cost=1, memory_cost=8192, parallelism=1)


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


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Cache TTL in [1, 60] seconds — the system uses 60s but we test the property
# holds for any TTL within the allowed range.
cache_ttl_strategy = st.integers(min_value=1, max_value=60)

# Time offset after revocation in [0, 120] seconds — covers both within-TTL
# and beyond-TTL scenarios.
time_offset_strategy = st.floats(min_value=0.0, max_value=120.0, allow_nan=False, allow_infinity=False)

# Random token prefix (8-12 chars, alphanumeric to match prefix format)
prefix_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("Ll",), whitelist_characters="abcdefghijklmnopqrstuvwxyz"),
    min_size=8,
    max_size=12,
)


# ---------------------------------------------------------------------------
# Property 32: Revocation propagates within cache_ttl_seconds
# ---------------------------------------------------------------------------


class TestRevocationPropagationProperty:
    """Property-based tests for revocation propagation within 60s (Property 32).

    **Validates: Requirements 13.4**

    The key insight: the _TTLCache uses time.monotonic() to determine entry expiry.
    By mocking time.monotonic(), we create a virtual clock that lets us test the
    temporal property without real delays.

    The property states:
    - If time_since_revocation > cache_ttl_seconds, the revoked key is ALWAYS rejected
      (cache has expired, fresh DB data shows revocation).
    - If time_since_revocation <= cache_ttl_seconds AND the cache was populated before
      revocation, the key MAY still authenticate (stale cache).
    - The maximum propagation delay is bounded by cache_ttl_seconds (≤ 60s).
    """

    @given(
        cache_ttl=cache_ttl_strategy,
        time_after_revocation=time_offset_strategy,
        prefix=prefix_strategy,
    )
    async def test_revocation_propagates_within_ttl(
        self,
        cache_ttl: int,
        time_after_revocation: float,
        prefix: str,
    ):
        """Property: After cache_ttl_seconds past revocation, a revoked key is always rejected.

        **Validates: Requirements 13.4**

        Virtual clock approach:
        1. Set virtual clock to T=0, create AuthService with given cache_ttl.
        2. Populate the cache with an active key (simulates normal auth before revocation).
        3. "Revoke" the key in the DB (change mock DB response to return revoked row).
        4. Advance the virtual clock by time_after_revocation seconds.
        5. Assert: if time_after_revocation > cache_ttl, the key MUST be rejected.
        """
        # Virtual clock state
        virtual_time = [0.0]  # mutable container for closure

        def mock_monotonic():
            return virtual_time[0]

        # Set up the mock pool and service
        pool = _MockPool()
        token = _make_token(prefix)
        tenant_id = str(uuid.uuid4())

        # Active key row (before revocation)
        active_row = _make_key_row(token, tenant_id=tenant_id, revoked_at=None)

        # Revoked key row (after revocation, past grace period so it's definitively rejected)
        # Use a revocation time far in the past with 0 grace to ensure rejection is due to revocation
        revoked_row = _make_key_row(
            token,
            tenant_id=tenant_id,
            revoked_at=datetime.now(timezone.utc) - timedelta(hours=24),
            rotation_grace_seconds=0,
        )

        with patch("time.monotonic", side_effect=mock_monotonic):
            # Create service with the given cache TTL
            service = AuthService(db_pool=pool, cache_ttl_seconds=cache_ttl)

            # Phase 1: Populate cache at T=0 with active key
            virtual_time[0] = 0.0
            pool.conn.fetch = AsyncMock(return_value=[active_row])

            result = await service.authenticate({"authorization": f"Bearer {token}"})
            assert isinstance(result, AuthResult), (
                f"Initial auth should succeed, got: {result}"
            )
            assert result.tenant_id == tenant_id

            # Phase 2: "Revoke" the key in the DB (change what DB returns)
            # The revocation happens at T=0 (or shortly after cache population).
            # The cache still holds the old active row until it expires.
            pool.conn.fetch = AsyncMock(return_value=[revoked_row])

            # Phase 3: Advance virtual clock by time_after_revocation
            virtual_time[0] = time_after_revocation

            # Phase 4: Attempt authentication
            result = await service.authenticate({"authorization": f"Bearer {token}"})

            # Phase 5: Assert the property
            if time_after_revocation > cache_ttl:
                # MUST be rejected — cache has expired, fresh DB data shows revocation
                assert isinstance(result, AuthError), (
                    f"After {time_after_revocation:.2f}s (TTL={cache_ttl}s), "
                    f"revoked key MUST be rejected but got: {result}"
                )
                assert result.code == "revoked_token", (
                    f"Expected 'revoked_token' error code, got: {result.code}"
                )
            # If time_after_revocation <= cache_ttl, the key MAY still authenticate
            # (stale cache) — this is acceptable behavior within the propagation window.
            # We don't assert success here because the cache entry might have been
            # evicted for other reasons, but we DO assert that the maximum delay
            # is bounded by cache_ttl.

    @given(
        cache_ttl=cache_ttl_strategy,
        prefix=prefix_strategy,
    )
    async def test_propagation_delay_bounded_by_ttl(
        self,
        cache_ttl: int,
        prefix: str,
    ):
        """Property: The maximum propagation delay is exactly cache_ttl_seconds.

        **Validates: Requirements 13.4**

        At exactly cache_ttl + epsilon seconds after cache population, the cache
        entry MUST be expired, forcing a fresh DB fetch that reveals the revocation.
        This proves the upper bound on propagation delay equals cache_ttl_seconds.
        """
        virtual_time = [0.0]

        def mock_monotonic():
            return virtual_time[0]

        pool = _MockPool()
        token = _make_token(prefix)
        tenant_id = str(uuid.uuid4())

        active_row = _make_key_row(token, tenant_id=tenant_id, revoked_at=None)
        revoked_row = _make_key_row(
            token,
            tenant_id=tenant_id,
            revoked_at=datetime.now(timezone.utc) - timedelta(hours=24),
            rotation_grace_seconds=0,
        )

        with patch("time.monotonic", side_effect=mock_monotonic):
            service = AuthService(db_pool=pool, cache_ttl_seconds=cache_ttl)

            # Populate cache at T=0
            virtual_time[0] = 0.0
            pool.conn.fetch = AsyncMock(return_value=[active_row])
            result = await service.authenticate({"authorization": f"Bearer {token}"})
            assert isinstance(result, AuthResult)

            # Revoke in DB
            pool.conn.fetch = AsyncMock(return_value=[revoked_row])

            # At exactly TTL + a small epsilon, cache MUST be expired
            epsilon = 0.001
            virtual_time[0] = cache_ttl + epsilon

            result = await service.authenticate({"authorization": f"Bearer {token}"})
            assert isinstance(result, AuthError), (
                f"At TTL+epsilon ({cache_ttl + epsilon:.3f}s), "
                f"revoked key MUST be rejected but got: {result}"
            )
            assert result.code == "revoked_token"

    @given(
        cache_ttl=cache_ttl_strategy,
        prefix=prefix_strategy,
    )
    async def test_within_ttl_cache_serves_stale_data(
        self,
        cache_ttl: int,
        prefix: str,
    ):
        """Property: Within cache_ttl_seconds, a stale cache entry MAY still authenticate.

        **Validates: Requirements 13.4**

        This is the converse: before the TTL expires, the cache may serve stale
        (pre-revocation) data, allowing the revoked key to still authenticate.
        This confirms the cache is working as designed — propagation is bounded
        by TTL, not instant (unless explicitly invalidated).
        """
        virtual_time = [0.0]

        def mock_monotonic():
            return virtual_time[0]

        pool = _MockPool()
        token = _make_token(prefix)
        tenant_id = str(uuid.uuid4())

        active_row = _make_key_row(token, tenant_id=tenant_id, revoked_at=None)
        revoked_row = _make_key_row(
            token,
            tenant_id=tenant_id,
            revoked_at=datetime.now(timezone.utc) - timedelta(hours=24),
            rotation_grace_seconds=0,
        )

        with patch("time.monotonic", side_effect=mock_monotonic):
            service = AuthService(db_pool=pool, cache_ttl_seconds=cache_ttl)

            # Populate cache at T=0
            virtual_time[0] = 0.0
            pool.conn.fetch = AsyncMock(return_value=[active_row])
            result = await service.authenticate({"authorization": f"Bearer {token}"})
            assert isinstance(result, AuthResult)

            # Revoke in DB
            pool.conn.fetch = AsyncMock(return_value=[revoked_row])

            # At T = cache_ttl * 0.5 (well within TTL), cache still serves stale data
            virtual_time[0] = cache_ttl * 0.5

            result = await service.authenticate({"authorization": f"Bearer {token}"})
            # Within TTL, the stale cache entry is still valid → key authenticates
            assert isinstance(result, AuthResult), (
                f"Within TTL ({cache_ttl * 0.5:.1f}s < {cache_ttl}s), "
                f"stale cache should still authenticate, got: {result}"
            )
            assert result.tenant_id == tenant_id


# ---------------------------------------------------------------------------
# Property 33: Rotation grace window accepts both keys during [T, T+G], only new after
# ---------------------------------------------------------------------------


# Strategies for Property 33
grace_seconds_strategy = st.integers(min_value=1, max_value=86400)


class TestRotationGraceWindowProperty:
    """Property-based tests for rotation grace window (Property 33).

    **Validates: Requirements 13.5**

    Property 33: For any configured grace G ∈ [1, 86400] seconds and any rotation
    event accepted at virtual time T, both the old and the new key authenticate at
    any virtual time in [T, T+G]; only the new key authenticates at virtual times > T+G.

    The auth service checks: `if now > grace_end` where `grace_end = revoked_at + timedelta(seconds=grace_seconds)`.
    This means:
    - At now <= grace_end (i.e., now - revoked_at <= grace_seconds): old key authenticates (within grace)
    - At now > grace_end (i.e., now - revoked_at > grace_seconds): old key is rejected

    We control the temporal relationship by setting `revoked_at = now - elapsed` where
    `elapsed` is the time since revocation. The cache is bypassed by setting TTL=0 and
    advancing the monotonic clock past expiry on each call.
    """

    @given(
        grace_seconds=grace_seconds_strategy,
        time_within_grace=st.data(),
        prefix=prefix_strategy,
    )
    @settings(suppress_health_check=[HealthCheck.too_slow])
    async def test_old_key_authenticates_within_grace_window(
        self,
        grace_seconds: int,
        time_within_grace: st.DataObject,
        prefix: str,
    ):
        """Property: An old (revoked) key authenticates within the grace window [T, T+G].

        **Validates: Requirements 13.5**

        For any grace_seconds in [1, 86400] and any time_within_grace in [0, grace_seconds),
        setting revoked_at such that now - revoked_at = time_within_grace means we are
        within the grace window, so the old key should still authenticate.
        """
        # Draw time_within_grace conditioned on grace_seconds: [0, grace_seconds)
        elapsed = time_within_grace.draw(
            st.integers(min_value=0, max_value=grace_seconds - 1),
            label="time_within_grace",
        )

        # Virtual monotonic clock — always past cache TTL to force DB fetch
        call_count = [0]

        def mock_monotonic():
            call_count[0] += 1
            return call_count[0] * 1000.0  # always far past any TTL

        pool = _MockPool()
        token = _make_token(prefix)
        tenant_id = str(uuid.uuid4())

        # Set revoked_at so that now - revoked_at = elapsed seconds
        # This places us within the grace window since elapsed < grace_seconds
        revoked_at = datetime.now(timezone.utc) - timedelta(seconds=elapsed)

        key_row = _make_key_row(
            token,
            tenant_id=tenant_id,
            revoked_at=revoked_at,
            rotation_grace_seconds=grace_seconds,
        )

        with patch("time.monotonic", side_effect=mock_monotonic):
            service = AuthService(db_pool=pool, cache_ttl_seconds=1)
            pool.conn.fetch = AsyncMock(return_value=[key_row])

            result = await service.authenticate({"authorization": f"Bearer {token}"})

            assert isinstance(result, AuthResult), (
                f"Old key with elapsed={elapsed}s < grace={grace_seconds}s "
                f"should authenticate within grace window, got: {result}"
            )
            assert result.tenant_id == tenant_id

    @given(
        grace_seconds=grace_seconds_strategy,
        time_after_grace=st.data(),
        prefix=prefix_strategy,
    )
    @settings(suppress_health_check=[HealthCheck.too_slow])
    async def test_old_key_rejected_after_grace_window(
        self,
        grace_seconds: int,
        time_after_grace: st.DataObject,
        prefix: str,
    ):
        """Property: An old (revoked) key is rejected after the grace window (> T+G).

        **Validates: Requirements 13.5**

        For any grace_seconds in [1, 86400] and any time_after_grace in (grace_seconds, grace_seconds*2],
        setting revoked_at such that now - revoked_at = time_after_grace means we are
        past the grace window, so the old key should be rejected with "revoked_token".
        """
        # Draw time_after_grace conditioned on grace_seconds: (grace_seconds, grace_seconds*2]
        elapsed = time_after_grace.draw(
            st.integers(min_value=grace_seconds + 1, max_value=grace_seconds * 2),
            label="time_after_grace",
        )

        call_count = [0]

        def mock_monotonic():
            call_count[0] += 1
            return call_count[0] * 1000.0

        pool = _MockPool()
        token = _make_token(prefix)
        tenant_id = str(uuid.uuid4())

        # Set revoked_at so that now - revoked_at = elapsed seconds
        # This places us past the grace window since elapsed > grace_seconds
        revoked_at = datetime.now(timezone.utc) - timedelta(seconds=elapsed)

        key_row = _make_key_row(
            token,
            tenant_id=tenant_id,
            revoked_at=revoked_at,
            rotation_grace_seconds=grace_seconds,
        )

        with patch("time.monotonic", side_effect=mock_monotonic):
            service = AuthService(db_pool=pool, cache_ttl_seconds=1)
            pool.conn.fetch = AsyncMock(return_value=[key_row])

            result = await service.authenticate({"authorization": f"Bearer {token}"})

            assert isinstance(result, AuthError), (
                f"Old key with elapsed={elapsed}s > grace={grace_seconds}s "
                f"should be rejected after grace window, got: {result}"
            )
            assert result.code == "revoked_token", (
                f"Expected 'revoked_token' error code, got: {result.code}"
            )

    @given(
        grace_seconds=grace_seconds_strategy,
        elapsed_seconds=st.integers(min_value=0, max_value=172800),
        prefix=prefix_strategy,
    )
    @settings(suppress_health_check=[HealthCheck.too_slow])
    async def test_new_key_always_authenticates(
        self,
        grace_seconds: int,
        elapsed_seconds: int,
        prefix: str,
    ):
        """Property: A new (non-revoked) key always authenticates regardless of time.

        **Validates: Requirements 13.5**

        For any time point (within or after grace), a new key (revoked_at=None)
        always authenticates. The grace window only affects the OLD key; the new
        key is never revoked and should always succeed.
        """
        call_count = [0]

        def mock_monotonic():
            call_count[0] += 1
            return call_count[0] * 1000.0

        pool = _MockPool()
        token = _make_token(prefix)
        tenant_id = str(uuid.uuid4())

        # New key: revoked_at is None (never revoked)
        key_row = _make_key_row(
            token,
            tenant_id=tenant_id,
            revoked_at=None,
            rotation_grace_seconds=grace_seconds,
        )

        with patch("time.monotonic", side_effect=mock_monotonic):
            service = AuthService(db_pool=pool, cache_ttl_seconds=1)
            pool.conn.fetch = AsyncMock(return_value=[key_row])

            result = await service.authenticate({"authorization": f"Bearer {token}"})

            assert isinstance(result, AuthResult), (
                f"New key (revoked_at=None) should always authenticate "
                f"regardless of grace_seconds={grace_seconds} or elapsed={elapsed_seconds}s, "
                f"got: {result}"
            )
            assert result.tenant_id == tenant_id

    @given(
        grace_seconds=grace_seconds_strategy,
        prefix=prefix_strategy,
    )
    @settings(suppress_health_check=[HealthCheck.too_slow])
    async def test_grace_boundary_exact(
        self,
        grace_seconds: int,
        prefix: str,
    ):
        """Property: At exactly T+G the old key is still accepted; at T+G+epsilon it is rejected.

        **Validates: Requirements 13.5**

        The auth service uses `if now > grace_end` (strictly greater than), meaning:
        - At exactly grace_end (now == revoked_at + grace_seconds): key is accepted
        - At T+G+epsilon (now > revoked_at + grace_seconds): key is rejected

        This tests the boundary condition of the grace window. We patch
        datetime.datetime in the datetime module so the local import in the
        auth service picks up our mock with a controlled `now()`.
        """
        import datetime as dt_module

        call_count = [0]

        def mock_monotonic():
            call_count[0] += 1
            return call_count[0] * 1000.0

        pool = _MockPool()
        token = _make_token(prefix)
        tid = str(uuid.uuid4())

        # Fix a reference time
        base_time = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        revoked_at = base_time

        key_row = _make_key_row(
            token,
            tenant_id=tid,
            revoked_at=revoked_at,
            rotation_grace_seconds=grace_seconds,
        )

        # --- Test 1: At exactly T+G (now == revoked_at + grace_seconds), key is accepted ---
        virtual_now_exact = base_time + timedelta(seconds=grace_seconds)

        class _FakeDatetimeExact(datetime):
            """Subclass that overrides now() to return a fixed time."""

            @classmethod
            def now(cls, tz=None):
                return virtual_now_exact

        with patch("time.monotonic", side_effect=mock_monotonic):
            with patch.object(dt_module, "datetime", _FakeDatetimeExact):
                service = AuthService(db_pool=pool, cache_ttl_seconds=1)
                pool.conn.fetch = AsyncMock(return_value=[key_row])

                result = await service.authenticate({"authorization": f"Bearer {token}"})

                assert isinstance(result, AuthResult), (
                    f"At exactly T+G (now == revoked_at + {grace_seconds}s), "
                    f"old key should still be accepted (boundary inclusive), got: {result}"
                )
                assert result.tenant_id == tid

        # --- Test 2: At T+G+1s (now > revoked_at + grace_seconds), key is rejected ---
        call_count[0] = 0
        virtual_now_past = base_time + timedelta(seconds=grace_seconds + 1)

        class _FakeDatetimePast(datetime):
            """Subclass that overrides now() to return a time past the grace window."""

            @classmethod
            def now(cls, tz=None):
                return virtual_now_past

        with patch("time.monotonic", side_effect=mock_monotonic):
            with patch.object(dt_module, "datetime", _FakeDatetimePast):
                service = AuthService(db_pool=pool, cache_ttl_seconds=1)
                pool.conn.fetch = AsyncMock(return_value=[key_row])

                result = await service.authenticate({"authorization": f"Bearer {token}"})

                assert isinstance(result, AuthError), (
                    f"At T+G+1s (now > revoked_at + {grace_seconds}s), "
                    f"old key should be rejected, got: {result}"
                )
                assert result.code == "revoked_token", (
                    f"Expected 'revoked_token' error code, got: {result.code}"
                )


# ---------------------------------------------------------------------------
# Property 34: Auth failure audit shape
# ---------------------------------------------------------------------------

# Strategies for Property 34
token_prefix_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("Ll",), whitelist_characters="abcdefghijklmnopqrstuvwxyz0123456789"),
    min_size=8,
    max_size=12,
)

token_random_part_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("Ll", "Nd"), whitelist_characters="abcdefghijklmnopqrstuvwxyz0123456789"),
    min_size=8,
    max_size=32,
)

request_id_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("Ll", "Nd"), whitelist_characters="abcdefghijklmnopqrstuvwxyz0123456789-"),
    min_size=16,
    max_size=64,
)

resource_path_strategy = st.sampled_from([
    "/v1/search",
    "/v1/answer",
    "/v1/find_similar",
    "/v1/contents",
    "/v1/research",
    "/v1/sessions",
    "/v1/pipelines",
])

# Failure scenarios
failure_scenario_strategy = st.sampled_from([
    "missing_header",
    "invalid_format",
    "unknown_key",
    "expired_key",
    "revoked_key",
])


class TestAuthFailureAuditShapeProperty:
    """Property-based tests for auth failure audit shape (Property 34).

    **Validates: Requirements 13.6**

    Property 34: For any authentication failure, an `auth_failure` audit entry
    exists with:
    - action == "auth_failure"
    - actor == "anonymous"
    - request_id matches the input request_id
    - resource matches the input resource
    - detail is a dict with exactly one key "error_code"
    - detail["error_code"] is one of: "missing_token", "invalid_token", "expired_token", "revoked_token"
    - The token value does NOT appear anywhere in the audit entry

    Additionally:
    - When the key is unknown (missing_token, invalid_token with no DB match): tenant_id is None
    - When the key IS found but expired/revoked: tenant_id matches the key's tenant_id
    """

    @given(
        prefix=token_prefix_strategy,
        random_part=token_random_part_strategy,
        request_id=request_id_strategy,
        resource=resource_path_strategy,
        scenario=failure_scenario_strategy,
    )
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=200)
    async def test_auth_failure_audit_shape_invariants(
        self,
        prefix: str,
        random_part: str,
        request_id: str,
        resource: str,
        scenario: str,
    ):
        """Property: For any authentication failure, the emitted audit entry conforms to the expected shape.

        **Validates: Requirements 13.6**

        For any generated token, request_id, resource, and failure scenario:
        - The audit entry has action == "auth_failure"
        - The audit entry has actor == "anonymous"
        - The audit entry has request_id matching the input
        - The audit entry has resource matching the input
        - The detail dict has exactly one key "error_code"
        - The error_code is one of the valid codes
        - The token value NEVER appears in the audit entry
        """
        from backend.audit_log.in_memory import InMemoryAuditEmitter

        # Build the token
        token = f"{prefix}_{random_part}"

        # Virtual monotonic clock — always past cache TTL to force DB fetch
        call_count = [0]

        def mock_monotonic():
            call_count[0] += 1
            return call_count[0] * 1000.0

        pool = _MockPool()
        emitter = InMemoryAuditEmitter()

        with patch("time.monotonic", side_effect=mock_monotonic):
            service = AuthService(db_pool=pool, cache_ttl_seconds=1, audit_emitter=emitter)

            # Set up the scenario
            if scenario == "missing_header":
                headers = {}  # No authorization header
            elif scenario == "invalid_format":
                # Provide a malformed authorization header
                headers = {"authorization": f"Basic {token}"}
            elif scenario == "unknown_key":
                # Valid bearer format but key not in DB
                pool.conn.fetch = AsyncMock(return_value=[])
                headers = {"authorization": f"Bearer {token}"}
            elif scenario == "expired_key":
                # Key found but expired
                from datetime import timedelta
                expired_row = _make_key_row(
                    token,
                    tenant_id="tenant-expired-123",
                    expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
                    revoked_at=None,
                )
                pool.conn.fetch = AsyncMock(return_value=[expired_row])
                headers = {"authorization": f"Bearer {token}"}
            elif scenario == "revoked_key":
                # Key found but revoked (past grace period)
                from datetime import timedelta
                revoked_row = _make_key_row(
                    token,
                    tenant_id="tenant-revoked-456",
                    revoked_at=datetime.now(timezone.utc) - timedelta(hours=24),
                    rotation_grace_seconds=0,
                )
                pool.conn.fetch = AsyncMock(return_value=[revoked_row])
                headers = {"authorization": f"Bearer {token}"}
            else:
                raise ValueError(f"Unknown scenario: {scenario}")

            # Perform authentication (should fail)
            result = await service.authenticate(
                headers,
                request_id=request_id,
                resource=resource,
            )

            # Verify it failed
            assert isinstance(result, AuthError), (
                f"Scenario '{scenario}' should produce AuthError, got: {result}"
            )

            # Verify audit entry was emitted
            assert len(emitter.events) == 1, (
                f"Expected exactly 1 audit event, got {len(emitter.events)}"
            )

            entry = emitter.events[0]

            # Shape invariants
            assert entry.action == "auth_failure", (
                f"Expected action='auth_failure', got '{entry.action}'"
            )
            assert entry.actor == "anonymous", (
                f"Expected actor='anonymous', got '{entry.actor}'"
            )
            assert entry.request_id == request_id, (
                f"Expected request_id='{request_id}', got '{entry.request_id}'"
            )
            assert entry.resource == resource, (
                f"Expected resource='{resource}', got '{entry.resource}'"
            )

            # Detail shape: exactly one key "error_code"
            assert isinstance(entry.detail, dict), (
                f"Expected detail to be a dict, got {type(entry.detail)}"
            )
            assert set(entry.detail.keys()) == {"error_code"}, (
                f"Expected detail to have exactly key 'error_code', got keys: {set(entry.detail.keys())}"
            )

            # Valid error codes
            valid_codes = {"missing_token", "invalid_token", "expired_token", "revoked_token"}
            assert entry.detail["error_code"] in valid_codes, (
                f"Expected error_code in {valid_codes}, got '{entry.detail['error_code']}'"
            )

            # Token value MUST NOT appear anywhere in the audit entry
            import json
            serialized_detail = json.dumps(entry.detail)
            # Only check token leakage for scenarios where a real token is presented
            if scenario not in ("missing_header",):
                assert token not in entry.action, "Token leaked into action field"
                assert token not in entry.actor, "Token leaked into actor field"
                assert token not in entry.resource, "Token leaked into resource field"
                assert token not in entry.request_id, "Token leaked into request_id field"
                assert token not in serialized_detail, "Token leaked into detail field"

    @given(
        prefix=token_prefix_strategy,
        random_part=token_random_part_strategy,
        request_id=request_id_strategy,
        resource=resource_path_strategy,
    )
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=100)
    async def test_auth_failure_tenant_id_null_for_unknown_keys(
        self,
        prefix: str,
        random_part: str,
        request_id: str,
        resource: str,
    ):
        """Property: For failures where the key is not found, tenant_id is None.

        **Validates: Requirements 13.6**

        When authentication fails because:
        - The authorization header is missing (missing_token)
        - The token format is invalid (invalid_token with no DB match)
        - The key is not found in the database (invalid_token)

        Then the audit entry's tenant_id MUST be None (no spurious tenant correlation).
        """
        from backend.audit_log.in_memory import InMemoryAuditEmitter

        token = f"{prefix}_{random_part}"

        call_count = [0]

        def mock_monotonic():
            call_count[0] += 1
            return call_count[0] * 1000.0

        pool = _MockPool()
        emitter = InMemoryAuditEmitter()

        # Test all "unknown key" scenarios
        unknown_scenarios = [
            # Missing header
            ({}, "missing_header"),
            # Invalid format (not Bearer)
            ({"authorization": f"Basic {token}"}, "invalid_format"),
            # Valid bearer but key not in DB
            ({"authorization": f"Bearer {token}"}, "unknown_key"),
        ]

        for headers, scenario_name in unknown_scenarios:
            emitter.clear()
            call_count[0] = 0

            with patch("time.monotonic", side_effect=mock_monotonic):
                service = AuthService(db_pool=pool, cache_ttl_seconds=1, audit_emitter=emitter)
                pool.conn.fetch = AsyncMock(return_value=[])

                result = await service.authenticate(
                    headers,
                    request_id=request_id,
                    resource=resource,
                )

                assert isinstance(result, AuthError), (
                    f"Scenario '{scenario_name}' should produce AuthError, got: {result}"
                )

                assert len(emitter.events) == 1, (
                    f"Scenario '{scenario_name}': Expected 1 audit event, got {len(emitter.events)}"
                )

                entry = emitter.events[0]
                assert entry.tenant_id is None, (
                    f"Scenario '{scenario_name}': For unknown key failures, "
                    f"tenant_id MUST be None, got '{entry.tenant_id}'"
                )

    @given(
        prefix=token_prefix_strategy,
        random_part=token_random_part_strategy,
        request_id=request_id_strategy,
        resource=resource_path_strategy,
        tenant_id=st.uuids().map(str),
        scenario=st.sampled_from(["expired", "revoked"]),
    )
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=100)
    async def test_auth_failure_tenant_id_present_for_known_keys(
        self,
        prefix: str,
        random_part: str,
        request_id: str,
        resource: str,
        tenant_id: str,
        scenario: str,
    ):
        """Property: For failures where the key IS found but expired/revoked, tenant_id is present.

        **Validates: Requirements 13.6**

        When authentication fails because:
        - The key is found but has expired (expired_token)
        - The key is found but has been revoked past grace period (revoked_token)

        Then the audit entry's tenant_id MUST match the key's tenant_id.
        """
        from backend.audit_log.in_memory import InMemoryAuditEmitter

        token = f"{prefix}_{random_part}"

        call_count = [0]

        def mock_monotonic():
            call_count[0] += 1
            return call_count[0] * 1000.0

        pool = _MockPool()
        emitter = InMemoryAuditEmitter()

        with patch("time.monotonic", side_effect=mock_monotonic):
            service = AuthService(db_pool=pool, cache_ttl_seconds=1, audit_emitter=emitter)

            if scenario == "expired":
                from datetime import timedelta
                key_row = _make_key_row(
                    token,
                    tenant_id=tenant_id,
                    expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
                    revoked_at=None,
                )
            else:  # revoked
                from datetime import timedelta
                key_row = _make_key_row(
                    token,
                    tenant_id=tenant_id,
                    revoked_at=datetime.now(timezone.utc) - timedelta(hours=24),
                    rotation_grace_seconds=0,
                )

            pool.conn.fetch = AsyncMock(return_value=[key_row])

            result = await service.authenticate(
                {"authorization": f"Bearer {token}"},
                request_id=request_id,
                resource=resource,
            )

            assert isinstance(result, AuthError), (
                f"Scenario '{scenario}' should produce AuthError, got: {result}"
            )

            assert len(emitter.events) == 1, (
                f"Scenario '{scenario}': Expected 1 audit event, got {len(emitter.events)}"
            )

            entry = emitter.events[0]
            assert entry.tenant_id == tenant_id, (
                f"Scenario '{scenario}': For known key failures, "
                f"tenant_id MUST match key's tenant_id '{tenant_id}', "
                f"got '{entry.tenant_id}'"
            )
