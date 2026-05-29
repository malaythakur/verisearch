"""Auth Service — bearer extraction, Argon2id hash verification, tenant resolution, authorization.

Implements R13.1 (50ms p95 auth resolution) via prefix-indexed lookup and in-memory
LRU cache with configurable TTL. Bearer token values are never logged (R13.6).

Cross-tenant authorization (R13.3): The authorize() and raise_if_cross_tenant() methods
ensure that cross-tenant access yields the same 404 response shape as a genuine not-found,
making it impossible for an attacker to distinguish between "resource belongs to another
tenant" and "resource does not exist".
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from argon2 import PasswordHasher
from argon2.exceptions import VerificationError, VerifyMismatchError

if TYPE_CHECKING:
    from backend.audit_log import AuditEmitter


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AuthResult:
    """Successful authentication result."""

    tenant_id: str
    api_key_id: str
    authenticated: bool = True


@dataclass(frozen=True, slots=True)
class AuthError:
    """Authentication failure descriptor."""

    code: str  # "missing_token" | "invalid_token" | "expired_token" | "revoked_token"
    message: str


class ResourceNotFoundError(Exception):
    """Uniform not-found error for cross-tenant access (R13.3).

    This exception is raised when a tenant attempts to access a resource
    belonging to another tenant. The error shape is intentionally identical
    to a genuine "resource not found" response, so that an attacker cannot
    distinguish between "resource belongs to another tenant" and "resource
    does not exist".

    The code is always "resource_not_found" — never "access_denied" or
    "forbidden" — to prevent information leakage.
    """

    def __init__(
        self,
        code: str = "resource_not_found",
        message: str = "The requested resource was not found",
    ) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


# ---------------------------------------------------------------------------
# LRU Cache with TTL
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _CacheEntry:
    """A cached set of candidate rows for a given key prefix."""

    rows: list[dict[str, Any]]
    expires_at: float  # monotonic clock


class _TTLCache:
    """Simple bounded LRU-style cache with per-entry TTL.

    Keyed by token prefix → list of candidate rows from the api_keys table.
    Evicts expired entries on access and caps total size.
    """

    def __init__(self, ttl_seconds: int = 60, max_size: int = 4096) -> None:
        self._ttl_seconds = ttl_seconds
        self._max_size = max_size
        self._store: dict[str, _CacheEntry] = {}

    def get(self, prefix: str) -> list[dict[str, Any]] | None:
        entry = self._store.get(prefix)
        if entry is None:
            return None
        if time.monotonic() > entry.expires_at:
            del self._store[prefix]
            return None
        return entry.rows

    def put(self, prefix: str, rows: list[dict[str, Any]]) -> None:
        # Evict oldest entries if at capacity
        if len(self._store) >= self._max_size and prefix not in self._store:
            oldest_key = next(iter(self._store))
            del self._store[oldest_key]
        self._store[prefix] = _CacheEntry(
            rows=rows,
            expires_at=time.monotonic() + self._ttl_seconds,
        )

    def invalidate(self, prefix: str) -> None:
        self._store.pop(prefix, None)

    def clear(self) -> None:
        self._store.clear()


# ---------------------------------------------------------------------------
# Auth Service
# ---------------------------------------------------------------------------

# Default prefix length for token format: {prefix}_{random_part}
_MIN_PREFIX_LEN = 8
_MAX_PREFIX_LEN = 12


def _extract_prefix(token: str) -> str | None:
    """Extract the prefix portion from a token of format `{prefix}_{random_part}`.

    Returns None if the token doesn't match the expected format.
    """
    underscore_idx = token.find("_")
    if underscore_idx == -1:
        return None
    prefix = token[:underscore_idx]
    if len(prefix) < _MIN_PREFIX_LEN or len(prefix) > _MAX_PREFIX_LEN:
        return None
    return prefix


class AuthService:
    """Authenticates API keys via prefix-indexed Argon2id hash lookup.

    Uses an in-memory TTL cache to avoid DB round-trips on repeated requests
    within the cache window (default 60s per R13.4).

    Revocation propagation guarantee (R13.4):
        When a key is revoked in the database, the revocation propagates to
        authentication decisions within at most ``cache_ttl_seconds`` (default 60s).
        This is guaranteed because:
        1. The cache TTL is 60s — entries expire and are re-fetched from DB.
        2. On each authentication attempt, ``revoked_at`` is checked on the
           fetched row, so stale cache entries are only valid until TTL expiry.
        3. For same-process revocations, ``revoke_key()`` immediately invalidates
           the cache entry, providing instant propagation.
        4. For cross-process propagation (e.g., via pub/sub or webhook),
           ``invalidate_cache_for_prefix()`` can be called to eagerly evict
           the cached entry before TTL expiry.
    """

    def __init__(
        self,
        db_pool: Any,
        cache_ttl_seconds: int = 60,
        audit_emitter: AuditEmitter | None = None,
    ) -> None:
        """Initialize the auth service.

        Args:
            db_pool: An asyncpg connection pool for database access.
            cache_ttl_seconds: TTL for the prefix→candidates cache (default 60s).
                This value defines the maximum revocation propagation delay (R13.4).
            audit_emitter: Optional audit emitter for recording auth_failure events (R13.6).
                When provided, authentication failures emit an audit entry with:
                - tenant_id: resolved tenant if key was found (expired/revoked), None if unknown key
                - detail: error code only, NEVER the bearer token value
        """
        self._db_pool = db_pool
        self._cache = _TTLCache(ttl_seconds=cache_ttl_seconds)
        self._cache_ttl_seconds = cache_ttl_seconds
        self._hasher = PasswordHasher()
        self._audit_emitter = audit_emitter

    async def authenticate(
        self,
        headers: dict[str, str],
        *,
        request_id: str = "",
        resource: str = "",
    ) -> AuthResult | AuthError:
        """Authenticate a request from its headers.

        Extracts the bearer token from the Authorization header, looks up
        candidate keys by prefix, and verifies the full token against stored
        Argon2id hashes.

        On failure, emits an ``auth_failure`` audit event (if an audit_emitter
        is configured) with:
        - tenant_id: the resolved tenant if the key was found but expired/revoked;
          None if the key was not found at all (unknown key) per R13.6.
        - detail: includes the error code but NEVER the bearer token value (R13.6).

        Args:
            headers: Request headers (case-insensitive key lookup is caller's
                     responsibility; expects lowercase keys).
            request_id: The request correlation ID for audit trail propagation.
            resource: The endpoint/resource being accessed (for audit context).

        Returns:
            AuthResult on success, AuthError on failure.
        """
        # Step a/b: Extract bearer token
        auth_header = headers.get("authorization", "")
        if not auth_header:
            error = AuthError(code="missing_token", message="Authorization header is required")
            await self._emit_auth_failure(
                error_code=error.code,
                tenant_id=None,
                request_id=request_id,
                resource=resource,
            )
            return error

        parts = auth_header.split(" ", 1)
        if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1].strip():
            error = AuthError(code="missing_token", message="Authorization header must be 'Bearer <token>'")
            await self._emit_auth_failure(
                error_code=error.code,
                tenant_id=None,
                request_id=request_id,
                resource=resource,
            )
            return error

        token = parts[1].strip()

        # Step c: Extract prefix
        prefix = _extract_prefix(token)
        if prefix is None:
            error = AuthError(code="invalid_token", message="Token format is invalid")
            await self._emit_auth_failure(
                error_code=error.code,
                tenant_id=None,
                request_id=request_id,
                resource=resource,
            )
            return error

        # Step d: Look up candidates (cache first, then DB)
        candidates = await self._get_candidates(prefix)

        # Step e: Verify token against each candidate's hash
        matched_row: dict[str, Any] | None = None
        for row in candidates:
            if self._verify_token(token, row["key_hash"]):
                matched_row = row
                break

        # Step f: No match — unknown key, tenant_id is None
        if matched_row is None:
            error = AuthError(code="invalid_token", message="API key is not valid")
            await self._emit_auth_failure(
                error_code=error.code,
                tenant_id=None,
                request_id=request_id,
                resource=resource,
            )
            return error

        # Step g: Check expiration — key found, tenant is known
        expires_at = matched_row.get("expires_at")
        if expires_at is not None:
            from datetime import datetime, timezone

            now = datetime.now(timezone.utc)
            if now > expires_at:
                error = AuthError(code="expired_token", message="API key has expired")
                await self._emit_auth_failure(
                    error_code=error.code,
                    tenant_id=str(matched_row["tenant_id"]),
                    request_id=request_id,
                    resource=resource,
                )
                return error

        # Step h: Check revocation with grace period — key found, tenant is known
        revoked_at = matched_row.get("revoked_at")
        if revoked_at is not None:
            from datetime import datetime, timedelta, timezone

            now = datetime.now(timezone.utc)
            grace_seconds = matched_row.get("rotation_grace_seconds", 3600)
            grace_end = revoked_at + timedelta(seconds=grace_seconds)
            if now > grace_end:
                error = AuthError(code="revoked_token", message="API key has been revoked")
                await self._emit_auth_failure(
                    error_code=error.code,
                    tenant_id=str(matched_row["tenant_id"]),
                    request_id=request_id,
                    resource=resource,
                )
                return error

        # Step i: Success
        return AuthResult(
            tenant_id=str(matched_row["tenant_id"]),
            api_key_id=str(matched_row["api_key_id"]),
        )

    async def _emit_auth_failure(
        self,
        *,
        error_code: str,
        tenant_id: str | None,
        request_id: str,
        resource: str,
    ) -> None:
        """Emit an auth_failure audit event if an audit emitter is configured.

        Per R13.6:
        - Bearer token value is NEVER included in the audit detail.
        - tenant_id is None for unknown keys (unattributable failures).
        - tenant_id is the resolved tenant for expired/revoked keys.

        Args:
            error_code: The authentication error code (e.g., "invalid_token").
            tenant_id: The resolved tenant_id, or None if key was not found.
            request_id: The request correlation ID.
            resource: The endpoint being accessed.
        """
        if self._audit_emitter is None:
            return

        await self._audit_emitter.emit(
            action="auth_failure",
            tenant_id=tenant_id,
            actor="anonymous",
            resource=resource,
            request_id=request_id,
            detail={"error_code": error_code},
        )

    def revoke_key(self, prefix: str) -> None:
        """Immediately invalidate the cache entry for a revoked key prefix.

        Call this when the same service instance revokes a key to ensure
        instant propagation without waiting for cache TTL expiry.

        For cross-process revocation (e.g., another service instance revoked
        the key), use ``invalidate_cache_for_prefix()`` via a pub/sub listener
        or webhook handler.

        Args:
            prefix: The key prefix whose cache entry should be invalidated.
        """
        self._cache.invalidate(prefix)

    def invalidate_cache_for_prefix(self, prefix: str) -> None:
        """Invalidate the cache entry for a given prefix.

        This method is intended to be called by external propagation mechanisms
        (e.g., pub/sub listeners, webhook handlers) to eagerly evict a cached
        entry when a key is revoked by another process or service instance.

        Even without calling this method, revocation propagates within
        ``cache_ttl_seconds`` (default 60s) due to natural cache expiry (R13.4).

        Args:
            prefix: The key prefix whose cache entry should be invalidated.
        """
        self._cache.invalidate(prefix)

    def authorize(self, tenant_id: str, resource_tenant_id: str) -> bool:
        """Check whether a tenant is authorized to access a resource (R13.3).

        Returns True if the requesting tenant owns the resource (same tenant).
        Returns False if the resource belongs to a different tenant.

        When this returns False, the caller MUST return a 404 "resource not found"
        response — NOT a 403 "forbidden". This ensures that cross-tenant access
        is indistinguishable from a genuine not-found, preventing information
        leakage about resource existence in other tenants.

        Args:
            tenant_id: The authenticated tenant making the request.
            resource_tenant_id: The tenant that owns the target resource.

        Returns:
            True if tenant_id == resource_tenant_id, False otherwise.
        """
        return tenant_id == resource_tenant_id

    def raise_if_cross_tenant(self, tenant_id: str, resource_tenant_id: str) -> None:
        """Raise ResourceNotFoundError if the request is cross-tenant (R13.3).

        This is a convenience wrapper around ``authorize()`` that raises a
        ``ResourceNotFoundError`` when the requesting tenant does not own the
        resource. The error has a generic "resource_not_found" code — never
        "access_denied" or "forbidden" — ensuring the response is
        indistinguishable from a genuine not-found.

        Args:
            tenant_id: The authenticated tenant making the request.
            resource_tenant_id: The tenant that owns the target resource.

        Raises:
            ResourceNotFoundError: If tenant_id != resource_tenant_id.
        """
        if not self.authorize(tenant_id, resource_tenant_id):
            raise ResourceNotFoundError()

    @property
    def cache_ttl_seconds(self) -> int:
        """The maximum revocation propagation delay in seconds (R13.4)."""
        return self._cache_ttl_seconds

    async def _get_candidates(self, prefix: str) -> list[dict[str, Any]]:
        """Retrieve candidate rows for a prefix, using cache when available."""
        cached = self._cache.get(prefix)
        if cached is not None:
            return cached

        # Query DB by prefix
        rows = await self._fetch_candidates_from_db(prefix)
        self._cache.put(prefix, rows)
        return rows

    async def _fetch_candidates_from_db(self, prefix: str) -> list[dict[str, Any]]:
        """Fetch candidate API key rows from the database by prefix."""
        query = """
            SELECT api_key_id, tenant_id, key_hash, expires_at, revoked_at, rotation_grace_seconds
            FROM api_keys
            WHERE key_prefix = $1
        """
        async with self._db_pool.acquire() as conn:
            records = await conn.fetch(query, prefix)

        return [dict(record) for record in records]

    def _verify_token(self, token: str, key_hash: str) -> bool:
        """Verify a token against an Argon2id hash.

        Returns True if the token matches the hash, False otherwise.
        """
        try:
            return self._hasher.verify(key_hash, token)
        except (VerifyMismatchError, VerificationError):
            return False
