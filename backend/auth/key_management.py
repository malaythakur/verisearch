"""Key Management — rotation with grace period logic.

Implements R13.5: Key rotation grace period [1, 86400]s default 3600.
During the rotation grace window [T, T+G], both old and new keys authenticate
successfully. After the grace window, only the new key works.
"""

from __future__ import annotations

import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from argon2 import PasswordHasher


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROTATION_GRACE_MIN = 1
ROTATION_GRACE_MAX = 86400
ROTATION_GRACE_DEFAULT = 3600

# Key format: {prefix}_{random_part}
_PREFIX_LENGTH = 8
_RANDOM_PART_LENGTH = 32  # hex chars


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class ValidationError(Exception):
    """Raised when input validation fails."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


def validate_rotation_grace_seconds(value: int) -> int:
    """Validate that rotation_grace_seconds is within [1, 86400].

    Args:
        value: The grace period in seconds.

    Returns:
        The validated value.

    Raises:
        ValidationError: If value is outside the allowed range.
    """
    if not isinstance(value, int):
        raise ValidationError(
            code="invalid_rotation_grace_seconds",
            message=f"rotation_grace_seconds must be an integer, got {type(value).__name__}",
        )
    if value < ROTATION_GRACE_MIN or value > ROTATION_GRACE_MAX:
        raise ValidationError(
            code="invalid_rotation_grace_seconds",
            message=(
                f"rotation_grace_seconds must be in [{ROTATION_GRACE_MIN}, {ROTATION_GRACE_MAX}], "
                f"got {value}"
            ),
        )
    return value


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RotationResult:
    """Result of a successful key rotation."""

    new_api_key_id: str
    new_key_plaintext: str  # Only available at creation time
    new_key_prefix: str
    old_key_revoked_at: datetime
    rotation_grace_seconds: int


# ---------------------------------------------------------------------------
# Key Management Service
# ---------------------------------------------------------------------------


def _generate_api_key() -> tuple[str, str]:
    """Generate a new API key in the format {prefix}_{random_part}.

    Returns:
        A tuple of (full_key, prefix).
    """
    prefix = secrets.token_hex(_PREFIX_LENGTH // 2 + 4)[:_PREFIX_LENGTH]
    random_part = secrets.token_hex(_RANDOM_PART_LENGTH // 2)
    full_key = f"{prefix}_{random_part}"
    return full_key, prefix


class KeyManagementService:
    """Manages API key lifecycle including rotation with grace periods.

    Key rotation (R13.5):
        When rotating a key, the old key's `revoked_at` is set to NOW().
        During the grace window [T, T+G] where T = revoked_at and G = rotation_grace_seconds,
        both old and new keys authenticate successfully.
        After T+G, only the new key works.

    The grace period is validated to be in [1, 86400] seconds (default 3600).
    """

    def __init__(self, db_pool: Any, auth_service: Any | None = None) -> None:
        """Initialize the key management service.

        Args:
            db_pool: An asyncpg connection pool for database access.
            auth_service: Optional AuthService instance for cache invalidation.
        """
        self._db_pool = db_pool
        self._auth_service = auth_service
        self._hasher = PasswordHasher()

    async def rotate_key(
        self,
        tenant_id: str,
        old_key_prefix: str,
        rotation_grace_seconds: int = ROTATION_GRACE_DEFAULT,
    ) -> RotationResult:
        """Rotate an API key: revoke the old key and create a new one.

        This operation:
        1. Validates rotation_grace_seconds is in [1, 86400]
        2. Sets `revoked_at = NOW()` on the old key
        3. Creates a new key row with a fresh hash and prefix
        4. Invalidates the cache for the old key's prefix
        5. Returns the new key's plaintext (only time it's available)

        During the grace window [T, T+G], both old and new keys authenticate.
        After T+G, only the new key works.

        Args:
            tenant_id: The tenant that owns the key being rotated.
            old_key_prefix: The prefix of the key to revoke.
            rotation_grace_seconds: Grace period in seconds [1, 86400], default 3600.

        Returns:
            RotationResult with the new key details.

        Raises:
            ValidationError: If rotation_grace_seconds is out of range.
            KeyNotFoundError: If no active key with the given prefix exists for the tenant.
        """
        # Step 1: Validate grace period
        validate_rotation_grace_seconds(rotation_grace_seconds)

        # Step 2: Generate new key
        new_key_plaintext, new_prefix = _generate_api_key()
        new_key_hash = self._hasher.hash(new_key_plaintext)
        new_api_key_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        # Step 3: Execute rotation in a transaction
        async with self._db_pool.acquire() as conn:
            async with conn.transaction():
                # Revoke the old key
                result = await conn.execute(
                    """
                    UPDATE api_keys
                    SET revoked_at = $1, rotation_grace_seconds = $2
                    WHERE key_prefix = $3 AND tenant_id = $4 AND revoked_at IS NULL
                    """,
                    now,
                    rotation_grace_seconds,
                    old_key_prefix,
                    tenant_id,
                )

                # Check if the old key was found
                if result == "UPDATE 0":
                    raise KeyNotFoundError(
                        f"No active key with prefix '{old_key_prefix}' found for tenant"
                    )

                # Create the new key
                await conn.execute(
                    """
                    INSERT INTO api_keys (api_key_id, tenant_id, key_prefix, key_hash,
                                          created_at, rotation_grace_seconds)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                    new_api_key_id,
                    tenant_id,
                    new_prefix,
                    new_key_hash,
                    now,
                    rotation_grace_seconds,
                )

        # Step 4: Invalidate cache for old prefix
        if self._auth_service is not None:
            self._auth_service.invalidate_cache_for_prefix(old_key_prefix)

        # Step 5: Return result
        return RotationResult(
            new_api_key_id=new_api_key_id,
            new_key_plaintext=new_key_plaintext,
            new_key_prefix=new_prefix,
            old_key_revoked_at=now,
            rotation_grace_seconds=rotation_grace_seconds,
        )

    async def create_key(
        self,
        tenant_id: str,
        rotation_grace_seconds: int = ROTATION_GRACE_DEFAULT,
        expires_at: datetime | None = None,
    ) -> RotationResult:
        """Create a new API key for a tenant.

        Args:
            tenant_id: The tenant to create the key for.
            rotation_grace_seconds: Grace period for future rotations [1, 86400], default 3600.
            expires_at: Optional expiration timestamp.

        Returns:
            RotationResult with the new key details.

        Raises:
            ValidationError: If rotation_grace_seconds is out of range.
        """
        validate_rotation_grace_seconds(rotation_grace_seconds)

        new_key_plaintext, new_prefix = _generate_api_key()
        new_key_hash = self._hasher.hash(new_key_plaintext)
        new_api_key_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        async with self._db_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO api_keys (api_key_id, tenant_id, key_prefix, key_hash,
                                      created_at, expires_at, rotation_grace_seconds)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                new_api_key_id,
                tenant_id,
                new_prefix,
                new_key_hash,
                now,
                expires_at,
                rotation_grace_seconds,
            )

        return RotationResult(
            new_api_key_id=new_api_key_id,
            new_key_plaintext=new_key_plaintext,
            new_key_prefix=new_prefix,
            old_key_revoked_at=now,
            rotation_grace_seconds=rotation_grace_seconds,
        )


class KeyNotFoundError(Exception):
    """Raised when a key to be rotated is not found."""

    pass
