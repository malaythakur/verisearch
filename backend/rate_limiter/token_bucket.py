"""Token-bucket rate limiter backed by Redis.

Implements per-(tenant_id, endpoint) rate limiting using a Lua script
for atomic token-bucket operations. Each bucket refills at a rate of
limit_per_minute / 60 tokens per second, with a maximum capacity equal
to limit_per_minute.

Satisfies R14.1 (per-tenant rate limits using token-bucket algorithm).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from redis.asyncio import Redis


@dataclass(frozen=True, slots=True)
class RateLimitResult:
    """Result of a rate limit check.

    Attributes:
        allowed: Whether the request is allowed.
        limit: The configured limit (max tokens / bucket capacity).
        remaining: Tokens remaining after this request.
        reset_at: Unix timestamp when the bucket fully refills.
        retry_after: Seconds until next token available (only when denied).
    """

    allowed: bool
    limit: int
    remaining: int
    reset_at: int
    retry_after: int | None = None


# Lua script for atomic token-bucket rate limiting.
# KEYS[1] = ratelimit:{tenant_id}:{endpoint}
# ARGV[1] = limit (max bucket capacity / tokens per minute)
# ARGV[2] = current time in seconds (float as string)
# ARGV[3] = refill rate (tokens per second = limit / 60)
#
# Stored hash fields:
#   tokens   - current token count (float)
#   last_refill - last refill timestamp (float seconds)
#
# Returns: [allowed (0/1), remaining (int), reset_at (int), retry_after (int or -1)]
_TOKEN_BUCKET_LUA = """
local key = KEYS[1]
local limit = tonumber(ARGV[1])
local now = tonumber(ARGV[2])
local refill_rate = tonumber(ARGV[3])

local data = redis.call('HMGET', key, 'tokens', 'last_refill')
local tokens = tonumber(data[1])
local last_refill = tonumber(data[2])

if tokens == nil then
    -- First request: initialize bucket to full capacity minus 1 (this request)
    tokens = limit
    last_refill = now
end

-- Refill tokens based on elapsed time
local elapsed = now - last_refill
if elapsed > 0 then
    tokens = math.min(limit, tokens + elapsed * refill_rate)
    last_refill = now
end

local allowed = 0
local remaining = 0
local retry_after = -1
local reset_at = 0

if tokens >= 1 then
    -- Allow: consume one token
    tokens = tokens - 1
    allowed = 1
    remaining = math.floor(tokens)
else
    -- Deny: calculate when next token will be available
    remaining = 0
    if refill_rate > 0 then
        retry_after = math.ceil((1 - tokens) / refill_rate)
    else
        retry_after = 60
    end
end

-- Calculate reset_at: time when bucket will be fully refilled
local tokens_needed = limit - tokens
if tokens_needed > 0 and refill_rate > 0 then
    reset_at = math.ceil(now + tokens_needed / refill_rate)
else
    reset_at = math.ceil(now)
end

-- Store updated state with TTL of 2 minutes (bucket expires if unused)
redis.call('HSET', key, 'tokens', tostring(tokens), 'last_refill', tostring(last_refill))
redis.call('EXPIRE', key, 120)

return {allowed, remaining, reset_at, retry_after}
"""


class TokenBucketRateLimiter:
    """Per-(tenant_id, endpoint) token-bucket rate limiter using Redis.

    The bucket refills at `limit_per_minute / 60` tokens per second,
    with a maximum capacity of `limit_per_minute`. Each request consumes
    one token.

    Args:
        redis_client: An async Redis client instance.
        default_limit_per_minute: Default rate limit when no tenant-specific
            override is configured.
    """

    def __init__(self, redis_client: Redis, default_limit_per_minute: int = 60) -> None:
        self._redis = redis_client
        self._default_limit_per_minute = default_limit_per_minute
        self._script: object | None = None

    def _get_key(self, tenant_id: str, endpoint: str) -> str:
        """Build the Redis key for a tenant+endpoint bucket."""
        return f"ratelimit:{tenant_id}:{endpoint}"

    async def _ensure_script(self) -> object:
        """Register the Lua script with Redis (cached after first call)."""
        if self._script is None:
            self._script = self._redis.register_script(_TOKEN_BUCKET_LUA)
        return self._script

    async def check_rate_limit(
        self,
        tenant_id: str,
        endpoint: str,
        limit_per_minute: int | None = None,
    ) -> RateLimitResult:
        """Check and consume a rate limit token for the given tenant+endpoint.

        Args:
            tenant_id: The tenant making the request.
            endpoint: The API endpoint being accessed.
            limit_per_minute: Optional per-tenant override. Falls back to
                default_limit_per_minute if not provided.

        Returns:
            RateLimitResult with allowed/denied status and header values.
        """
        limit = limit_per_minute if limit_per_minute is not None else self._default_limit_per_minute
        refill_rate = limit / 60.0
        now = time.time()

        key = self._get_key(tenant_id, endpoint)
        script = await self._ensure_script()

        result = await script(keys=[key], args=[str(limit), str(now), str(refill_rate)])

        allowed = bool(result[0])
        remaining = int(result[1])
        reset_at = int(result[2])
        retry_after_raw = int(result[3])
        retry_after = retry_after_raw if retry_after_raw >= 0 else None

        return RateLimitResult(
            allowed=allowed,
            limit=limit,
            remaining=remaining,
            reset_at=reset_at,
            retry_after=retry_after,
        )
