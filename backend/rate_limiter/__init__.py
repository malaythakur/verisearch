"""Rate Limiter - Per-tenant rate limiting and metering event emission.

Provides a token-bucket rate limiter backed by Redis for per-(tenant_id, endpoint)
rate limiting as specified in R14.1, and metering event emission for billable
responses as specified in R14.2 and R14.3.
"""

from backend.rate_limiter.metering import MeteringEvent, MeteringService
from backend.rate_limiter.metering_buffer import DurableMeteringBuffer
from backend.rate_limiter.token_bucket import RateLimitResult, TokenBucketRateLimiter

__all__ = [
    "DurableMeteringBuffer",
    "MeteringEvent",
    "MeteringService",
    "RateLimitResult",
    "TokenBucketRateLimiter",
]
