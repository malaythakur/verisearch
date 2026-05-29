"""Immutable system constants that are not configurable at runtime.

These values are fixed by the system specification and should not be changed
without a corresponding requirements update.
"""


class Constants:
    """Immutable system-wide constants.

    These are not loaded from environment variables — they are fixed by design.
    """

    # Request ID bounds
    REQUEST_ID_MIN_LENGTH: int = 16
    REQUEST_ID_MAX_LENGTH: int = 64

    # Input size limits
    QUERY_MAX_CODE_POINTS: int = 2048
    URL_MAX_CODE_POINTS: int = 2048
    RESEARCH_GOAL_MAX_CODE_POINTS: int = 4096

    # Cache TTLs
    WARM_CACHE_TTL_SECONDS: int = 300
    AUTH_CACHE_TTL_SECONDS: int = 60
    ROBOTS_CACHE_TTL_HOURS: int = 24

    # Opt-out activation
    OPT_OUT_ACTIVATION_HOURS: int = 24

    # Dead-letter queue
    DLQ_MAX_RETRIES: int = 3
    DLQ_RETRY_SPACING_SECONDS: int = 60

    # Streaming
    SSE_KEEPALIVE_SECONDS: int = 15
    ANSWER_SILENCE_TIMEOUT_SECONDS: int = 30
    ANSWER_ERROR_CLOSE_SECONDS: int = 2
    CITATION_EMISSION_DEADLINE_MS: int = 500
