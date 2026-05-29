"""Data models for the Crawler subsystem.

Defines FetchResult and SkipReason used throughout the crawler pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class SkipReasonType(str, Enum):
    """Enumeration of reasons a URL may be skipped during crawling."""

    DISALLOWED_BY_ROBOTS = "disallowed_by_robots"
    ROBOTS_UNAVAILABLE = "robots_unavailable"
    DOMAIN_OPTED_OUT = "domain_opted_out"


@dataclass(frozen=True, slots=True)
class SkipReason:
    """Represents a skipped URL with the reason and context (R1.2, R1.3, R1.6).

    Attributes:
        reason: The skip reason type.
        url: The URL that was skipped.
        detail: Additional context (e.g., matched directive, error message).
    """

    reason: SkipReasonType
    url: str
    detail: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FetchResult:
    """Result of a successful page fetch (R1.7).

    Records the content alongside fetch metadata:
    - fetch_timestamp_utc: When the fetch completed (UTC).
    - http_status: The HTTP response status code.
    - content_type: The Content-Type header value.
    - canonical_url: The canonical source URL (after redirects).
    """

    content: bytes
    fetch_timestamp_utc: datetime
    http_status: int
    content_type: str
    canonical_url: str


@dataclass(slots=True)
class RobotsResult:
    """Result of fetching and parsing a robots.txt file.

    Attributes:
        available: Whether the robots.txt was successfully fetched.
        rules: Parsed rules as a list of (user_agent, disallow_paths) tuples.
        crawl_delay: The Crawl-Delay value if specified, else None.
        fetched_at: When the robots.txt was fetched.
        error: Error message if fetch failed.
    """

    available: bool
    rules: list[tuple[str, list[str]]] = field(default_factory=list)
    crawl_delay: float | None = None
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    error: str | None = None
