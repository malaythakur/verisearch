"""Robots.txt fetcher and cache (R1.1, R1.2, R1.3).

Implements:
- RobotsCache: TTL-based cache for robots.txt results (≤24h).
- fetch_robots: Async fetcher with 10s timeout.
- is_allowed: Check if a URL is allowed for a given user agent.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib.parse import urlparse

import httpx

from crawler.models import RobotsResult


# Maximum cache TTL in seconds (24 hours)
_MAX_CACHE_TTL_SECONDS = 24 * 60 * 60

# Fetch timeout in seconds
_FETCH_TIMEOUT_SECONDS = 10.0

# Default user agent for robots.txt compliance
DEFAULT_USER_AGENT = "AgenticResearchBot"


@dataclass
class _CacheEntry:
    """Internal cache entry for a robots.txt result."""

    result: RobotsResult
    cached_at: float  # monotonic time


class RobotsCache:
    """TTL-based cache for robots.txt results (R1.1).

    Caches parsed robots.txt results per host for at most 24 hours.
    On cache miss or expiry, fetches fresh robots.txt with a 10s timeout.

    Args:
        ttl_seconds: Cache TTL in seconds. Clamped to [0, 86400].
        timeout_seconds: HTTP fetch timeout. Defaults to 10s.
        http_client: Optional httpx.AsyncClient for dependency injection.
        user_agent: The user agent string to identify as.
    """

    def __init__(
        self,
        *,
        ttl_seconds: float = _MAX_CACHE_TTL_SECONDS,
        timeout_seconds: float = _FETCH_TIMEOUT_SECONDS,
        http_client: httpx.AsyncClient | None = None,
        user_agent: str = DEFAULT_USER_AGENT,
    ) -> None:
        self._ttl = min(max(ttl_seconds, 0), _MAX_CACHE_TTL_SECONDS)
        self._timeout = timeout_seconds
        self._client = http_client
        self._user_agent = user_agent
        self._cache: dict[str, _CacheEntry] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def _get_lock(self, host: str) -> asyncio.Lock:
        """Get or create a per-host lock to prevent thundering herd."""
        if host not in self._locks:
            self._locks[host] = asyncio.Lock()
        return self._locks[host]

    def _is_fresh(self, entry: _CacheEntry) -> bool:
        """Check if a cache entry is still within TTL."""
        elapsed = time.monotonic() - entry.cached_at
        return elapsed < self._ttl

    async def get(self, host: str) -> RobotsResult:
        """Get robots.txt result for a host, fetching if not cached or expired.

        Args:
            host: The hostname to look up robots.txt for.

        Returns:
            RobotsResult with availability status and parsed rules.
        """
        # Check cache first (no lock needed for read)
        entry = self._cache.get(host)
        if entry is not None and self._is_fresh(entry):
            return entry.result

        # Fetch under per-host lock to prevent duplicate fetches
        lock = self._get_lock(host)
        async with lock:
            # Double-check after acquiring lock
            entry = self._cache.get(host)
            if entry is not None and self._is_fresh(entry):
                return entry.result

            # Fetch fresh robots.txt
            result = await self._fetch_robots(host)
            self._cache[host] = _CacheEntry(
                result=result,
                cached_at=time.monotonic(),
            )
            return result

    async def _fetch_robots(self, host: str) -> RobotsResult:
        """Fetch and parse robots.txt for a host with 10s timeout (R1.1).

        On failure (network error, timeout, HTTP 5xx), returns an unavailable
        result per R1.3.
        """
        url = f"https://{host}/robots.txt"
        client = self._client or httpx.AsyncClient()
        should_close = self._client is None

        try:
            response = await client.get(
                url,
                timeout=self._timeout,
                follow_redirects=True,
            )

            # HTTP 5xx → robots unavailable (R1.3)
            if response.status_code >= 500:
                return RobotsResult(
                    available=False,
                    error=f"HTTP {response.status_code} from {url}",
                )

            # HTTP 4xx → treat as no restrictions (standard behavior)
            if response.status_code >= 400:
                return RobotsResult(available=True, rules=[], crawl_delay=None)

            # Parse the robots.txt content
            return _parse_robots_txt(response.text)

        except (httpx.TimeoutException, httpx.ConnectTimeout) as exc:
            return RobotsResult(
                available=False,
                error=f"Timeout fetching {url}: {exc}",
            )
        except (httpx.NetworkError, httpx.HTTPError, OSError) as exc:
            return RobotsResult(
                available=False,
                error=f"Network error fetching {url}: {exc}",
            )
        finally:
            if should_close:
                await client.aclose()

    def is_allowed(self, url: str, user_agent: str | None = None) -> bool | None:
        """Check if a URL is allowed by cached robots.txt rules (R1.2).

        Args:
            url: The full URL to check.
            user_agent: The user agent to check against. Defaults to configured agent.

        Returns:
            True if allowed, False if disallowed, None if no cache entry exists.
        """
        parsed = urlparse(url)
        host = parsed.hostname or ""
        entry = self._cache.get(host)

        if entry is None or not self._is_fresh(entry):
            return None  # No cached data available

        result = entry.result
        if not result.available:
            return None  # robots.txt unavailable — caller decides

        ua = (user_agent or self._user_agent).lower()
        path = parsed.path or "/"

        return _check_rules(result.rules, ua, path)

    def invalidate(self, host: str) -> None:
        """Remove a host from the cache."""
        self._cache.pop(host, None)

    def clear(self) -> None:
        """Clear the entire cache."""
        self._cache.clear()


def _parse_robots_txt(content: str) -> RobotsResult:
    """Parse robots.txt content into structured rules.

    Handles standard robots.txt format with User-agent, Disallow, Allow,
    and Crawl-delay directives.
    """
    rules: list[tuple[str, list[str]]] = []
    current_agents: list[str] = []
    current_disallows: list[str] = []
    current_allows: list[str] = []
    crawl_delay: float | None = None

    for line in content.splitlines():
        # Strip comments
        line = line.split("#", 1)[0].strip()
        if not line:
            continue

        if ":" not in line:
            continue

        directive, _, value = line.partition(":")
        directive = directive.strip().lower()
        value = value.strip()

        if directive == "user-agent":
            # If we have accumulated rules, save them
            if current_agents and (current_disallows or current_allows):
                for agent in current_agents:
                    rules.append((agent.lower(), list(current_disallows)))
                current_disallows = []
                current_allows = []
                current_agents = []
            elif not current_disallows and not current_allows:
                # Multiple user-agent lines in a row
                pass
            current_agents.append(value)
        elif directive == "disallow":
            if not current_agents:
                current_agents = ["*"]
            if value:  # Empty disallow means allow all
                current_disallows.append(value)
        elif directive == "allow":
            if not current_agents:
                current_agents = ["*"]
            current_allows.append(value)
        elif directive == "crawl-delay":
            try:
                crawl_delay = float(value)
            except ValueError:
                pass

    # Save final group
    if current_agents and current_disallows:
        for agent in current_agents:
            rules.append((agent.lower(), list(current_disallows)))

    return RobotsResult(
        available=True,
        rules=rules,
        crawl_delay=crawl_delay,
    )


def _check_rules(
    rules: list[tuple[str, list[str]]],
    user_agent: str,
    path: str,
) -> bool:
    """Check if a path is allowed given parsed robots.txt rules.

    Matching logic:
    1. Find rules for the specific user agent first.
    2. Fall back to wildcard (*) rules.
    3. If no matching rules, allow by default.
    """
    ua_lower = user_agent.lower()

    # Find matching rule groups
    specific_disallows: list[str] = []
    wildcard_disallows: list[str] = []

    for agent, disallows in rules:
        if agent == ua_lower:
            specific_disallows.extend(disallows)
        elif agent == "*":
            wildcard_disallows.extend(disallows)

    # Use specific rules if available, otherwise wildcard
    applicable = specific_disallows if specific_disallows else wildcard_disallows

    if not applicable:
        return True  # No rules → allowed

    # Check if path matches any disallow pattern
    for pattern in applicable:
        if _path_matches(path, pattern):
            return False

    return True


def _path_matches(path: str, pattern: str) -> bool:
    """Check if a path matches a robots.txt disallow pattern.

    Supports:
    - Prefix matching (default)
    - Wildcard (*) in patterns
    - End-of-string anchor ($)
    """
    if not pattern:
        return False

    # Handle $ anchor
    if pattern.endswith("$"):
        pattern_base = pattern[:-1]
        if "*" in pattern_base:
            return _wildcard_match(path, pattern_base, anchored=True)
        return path == pattern_base

    # Handle wildcard
    if "*" in pattern:
        return _wildcard_match(path, pattern, anchored=False)

    # Simple prefix match
    return path.startswith(pattern)


def _wildcard_match(path: str, pattern: str, *, anchored: bool) -> bool:
    """Match a path against a pattern with wildcards."""
    parts = pattern.split("*")

    pos = 0
    for i, part in enumerate(parts):
        if not part:
            continue
        idx = path.find(part, pos)
        if idx == -1:
            return False
        if i == 0 and not pattern.startswith("*"):
            # First part must match at start
            if idx != 0:
                return False
        pos = idx + len(part)

    if anchored:
        return pos == len(path)

    return True
