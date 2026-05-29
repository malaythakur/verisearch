"""Main fetch orchestration for the Crawler subsystem (R1).

Implements CrawlFetcher which coordinates:
- Opt-out check (R1.6)
- Robots.txt check (R1.1, R1.2, R1.3)
- Per-host throttling (R1.4, R1.5)
- Content fetch with metadata recording (R1.7)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Protocol
from urllib.parse import urlparse

import httpx

from backend.crawler.models import FetchResult, RobotsResult, SkipReason, SkipReasonType
from backend.crawler.opt_out import OptOutRegistry
from backend.crawler.robots import RobotsCache
from backend.crawler.throttle import HostThrottle


class AuditEmitter(Protocol):
    """Protocol for audit event emission."""

    async def emit(
        self,
        *,
        action: str,
        tenant_id: str | None,
        actor: str,
        resource: str,
        request_id: str,
        detail: dict,
    ) -> None: ...


class CrawlFetcher:
    """Orchestrates the full crawl pipeline for a single URL.

    Coordinates robots check, opt-out check, throttle, and fetch.
    Records metadata per R1.7.

    Args:
        robots_cache: RobotsCache instance for robots.txt lookups.
        throttle: HostThrottle instance for concurrency/delay control.
        opt_out_registry: OptOutRegistry for domain opt-out checks.
        audit_emitter: AuditEmitter for recording skip reasons.
        http_client: Optional httpx.AsyncClient for content fetching.
        user_agent: User agent string for robots.txt matching.
        fetch_timeout: Timeout for content fetch requests in seconds.
    """

    def __init__(
        self,
        *,
        robots_cache: RobotsCache,
        throttle: HostThrottle,
        opt_out_registry: OptOutRegistry,
        audit_emitter: AuditEmitter,
        http_client: httpx.AsyncClient | None = None,
        user_agent: str = "AgenticResearchBot",
        fetch_timeout: float = 30.0,
    ) -> None:
        self._robots_cache = robots_cache
        self._throttle = throttle
        self._opt_out = opt_out_registry
        self._audit = audit_emitter
        self._client = http_client
        self._user_agent = user_agent
        self._fetch_timeout = fetch_timeout

    async def fetch(self, url: str) -> FetchResult | SkipReason:
        """Fetch a URL, respecting robots.txt, opt-outs, and throttling.

        Pipeline:
        1. Check opt-out registry (R1.6)
        2. Fetch/check robots.txt (R1.1, R1.2, R1.3)
        3. Acquire throttle slot (R1.4, R1.5)
        4. Fetch content and record metadata (R1.7)

        Args:
            url: The URL to fetch.

        Returns:
            FetchResult on success, SkipReason if the URL should be skipped.
        """
        parsed = urlparse(url)
        host = parsed.hostname or ""
        domain = _extract_domain(host)
        request_id = _generate_request_id()

        # Step 1: Check opt-out registry (R1.6)
        if self._opt_out.is_opted_out(domain):
            skip = SkipReason(
                reason=SkipReasonType.DOMAIN_OPTED_OUT,
                url=url,
                detail={"domain": domain},
            )
            await self._audit.emit(
                action="domain_opted_out",
                tenant_id=None,
                actor="crawler",
                resource=url,
                request_id=request_id,
                detail={"domain": domain, "url": url},
            )
            return skip

        # Step 2: Check robots.txt (R1.1, R1.2, R1.3)
        robots_result = await self._robots_cache.get(host)

        if not robots_result.available:
            # R1.3: robots.txt unavailable → skip
            skip = SkipReason(
                reason=SkipReasonType.ROBOTS_UNAVAILABLE,
                url=url,
                detail={"host": host, "error": robots_result.error or "unknown"},
            )
            await self._audit.emit(
                action="robots_unavailable",
                tenant_id=None,
                actor="crawler",
                resource=url,
                request_id=request_id,
                detail={"host": host, "error": robots_result.error or "unknown"},
            )
            return skip

        # Check if URL is allowed (R1.2)
        path = parsed.path or "/"
        if not _is_url_allowed(robots_result, path, self._user_agent):
            skip = SkipReason(
                reason=SkipReasonType.DISALLOWED_BY_ROBOTS,
                url=url,
                detail={"host": host, "path": path},
            )
            await self._audit.emit(
                action="disallowed_by_robots",
                tenant_id=None,
                actor="crawler",
                resource=url,
                request_id=request_id,
                detail={"host": host, "path": path, "user_agent": self._user_agent},
            )
            return skip

        # Set crawl delay from robots.txt (R1.5)
        if robots_result.crawl_delay is not None:
            self._throttle.set_crawl_delay(host, robots_result.crawl_delay)

        # Step 3: Acquire throttle slot (R1.4, R1.5)
        await self._throttle.acquire(host)
        try:
            # Step 4: Fetch content (R1.7)
            return await self._fetch_content(url)
        finally:
            self._throttle.release(host)

    async def _fetch_content(self, url: str) -> FetchResult:
        """Perform the actual HTTP fetch and record metadata (R1.7)."""
        client = self._client or httpx.AsyncClient()
        should_close = self._client is None

        try:
            response = await client.get(
                url,
                timeout=self._fetch_timeout,
                follow_redirects=True,
                headers={"User-Agent": self._user_agent},
            )

            fetch_timestamp = datetime.now(timezone.utc)

            # Determine canonical URL (after redirects)
            canonical_url = str(response.url)

            return FetchResult(
                content=response.content,
                fetch_timestamp_utc=fetch_timestamp,
                http_status=response.status_code,
                content_type=response.headers.get("content-type", ""),
                canonical_url=canonical_url,
            )
        finally:
            if should_close:
                await client.aclose()


def _extract_domain(host: str) -> str:
    """Extract the registrable domain from a hostname.

    Simple implementation: returns the last two parts of the hostname.
    For production, use a proper public suffix list library.
    """
    parts = host.lower().split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return host.lower()


def _is_url_allowed(robots_result: RobotsResult, path: str, user_agent: str) -> bool:
    """Check if a path is allowed by the robots.txt rules."""
    from crawler.robots import _check_rules

    return _check_rules(robots_result.rules, user_agent, path)


def _generate_request_id() -> str:
    """Generate a request ID (16-64 code points per R15.1)."""
    return f"crawl-{uuid.uuid4().hex}"
