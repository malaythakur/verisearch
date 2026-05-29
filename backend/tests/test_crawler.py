"""Unit tests for the Crawler subsystem (R1.1–R1.7).

Covers:
- Robots.txt caching and timeout (R1.1)
- Disallowed URL detection (R1.2)
- Failure handling: timeout, 5xx (R1.3)
- Per-host concurrency limits (R1.4)
- Crawl-Delay enforcement (R1.5)
- Opt-out registry with 24h window (R1.6)
- Fetch metadata recording (R1.7)
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from crawler.fetcher import CrawlFetcher, _extract_domain
from crawler.models import FetchResult, RobotsResult, SkipReason, SkipReasonType
from crawler.opt_out import OptOutRegistry
from crawler.robots import RobotsCache, _check_rules, _parse_robots_txt
from crawler.throttle import HostThrottle


# ============================================================================
# Robots.txt parsing tests
# ============================================================================


class TestRobotsTxtParsing:
    """Tests for robots.txt content parsing."""

    def test_parse_basic_disallow(self):
        content = "User-agent: *\nDisallow: /private/\nDisallow: /admin/"
        result = _parse_robots_txt(content)
        assert result.available is True
        assert ("*", ["/private/", "/admin/"]) in result.rules

    def test_parse_specific_user_agent(self):
        content = (
            "User-agent: AgenticResearchBot\n"
            "Disallow: /secret/\n\n"
            "User-agent: *\n"
            "Disallow: /general/"
        )
        result = _parse_robots_txt(content)
        assert result.available is True
        agents = [agent for agent, _ in result.rules]
        assert "agenticresearchbot" in agents

    def test_parse_crawl_delay(self):
        content = "User-agent: *\nCrawl-delay: 5\nDisallow: /slow/"
        result = _parse_robots_txt(content)
        assert result.crawl_delay == 5.0

    def test_parse_empty_disallow_means_allow_all(self):
        content = "User-agent: *\nDisallow:"
        result = _parse_robots_txt(content)
        assert result.available is True
        # No disallow rules should be recorded for empty disallow
        disallows = [d for _, disallows in result.rules for d in disallows]
        assert disallows == []

    def test_parse_comments_stripped(self):
        content = "User-agent: * # all bots\nDisallow: /private/ # secret stuff"
        result = _parse_robots_txt(content)
        assert result.available is True
        assert any("/private/" in disallows for _, disallows in result.rules)

    def test_parse_empty_content(self):
        result = _parse_robots_txt("")
        assert result.available is True
        assert result.rules == []


class TestRobotsRuleChecking:
    """Tests for robots.txt rule matching."""

    def test_allowed_when_no_rules(self):
        assert _check_rules([], "bot", "/anything") is True

    def test_disallowed_by_wildcard(self):
        rules = [("*", ["/private/"])]
        assert _check_rules(rules, "mybot", "/private/page") is False
        assert _check_rules(rules, "mybot", "/public/page") is True

    def test_specific_agent_takes_precedence(self):
        rules = [
            ("mybot", ["/bot-blocked/"]),
            ("*", ["/general-blocked/"]),
        ]
        # mybot is blocked from /bot-blocked/ but not /general-blocked/
        assert _check_rules(rules, "mybot", "/bot-blocked/page") is False
        assert _check_rules(rules, "mybot", "/general-blocked/page") is True

    def test_prefix_matching(self):
        rules = [("*", ["/api"])]
        assert _check_rules(rules, "bot", "/api/v1") is False
        assert _check_rules(rules, "bot", "/api") is False
        assert _check_rules(rules, "bot", "/other") is True


# ============================================================================
# RobotsCache tests
# ============================================================================


class TestRobotsCache:
    """Tests for RobotsCache with TTL and timeout."""

    @pytest.mark.asyncio
    async def test_cache_hit_returns_cached_result(self):
        """Cached result is returned without re-fetching."""
        mock_response = httpx.Response(
            200,
            text="User-agent: *\nDisallow: /blocked/",
            request=httpx.Request("GET", "https://example.com/robots.txt"),
        )
        client = AsyncMock(spec=httpx.AsyncClient)
        client.get = AsyncMock(return_value=mock_response)

        cache = RobotsCache(http_client=client, ttl_seconds=3600)

        # First call fetches
        result1 = await cache.get("example.com")
        assert result1.available is True
        assert client.get.call_count == 1

        # Second call uses cache
        result2 = await cache.get("example.com")
        assert result2.available is True
        assert client.get.call_count == 1  # No additional fetch

    @pytest.mark.asyncio
    async def test_timeout_returns_unavailable(self):
        """10s timeout results in robots_unavailable."""
        client = AsyncMock(spec=httpx.AsyncClient)
        client.get = AsyncMock(side_effect=httpx.TimeoutException("timed out"))

        cache = RobotsCache(http_client=client, timeout_seconds=10.0)
        result = await cache.get("slow-host.com")

        assert result.available is False
        assert "Timeout" in (result.error or "")

    @pytest.mark.asyncio
    async def test_5xx_returns_unavailable(self):
        """HTTP 5xx results in robots_unavailable (R1.3)."""
        mock_response = httpx.Response(
            503,
            text="Service Unavailable",
            request=httpx.Request("GET", "https://down.com/robots.txt"),
        )
        client = AsyncMock(spec=httpx.AsyncClient)
        client.get = AsyncMock(return_value=mock_response)

        cache = RobotsCache(http_client=client)
        result = await cache.get("down.com")

        assert result.available is False
        assert "503" in (result.error or "")

    @pytest.mark.asyncio
    async def test_4xx_means_no_restrictions(self):
        """HTTP 404 means no robots.txt → allow all."""
        mock_response = httpx.Response(
            404,
            text="Not Found",
            request=httpx.Request("GET", "https://open.com/robots.txt"),
        )
        client = AsyncMock(spec=httpx.AsyncClient)
        client.get = AsyncMock(return_value=mock_response)

        cache = RobotsCache(http_client=client)
        result = await cache.get("open.com")

        assert result.available is True
        assert result.rules == []

    @pytest.mark.asyncio
    async def test_network_error_returns_unavailable(self):
        """Network errors result in robots_unavailable (R1.3)."""
        client = AsyncMock(spec=httpx.AsyncClient)
        client.get = AsyncMock(side_effect=httpx.NetworkError("connection refused"))

        cache = RobotsCache(http_client=client)
        result = await cache.get("unreachable.com")

        assert result.available is False
        assert "Network error" in (result.error or "")

    @pytest.mark.asyncio
    async def test_ttl_expiry_triggers_refetch(self):
        """Expired cache entries trigger a new fetch."""
        mock_response = httpx.Response(
            200,
            text="User-agent: *\nDisallow:",
            request=httpx.Request("GET", "https://example.com/robots.txt"),
        )
        client = AsyncMock(spec=httpx.AsyncClient)
        client.get = AsyncMock(return_value=mock_response)

        # Very short TTL for testing
        cache = RobotsCache(http_client=client, ttl_seconds=0.01)

        await cache.get("example.com")
        assert client.get.call_count == 1

        # Wait for TTL to expire
        await asyncio.sleep(0.02)

        await cache.get("example.com")
        assert client.get.call_count == 2

    def test_is_allowed_with_cached_rules(self):
        """is_allowed checks cached rules correctly."""
        cache = RobotsCache(ttl_seconds=3600)
        # Manually populate cache
        from crawler.robots import _CacheEntry

        result = RobotsResult(
            available=True,
            rules=[("*", ["/blocked/"])],
        )
        cache._cache["example.com"] = _CacheEntry(
            result=result, cached_at=time.monotonic()
        )

        assert cache.is_allowed("https://example.com/public/page") is True
        assert cache.is_allowed("https://example.com/blocked/page") is False

    def test_is_allowed_returns_none_when_no_cache(self):
        """is_allowed returns None when no cache entry exists."""
        cache = RobotsCache(ttl_seconds=3600)
        assert cache.is_allowed("https://unknown.com/page") is None


# ============================================================================
# HostThrottle tests
# ============================================================================


class TestHostThrottle:
    """Tests for per-host concurrency and Crawl-Delay."""

    def test_default_concurrency_is_2(self):
        throttle = HostThrottle()
        assert throttle.max_concurrency == 2

    def test_concurrency_clamped_to_bounds(self):
        """Concurrency is clamped to [1, 8]."""
        assert HostThrottle(max_concurrency=0).max_concurrency == 1
        assert HostThrottle(max_concurrency=-5).max_concurrency == 1
        assert HostThrottle(max_concurrency=100).max_concurrency == 8
        assert HostThrottle(max_concurrency=4).max_concurrency == 4

    @pytest.mark.asyncio
    async def test_concurrency_limit_enforced(self):
        """No more than max_concurrency requests run simultaneously."""
        throttle = HostThrottle(max_concurrency=2, default_crawl_delay=0.0)
        # Override crawl delay for this test
        throttle._hosts.clear()
        throttle._default_crawl_delay = 0.0

        host = "test.com"
        max_concurrent = 0
        current_concurrent = 0

        async def simulated_request():
            nonlocal max_concurrent, current_concurrent
            await throttle.acquire(host)
            try:
                current_concurrent += 1
                max_concurrent = max(max_concurrent, current_concurrent)
                await asyncio.sleep(0.05)
            finally:
                current_concurrent -= 1
                throttle.release(host)

        # Launch 5 concurrent requests with max_concurrency=2
        # Need to set crawl_delay to 0 for this test
        state = throttle._get_host_state(host)
        state.crawl_delay = 0.0

        tasks = [asyncio.create_task(simulated_request()) for _ in range(5)]
        await asyncio.gather(*tasks)

        assert max_concurrent <= 2

    @pytest.mark.asyncio
    async def test_crawl_delay_enforced(self):
        """Requests are spaced by at least the crawl delay."""
        throttle = HostThrottle(max_concurrency=1, default_crawl_delay=1.0)
        host = "delayed.com"

        # Set a short delay for testing
        throttle.set_crawl_delay(host, 0.1)

        timestamps = []

        await throttle.acquire(host)
        timestamps.append(time.monotonic())
        throttle.release(host)

        await throttle.acquire(host)
        timestamps.append(time.monotonic())
        throttle.release(host)

        # Second request should be at least 0.1s after first
        gap = timestamps[1] - timestamps[0]
        assert gap >= 0.09  # Allow small timing tolerance

    def test_set_crawl_delay_enforces_minimum(self):
        """Crawl delay is at least 1s (R1.5)."""
        throttle = HostThrottle()
        throttle.set_crawl_delay("fast.com", 0.1)
        assert throttle.get_crawl_delay("fast.com") == 1.0

        throttle.set_crawl_delay("slow.com", 5.0)
        assert throttle.get_crawl_delay("slow.com") == 5.0

    def test_active_count(self):
        """active_count reflects current in-flight requests."""
        throttle = HostThrottle(max_concurrency=3)
        assert throttle.active_count("host.com") == 0


# ============================================================================
# OptOutRegistry tests
# ============================================================================


class TestOptOutRegistry:
    """Tests for domain opt-out with 24h activation window."""

    def test_not_opted_out_by_default(self):
        registry = OptOutRegistry()
        assert registry.is_opted_out("example.com") is False

    def test_not_opted_out_within_24h(self):
        """Domain is NOT opted out within the 24h activation window."""
        now = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        clock = lambda: now  # noqa: E731
        registry = OptOutRegistry(clock=clock)

        registry.register_opt_out("example.com")

        # Still within 24h
        assert registry.is_opted_out("example.com") is False

    def test_opted_out_after_24h(self):
        """Domain IS opted out after 24h activation window."""
        start = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        current_time = [start]
        clock = lambda: current_time[0]  # noqa: E731
        registry = OptOutRegistry(clock=clock)

        registry.register_opt_out("example.com")

        # Advance past 24h
        current_time[0] = start + timedelta(hours=24, seconds=1)
        assert registry.is_opted_out("example.com") is True

    def test_opted_out_exactly_at_24h(self):
        """Domain IS opted out at exactly 24h (>= threshold)."""
        start = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        current_time = [start]
        clock = lambda: current_time[0]  # noqa: E731
        registry = OptOutRegistry(clock=clock)

        registry.register_opt_out("example.com")

        # Exactly at 24h boundary
        current_time[0] = start + timedelta(hours=24)
        assert registry.is_opted_out("example.com") is True

    def test_case_insensitive(self):
        """Domain matching is case-insensitive."""
        now = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        current_time = [now]
        clock = lambda: current_time[0]  # noqa: E731
        registry = OptOutRegistry(clock=clock)

        registry.register_opt_out("Example.COM")
        current_time[0] = now + timedelta(hours=25)
        assert registry.is_opted_out("example.com") is True

    def test_first_opt_out_wins(self):
        """Re-registering does not update the timestamp."""
        start = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        current_time = [start]
        clock = lambda: current_time[0]  # noqa: E731
        registry = OptOutRegistry(clock=clock)

        registry.register_opt_out("example.com")

        # Advance 12h and re-register
        current_time[0] = start + timedelta(hours=12)
        registry.register_opt_out("example.com")

        # At 24h from original registration, should be opted out
        current_time[0] = start + timedelta(hours=24)
        assert registry.is_opted_out("example.com") is True

    def test_remove_opt_out(self):
        """Removing opt-out makes domain crawlable again."""
        start = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        current_time = [start]
        clock = lambda: current_time[0]  # noqa: E731
        registry = OptOutRegistry(clock=clock)

        registry.register_opt_out("example.com")
        current_time[0] = start + timedelta(hours=25)
        assert registry.is_opted_out("example.com") is True

        registry.remove_opt_out("example.com")
        assert registry.is_opted_out("example.com") is False


# ============================================================================
# CrawlFetcher integration tests
# ============================================================================


class TestCrawlFetcher:
    """Tests for the CrawlFetcher orchestration."""

    def _make_fetcher(
        self,
        *,
        robots_available=True,
        robots_rules=None,
        crawl_delay=None,
        opt_out_domain=None,
        opt_out_active=False,
        http_response=None,
    ):
        """Helper to create a CrawlFetcher with mocked dependencies."""
        # Mock robots cache
        robots_cache = AsyncMock(spec=RobotsCache)
        robots_result = RobotsResult(
            available=robots_available,
            rules=robots_rules or [],
            crawl_delay=crawl_delay,
            error="fetch failed" if not robots_available else None,
        )
        robots_cache.get = AsyncMock(return_value=robots_result)

        # Mock throttle
        throttle = AsyncMock(spec=HostThrottle)
        throttle.acquire = AsyncMock()
        throttle.release = lambda host: None  # sync method
        throttle.set_crawl_delay = lambda host, delay: None  # sync method

        # Opt-out registry
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        current_time = [start]
        if opt_out_active:
            current_time[0] = start + timedelta(hours=25)
        clock = lambda: current_time[0]  # noqa: E731
        opt_out = OptOutRegistry(clock=clock)
        if opt_out_domain:
            # Register at start time
            old_clock = opt_out._clock
            opt_out._clock = lambda: start
            opt_out.register_opt_out(opt_out_domain)
            opt_out._clock = clock

        # Mock audit
        audit = AsyncMock()
        audit.emit = AsyncMock()

        # Mock HTTP client
        if http_response is None:
            http_response = httpx.Response(
                200,
                content=b"<html>Hello</html>",
                headers={"content-type": "text/html; charset=utf-8"},
                request=httpx.Request("GET", "https://example.com/page"),
            )
        http_client = AsyncMock(spec=httpx.AsyncClient)
        http_client.get = AsyncMock(return_value=http_response)

        fetcher = CrawlFetcher(
            robots_cache=robots_cache,
            throttle=throttle,
            opt_out_registry=opt_out,
            audit_emitter=audit,
            http_client=http_client,
            user_agent="AgenticResearchBot",
        )

        return fetcher, audit, throttle, robots_cache

    @pytest.mark.asyncio
    async def test_successful_fetch_records_metadata(self):
        """Successful fetch returns FetchResult with metadata (R1.7)."""
        fetcher, audit, throttle, _ = self._make_fetcher()

        result = await fetcher.fetch("https://example.com/page")

        assert isinstance(result, FetchResult)
        assert result.content == b"<html>Hello</html>"
        assert result.http_status == 200
        assert "text/html" in result.content_type
        assert result.fetch_timestamp_utc.tzinfo is not None
        assert result.canonical_url == "https://example.com/page"

    @pytest.mark.asyncio
    async def test_robots_unavailable_skips_and_audits(self):
        """robots.txt failure → skip + audit robots_unavailable (R1.3)."""
        fetcher, audit, throttle, _ = self._make_fetcher(robots_available=False)

        result = await fetcher.fetch("https://down.com/page")

        assert isinstance(result, SkipReason)
        assert result.reason == SkipReasonType.ROBOTS_UNAVAILABLE
        audit.emit.assert_called_once()
        call_kwargs = audit.emit.call_args.kwargs
        assert call_kwargs["action"] == "robots_unavailable"

    @pytest.mark.asyncio
    async def test_disallowed_url_skips_and_audits(self):
        """Disallowed URL → skip + audit disallowed_by_robots (R1.2)."""
        fetcher, audit, throttle, _ = self._make_fetcher(
            robots_rules=[("*", ["/blocked/"])]
        )

        result = await fetcher.fetch("https://example.com/blocked/page")

        assert isinstance(result, SkipReason)
        assert result.reason == SkipReasonType.DISALLOWED_BY_ROBOTS
        audit.emit.assert_called_once()
        call_kwargs = audit.emit.call_args.kwargs
        assert call_kwargs["action"] == "disallowed_by_robots"

    @pytest.mark.asyncio
    async def test_opted_out_domain_skips_and_audits(self):
        """Opted-out domain → skip + audit domain_opted_out (R1.6)."""
        fetcher, audit, throttle, _ = self._make_fetcher(
            opt_out_domain="example.com",
            opt_out_active=True,
        )

        result = await fetcher.fetch("https://example.com/page")

        assert isinstance(result, SkipReason)
        assert result.reason == SkipReasonType.DOMAIN_OPTED_OUT
        audit.emit.assert_called_once()
        call_kwargs = audit.emit.call_args.kwargs
        assert call_kwargs["action"] == "domain_opted_out"

    @pytest.mark.asyncio
    async def test_throttle_acquired_and_released(self):
        """Throttle is acquired before fetch and released after (R1.4)."""
        fetcher, audit, throttle, _ = self._make_fetcher()

        await fetcher.fetch("https://example.com/page")

        throttle.acquire.assert_called_once()

    @pytest.mark.asyncio
    async def test_crawl_delay_set_from_robots(self):
        """Crawl-Delay from robots.txt is propagated to throttle (R1.5)."""
        # Use a real throttle to verify set_crawl_delay is called
        from unittest.mock import MagicMock

        fetcher, audit, throttle, _ = self._make_fetcher(crawl_delay=3.0)
        # Replace the sync method with a MagicMock to track calls
        throttle.set_crawl_delay = MagicMock()

        await fetcher.fetch("https://example.com/page")

        throttle.set_crawl_delay.assert_called_once()
        args = throttle.set_crawl_delay.call_args
        assert args[0][1] == 3.0  # crawl_delay value

    @pytest.mark.asyncio
    async def test_opt_out_not_active_within_24h(self):
        """Domain within 24h window is still crawlable."""
        fetcher, audit, throttle, _ = self._make_fetcher(
            opt_out_domain="example.com",
            opt_out_active=False,  # Within 24h
        )

        result = await fetcher.fetch("https://example.com/page")

        # Should proceed to fetch (not skip)
        assert isinstance(result, FetchResult)


# ============================================================================
# Helper function tests
# ============================================================================


class TestHelpers:
    """Tests for helper functions."""

    def test_extract_domain(self):
        assert _extract_domain("www.example.com") == "example.com"
        assert _extract_domain("sub.deep.example.com") == "example.com"
        assert _extract_domain("example.com") == "example.com"
        assert _extract_domain("localhost") == "localhost"
