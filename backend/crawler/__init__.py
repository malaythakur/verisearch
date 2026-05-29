"""Crawler - Ethical web crawling with robots.txt compliance and opt-out support.

Provides:
- RobotsCache: TTL-based robots.txt caching with 10s fetch timeout (R1.1).
- HostThrottle: Per-host concurrency control [1,8] and Crawl-Delay enforcement (R1.4, R1.5).
- OptOutRegistry: Domain opt-out with 24h activation window (R1.6).
- CrawlFetcher: Main fetch orchestrator coordinating all checks (R1).
- FetchResult / SkipReason: Data models for fetch outcomes (R1.7).
"""

from backend.crawler.fetcher import CrawlFetcher
from backend.crawler.models import FetchResult, RobotsResult, SkipReason, SkipReasonType
from backend.crawler.opt_out import OptOutRegistry
from backend.crawler.robots import DEFAULT_USER_AGENT, RobotsCache
from backend.crawler.throttle import (
    DEFAULT_CONCURRENCY,
    MAX_CONCURRENCY,
    MIN_CONCURRENCY,
    MIN_CRAWL_DELAY,
    HostThrottle,
)

__all__ = [
    "CrawlFetcher",
    "DEFAULT_CONCURRENCY",
    "DEFAULT_USER_AGENT",
    "FetchResult",
    "HostThrottle",
    "MAX_CONCURRENCY",
    "MIN_CONCURRENCY",
    "MIN_CRAWL_DELAY",
    "OptOutRegistry",
    "RobotsCache",
    "RobotsResult",
    "SkipReason",
    "SkipReasonType",
]
