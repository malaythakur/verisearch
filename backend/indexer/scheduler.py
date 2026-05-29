"""Priority-source re-crawl scheduling (Task 10.6, R2.2).

Ensures priority sources are re-crawled at least once per rolling 24-hour
window measured in UTC.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone


# Maximum time between re-crawls for priority sources (R2.2)
PRIORITY_RECRAWL_WINDOW_HOURS = 24


@dataclass
class PrioritySource:
    """A URL designated as a priority source for frequent re-crawling.

    Attributes:
        url: The URL to re-crawl.
        last_crawled_at: When this URL was last crawled (UTC).
        added_at: When this URL was added as a priority source.
    """

    url: str
    last_crawled_at: datetime | None = None
    added_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class PriorityScheduler:
    """Scheduler that ensures priority sources are re-crawled within 24h windows.

    Maintains a registry of priority source URLs and tracks their last crawl
    time. Provides methods to determine which sources are due for re-crawling.

    In production, this would be backed by a persistent store and integrated
    with the Crawler's URL frontier.
    """

    def __init__(self) -> None:
        self._sources: dict[str, PrioritySource] = {}

    @property
    def sources(self) -> dict[str, PrioritySource]:
        """Return all registered priority sources."""
        return dict(self._sources)

    def add_source(self, url: str) -> PrioritySource:
        """Register a URL as a priority source.

        Args:
            url: The URL to register.

        Returns:
            The PrioritySource entry.
        """
        if url not in self._sources:
            self._sources[url] = PrioritySource(url=url)
        return self._sources[url]

    def remove_source(self, url: str) -> bool:
        """Remove a URL from priority sources.

        Args:
            url: The URL to remove.

        Returns:
            True if the URL was removed, False if not found.
        """
        if url in self._sources:
            del self._sources[url]
            return True
        return False

    def record_crawl(self, url: str, crawled_at: datetime | None = None) -> None:
        """Record that a priority source was crawled.

        Args:
            url: The URL that was crawled.
            crawled_at: When the crawl occurred (defaults to now UTC).
        """
        if url in self._sources:
            self._sources[url].last_crawled_at = crawled_at or datetime.now(timezone.utc)

    def get_due_sources(self, now: datetime | None = None) -> list[PrioritySource]:
        """Get priority sources that are due for re-crawling.

        A source is due if:
        - It has never been crawled, OR
        - Its last crawl was more than 24 hours ago.

        Args:
            now: Current time (defaults to now UTC). Useful for testing.

        Returns:
            List of PrioritySource entries that need re-crawling.
        """
        if now is None:
            now = datetime.now(timezone.utc)

        window = timedelta(hours=PRIORITY_RECRAWL_WINDOW_HOURS)
        due: list[PrioritySource] = []

        for source in self._sources.values():
            if source.last_crawled_at is None:
                due.append(source)
            elif (now - source.last_crawled_at) >= window:
                due.append(source)

        return due

    def is_source_due(self, url: str, now: datetime | None = None) -> bool:
        """Check if a specific priority source is due for re-crawling.

        Args:
            url: The URL to check.
            now: Current time (defaults to now UTC).

        Returns:
            True if the source is due, False otherwise.
            Returns False if the URL is not a registered priority source.
        """
        source = self._sources.get(url)
        if source is None:
            return False

        if now is None:
            now = datetime.now(timezone.utc)

        if source.last_crawled_at is None:
            return True

        window = timedelta(hours=PRIORITY_RECRAWL_WINDOW_HOURS)
        return (now - source.last_crawled_at) >= window
