"""Per-host concurrency and Crawl-Delay enforcement (R1.4, R1.5).

Implements:
- HostThrottle: Manages per-host asyncio.Semaphore for concurrency control
  and enforces minimum delay between sequential requests.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field


# Concurrency bounds (R1.4)
MIN_CONCURRENCY = 1
MAX_CONCURRENCY = 8
DEFAULT_CONCURRENCY = 2

# Minimum crawl delay in seconds (R1.5)
MIN_CRAWL_DELAY = 1.0


@dataclass
class _HostState:
    """Internal state for a single host."""

    semaphore: asyncio.Semaphore
    last_request_time: float = 0.0  # monotonic time of last request completion
    crawl_delay: float = MIN_CRAWL_DELAY
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class HostThrottle:
    """Per-host concurrency and delay management (R1.4, R1.5).

    Controls:
    - Concurrent in-flight requests per host: capped at [1, 8], default 2.
    - Sequential request spacing: max(host_crawl_delay, 1s) between requests.

    Args:
        max_concurrency: Maximum concurrent requests per host. Clamped to [1, 8].
        default_crawl_delay: Default delay between requests when no Crawl-Delay
            directive is present. Must be >= 1.0s.
    """

    def __init__(
        self,
        *,
        max_concurrency: int = DEFAULT_CONCURRENCY,
        default_crawl_delay: float = MIN_CRAWL_DELAY,
    ) -> None:
        self._max_concurrency = max(MIN_CONCURRENCY, min(max_concurrency, MAX_CONCURRENCY))
        self._default_crawl_delay = max(default_crawl_delay, MIN_CRAWL_DELAY)
        self._hosts: dict[str, _HostState] = {}

    @property
    def max_concurrency(self) -> int:
        """Return the configured max concurrency per host."""
        return self._max_concurrency

    def _get_host_state(self, host: str) -> _HostState:
        """Get or create state for a host."""
        if host not in self._hosts:
            self._hosts[host] = _HostState(
                semaphore=asyncio.Semaphore(self._max_concurrency),
                crawl_delay=self._default_crawl_delay,
            )
        return self._hosts[host]

    def set_crawl_delay(self, host: str, crawl_delay: float | None) -> None:
        """Set the Crawl-Delay for a host (R1.5).

        The effective delay is max(crawl_delay, 1s).

        Args:
            host: The hostname.
            crawl_delay: The Crawl-Delay value from robots.txt, or None.
        """
        state = self._get_host_state(host)
        if crawl_delay is not None:
            state.crawl_delay = max(crawl_delay, MIN_CRAWL_DELAY)
        else:
            state.crawl_delay = self._default_crawl_delay

    def get_crawl_delay(self, host: str) -> float:
        """Get the effective crawl delay for a host."""
        state = self._get_host_state(host)
        return state.crawl_delay

    async def acquire(self, host: str) -> None:
        """Acquire permission to make a request to a host.

        This method:
        1. Acquires the per-host semaphore (blocks if at concurrency limit).
        2. Waits for the crawl delay to elapse since the last request.

        Args:
            host: The hostname to acquire a slot for.
        """
        state = self._get_host_state(host)

        # Acquire semaphore (concurrency control)
        await state.semaphore.acquire()

        # Enforce crawl delay under lock
        async with state.lock:
            now = time.monotonic()
            elapsed = now - state.last_request_time
            remaining_delay = state.crawl_delay - elapsed

            if remaining_delay > 0 and state.last_request_time > 0:
                await asyncio.sleep(remaining_delay)

    def release(self, host: str) -> None:
        """Release a request slot for a host.

        Must be called after the request completes (in a finally block).
        Updates the last_request_time for crawl delay enforcement.

        Args:
            host: The hostname to release the slot for.
        """
        state = self._get_host_state(host)
        state.last_request_time = time.monotonic()
        state.semaphore.release()

    def active_count(self, host: str) -> int:
        """Return the number of currently active requests for a host.

        Useful for testing and monitoring.
        """
        state = self._get_host_state(host)
        # Semaphore value = max_concurrency - active_count
        return self._max_concurrency - state.semaphore._value

    def reset(self, host: str | None = None) -> None:
        """Reset throttle state for a host or all hosts.

        Args:
            host: If provided, reset only this host. Otherwise reset all.
        """
        if host is not None:
            self._hosts.pop(host, None)
        else:
            self._hosts.clear()
