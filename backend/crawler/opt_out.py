"""Opt-out registry for domain-level crawl exclusion (R1.6).

Implements:
- OptOutRegistry: In-memory store of opted-out domains with 24h activation window.
- Domains are excluded from crawling only after 24 hours have passed since opt-out.
- Emits `domain_opted_out` audit entries when a URL is skipped due to opt-out.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Callable


# Activation window: 24 hours after opt-out acceptance
_ACTIVATION_WINDOW_HOURS = 24


class OptOutRegistry:
    """Registry of domains that have opted out of crawling (R1.6).

    A domain's opt-out becomes effective 24 hours after the opt-out is recorded.
    Before that window elapses, the domain is still crawlable.

    Args:
        clock: Optional callable returning current UTC datetime. Defaults to
            datetime.now(timezone.utc). Useful for testing with virtual clocks.
    """

    def __init__(self, *, clock: Callable[[], datetime] | None = None) -> None:
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        # domain -> opt-out acceptance timestamp (UTC)
        self._registry: dict[str, datetime] = {}

    def register_opt_out(self, domain: str) -> None:
        """Record a domain opt-out with the current timestamp.

        The opt-out becomes effective 24 hours after this call.
        If the domain is already registered, the timestamp is NOT updated
        (first opt-out wins).

        Args:
            domain: The registrable domain (e.g., "example.com").
        """
        domain = domain.lower().strip()
        if domain not in self._registry:
            self._registry[domain] = self._clock()

    def is_opted_out(self, domain: str) -> bool:
        """Check if a domain's opt-out is active (>24h since registration).

        Args:
            domain: The registrable domain to check.

        Returns:
            True if the domain opted out more than 24 hours ago.
            False if not opted out or within the 24h activation window.
        """
        domain = domain.lower().strip()
        opt_out_time = self._registry.get(domain)

        if opt_out_time is None:
            return False

        now = self._clock()
        activation_threshold = opt_out_time + timedelta(hours=_ACTIVATION_WINDOW_HOURS)
        return now >= activation_threshold

    def get_opt_out_time(self, domain: str) -> datetime | None:
        """Get the opt-out timestamp for a domain, or None if not registered."""
        domain = domain.lower().strip()
        return self._registry.get(domain)

    def remove_opt_out(self, domain: str) -> bool:
        """Remove a domain from the opt-out registry.

        Args:
            domain: The domain to remove.

        Returns:
            True if the domain was in the registry, False otherwise.
        """
        domain = domain.lower().strip()
        return self._registry.pop(domain, None) is not None

    @property
    def registered_domains(self) -> list[str]:
        """Return all registered domains (regardless of activation status)."""
        return list(self._registry.keys())

    def clear(self) -> None:
        """Clear all opt-out registrations."""
        self._registry.clear()
