"""Property-based tests for the Crawler subsystem.

Property 3: Per-host concurrency never exceeds configured max;
            Crawl-Delay gap is always respected (simulated).
Property 4: Opt-out retroactively excludes domain after 24h (virtual clock).

**Validates: Requirements 1.4, 1.5, 1.6**
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from crawler.opt_out import OptOutRegistry
from crawler.throttle import HostThrottle, MIN_CRAWL_DELAY, MAX_CONCURRENCY, MIN_CONCURRENCY


# ============================================================================
# Property 3: Per-host concurrency and Crawl-Delay (simulated)
# ============================================================================


class TestProperty3ConcurrencyAndDelay:
    """Property 3: Per-host concurrency never exceeds configured max;
    Crawl-Delay gap is always respected.

    **Validates: Requirements 1.4, 1.5**
    """

    @settings(max_examples=50, deadline=10000)
    @given(
        max_concurrency=st.integers(min_value=MIN_CONCURRENCY, max_value=MAX_CONCURRENCY),
        num_requests=st.integers(min_value=2, max_value=10),
    )
    @pytest.mark.asyncio
    async def test_concurrency_never_exceeds_max(self, max_concurrency: int, num_requests: int):
        """Concurrent in-flight requests per host never exceed the configured maximum.

        **Validates: Requirements 1.4**
        """
        throttle = HostThrottle(max_concurrency=max_concurrency, default_crawl_delay=0.0)
        host = "property-test.com"
        # Override crawl delay to 0 for pure concurrency testing
        state = throttle._get_host_state(host)
        state.crawl_delay = 0.0

        max_observed = 0
        current_count = 0
        lock = asyncio.Lock()

        async def worker():
            nonlocal max_observed, current_count
            await throttle.acquire(host)
            try:
                async with lock:
                    current_count += 1
                    max_observed = max(max_observed, current_count)
                await asyncio.sleep(0.01)
                async with lock:
                    current_count -= 1
            finally:
                throttle.release(host)

        tasks = [asyncio.create_task(worker()) for _ in range(num_requests)]
        await asyncio.gather(*tasks)

        assert max_observed <= max_concurrency, (
            f"Observed {max_observed} concurrent requests, max allowed is {max_concurrency}"
        )

    @settings(max_examples=30, deadline=15000)
    @given(
        crawl_delay=st.floats(min_value=0.05, max_value=0.3),
        num_requests=st.integers(min_value=2, max_value=5),
    )
    @pytest.mark.asyncio
    async def test_crawl_delay_gap_always_respected(self, crawl_delay: float, num_requests: int):
        """Sequential requests to the same host are spaced by at least the crawl delay.

        **Validates: Requirements 1.5**

        Note: We use small delays (0.05-0.3s) for test speed. The real minimum
        is max(host_crawl_delay, 1s) but the logic is the same.
        """
        throttle = HostThrottle(max_concurrency=1, default_crawl_delay=crawl_delay)
        host = "delay-test.com"
        # Set the specific crawl delay (bypassing the 1s minimum for testing the logic)
        state = throttle._get_host_state(host)
        state.crawl_delay = crawl_delay

        timestamps: list[float] = []

        for _ in range(num_requests):
            await throttle.acquire(host)
            timestamps.append(time.monotonic())
            throttle.release(host)

        # Verify gaps between consecutive requests
        for i in range(1, len(timestamps)):
            gap = timestamps[i] - timestamps[i - 1]
            # Allow 10ms tolerance for timing jitter
            assert gap >= crawl_delay - 0.01, (
                f"Gap between request {i-1} and {i} was {gap:.4f}s, "
                f"expected >= {crawl_delay:.4f}s"
            )

    @settings(max_examples=30, deadline=10000)
    @given(
        max_concurrency=st.integers(min_value=MIN_CONCURRENCY, max_value=MAX_CONCURRENCY),
    )
    def test_concurrency_clamped_to_valid_range(self, max_concurrency: int):
        """Configured concurrency is always within [1, 8].

        **Validates: Requirements 1.4**
        """
        throttle = HostThrottle(max_concurrency=max_concurrency)
        assert MIN_CONCURRENCY <= throttle.max_concurrency <= MAX_CONCURRENCY

    @settings(max_examples=30, deadline=10000)
    @given(
        raw_concurrency=st.integers(min_value=-100, max_value=200),
    )
    def test_out_of_range_concurrency_clamped(self, raw_concurrency: int):
        """Out-of-range concurrency values are clamped to [1, 8].

        **Validates: Requirements 1.4**
        """
        throttle = HostThrottle(max_concurrency=raw_concurrency)
        assert MIN_CONCURRENCY <= throttle.max_concurrency <= MAX_CONCURRENCY

    @settings(max_examples=30, deadline=10000)
    @given(
        crawl_delay_input=st.floats(min_value=0.0, max_value=100.0),
    )
    def test_crawl_delay_minimum_enforced(self, crawl_delay_input: float):
        """Effective crawl delay is always >= 1s (R1.5).

        **Validates: Requirements 1.5**
        """
        throttle = HostThrottle()
        throttle.set_crawl_delay("test.com", crawl_delay_input)
        effective = throttle.get_crawl_delay("test.com")
        assert effective >= MIN_CRAWL_DELAY, (
            f"Effective delay {effective} < minimum {MIN_CRAWL_DELAY} "
            f"for input {crawl_delay_input}"
        )


# ============================================================================
# Property 4: Opt-out retroactively excludes domain after 24h (virtual clock)
# ============================================================================


class TestProperty4OptOutRetroactive:
    """Property 4: Opt-out retroactively excludes domain after 24h.

    Uses a virtual clock to test the 24h activation window without real delays.

    **Validates: Requirements 1.6**
    """

    @settings(max_examples=100, deadline=5000)
    @given(
        domain=st.from_regex(r"[a-z]{3,10}\.(com|org|net)", fullmatch=True),
        hours_before_check=st.floats(min_value=0.0, max_value=23.99),
    )
    def test_domain_not_excluded_within_24h(self, domain: str, hours_before_check: float):
        """A domain is NOT excluded from crawling within 24h of opt-out.

        **Validates: Requirements 1.6**
        """
        start = datetime(2024, 6, 1, 0, 0, 0, tzinfo=timezone.utc)
        current_time = [start]
        clock = lambda: current_time[0]  # noqa: E731

        registry = OptOutRegistry(clock=clock)
        registry.register_opt_out(domain)

        # Check at some point within 24h
        current_time[0] = start + timedelta(hours=hours_before_check)
        assert registry.is_opted_out(domain) is False, (
            f"Domain {domain} should NOT be opted out at {hours_before_check}h "
            f"(within 24h window)"
        )

    @settings(max_examples=100, deadline=5000)
    @given(
        domain=st.from_regex(r"[a-z]{3,10}\.(com|org|net)", fullmatch=True),
        hours_after_activation=st.floats(min_value=24.0, max_value=720.0),
    )
    def test_domain_excluded_after_24h(self, domain: str, hours_after_activation: float):
        """A domain IS excluded from crawling after 24h of opt-out.

        **Validates: Requirements 1.6**
        """
        start = datetime(2024, 6, 1, 0, 0, 0, tzinfo=timezone.utc)
        current_time = [start]
        clock = lambda: current_time[0]  # noqa: E731

        registry = OptOutRegistry(clock=clock)
        registry.register_opt_out(domain)

        # Check after 24h
        current_time[0] = start + timedelta(hours=hours_after_activation)
        assert registry.is_opted_out(domain) is True, (
            f"Domain {domain} SHOULD be opted out at {hours_after_activation}h "
            f"(past 24h window)"
        )

    @settings(max_examples=100, deadline=5000)
    @given(
        domain=st.from_regex(r"[a-z]{3,10}\.(com|org|net)", fullmatch=True),
    )
    def test_exact_24h_boundary_is_opted_out(self, domain: str):
        """At exactly 24h, the domain IS opted out (>= threshold).

        **Validates: Requirements 1.6**
        """
        start = datetime(2024, 6, 1, 0, 0, 0, tzinfo=timezone.utc)
        current_time = [start]
        clock = lambda: current_time[0]  # noqa: E731

        registry = OptOutRegistry(clock=clock)
        registry.register_opt_out(domain)

        # Exactly at 24h
        current_time[0] = start + timedelta(hours=24)
        assert registry.is_opted_out(domain) is True, (
            f"Domain {domain} should be opted out at exactly 24h"
        )

    @settings(max_examples=50, deadline=5000)
    @given(
        domain=st.from_regex(r"[a-z]{3,10}\.(com|org|net)", fullmatch=True),
        re_register_hours=st.floats(min_value=1.0, max_value=23.0),
    )
    def test_re_registration_does_not_reset_window(self, domain: str, re_register_hours: float):
        """Re-registering a domain does not reset the 24h activation window.

        **Validates: Requirements 1.6**
        """
        start = datetime(2024, 6, 1, 0, 0, 0, tzinfo=timezone.utc)
        current_time = [start]
        clock = lambda: current_time[0]  # noqa: E731

        registry = OptOutRegistry(clock=clock)
        registry.register_opt_out(domain)

        # Re-register at a later time
        current_time[0] = start + timedelta(hours=re_register_hours)
        registry.register_opt_out(domain)

        # At 24h from ORIGINAL registration, should be opted out
        current_time[0] = start + timedelta(hours=24)
        assert registry.is_opted_out(domain) is True, (
            f"Re-registration at {re_register_hours}h should not reset the window"
        )

    @settings(max_examples=50, deadline=5000)
    @given(
        domain=st.from_regex(r"[a-z]{3,10}\.(com|org|net)", fullmatch=True),
    )
    def test_unregistered_domain_never_opted_out(self, domain: str):
        """A domain that was never registered is never opted out.

        **Validates: Requirements 1.6**
        """
        start = datetime(2024, 6, 1, 0, 0, 0, tzinfo=timezone.utc)
        current_time = [start]
        clock = lambda: current_time[0]  # noqa: E731

        registry = OptOutRegistry(clock=clock)

        # Check at various times without registering
        for hours in [0, 12, 24, 48, 100]:
            current_time[0] = start + timedelta(hours=hours)
            assert registry.is_opted_out(domain) is False
