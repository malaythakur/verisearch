"""Shared fixtures and utilities for load tests."""

import asyncio
import statistics
import time
from dataclasses import dataclass, field
from typing import Callable, Awaitable

import pytest


@dataclass
class LatencyResult:
    """Captures latency measurements from a load test run."""

    latencies_ms: list[float] = field(default_factory=list)
    errors: int = 0
    total_requests: int = 0

    @property
    def p50(self) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_lats = sorted(self.latencies_ms)
        idx = int(len(sorted_lats) * 0.50)
        return sorted_lats[min(idx, len(sorted_lats) - 1)]

    @property
    def p95(self) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_lats = sorted(self.latencies_ms)
        idx = int(len(sorted_lats) * 0.95)
        return sorted_lats[min(idx, len(sorted_lats) - 1)]

    @property
    def p99(self) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_lats = sorted(self.latencies_ms)
        idx = int(len(sorted_lats) * 0.99)
        return sorted_lats[min(idx, len(sorted_lats) - 1)]

    @property
    def mean(self) -> float:
        if not self.latencies_ms:
            return 0.0
        return statistics.mean(self.latencies_ms)

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return (self.total_requests - self.errors) / self.total_requests

    def summary(self) -> str:
        return (
            f"Requests: {self.total_requests} | Errors: {self.errors} | "
            f"Success Rate: {self.success_rate:.2%}\n"
            f"Latency — p50: {self.p50:.1f}ms | p95: {self.p95:.1f}ms | "
            f"p99: {self.p99:.1f}ms | mean: {self.mean:.1f}ms"
        )


async def run_load_test(
    func: Callable[[], Awaitable[None]],
    num_requests: int = 100,
    concurrency: int = 10,
) -> LatencyResult:
    """Run a load test with the given async function.

    Args:
        func: Async callable that performs one request. Should raise on error.
        num_requests: Total number of requests to make.
        concurrency: Maximum concurrent requests.

    Returns:
        LatencyResult with timing data.
    """
    result = LatencyResult()
    semaphore = asyncio.Semaphore(concurrency)

    async def _run_one():
        async with semaphore:
            start = time.perf_counter()
            try:
                await func()
                elapsed_ms = (time.perf_counter() - start) * 1000
                result.latencies_ms.append(elapsed_ms)
            except Exception:
                result.errors += 1
            finally:
                result.total_requests += 1

    tasks = [asyncio.create_task(_run_one()) for _ in range(num_requests)]
    await asyncio.gather(*tasks)
    return result


@pytest.fixture
def slo_runner():
    """Fixture providing the load test runner."""
    return run_load_test
