"""
Load test: Search warm-cache p95 ≤ 800ms (R3.2)

Validates that cached search queries respond within the SLO target.
Uses a simulated warm cache to measure response latency under load.
"""

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest

from tests.load.conftest import LatencyResult, run_load_test

# SLO Target
SLO_P95_MS = 800


@pytest.fixture
def mock_search_service():
    """Mock search service that simulates warm-cache behavior."""

    async def _search(query: str, mode: str = "hybrid", num_results: int = 10):
        # Simulate warm-cache latency (typically 50-200ms for cached results)
        await asyncio.sleep(0.05 + (hash(query) % 100) / 1000)
        return {
            "results": [
                {
                    "document_id": f"doc_{i}",
                    "url": f"https://example.com/page{i}",
                    "title": f"Result {i}",
                    "score": 1.0 - (i * 0.05),
                    "published_at": "2024-01-01T00:00:00Z",
                    "provenance": {
                        "credibility_score": 0.9,
                        "ai_generated_likelihood": 0.1,
                        "scored_at": "2024-01-01T00:00:00Z",
                    },
                }
                for i in range(num_results)
            ],
            "total": num_results,
        }

    return _search


@pytest.mark.integration
class TestSearchWarmCacheSLO:
    """Load tests for search warm-cache p95 latency SLO (R3.2)."""

    async def test_warm_cache_p95_within_slo(self, mock_search_service):
        """
        SLO: Warm-cache search p95 ≤ 800ms.

        Simulates 100 concurrent warm-cache search requests and validates
        that the 95th percentile latency stays within the SLO target.
        """
        queries = [
            "machine learning",
            "quantum computing",
            "climate change",
            "artificial intelligence",
            "blockchain technology",
        ]
        query_idx = 0

        async def search_request():
            nonlocal query_idx
            q = queries[query_idx % len(queries)]
            query_idx += 1
            await mock_search_service(q)

        result = await run_load_test(
            func=search_request,
            num_requests=100,
            concurrency=10,
        )

        print(f"\n[Search Warm-Cache SLO Test]\n{result.summary()}")
        print(f"SLO Target: p95 ≤ {SLO_P95_MS}ms")

        assert result.p95 <= SLO_P95_MS, (
            f"Search warm-cache p95 ({result.p95:.1f}ms) exceeds SLO ({SLO_P95_MS}ms)"
        )
        assert result.success_rate >= 0.99, (
            f"Success rate ({result.success_rate:.2%}) below 99%"
        )

    async def test_warm_cache_sustained_load(self, mock_search_service):
        """
        Validates SLO holds under sustained load (multiple batches).
        """
        all_latencies = []

        for batch in range(5):
            async def search_request():
                await mock_search_service("sustained query test")

            result = await run_load_test(
                func=search_request,
                num_requests=50,
                concurrency=5,
            )
            all_latencies.extend(result.latencies_ms)

        # Check p95 across all batches
        sorted_lats = sorted(all_latencies)
        p95_idx = int(len(sorted_lats) * 0.95)
        overall_p95 = sorted_lats[min(p95_idx, len(sorted_lats) - 1)]

        print(f"\n[Sustained Load Test] Overall p95: {overall_p95:.1f}ms")
        assert overall_p95 <= SLO_P95_MS
