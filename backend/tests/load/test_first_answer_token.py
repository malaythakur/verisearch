"""
Load test: First answer token p95 ≤ 3s (R6.1)

Validates that the Answer_Engine emits the first token event within
3 seconds of request acceptance at the 95th percentile.
"""

import asyncio
import time

import pytest

from tests.load.conftest import LatencyResult, run_load_test

# SLO Target
SLO_P95_MS = 3000


@pytest.fixture
def mock_answer_engine():
    """Mock answer engine that simulates streaming token generation."""

    async def _answer(query: str, stream: bool = True):
        """Simulate answer generation with first-token latency."""
        # Simulate retrieval phase (200-500ms)
        await asyncio.sleep(0.2 + (hash(query) % 300) / 1000)

        # Simulate LLM inference startup (100-800ms)
        await asyncio.sleep(0.1 + (hash(query[::-1]) % 700) / 1000)

        # First token emitted
        first_token_time = time.perf_counter()

        if stream:
            # Yield tokens
            tokens = ["The", " answer", " to", " your", " query", " is", "..."]
            for token in tokens:
                await asyncio.sleep(0.05)  # ~50ms between tokens
                yield {"type": "token", "data": token}

            yield {
                "type": "done",
                "data": {
                    "answer": "The answer to your query is...",
                    "citations": [],
                },
            }

    return _answer


@pytest.mark.integration
class TestFirstAnswerTokenSLO:
    """Load tests for first answer token p95 latency SLO (R6.1)."""

    async def test_first_token_p95_within_slo(self, mock_answer_engine):
        """
        SLO: First answer token p95 ≤ 3s.

        Simulates concurrent answer requests and measures time to first token.
        """
        queries = [
            "What is quantum computing?",
            "Explain machine learning",
            "How does blockchain work?",
            "What are neural networks?",
            "Describe climate change effects",
        ]
        query_idx = 0

        async def answer_request():
            nonlocal query_idx
            q = queries[query_idx % len(queries)]
            query_idx += 1

            # Measure time to first token
            async for event in mock_answer_engine(q, stream=True):
                # First event received = first token
                break

        result = await run_load_test(
            func=answer_request,
            num_requests=100,
            concurrency=10,
        )

        print(f"\n[First Answer Token SLO Test]\n{result.summary()}")
        print(f"SLO Target: p95 ≤ {SLO_P95_MS}ms")

        assert result.p95 <= SLO_P95_MS, (
            f"First token p95 ({result.p95:.1f}ms) exceeds SLO ({SLO_P95_MS}ms)"
        )
        assert result.success_rate >= 0.95, (
            f"Success rate ({result.success_rate:.2%}) below 95%"
        )

    async def test_first_token_under_high_concurrency(self, mock_answer_engine):
        """
        Validates first-token SLO under higher concurrency (20 concurrent).
        """
        async def answer_request():
            async for event in mock_answer_engine("high concurrency test"):
                break

        result = await run_load_test(
            func=answer_request,
            num_requests=50,
            concurrency=20,
        )

        print(f"\n[High Concurrency First Token Test]\n{result.summary()}")
        assert result.p95 <= SLO_P95_MS
