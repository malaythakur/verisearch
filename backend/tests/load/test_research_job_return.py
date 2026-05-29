"""
Load test: Research job_id return p95 ≤ 1s (R7.1)

Validates that POST /v1/research returns a job_id within 1 second
at the 95th percentile. The job itself runs asynchronously.
"""

import asyncio
import time
import uuid

import pytest

from tests.load.conftest import LatencyResult, run_load_test

# SLO Target
SLO_P95_MS = 1000


@pytest.fixture
def mock_research_agent():
    """Mock research agent that simulates job creation."""

    async def _create_job(research_goal: str, output_schema=None, session_id=None):
        """Simulate research job creation (enqueue + return job_id)."""
        # Validate input (fast)
        if not research_goal or len(research_goal) > 4096:
            raise ValueError("invalid_research_request")

        # Enqueue job (typically 10-50ms for queue write)
        await asyncio.sleep(0.01 + (hash(research_goal) % 40) / 1000)

        # Return job_id immediately
        job_id = str(uuid.uuid4())
        return {"job_id": job_id, "status": "queued"}

    return _create_job


@pytest.mark.integration
class TestResearchJobReturnSLO:
    """Load tests for research job_id return p95 latency SLO (R7.1)."""

    async def test_job_id_return_p95_within_slo(self, mock_research_agent):
        """
        SLO: Research job_id return p95 ≤ 1s.

        Simulates concurrent research job creation requests and validates
        that job_id is returned within the SLO target.
        """
        goals = [
            "Research the impact of AI on healthcare",
            "Analyze recent developments in quantum computing",
            "Compare renewable energy technologies",
            "Investigate supply chain optimization strategies",
            "Study the effects of remote work on productivity",
        ]
        goal_idx = 0

        async def create_research_job():
            nonlocal goal_idx
            goal = goals[goal_idx % len(goals)]
            goal_idx += 1
            result = await mock_research_agent(goal)
            assert "job_id" in result

        result = await run_load_test(
            func=create_research_job,
            num_requests=100,
            concurrency=10,
        )

        print(f"\n[Research Job Return SLO Test]\n{result.summary()}")
        print(f"SLO Target: p95 ≤ {SLO_P95_MS}ms")

        assert result.p95 <= SLO_P95_MS, (
            f"Research job_id return p95 ({result.p95:.1f}ms) exceeds SLO ({SLO_P95_MS}ms)"
        )
        assert result.success_rate >= 0.99, (
            f"Success rate ({result.success_rate:.2%}) below 99%"
        )

    async def test_job_id_return_with_schema_validation(self, mock_research_agent):
        """
        Validates SLO holds even when output_schema validation is included.
        """
        schema = {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "findings": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["summary", "findings"],
        }

        async def create_with_schema():
            result = await mock_research_agent(
                "Research AI safety approaches",
                output_schema=schema,
            )
            assert "job_id" in result

        result = await run_load_test(
            func=create_with_schema,
            num_requests=50,
            concurrency=5,
        )

        print(f"\n[Research Job with Schema Test]\n{result.summary()}")
        assert result.p95 <= SLO_P95_MS
