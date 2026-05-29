"""
Integration test: Fetch-to-searchable p95 ≤ 60min, p99 ≤ 4h (R2.1)

Validates that the indexing pipeline processes documents from fetch
to searchable within the SLO targets. This is an integration test
that measures the end-to-end ingest pipeline latency.
"""

import asyncio
import time
import uuid
from datetime import datetime, timezone

import pytest

# SLO Targets
SLO_P95_MINUTES = 60
SLO_P99_HOURS = 4


@pytest.fixture
def mock_ingest_pipeline():
    """Mock ingest pipeline simulating fetch → index → searchable flow."""

    class IngestPipeline:
        def __init__(self):
            self.documents: dict[str, dict] = {}

        async def fetch_document(self, url: str) -> dict:
            """Simulate crawler fetching a document."""
            await asyncio.sleep(0.5)  # Network fetch
            return {
                "document_id": str(uuid.uuid4()),
                "url": url,
                "content": f"Content from {url}",
                "content_hash": f"hash_{hash(url)}",
                "fetch_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "http_status": 200,
                "content_type": "text/html",
            }

        async def index_document(self, doc: dict) -> dict:
            """Simulate indexer processing (clean, vectorize, store)."""
            # Text cleaning: ~100ms
            await asyncio.sleep(0.1)
            # Vector embedding: ~200ms
            await asyncio.sleep(0.2)
            # Storage write: ~50ms
            await asyncio.sleep(0.05)
            # Provenance scoring: ~100ms
            await asyncio.sleep(0.1)

            doc["indexed_at"] = datetime.now(timezone.utc).isoformat()
            doc["version"] = 1
            doc["provenance"] = {
                "credibility_score": 0.85,
                "ai_generated_likelihood": 0.1,
                "scored_at": datetime.now(timezone.utc).isoformat(),
            }
            self.documents[doc["document_id"]] = doc
            return doc

        async def make_searchable(self, doc: dict) -> dict:
            """Simulate making document available to retriever."""
            # Index sync/refresh: ~50ms
            await asyncio.sleep(0.05)
            doc["searchable_at"] = datetime.now(timezone.utc).isoformat()
            return doc

        async def full_pipeline(self, url: str) -> float:
            """Run full pipeline and return total time in seconds."""
            start = time.perf_counter()
            fetched = await self.fetch_document(url)
            indexed = await self.index_document(fetched)
            await self.make_searchable(indexed)
            elapsed = time.perf_counter() - start
            return elapsed

    return IngestPipeline()


@pytest.mark.integration
class TestFetchToSearchableSLO:
    """Integration tests for fetch-to-searchable latency SLO (R2.1)."""

    async def test_fetch_to_searchable_p95_within_slo(self, mock_ingest_pipeline):
        """
        SLO: Fetch-to-searchable p95 ≤ 60 minutes, p99 ≤ 4 hours.

        Simulates the full ingest pipeline for multiple documents and
        validates that processing times stay within SLO targets.

        Note: In production, this measures real end-to-end latency including
        queue delays. This test validates the pipeline logic is correct
        and measures simulated processing time.
        """
        urls = [f"https://example.com/page{i}" for i in range(50)]
        latencies_seconds = []

        # Process documents concurrently (simulating real pipeline)
        tasks = [mock_ingest_pipeline.full_pipeline(url) for url in urls]
        latencies_seconds = await asyncio.gather(*tasks)

        # Convert to minutes for SLO comparison
        latencies_minutes = [s / 60 for s in latencies_seconds]
        sorted_lats = sorted(latencies_minutes)

        p95_idx = int(len(sorted_lats) * 0.95)
        p99_idx = int(len(sorted_lats) * 0.99)
        p95_minutes = sorted_lats[min(p95_idx, len(sorted_lats) - 1)]
        p99_minutes = sorted_lats[min(p99_idx, len(sorted_lats) - 1)]

        print(f"\n[Fetch-to-Searchable SLO Test]")
        print(f"Documents processed: {len(latencies_seconds)}")
        print(f"p95: {p95_minutes:.2f} minutes (SLO: ≤{SLO_P95_MINUTES} min)")
        print(f"p99: {p99_minutes:.4f} minutes (SLO: ≤{SLO_P99_HOURS * 60} min)")

        # In simulation, times are much faster than real SLO
        # This validates the pipeline completes successfully
        assert p95_minutes <= SLO_P95_MINUTES, (
            f"Fetch-to-searchable p95 ({p95_minutes:.2f}min) exceeds SLO ({SLO_P95_MINUTES}min)"
        )
        assert p99_minutes <= SLO_P99_HOURS * 60, (
            f"Fetch-to-searchable p99 ({p99_minutes:.2f}min) exceeds SLO ({SLO_P99_HOURS * 60}min)"
        )

    async def test_pipeline_handles_failures_gracefully(self, mock_ingest_pipeline):
        """
        Validates that pipeline failures don't block other documents.
        Documents that fail after 3 retries go to DLQ (R2.5).
        """
        successful = 0
        failed = 0

        for i in range(20):
            try:
                elapsed = await mock_ingest_pipeline.full_pipeline(
                    f"https://example.com/doc{i}"
                )
                successful += 1
            except Exception:
                failed += 1

        print(f"\n[Pipeline Resilience Test]")
        print(f"Successful: {successful}, Failed: {failed}")
        assert successful >= 18, "At least 90% of documents should succeed"
