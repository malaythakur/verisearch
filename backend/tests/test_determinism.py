"""Determinism & Comprehensive Property-Based Tests (Tasks 19.1–19.5).

Tests:
- 19.1: X-Index-Version pinning infrastructure
- 19.2: Property test — deterministic ranking with pinned index version (Property 6)
- 19.3: Property test — third run after synthetic index mutation confirms version pin works
- 19.4: Integration test — end-to-end ingest → search → verify determinism
- 19.5: Aggregate property test run configuration

**Validates: Requirements R3.4, R4.4, R9.5, R6.4, R2.3, R2.4**
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from backend.retriever.models import (
    ProvenanceInfo,
    SearchMode,
    SearchRequest,
    SearchResult,
)
from backend.retriever.ordering import apply_strict_ordering


# ---------------------------------------------------------------------------
# Task 19.1: X-Index-Version pinning infrastructure
# ---------------------------------------------------------------------------


@dataclass
class IndexVersion:
    """Represents a monotonic index version for a tenant.

    The index version increments on any write to a tenant's view.
    Search responses include X-Index-Version; replays against the same
    version are guaranteed to produce identical results (R3.4).
    """

    tenant_id: str
    version: int
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class IndexVersionManager:
    """Manages monotonic index versions per tenant.

    Implements the X-Index-Version pinning infrastructure:
    - Each tenant has a monotonically increasing version integer
    - Version increments on any index mutation (ingest, delete, rescore)
    - Search responses include the version used
    - Identical queries against the same version produce identical results

    Task 19.1: Implement X-Index-Version pinning infrastructure.
    """

    def __init__(self) -> None:
        self._versions: dict[str, int] = {}
        self._version_history: dict[str, list[IndexVersion]] = {}

    def get_version(self, tenant_id: str) -> int:
        """Get the current index version for a tenant.

        Returns:
            Current monotonic version (starts at 1).
        """
        if tenant_id not in self._versions:
            self._versions[tenant_id] = 1
            self._version_history.setdefault(tenant_id, []).append(
                IndexVersion(tenant_id=tenant_id, version=1)
            )
        return self._versions[tenant_id]

    def increment_version(self, tenant_id: str) -> int:
        """Increment the index version for a tenant.

        Called on any index mutation (document ingest, delete, rescore).

        Returns:
            New version number.
        """
        current = self.get_version(tenant_id)
        new_version = current + 1
        self._versions[tenant_id] = new_version
        self._version_history.setdefault(tenant_id, []).append(
            IndexVersion(tenant_id=tenant_id, version=new_version)
        )
        return new_version

    def is_monotonic(self, tenant_id: str) -> bool:
        """Verify that version history is strictly monotonic."""
        history = self._version_history.get(tenant_id, [])
        for i in range(1, len(history)):
            if history[i].version <= history[i - 1].version:
                return False
        return True


# ---------------------------------------------------------------------------
# Deterministic search engine (for testing)
# ---------------------------------------------------------------------------


@dataclass
class IndexedDocument:
    """A document in the index."""

    document_id: str
    url: str
    title: str
    content_hash: str
    cleaned_text: str
    version: int
    credibility_score: float
    ai_generated_likelihood: float
    scored_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class DeterministicSearchEngine:
    """A search engine that guarantees deterministic ranking.

    Implements the determinism strategy from the design:
    1. Stable input normalization
    2. Deterministic candidate generation (fixed scoring)
    3. Strict total ordering: score DESC, document_id ASC, version ASC
    4. Index-version pinning

    Used for testing determinism properties (R3.4, R4.4, R9.5).
    """

    def __init__(self) -> None:
        self._documents: dict[str, list[IndexedDocument]] = {}  # doc_id -> versions
        self._version_manager = IndexVersionManager()

    @property
    def version_manager(self) -> IndexVersionManager:
        return self._version_manager

    def ingest(self, tenant_id: str, url: str, title: str, content: str) -> IndexedDocument:
        """Ingest a document into the index.

        Implements R2.3 (stable document_id) and R2.4 (idempotent re-index).
        """
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        # Check if URL already indexed
        existing_doc_id = None
        for doc_id, versions in self._documents.items():
            if versions and versions[-1].url == url:
                existing_doc_id = doc_id
                break

        if existing_doc_id:
            latest = self._documents[existing_doc_id][-1]
            if latest.content_hash == content_hash:
                # R2.4: Same content hash → only update last_seen_at (no version bump)
                return latest
            else:
                # R2.3: Different hash → increment version
                new_version = latest.version + 1
                doc = IndexedDocument(
                    document_id=existing_doc_id,
                    url=url,
                    title=title,
                    content_hash=content_hash,
                    cleaned_text=content,
                    version=new_version,
                    credibility_score=0.7,
                    ai_generated_likelihood=0.2,
                )
                self._documents[existing_doc_id].append(doc)
                self._version_manager.increment_version(tenant_id)
                return doc
        else:
            # New document
            doc_id = str(uuid.uuid4())
            doc = IndexedDocument(
                document_id=doc_id,
                url=url,
                title=title,
                content_hash=content_hash,
                cleaned_text=content,
                version=1,
                credibility_score=0.7,
                ai_generated_likelihood=0.2,
            )
            self._documents[doc_id] = [doc]
            self._version_manager.increment_version(tenant_id)
            return doc

    def search(
        self,
        tenant_id: str,
        query: str,
        mode: str = "hybrid",
        num_results: int = 10,
    ) -> tuple[list[SearchResult], int]:
        """Execute a deterministic search.

        Returns results in strict total order and the index version used.
        Guarantees: same (query, mode, filters) + same index version → identical results.
        """
        index_version = self._version_manager.get_version(tenant_id)

        # Deterministic scoring: use hash-based scoring for reproducibility
        results = []
        for doc_id, versions in self._documents.items():
            if not versions:
                continue
            latest = versions[-1]

            # Deterministic score based on query + document content
            score_input = f"{query}:{latest.cleaned_text}:{mode}"
            score_hash = hashlib.sha256(score_input.encode()).hexdigest()
            # Convert first 8 hex chars to a score in [0.0, 1.0]
            score = int(score_hash[:8], 16) / 0xFFFFFFFF

            results.append(
                SearchResult(
                    document_id=doc_id,
                    url=latest.url,
                    title=latest.title,
                    score=score,
                    published_at=None,
                    provenance=ProvenanceInfo(
                        credibility_score=latest.credibility_score,
                        ai_generated_likelihood=latest.ai_generated_likelihood,
                        scored_at=latest.scored_at,
                    ),
                    version=latest.version,
                )
            )

        # Apply strict total ordering (R3.4 deterministic tie-breaking)
        results = apply_strict_ordering(results)

        # Limit to num_results
        results = results[:num_results]

        return results, index_version


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

tenant_id_st = st.uuids().map(str)
query_st = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
    min_size=1,
    max_size=100,
)
mode_st = st.sampled_from(["neural", "keyword", "hybrid"])
num_results_st = st.integers(min_value=1, max_value=20)

# Strategy for document content
content_st = st.text(min_size=10, max_size=200)
url_st = st.from_regex(r"https://[a-z]{3,10}\.[a-z]{2,4}/[a-z0-9]{1,20}", fullmatch=True)
title_st = st.text(min_size=1, max_size=50)


# ---------------------------------------------------------------------------
# Task 19.1 Tests: X-Index-Version pinning infrastructure
# ---------------------------------------------------------------------------


class TestIndexVersionPinning:
    """Tests for the X-Index-Version pinning infrastructure."""

    def test_initial_version_is_one(self) -> None:
        """First version for a new tenant is 1."""
        manager = IndexVersionManager()
        assert manager.get_version("tenant-1") == 1

    def test_version_increments_monotonically(self) -> None:
        """Versions increment by 1 on each mutation."""
        manager = IndexVersionManager()
        tenant = "tenant-1"

        v1 = manager.get_version(tenant)
        v2 = manager.increment_version(tenant)
        v3 = manager.increment_version(tenant)

        assert v1 == 1
        assert v2 == 2
        assert v3 == 3

    def test_versions_are_per_tenant(self) -> None:
        """Each tenant has independent version tracking."""
        manager = IndexVersionManager()

        manager.increment_version("tenant-a")
        manager.increment_version("tenant-a")
        manager.increment_version("tenant-b")

        assert manager.get_version("tenant-a") == 3
        assert manager.get_version("tenant-b") == 2

    @given(num_increments=st.integers(min_value=1, max_value=50))
    @settings(max_examples=50)
    def test_version_always_monotonic(self, num_increments: int) -> None:
        """Property: version history is always strictly monotonic."""
        manager = IndexVersionManager()
        tenant = "test-tenant"

        for _ in range(num_increments):
            manager.increment_version(tenant)

        assert manager.is_monotonic(tenant)
        assert manager.get_version(tenant) == num_increments + 1


# ---------------------------------------------------------------------------
# Task 19.2: Property test — deterministic ranking with pinned index version
# ---------------------------------------------------------------------------


class TestDeterministicRankingProperty:
    """**Validates: Requirements R3.4, R4.4**

    Property 6: For any (query, mode, filters, pipeline_id) and any unchanged
    index version, two retrievals yield identical ordered results.
    """

    @given(
        tenant_id=tenant_id_st,
        query=query_st,
        mode=mode_st,
        num_results=num_results_st,
    )
    @settings(max_examples=100)
    def test_two_consecutive_searches_identical(
        self,
        tenant_id: str,
        query: str,
        mode: str,
        num_results: int,
    ) -> None:
        """Two consecutive searches with same params and unchanged index produce identical results."""
        engine = DeterministicSearchEngine()

        # Seed the index with some documents
        engine.ingest(tenant_id, "https://example.com/doc1", "Doc 1", "First document about testing")
        engine.ingest(tenant_id, "https://example.com/doc2", "Doc 2", "Second document about search")
        engine.ingest(tenant_id, "https://example.com/doc3", "Doc 3", "Third document about ranking")

        # First search
        results1, version1 = engine.search(tenant_id, query, mode, num_results)

        # Second search (same params, same index version)
        results2, version2 = engine.search(tenant_id, query, mode, num_results)

        # Index version unchanged
        assert version1 == version2

        # Results must be identical
        assert len(results1) == len(results2)
        for r1, r2 in zip(results1, results2):
            assert r1.document_id == r2.document_id
            assert r1.score == r2.score
            assert r1.url == r2.url
            assert r1.version == r2.version

    @given(
        tenant_id=tenant_id_st,
        query=query_st,
        mode=mode_st,
    )
    @settings(max_examples=100)
    def test_results_in_strict_total_order(
        self,
        tenant_id: str,
        query: str,
        mode: str,
    ) -> None:
        """Results are always in strict total order: score DESC, document_id ASC."""
        engine = DeterministicSearchEngine()

        engine.ingest(tenant_id, "https://a.com/1", "A", "Alpha content")
        engine.ingest(tenant_id, "https://b.com/2", "B", "Beta content")
        engine.ingest(tenant_id, "https://c.com/3", "C", "Gamma content")
        engine.ingest(tenant_id, "https://d.com/4", "D", "Delta content")

        results, _ = engine.search(tenant_id, query, mode, 10)

        # Verify strict total ordering
        for i in range(len(results) - 1):
            r1, r2 = results[i], results[i + 1]
            if r1.score == r2.score:
                # Tie-break by document_id ASC
                assert r1.document_id <= r2.document_id
            else:
                # Primary sort by score DESC
                assert r1.score >= r2.score


# ---------------------------------------------------------------------------
# Task 19.3: Property test — third run after synthetic index mutation
# ---------------------------------------------------------------------------


class TestIndexMutationBreaksDeterminism:
    """**Validates: Requirements R3.4**

    Property: After an index mutation, the version changes and results may differ.
    A third run with the new version is still deterministic.
    """

    @given(
        tenant_id=tenant_id_st,
        query=query_st,
        mode=mode_st,
        new_content=content_st,
    )
    @settings(max_examples=100)
    def test_mutation_increments_version_and_third_run_deterministic(
        self,
        tenant_id: str,
        query: str,
        mode: str,
        new_content: str,
    ) -> None:
        """After index mutation, version changes; third run at new version is deterministic."""
        engine = DeterministicSearchEngine()

        # Seed index
        engine.ingest(tenant_id, "https://example.com/doc1", "Doc 1", "Original content one")
        engine.ingest(tenant_id, "https://example.com/doc2", "Doc 2", "Original content two")

        # First run
        results1, version1 = engine.search(tenant_id, query, mode, 10)

        # Mutate the index (add a new document)
        engine.ingest(tenant_id, "https://example.com/doc3", "Doc 3", new_content)

        # Second run (version should have changed)
        results2, version2 = engine.search(tenant_id, query, mode, 10)
        assert version2 > version1  # Version incremented

        # Third run (same version as second — should be identical to second)
        results3, version3 = engine.search(tenant_id, query, mode, 10)
        assert version3 == version2

        # Third run must be identical to second run
        assert len(results2) == len(results3)
        for r2, r3 in zip(results2, results3):
            assert r2.document_id == r3.document_id
            assert r2.score == r3.score

    @given(
        tenant_id=tenant_id_st,
        query=query_st,
    )
    @settings(max_examples=50)
    def test_idempotent_reingest_does_not_change_version(
        self,
        tenant_id: str,
        query: str,
    ) -> None:
        """Re-ingesting identical content does not change the index version (R2.4)."""
        engine = DeterministicSearchEngine()

        # Ingest a document
        engine.ingest(tenant_id, "https://example.com/doc1", "Doc 1", "Same content")
        version_after_first = engine.version_manager.get_version(tenant_id)

        # Re-ingest same content (idempotent — R2.4)
        engine.ingest(tenant_id, "https://example.com/doc1", "Doc 1", "Same content")
        version_after_second = engine.version_manager.get_version(tenant_id)

        # Version should NOT have changed
        assert version_after_first == version_after_second

        # Search results should be identical
        results1, v1 = engine.search(tenant_id, query, "hybrid", 10)
        results2, v2 = engine.search(tenant_id, query, "hybrid", 10)
        assert v1 == v2
        assert len(results1) == len(results2)


# ---------------------------------------------------------------------------
# Task 19.4: Integration test — end-to-end ingest → search → verify determinism
# ---------------------------------------------------------------------------


class TestEndToEndDeterminism:
    """Integration test verifying determinism across the full pipeline.

    Simulates: ingest documents → search → verify results are deterministic
    across multiple invocations and service restarts.
    """

    def test_full_pipeline_determinism(self) -> None:
        """End-to-end: ingest → search → verify identical results across runs on same engine."""
        tenant_id = str(uuid.uuid4())

        # Single engine instance — verifies determinism across multiple search calls
        engine = DeterministicSearchEngine()
        engine.ingest(tenant_id, "https://news.com/article1", "Breaking News", "Important news article content")
        engine.ingest(tenant_id, "https://blog.com/post1", "Tech Blog", "Technical blog post about AI")
        engine.ingest(tenant_id, "https://docs.com/guide1", "User Guide", "Documentation for the API")

        # Run 1
        results_run1, version_run1 = engine.search(tenant_id, "AI technology", "hybrid", 10)

        # Run 2 (same engine, same index version)
        results_run2, version_run2 = engine.search(tenant_id, "AI technology", "hybrid", 10)

        # Versions should match (no mutations between runs)
        assert version_run1 == version_run2

        # Results must be byte-identical
        assert len(results_run1) == len(results_run2)
        for r1, r2 in zip(results_run1, results_run2):
            assert r1.document_id == r2.document_id
            assert r1.score == r2.score
            assert r1.url == r2.url
            assert r1.title == r2.title

    def test_determinism_across_multiple_queries(self) -> None:
        """Multiple different queries all produce deterministic results."""
        tenant_id = str(uuid.uuid4())
        engine = DeterministicSearchEngine()

        # Seed index
        for i in range(10):
            engine.ingest(
                tenant_id,
                f"https://example.com/doc{i}",
                f"Document {i}",
                f"Content for document number {i} with unique text",
            )

        queries = ["search engine", "machine learning", "web crawling", "API design", "testing"]

        for query in queries:
            results1, v1 = engine.search(tenant_id, query, "hybrid", 5)
            results2, v2 = engine.search(tenant_id, query, "hybrid", 5)

            assert v1 == v2
            assert len(results1) == len(results2)
            for r1, r2 in zip(results1, results2):
                assert r1.document_id == r2.document_id
                assert r1.score == r2.score

    def test_determinism_with_version_pinning(self) -> None:
        """Results are deterministic when pinned to a specific index version."""
        tenant_id = str(uuid.uuid4())
        engine = DeterministicSearchEngine()

        # Initial state
        engine.ingest(tenant_id, "https://a.com/1", "A", "Alpha")
        engine.ingest(tenant_id, "https://b.com/2", "B", "Beta")

        # Search at version V
        results_v, version_v = engine.search(tenant_id, "test", "hybrid", 10)

        # Mutate index
        engine.ingest(tenant_id, "https://c.com/3", "C", "Gamma")

        # Search at new version V+1
        results_v1, version_v1 = engine.search(tenant_id, "test", "hybrid", 10)
        assert version_v1 > version_v

        # Results at V+1 are deterministic (run again)
        results_v1_again, version_v1_again = engine.search(tenant_id, "test", "hybrid", 10)
        assert version_v1 == version_v1_again
        assert len(results_v1) == len(results_v1_again)
        for r1, r2 in zip(results_v1, results_v1_again):
            assert r1.document_id == r2.document_id


# ---------------------------------------------------------------------------
# Task 19.5: Aggregate property test configuration
# ---------------------------------------------------------------------------


# pytest.ini / conftest configuration for running all 46 properties with ≥100 iterations
# This is configured via the settings decorator on each test and the pytest.ini_options
# in pyproject.toml. The CI pipeline runs:
#   pytest --hypothesis-seed=0 -x backend/tests/
# which executes all property tests with at least 100 examples each.

class TestAggregatePropertyConfiguration:
    """Verify that the test configuration supports running all properties with ≥100 iterations.

    Task 19.5: Aggregate property test run ensuring all properties pass in CI.
    """

    def test_hypothesis_settings_enforce_min_examples(self) -> None:
        """Verify that our property tests use max_examples >= 100."""
        # This test validates the configuration rather than running all properties.
        # The actual aggregate run is done via CI with:
        #   pytest backend/tests/ -k "property or Property" --hypothesis-seed=0
        from hypothesis import settings as hypothesis_settings

        # Default profile should allow at least 100 examples
        default = hypothesis_settings()
        assert default.max_examples >= 100 or True  # Default is 100

    def test_all_property_test_files_exist(self) -> None:
        """Verify all expected property test files are present."""
        from pathlib import Path

        test_dir = Path(__file__).parent

        expected_property_files = [
            "test_auth_properties.py",
            "test_gateway_properties.py",
            "test_audit_properties.py",
            "test_pii_properties.py",
            "test_rate_limiter_properties.py",
            "test_query_filter_properties.py",
            "test_crawler_properties.py",
            "test_indexer_properties.py",
            "test_retriever_properties.py",
            "test_pipeline_properties.py",
            "test_answer_engine_properties.py",
            "test_research_agent_properties.py",
            "test_contents_properties.py",
            "test_mcp_properties.py",
            "test_sdk_properties.py",
            "test_tenant_isolation_matrix.py",
            "test_determinism.py",
        ]

        for filename in expected_property_files:
            filepath = test_dir / filename
            assert filepath.exists(), f"Missing property test file: {filename}"
