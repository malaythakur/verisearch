"""Property-based tests for the Retriever subsystem (Tasks 11.11–11.14).

Properties tested:
- Property 5: Response shape and score ordering invariants.
- Property 6: Deterministic ranking against unchanged index version.
- Property 7: find_similar excludes all versions of input document.
- Property 21: Threshold boundary inclusion/exclusion.
"""

from datetime import datetime, timezone

import pytest
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from backend.indexer.service import IndexerService
from backend.provenance_scorer.scorer import ProvenanceScorer
from backend.retriever.filters import apply_max_ai_generated, apply_min_credibility
from backend.retriever.find_similar import canonicalize_url
from backend.retriever.models import (
    FindSimilarRequest,
    ProvenanceInfo,
    SearchMode,
    SearchRequest,
    SearchResult,
)
from backend.retriever.ordering import apply_strict_ordering
from backend.retriever.service import DocumentNotFoundError, RetrieverService


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy for valid search queries (1–100 chars for test speed)
query_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
    min_size=1,
    max_size=100,
).filter(lambda s: len(s.strip()) > 0)

# Strategy for search modes
mode_strategy = st.sampled_from([SearchMode.NEURAL, SearchMode.KEYWORD, SearchMode.HYBRID])

# Strategy for num_results (valid range)
num_results_strategy = st.integers(min_value=1, max_value=100)

# Strategy for threshold values in [0.0, 1.0]
threshold_strategy = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)

# Strategy for scores in [0.0, 1.0]
score_strategy = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)

# Strategy for document IDs
doc_id_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N")),
    min_size=1,
    max_size=36,
)

# Strategy for URLs
url_strategy = st.from_regex(
    r"https?://[a-z][a-z0-9]{0,20}\.[a-z]{2,4}/[a-z0-9/]{0,30}",
    fullmatch=True,
)


def make_result(doc_id: str, score: float, credibility: float, ai_likelihood: float, version: int = 1) -> SearchResult:
    """Helper to create a SearchResult with given parameters."""
    now = datetime.now(timezone.utc)
    return SearchResult(
        document_id=doc_id,
        url=f"https://example.com/{doc_id}",
        title=f"Title {doc_id}",
        score=score,
        published_at=None,
        provenance=ProvenanceInfo(credibility, ai_likelihood, now),
        version=version,
    )


# Strategy for generating lists of SearchResults
def results_strategy(min_size=0, max_size=20):
    """Generate a list of SearchResult objects."""
    return st.lists(
        st.builds(
            make_result,
            doc_id=st.uuids().map(str),
            score=score_strategy,
            credibility=score_strategy,
            ai_likelihood=score_strategy,
            version=st.integers(min_value=1, max_value=10),
        ),
        min_size=min_size,
        max_size=max_size,
    )


# ---------------------------------------------------------------------------
# Property 5: Response shape and score ordering invariants
# ---------------------------------------------------------------------------


class TestProperty5ResponseShapeAndOrdering:
    """Property 5: Response shape and score ordering invariants.

    **Validates: Requirements 3.1, 3.3**

    For any search results after ordering:
    - All scores are in [0.0, 1.0].
    - Results are in non-increasing score order.
    - Each result has required fields (document_id, url, title, score, provenance).
    - The strict total ordering is deterministic (same input → same output).
    """

    @given(results=results_strategy(min_size=0, max_size=20))
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_scores_in_valid_range(self, results: list[SearchResult]):
        """All scores are in [0.0, 1.0] after ordering."""
        ordered = apply_strict_ordering(results)
        for r in ordered:
            assert 0.0 <= r.score <= 1.0, f"Score {r.score} out of range"

    @given(results=results_strategy(min_size=0, max_size=20))
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_non_increasing_score_order(self, results: list[SearchResult]):
        """Results are in non-increasing score order after ordering."""
        ordered = apply_strict_ordering(results)
        for i in range(len(ordered) - 1):
            assert ordered[i].score >= ordered[i + 1].score, (
                f"Score at position {i} ({ordered[i].score}) < "
                f"score at position {i+1} ({ordered[i+1].score})"
            )

    @given(results=results_strategy(min_size=0, max_size=20))
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_required_fields_present(self, results: list[SearchResult]):
        """Each result has all required fields."""
        ordered = apply_strict_ordering(results)
        for r in ordered:
            assert r.document_id is not None and r.document_id != ""
            assert r.url is not None
            assert r.title is not None
            assert r.provenance is not None
            assert 0.0 <= r.provenance.credibility_score <= 1.0
            assert 0.0 <= r.provenance.ai_generated_likelihood <= 1.0

    @given(results=results_strategy(min_size=0, max_size=20))
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_ordering_is_deterministic(self, results: list[SearchResult]):
        """Same input produces same output (deterministic)."""
        ordered1 = apply_strict_ordering(results)
        ordered2 = apply_strict_ordering(results)
        assert ordered1 == ordered2

    @given(results=results_strategy(min_size=0, max_size=20))
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_ordering_preserves_length(self, results: list[SearchResult]):
        """Ordering does not add or remove results."""
        ordered = apply_strict_ordering(results)
        assert len(ordered) == len(results)


# ---------------------------------------------------------------------------
# Property 6: Deterministic ranking against unchanged index version
# ---------------------------------------------------------------------------


class TestProperty6DeterministicRanking:
    """Property 6: Deterministic ranking against unchanged index version.

    **Validates: Requirements 3.4, 4.4**

    For any valid query, mode, and filters against an unchanged index:
    - Two consecutive searches produce identical results in identical order.
    - The strict total ordering guarantees a unique position for every result.
    """

    @given(
        query=query_strategy,
        mode=mode_strategy,
        num_results=st.integers(min_value=1, max_value=10),
    )
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=10)
    def test_deterministic_search(self, query: str, mode: SearchMode, num_results: int):
        """Same query/mode/num_results produces identical results twice."""
        import asyncio

        # Set up a fresh indexer with known documents
        scorer = ProvenanceScorer()
        indexer = IndexerService(provenance_scorer=scorer)

        async def setup():
            await indexer.index_document(
                "<html><body>Machine learning and artificial intelligence research</body></html>",
                "https://example.com/ml",
            )
            await indexer.index_document(
                "<html><body>Python programming language tutorials and guides</body></html>",
                "https://example.com/python",
            )
            await indexer.index_document(
                "<html><body>Cloud computing and distributed systems architecture</body></html>",
                "https://example.com/cloud",
            )

        asyncio.run(setup())

        retriever = RetrieverService(indexer=indexer)

        request = SearchRequest(
            query=query,
            mode=mode,
            num_results=num_results,
            tenant_id="test-tenant",
        )

        # Clear cache to test actual retrieval
        retriever.cache.invalidate()
        response1 = retriever.search(request)
        retriever.cache.invalidate()
        response2 = retriever.search(request)

        # Results must be identical
        assert len(response1.results) == len(response2.results)
        for r1, r2 in zip(response1.results, response2.results):
            assert r1.document_id == r2.document_id
            assert r1.score == r2.score
            assert r1.version == r2.version

    @given(results=results_strategy(min_size=2, max_size=10))
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_total_ordering_unique_positions(self, results: list[SearchResult]):
        """Strict total ordering assigns unique positions (no ambiguity)."""
        ordered = apply_strict_ordering(results)
        # Check that the ordering key is unique for each position
        keys = [(-r.score, r.document_id, r.version) for r in ordered]
        # Keys should be in sorted order
        assert keys == sorted(keys)


# ---------------------------------------------------------------------------
# Property 7: find_similar excludes all versions of input document
# ---------------------------------------------------------------------------


class TestProperty7FindSimilarExclusion:
    """Property 7: find_similar excludes all versions of input document.

    **Validates: Requirements 4.2**

    For any document with multiple versions in the index:
    - find_similar never returns any version of the input document.
    - This holds regardless of how many versions exist.
    """

    @given(num_versions=st.integers(min_value=1, max_value=5))
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=10)
    def test_all_versions_excluded(self, num_versions: int):
        """find_similar excludes every version of the input document."""
        import asyncio

        scorer = ProvenanceScorer()
        indexer = IndexerService(provenance_scorer=scorer)

        async def setup():
            # Index the target document with multiple versions
            base_url = "https://example.com/target"
            for i in range(num_versions):
                content = f"<html><body>Target document version {i} with unique content {i * 7}</body></html>"
                await indexer.index_document(content, base_url)

            # Index other documents for similarity results
            await indexer.index_document(
                "<html><body>Similar content about targets and documents</body></html>",
                "https://example.com/similar1",
            )
            await indexer.index_document(
                "<html><body>Another document with related information</body></html>",
                "https://example.com/similar2",
            )

        asyncio.run(setup())

        retriever = RetrieverService(indexer=indexer)

        # Get the document_id for the target
        target_versions = indexer.get_document_by_url("https://example.com/target")
        assert target_versions is not None
        target_doc_id = target_versions[0].document_id

        # Verify multiple versions exist
        assert len(target_versions) == num_versions

        # Execute find_similar
        request = FindSimilarRequest(
            url="https://example.com/target",
            num_results=10,
            tenant_id="test-tenant",
        )
        response = retriever.find_similar(request)

        # No version of the target document should appear in results
        for result in response.results:
            assert result.document_id != target_doc_id, (
                f"find_similar returned version {result.version} of input document"
            )


# ---------------------------------------------------------------------------
# Property 21: Threshold boundary inclusion/exclusion
# ---------------------------------------------------------------------------


class TestProperty21ThresholdBoundary:
    """Property 21: Threshold boundary inclusion/exclusion.

    **Validates: Requirements 10.3, 10.4**

    - min_credibility: strict-less-than excluded, equality included (R10.3).
    - max_ai_generated_likelihood: strict-greater-than excluded, equality included (R10.4).
    """

    @given(
        threshold=threshold_strategy,
        scores=st.lists(score_strategy, min_size=1, max_size=20),
    )
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_min_credibility_boundary(self, threshold: float, scores: list[float]):
        """min_credibility: score < threshold excluded, score == threshold included."""
        now = datetime.now(timezone.utc)
        results = [
            make_result(f"doc-{i}", 0.5, score, 0.3, 1)
            for i, score in enumerate(scores)
        ]

        filtered = apply_min_credibility(results, threshold)

        for r in filtered:
            # All included results must have credibility >= threshold
            assert r.provenance.credibility_score >= threshold, (
                f"Included result has credibility {r.provenance.credibility_score} "
                f"< threshold {threshold}"
            )

        # Check that excluded results all have credibility < threshold
        included_ids = {r.document_id for r in filtered}
        for r in results:
            if r.document_id not in included_ids:
                assert r.provenance.credibility_score < threshold, (
                    f"Excluded result has credibility {r.provenance.credibility_score} "
                    f">= threshold {threshold}"
                )

    @given(
        threshold=threshold_strategy,
        scores=st.lists(score_strategy, min_size=1, max_size=20),
    )
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_max_ai_generated_boundary(self, threshold: float, scores: list[float]):
        """max_ai_generated: score > threshold excluded, score == threshold included."""
        now = datetime.now(timezone.utc)
        results = [
            make_result(f"doc-{i}", 0.5, 0.8, score, 1)
            for i, score in enumerate(scores)
        ]

        filtered = apply_max_ai_generated(results, threshold)

        for r in filtered:
            # All included results must have ai_likelihood <= threshold
            assert r.provenance.ai_generated_likelihood <= threshold, (
                f"Included result has ai_likelihood {r.provenance.ai_generated_likelihood} "
                f"> threshold {threshold}"
            )

        # Check that excluded results all have ai_likelihood > threshold
        included_ids = {r.document_id for r in filtered}
        for r in results:
            if r.document_id not in included_ids:
                assert r.provenance.ai_generated_likelihood > threshold, (
                    f"Excluded result has ai_likelihood {r.provenance.ai_generated_likelihood} "
                    f"<= threshold {threshold}"
                )

    @given(threshold=threshold_strategy)
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_min_credibility_exact_boundary(self, threshold: float):
        """A document with credibility exactly at threshold is included."""
        result = make_result("exact", 0.5, threshold, 0.3, 1)
        filtered = apply_min_credibility([result], threshold)
        assert len(filtered) == 1, (
            f"Document with credibility={threshold} should be included at threshold={threshold}"
        )

    @given(threshold=threshold_strategy)
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_max_ai_generated_exact_boundary(self, threshold: float):
        """A document with ai_likelihood exactly at threshold is included."""
        result = make_result("exact", 0.5, 0.8, threshold, 1)
        filtered = apply_max_ai_generated([result], threshold)
        assert len(filtered) == 1, (
            f"Document with ai_likelihood={threshold} should be included at threshold={threshold}"
        )

    @given(
        min_cred=threshold_strategy,
        max_ai=threshold_strategy,
        results=results_strategy(min_size=1, max_size=10),
    )
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_combined_filters_consistent(self, min_cred: float, max_ai: float, results: list[SearchResult]):
        """Combined filters are equivalent to applying each filter sequentially."""
        from backend.retriever.filters import apply_threshold_filters

        # Apply combined
        combined = apply_threshold_filters(results, min_credibility=min_cred, max_ai_generated_likelihood=max_ai)

        # Apply sequentially
        sequential = apply_min_credibility(results, min_cred)
        sequential = apply_max_ai_generated(sequential, max_ai)

        # Results should be identical
        assert len(combined) == len(sequential)
        for c, s in zip(combined, sequential):
            assert c.document_id == s.document_id
