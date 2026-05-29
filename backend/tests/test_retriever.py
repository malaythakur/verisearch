"""Unit tests for the Retriever subsystem (Tasks 11.1–11.10).

Tests cover:
- Neural retrieval (vector ANN)
- Keyword retrieval (BM25)
- Hybrid retrieval (RRF fusion)
- Strict total ordering
- Warm cache with TTL
- Threshold filtering (min_credibility, max_ai_generated_likelihood)
- find_similar with document exclusion
- URL canonicalization
- X-Index-Version tracking
"""

import time
from datetime import datetime, timezone

import pytest

from backend.indexer.service import IndexerService
from backend.provenance_scorer.scorer import ProvenanceScorer
from backend.retriever.cache import WarmCache
from backend.retriever.filters import (
    apply_max_ai_generated,
    apply_min_credibility,
    apply_threshold_filters,
)
from backend.retriever.find_similar import canonicalize_url
from backend.retriever.hybrid import reciprocal_rank_fusion
from backend.retriever.keyword import KeywordRetriever
from backend.retriever.models import (
    FindSimilarRequest,
    ProvenanceInfo,
    SearchMode,
    SearchRequest,
    SearchResult,
)
from backend.retriever.neural import NeuralRetriever
from backend.retriever.ordering import apply_strict_ordering
from backend.retriever.service import DocumentNotFoundError, RetrieverService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def scorer():
    """Create a ProvenanceScorer instance."""
    return ProvenanceScorer()


@pytest.fixture
def indexer(scorer):
    """Create an IndexerService with provenance scoring."""
    return IndexerService(provenance_scorer=scorer)


@pytest.fixture
async def populated_indexer(indexer):
    """Create an indexer with several documents indexed."""
    docs = [
        ("<html><body>Machine learning algorithms for natural language processing</body></html>",
         "https://example.com/ml-nlp"),
        ("<html><body>Deep learning neural networks and transformers architecture</body></html>",
         "https://example.com/deep-learning"),
        ("<html><body>Python programming best practices and design patterns</body></html>",
         "https://example.com/python"),
        ("<html><body>Database optimization and query performance tuning</body></html>",
         "https://example.com/databases"),
        ("<html><body>Cloud computing infrastructure and deployment strategies</body></html>",
         "https://example.com/cloud"),
    ]
    for content, url in docs:
        await indexer.index_document(content, url)
    return indexer


@pytest.fixture
def retriever(populated_indexer):
    """Create a RetrieverService with populated index."""
    return RetrieverService(indexer=populated_indexer)


# ---------------------------------------------------------------------------
# Task 11.1: Neural Retrieval
# ---------------------------------------------------------------------------


class TestNeuralRetrieval:
    """Tests for neural retrieval (vector ANN with seeded HNSW)."""

    async def test_neural_returns_results(self, populated_indexer):
        """Neural retrieval returns ranked results for a query."""
        neural = NeuralRetriever(populated_indexer.vector_index)
        results = neural.search("machine learning", num_results=3)
        assert len(results) > 0
        assert len(results) <= 3

    async def test_neural_scores_in_range(self, populated_indexer):
        """Neural retrieval scores are in [0.0, 1.0]."""
        neural = NeuralRetriever(populated_indexer.vector_index)
        results = neural.search("deep learning neural networks")
        for r in results:
            assert 0.0 <= r.score <= 1.0

    async def test_neural_deterministic(self, populated_indexer):
        """Same query produces same results (deterministic, seeded HNSW)."""
        neural = NeuralRetriever(populated_indexer.vector_index)
        results1 = neural.search("machine learning")
        results2 = neural.search("machine learning")
        assert results1 == results2

    async def test_neural_excludes_doc_ids(self, populated_indexer):
        """Neural retrieval respects exclude_doc_ids."""
        neural = NeuralRetriever(populated_indexer.vector_index)
        all_results = neural.search("programming", num_results=10)
        if all_results:
            exclude = {all_results[0].document_id}
            filtered = neural.search("programming", num_results=10, exclude_doc_ids=exclude)
            for r in filtered:
                assert r.document_id not in exclude

    async def test_neural_empty_query(self, populated_indexer):
        """Empty query returns no results."""
        neural = NeuralRetriever(populated_indexer.vector_index)
        results = neural.search("")
        assert results == []


# ---------------------------------------------------------------------------
# Task 11.2: Keyword Retrieval
# ---------------------------------------------------------------------------


class TestKeywordRetrieval:
    """Tests for keyword retrieval (BM25 with fixed analyzer)."""

    async def test_keyword_returns_results(self, populated_indexer):
        """Keyword retrieval returns results for matching terms."""
        keyword = KeywordRetriever(populated_indexer.lexical_index)
        results = keyword.search("machine learning")
        assert len(results) > 0

    async def test_keyword_scores_in_range(self, populated_indexer):
        """Keyword retrieval scores are normalized to [0.0, 1.0]."""
        keyword = KeywordRetriever(populated_indexer.lexical_index)
        results = keyword.search("python programming")
        for r in results:
            assert 0.0 <= r.score <= 1.0

    async def test_keyword_deterministic(self, populated_indexer):
        """Same query produces same BM25 results."""
        keyword = KeywordRetriever(populated_indexer.lexical_index)
        results1 = keyword.search("database optimization")
        results2 = keyword.search("database optimization")
        assert results1 == results2

    async def test_keyword_empty_query(self, populated_indexer):
        """Empty query returns no results."""
        keyword = KeywordRetriever(populated_indexer.lexical_index)
        results = keyword.search("")
        assert results == []

    async def test_keyword_no_match(self, populated_indexer):
        """Query with no matching terms returns empty."""
        keyword = KeywordRetriever(populated_indexer.lexical_index)
        results = keyword.search("xyzzyplugh")
        assert results == []


# ---------------------------------------------------------------------------
# Task 11.3: Hybrid Retrieval (RRF)
# ---------------------------------------------------------------------------


class TestHybridRetrieval:
    """Tests for hybrid retrieval (Reciprocal-Rank Fusion)."""

    async def test_hybrid_combines_results(self, populated_indexer):
        """Hybrid retrieval combines neural and keyword results."""
        neural = NeuralRetriever(populated_indexer.vector_index)
        keyword = KeywordRetriever(populated_indexer.lexical_index)

        neural_results = neural.search("machine learning", num_results=5)
        keyword_results = keyword.search("machine learning", num_results=5)

        hybrid = reciprocal_rank_fusion(neural_results, keyword_results, num_results=5)
        assert len(hybrid) > 0

    async def test_hybrid_scores_normalized(self, populated_indexer):
        """RRF scores are normalized to [0.0, 1.0]."""
        neural = NeuralRetriever(populated_indexer.vector_index)
        keyword = KeywordRetriever(populated_indexer.lexical_index)

        neural_results = neural.search("deep learning", num_results=5)
        keyword_results = keyword.search("deep learning", num_results=5)

        hybrid = reciprocal_rank_fusion(neural_results, keyword_results)
        for h in hybrid:
            assert 0.0 <= h.score <= 1.0

    async def test_hybrid_deterministic(self, populated_indexer):
        """RRF fusion is deterministic for same inputs."""
        neural = NeuralRetriever(populated_indexer.vector_index)
        keyword = KeywordRetriever(populated_indexer.lexical_index)

        n1 = neural.search("cloud computing", num_results=5)
        k1 = keyword.search("cloud computing", num_results=5)
        h1 = reciprocal_rank_fusion(n1, k1, num_results=5)

        n2 = neural.search("cloud computing", num_results=5)
        k2 = keyword.search("cloud computing", num_results=5)
        h2 = reciprocal_rank_fusion(n2, k2, num_results=5)

        assert h1 == h2

    def test_hybrid_empty_inputs(self):
        """RRF with empty inputs returns empty."""
        result = reciprocal_rank_fusion([], [])
        assert result == []


# ---------------------------------------------------------------------------
# Task 11.4: Strict Total Ordering
# ---------------------------------------------------------------------------


class TestStrictOrdering:
    """Tests for strict total ordering (score DESC, document_id ASC, version ASC)."""

    def test_orders_by_score_descending(self):
        """Results are ordered by score descending."""
        now = datetime.now(timezone.utc)
        prov = ProvenanceInfo(0.8, 0.2, now)
        results = [
            SearchResult("doc-a", "http://a.com", "A", 0.5, None, prov, 1),
            SearchResult("doc-b", "http://b.com", "B", 0.9, None, prov, 1),
            SearchResult("doc-c", "http://c.com", "C", 0.7, None, prov, 1),
        ]
        ordered = apply_strict_ordering(results)
        assert [r.score for r in ordered] == [0.9, 0.7, 0.5]

    def test_tiebreak_by_document_id_ascending(self):
        """Equal scores tie-break by document_id ascending."""
        now = datetime.now(timezone.utc)
        prov = ProvenanceInfo(0.8, 0.2, now)
        results = [
            SearchResult("doc-c", "http://c.com", "C", 0.8, None, prov, 1),
            SearchResult("doc-a", "http://a.com", "A", 0.8, None, prov, 1),
            SearchResult("doc-b", "http://b.com", "B", 0.8, None, prov, 1),
        ]
        ordered = apply_strict_ordering(results)
        assert [r.document_id for r in ordered] == ["doc-a", "doc-b", "doc-c"]

    def test_tiebreak_by_version_ascending(self):
        """Equal score and document_id tie-break by version ascending."""
        now = datetime.now(timezone.utc)
        prov = ProvenanceInfo(0.8, 0.2, now)
        results = [
            SearchResult("doc-a", "http://a.com", "A", 0.8, None, prov, 3),
            SearchResult("doc-a", "http://a.com", "A", 0.8, None, prov, 1),
            SearchResult("doc-a", "http://a.com", "A", 0.8, None, prov, 2),
        ]
        ordered = apply_strict_ordering(results)
        assert [r.version for r in ordered] == [1, 2, 3]

    def test_empty_list(self):
        """Empty input returns empty output."""
        assert apply_strict_ordering([]) == []


# ---------------------------------------------------------------------------
# Task 11.5: X-Index-Version
# ---------------------------------------------------------------------------


class TestIndexVersion:
    """Tests for X-Index-Version header tracking."""

    async def test_initial_version_is_one(self, retriever):
        """Initial index version for a tenant is 1."""
        version = retriever.get_index_version("tenant-1")
        assert version == 1

    async def test_version_increments(self, retriever):
        """Index version increments monotonically."""
        v1 = retriever.get_index_version("tenant-1")
        v2 = retriever.increment_index_version("tenant-1")
        v3 = retriever.increment_index_version("tenant-1")
        assert v2 == v1 + 1
        assert v3 == v2 + 1

    async def test_version_per_tenant(self, retriever):
        """Each tenant has independent index versions."""
        retriever.increment_index_version("tenant-a")
        retriever.increment_index_version("tenant-a")
        v_a = retriever.get_index_version("tenant-a")
        v_b = retriever.get_index_version("tenant-b")
        assert v_a == 3
        assert v_b == 1

    async def test_response_includes_version(self, retriever):
        """Search response includes the index version."""
        request = SearchRequest(
            query="machine learning",
            mode=SearchMode.NEURAL,
            tenant_id="tenant-1",
        )
        response = retriever.search(request)
        assert response.index_version >= 1


# ---------------------------------------------------------------------------
# Task 11.6: Warm Cache
# ---------------------------------------------------------------------------


class TestWarmCache:
    """Tests for warm-cache layer with 5-min TTL."""

    def test_cache_miss_returns_none(self):
        """Cache miss returns None."""
        cache = WarmCache()
        result = cache.get("t1", "query", "neural", None, None, 10)
        assert result is None

    def test_cache_hit_returns_results(self):
        """Cache hit returns stored results."""
        cache = WarmCache()
        now = datetime.now(timezone.utc)
        prov = ProvenanceInfo(0.8, 0.2, now)
        results = [SearchResult("doc-1", "http://a.com", "A", 0.9, None, prov, 1)]

        cache.put("t1", "query", "neural", None, None, 10, results, 1)
        cached = cache.get("t1", "query", "neural", None, None, 10)
        assert cached == results

    def test_cache_ttl_expiry(self):
        """Expired entries return None."""
        cache = WarmCache(ttl_seconds=0.01)  # Very short TTL
        now = datetime.now(timezone.utc)
        prov = ProvenanceInfo(0.8, 0.2, now)
        results = [SearchResult("doc-1", "http://a.com", "A", 0.9, None, prov, 1)]

        cache.put("t1", "query", "neural", None, None, 10, results, 1)
        time.sleep(0.02)
        cached = cache.get("t1", "query", "neural", None, None, 10)
        assert cached is None

    def test_cache_different_params_miss(self):
        """Different parameters produce cache miss."""
        cache = WarmCache()
        now = datetime.now(timezone.utc)
        prov = ProvenanceInfo(0.8, 0.2, now)
        results = [SearchResult("doc-1", "http://a.com", "A", 0.9, None, prov, 1)]

        cache.put("t1", "query", "neural", None, None, 10, results, 1)
        # Different mode
        assert cache.get("t1", "query", "keyword", None, None, 10) is None
        # Different query
        assert cache.get("t1", "other", "neural", None, None, 10) is None
        # Different tenant
        assert cache.get("t2", "query", "neural", None, None, 10) is None

    def test_cache_invalidate_all(self):
        """Invalidate all clears the cache."""
        cache = WarmCache()
        now = datetime.now(timezone.utc)
        prov = ProvenanceInfo(0.8, 0.2, now)
        results = [SearchResult("doc-1", "http://a.com", "A", 0.9, None, prov, 1)]

        cache.put("t1", "q1", "neural", None, None, 10, results, 1)
        cache.put("t2", "q2", "keyword", None, None, 10, results, 1)
        count = cache.invalidate()
        assert count == 2
        assert cache.size == 0

    async def test_search_uses_cache(self, retriever):
        """Second identical search returns cache hit."""
        request = SearchRequest(
            query="machine learning",
            mode=SearchMode.NEURAL,
            tenant_id="tenant-1",
        )
        response1 = retriever.search(request)
        assert not response1.cache_hit

        response2 = retriever.search(request)
        assert response2.cache_hit
        assert response2.results == response1.results


# ---------------------------------------------------------------------------
# Task 11.7: min_credibility Filtering
# ---------------------------------------------------------------------------


class TestMinCredibilityFilter:
    """Tests for min_credibility filtering (R10.3)."""

    def test_excludes_below_threshold(self):
        """Documents with credibility < threshold are excluded."""
        now = datetime.now(timezone.utc)
        results = [
            SearchResult("d1", "http://a.com", "A", 0.9, None,
                         ProvenanceInfo(0.3, 0.2, now), 1),
            SearchResult("d2", "http://b.com", "B", 0.8, None,
                         ProvenanceInfo(0.7, 0.2, now), 1),
            SearchResult("d3", "http://c.com", "C", 0.7, None,
                         ProvenanceInfo(0.5, 0.2, now), 1),
        ]
        filtered = apply_min_credibility(results, 0.5)
        assert len(filtered) == 2
        assert all(r.provenance.credibility_score >= 0.5 for r in filtered)

    def test_includes_at_threshold(self):
        """Documents with credibility == threshold are INCLUDED."""
        now = datetime.now(timezone.utc)
        results = [
            SearchResult("d1", "http://a.com", "A", 0.9, None,
                         ProvenanceInfo(0.5, 0.2, now), 1),
        ]
        filtered = apply_min_credibility(results, 0.5)
        assert len(filtered) == 1

    def test_excludes_strictly_below(self):
        """Documents with credibility strictly < threshold are excluded."""
        now = datetime.now(timezone.utc)
        results = [
            SearchResult("d1", "http://a.com", "A", 0.9, None,
                         ProvenanceInfo(0.4999, 0.2, now), 1),
        ]
        filtered = apply_min_credibility(results, 0.5)
        assert len(filtered) == 0


# ---------------------------------------------------------------------------
# Task 11.8: max_ai_generated_likelihood Filtering
# ---------------------------------------------------------------------------


class TestMaxAiGeneratedFilter:
    """Tests for max_ai_generated_likelihood filtering (R10.4)."""

    def test_excludes_above_threshold(self):
        """Documents with ai_likelihood > threshold are excluded."""
        now = datetime.now(timezone.utc)
        results = [
            SearchResult("d1", "http://a.com", "A", 0.9, None,
                         ProvenanceInfo(0.8, 0.9, now), 1),
            SearchResult("d2", "http://b.com", "B", 0.8, None,
                         ProvenanceInfo(0.8, 0.3, now), 1),
            SearchResult("d3", "http://c.com", "C", 0.7, None,
                         ProvenanceInfo(0.8, 0.5, now), 1),
        ]
        filtered = apply_max_ai_generated(results, 0.5)
        assert len(filtered) == 2
        assert all(r.provenance.ai_generated_likelihood <= 0.5 for r in filtered)

    def test_includes_at_threshold(self):
        """Documents with ai_likelihood == threshold are INCLUDED."""
        now = datetime.now(timezone.utc)
        results = [
            SearchResult("d1", "http://a.com", "A", 0.9, None,
                         ProvenanceInfo(0.8, 0.5, now), 1),
        ]
        filtered = apply_max_ai_generated(results, 0.5)
        assert len(filtered) == 1

    def test_excludes_strictly_above(self):
        """Documents with ai_likelihood strictly > threshold are excluded."""
        now = datetime.now(timezone.utc)
        results = [
            SearchResult("d1", "http://a.com", "A", 0.9, None,
                         ProvenanceInfo(0.8, 0.5001, now), 1),
        ]
        filtered = apply_max_ai_generated(results, 0.5)
        assert len(filtered) == 0


# ---------------------------------------------------------------------------
# Task 11.9: find_similar — Document Exclusion
# ---------------------------------------------------------------------------


class TestFindSimilar:
    """Tests for find_similar with document exclusion (R4.2)."""

    async def test_excludes_input_document(self, retriever):
        """find_similar excludes the input document from results."""
        # Get a document URL from the index
        docs = retriever._indexer.documents
        if not docs:
            pytest.skip("No documents in index")

        doc_id = next(iter(docs))
        versions = docs[doc_id]
        url = versions[0].source_url

        request = FindSimilarRequest(url=url, num_results=10, tenant_id="t1")
        response = retriever.find_similar(request)

        # The input document should NOT appear in results
        for result in response.results:
            assert result.document_id != doc_id

    async def test_excludes_all_versions(self, populated_indexer):
        """find_similar excludes ALL versions of the input document (R4.2)."""
        # Index a second version of a document
        await populated_indexer.index_document(
            "<html><body>Updated machine learning content with new algorithms</body></html>",
            "https://example.com/ml-nlp",
        )

        retriever = RetrieverService(indexer=populated_indexer)
        request = FindSimilarRequest(
            url="https://example.com/ml-nlp",
            num_results=10,
            tenant_id="t1",
        )
        response = retriever.find_similar(request)

        # Get the document_id for the URL
        doc_versions = populated_indexer.get_document_by_url("https://example.com/ml-nlp")
        doc_id = doc_versions[0].document_id

        # No version of this document should appear
        for result in response.results:
            assert result.document_id != doc_id

    async def test_unknown_url_raises_error(self, retriever):
        """find_similar with unknown URL raises DocumentNotFoundError."""
        request = FindSimilarRequest(
            url="https://nonexistent.com/page",
            num_results=10,
            tenant_id="t1",
        )
        with pytest.raises(DocumentNotFoundError):
            retriever.find_similar(request)


# ---------------------------------------------------------------------------
# Task 11.10: URL Canonicalization
# ---------------------------------------------------------------------------


class TestUrlCanonicalization:
    """Tests for URL canonicalization (R4.3)."""

    def test_lowercase_scheme(self):
        """Scheme is lowercased."""
        assert canonicalize_url("HTTP://Example.com/path") == "http://example.com/path"
        assert canonicalize_url("HTTPS://Example.com/path") == "https://example.com/path"

    def test_lowercase_host(self):
        """Host is lowercased."""
        assert canonicalize_url("https://EXAMPLE.COM/path") == "https://example.com/path"
        assert canonicalize_url("https://Sub.Domain.COM/path") == "https://sub.domain.com/path"

    def test_strip_fragment(self):
        """Fragment (#...) is stripped."""
        assert canonicalize_url("https://example.com/page#section") == "https://example.com/page"
        assert canonicalize_url("https://example.com/page#") == "https://example.com/page"

    def test_normalize_trailing_slash(self):
        """Trailing slash is normalized."""
        # Root path keeps slash
        assert canonicalize_url("https://example.com/") == "https://example.com/"
        # Trailing slash on non-root is removed
        assert canonicalize_url("https://example.com/path/") == "https://example.com/path"

    def test_remove_default_port(self):
        """Default ports (80/443) are removed."""
        assert canonicalize_url("http://example.com:80/path") == "http://example.com/path"
        assert canonicalize_url("https://example.com:443/path") == "https://example.com/path"

    def test_keep_non_default_port(self):
        """Non-default ports are preserved."""
        assert canonicalize_url("https://example.com:8080/path") == "https://example.com:8080/path"

    def test_combined_canonicalization(self):
        """Multiple canonicalization rules applied together."""
        url = "HTTPS://Example.COM:443/Path/Page/#section"
        expected = "https://example.com/Path/Page"
        assert canonicalize_url(url) == expected


# ---------------------------------------------------------------------------
# Integration: RetrieverService
# ---------------------------------------------------------------------------


class TestRetrieverService:
    """Integration tests for the full RetrieverService."""

    async def test_search_neural_mode(self, retriever):
        """Search in neural mode returns results."""
        request = SearchRequest(
            query="machine learning",
            mode=SearchMode.NEURAL,
            num_results=5,
            tenant_id="t1",
        )
        response = retriever.search(request)
        assert len(response.results) <= 5
        assert response.index_version >= 1

    async def test_search_keyword_mode(self, retriever):
        """Search in keyword mode returns results."""
        request = SearchRequest(
            query="python programming",
            mode=SearchMode.KEYWORD,
            num_results=5,
            tenant_id="t1",
        )
        response = retriever.search(request)
        assert len(response.results) > 0

    async def test_search_hybrid_mode(self, retriever):
        """Search in hybrid mode returns results."""
        request = SearchRequest(
            query="deep learning",
            mode=SearchMode.HYBRID,
            num_results=5,
            tenant_id="t1",
        )
        response = retriever.search(request)
        assert len(response.results) > 0

    async def test_results_have_required_fields(self, retriever):
        """Results include all required fields (R3.3)."""
        request = SearchRequest(
            query="machine learning",
            mode=SearchMode.NEURAL,
            tenant_id="t1",
        )
        response = retriever.search(request)
        for result in response.results:
            assert result.document_id
            assert result.url
            assert isinstance(result.title, str)
            assert 0.0 <= result.score <= 1.0
            assert result.provenance is not None
            assert 0.0 <= result.provenance.credibility_score <= 1.0
            assert 0.0 <= result.provenance.ai_generated_likelihood <= 1.0

    async def test_results_ordered_by_score(self, retriever):
        """Results are in non-increasing score order (R3.3)."""
        request = SearchRequest(
            query="machine learning",
            mode=SearchMode.NEURAL,
            num_results=10,
            tenant_id="t1",
        )
        response = retriever.search(request)
        scores = [r.score for r in response.results]
        assert scores == sorted(scores, reverse=True)

    async def test_deterministic_ranking(self, retriever):
        """Same query produces identical results (R3.4)."""
        request = SearchRequest(
            query="cloud computing",
            mode=SearchMode.HYBRID,
            num_results=5,
            tenant_id="t1",
        )
        # Clear cache to test actual retrieval determinism
        retriever.cache.invalidate()
        response1 = retriever.search(request)
        retriever.cache.invalidate()
        response2 = retriever.search(request)

        assert len(response1.results) == len(response2.results)
        for r1, r2 in zip(response1.results, response2.results):
            assert r1.document_id == r2.document_id
            assert r1.score == r2.score

    async def test_threshold_filters_applied(self, retriever):
        """Threshold filters are applied during search."""
        request = SearchRequest(
            query="machine learning",
            mode=SearchMode.NEURAL,
            num_results=10,
            min_credibility=0.99,  # Very high threshold
            tenant_id="t1",
        )
        response = retriever.search(request)
        for result in response.results:
            assert result.provenance.credibility_score >= 0.99

    async def test_num_results_respected(self, retriever):
        """num_results caps the number of returned results."""
        request = SearchRequest(
            query="machine learning",
            mode=SearchMode.NEURAL,
            num_results=2,
            tenant_id="t1",
        )
        response = retriever.search(request)
        assert len(response.results) <= 2
