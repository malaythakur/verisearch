"""Main RetrieverService — orchestrates neural, keyword, hybrid retrieval (Task 11).

Coordinates all retriever components:
- Neural retrieval (vector ANN)
- Keyword retrieval (BM25)
- Hybrid retrieval (RRF fusion)
- Strict total ordering (deterministic tie-breaking)
- Warm cache (5-min TTL)
- Threshold filtering (min_credibility, max_ai_generated_likelihood)
- find_similar with document exclusion and URL canonicalization
- X-Index-Version header tracking
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from backend.indexer.embeddings import VectorIndex
from backend.indexer.lexical import LexicalIndex
from backend.indexer.service import DocumentVersion, IndexerService
from backend.retriever.cache import WarmCache
from backend.retriever.filters import apply_threshold_filters
from backend.retriever.find_similar import canonicalize_url
from backend.retriever.hybrid import HybridCandidate, reciprocal_rank_fusion
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


@dataclass
class RetrievalResponse:
    """Response from the RetrieverService.

    Attributes:
        results: Ranked search results.
        index_version: The index version used for this response.
        cache_hit: Whether the response was served from cache.
    """

    results: list[SearchResult]
    index_version: int
    cache_hit: bool = False


class RetrieverService:
    """Main retriever service orchestrating all retrieval components.

    Provides search and find_similar operations with:
    - Deterministic ranking (R3.4, R4.4)
    - Warm caching (R3.2)
    - Threshold filtering (R10.3, R10.4)
    - Document exclusion for find_similar (R4.2)
    - URL canonicalization (R4.3)
    - X-Index-Version tracking (monotonic per tenant)
    """

    def __init__(
        self,
        *,
        indexer: IndexerService,
        vector_index: VectorIndex | None = None,
        lexical_index: LexicalIndex | None = None,
        cache: WarmCache | None = None,
    ) -> None:
        self._indexer = indexer
        self._vector_index = vector_index or indexer.vector_index
        self._lexical_index = lexical_index or indexer.lexical_index
        self._cache = cache or WarmCache()

        self._neural = NeuralRetriever(self._vector_index)
        self._keyword = KeywordRetriever(self._lexical_index)

        # OpenSearch client — used when available, falls back to in-memory
        from backend.retriever.opensearch_client import OpenSearchClient
        self._opensearch = OpenSearchClient()
        if self._opensearch.is_available:
            self._opensearch.ensure_index()

        # Monotonic index version per tenant (incremented on index changes)
        self._index_versions: dict[str, int] = {}

    @property
    def cache(self) -> WarmCache:
        """Return the warm cache instance."""
        return self._cache

    def get_index_version(self, tenant_id: str) -> int:
        """Get the current index version for a tenant.

        The index version is a monotonic integer that increments
        whenever the tenant's view of the index changes.

        Args:
            tenant_id: The tenant ID.

        Returns:
            Current index version (starts at 1).
        """
        if tenant_id not in self._index_versions:
            self._index_versions[tenant_id] = 1
        return self._index_versions[tenant_id]

    def increment_index_version(self, tenant_id: str) -> int:
        """Increment the index version for a tenant.

        Called when the index changes (new documents indexed, etc.).

        Args:
            tenant_id: The tenant ID.

        Returns:
            New index version.
        """
        current = self.get_index_version(tenant_id)
        self._index_versions[tenant_id] = current + 1
        return self._index_versions[tenant_id]

    def search(self, request: SearchRequest) -> RetrievalResponse:
        """Execute a search query.

        Orchestrates:
        1. Check warm cache.
        2. Execute retrieval (neural/keyword/hybrid).
        3. Build SearchResult objects with provenance.
        4. Apply threshold filters.
        5. Apply strict total ordering.
        6. Limit to num_results.
        7. Store in cache.

        Args:
            request: The search request parameters.

        Returns:
            RetrievalResponse with ranked results and metadata.
        """
        tenant_id = request.tenant_id
        index_version = self.get_index_version(tenant_id)

        # 1. Check warm cache
        cached = self._cache.get(
            tenant_id=tenant_id,
            query=request.query,
            mode=request.mode.value,
            filters=request.filters,
            pipeline_id=request.pipeline_id,
            num_results=request.num_results,
        )
        if cached is not None:
            return RetrievalResponse(
                results=cached,
                index_version=index_version,
                cache_hit=True,
            )

        # 2. Execute retrieval based on mode
        results = self._execute_retrieval(request)

        # 3. Apply threshold filters
        results = apply_threshold_filters(
            results,
            min_credibility=request.min_credibility,
            max_ai_generated_likelihood=request.max_ai_generated_likelihood,
        )

        # 4. Apply strict total ordering
        results = apply_strict_ordering(results)

        # 5. Limit to num_results
        results = results[: request.num_results]

        # 6. Store in cache
        self._cache.put(
            tenant_id=tenant_id,
            query=request.query,
            mode=request.mode.value,
            filters=request.filters,
            pipeline_id=request.pipeline_id,
            num_results=request.num_results,
            results=results,
            index_version=index_version,
        )

        return RetrievalResponse(
            results=results,
            index_version=index_version,
            cache_hit=False,
        )

    def find_similar(self, request: FindSimilarRequest) -> RetrievalResponse:
        """Find documents similar to a given URL (R4).

        Orchestrates:
        1. Canonicalize the input URL (R4.3).
        2. Look up the document by canonical URL.
        3. Get all versions of the document for exclusion (R4.2).
        4. Execute neural retrieval excluding the input document.
        5. Apply threshold filters.
        6. Apply strict total ordering.
        7. Limit to num_results.

        Args:
            request: The find-similar request parameters.

        Returns:
            RetrievalResponse with ranked results.

        Raises:
            DocumentNotFoundError: If the URL is not in the index.
        """
        tenant_id = request.tenant_id
        index_version = self.get_index_version(tenant_id)

        # 1. Canonicalize URL
        canonical_url = canonicalize_url(request.url)

        # 2. Look up document by URL
        doc_versions = self._indexer.get_document_by_url(canonical_url)
        if not doc_versions:
            # Also try the original URL in case it was indexed without canonicalization
            doc_versions = self._indexer.get_document_by_url(request.url)

        if not doc_versions:
            raise DocumentNotFoundError(
                f"URL not found in index: {request.url}"
            )

        # 3. Get document_id for exclusion (R4.2 — exclude ALL versions)
        document_id = doc_versions[0].document_id
        exclude_doc_ids = {document_id}

        # 4. Execute neural retrieval with exclusion
        # Use the document's text as the query for similarity
        latest_version = doc_versions[-1]
        neural_candidates = self._neural.search(
            latest_version.cleaned_text,
            num_results=request.num_results * 2,  # Over-fetch for filtering
            exclude_doc_ids=exclude_doc_ids,
        )

        # 5. Build SearchResult objects
        results = self._candidates_to_results(
            [(c.document_id, c.version, c.score) for c in neural_candidates]
        )

        # 6. Apply threshold filters
        results = apply_threshold_filters(
            results,
            min_credibility=request.min_credibility,
            max_ai_generated_likelihood=request.max_ai_generated_likelihood,
        )

        # 7. Apply strict total ordering
        results = apply_strict_ordering(results)

        # 8. Limit to num_results
        results = results[: request.num_results]

        return RetrievalResponse(
            results=results,
            index_version=index_version,
            cache_hit=False,
        )

    def _execute_retrieval(self, request: SearchRequest) -> list[SearchResult]:
        """Execute retrieval based on the requested mode.

        Uses OpenSearch when available, falls back to in-memory indexes.

        Args:
            request: The search request.

        Returns:
            List of SearchResult objects (unordered).
        """
        # Try OpenSearch first
        if self._opensearch.is_available:
            return self._execute_opensearch_retrieval(request)

        # Fall back to in-memory
        if request.mode == SearchMode.NEURAL:
            candidates = self._neural.search(
                request.query, num_results=request.num_results * 2
            )
            return self._candidates_to_results(
                [(c.document_id, c.version, c.score) for c in candidates]
            )

        elif request.mode == SearchMode.KEYWORD:
            candidates = self._keyword.search(
                request.query, num_results=request.num_results * 2
            )
            return self._candidates_to_results(
                [(c.document_id, c.version, c.score) for c in candidates]
            )

        else:  # HYBRID
            neural_candidates = self._neural.search(
                request.query, num_results=request.num_results * 2
            )
            keyword_candidates = self._keyword.search(
                request.query, num_results=request.num_results * 2
            )
            hybrid_candidates = reciprocal_rank_fusion(
                neural_candidates,
                keyword_candidates,
                num_results=request.num_results * 2,
            )
            return self._candidates_to_results(
                [(c.document_id, c.version, c.score) for c in hybrid_candidates]
            )

    def _execute_opensearch_retrieval(self, request: SearchRequest) -> list[SearchResult]:
        """Execute retrieval via OpenSearch."""
        from backend.indexer.embeddings import generate_embedding

        if request.mode == SearchMode.NEURAL:
            embedding = generate_embedding(request.query)
            hits = self._opensearch.vector_search(
                embedding, num_results=request.num_results * 2, tenant_id=request.tenant_id
            )
        elif request.mode == SearchMode.KEYWORD:
            hits = self._opensearch.keyword_search(
                request.query, num_results=request.num_results * 2, tenant_id=request.tenant_id
            )
        else:  # HYBRID
            embedding = generate_embedding(request.query)
            hits = self._opensearch.hybrid_search(
                request.query, embedding, num_results=request.num_results * 2, tenant_id=request.tenant_id
            )

        # Convert OpenSearch hits to SearchResult objects
        results = []
        for hit in hits:
            source = hit.source
            provenance = ProvenanceInfo(
                credibility_score=source.get("credibility_score", 0.5),
                ai_generated_likelihood=source.get("ai_generated_likelihood", 0.5),
                scored_at=datetime.now(timezone.utc),
            )
            # Normalize score to [0, 1]
            max_score = hits[0].score if hits else 1.0
            normalized_score = hit.score / max_score if max_score > 0 else 0.0

            results.append(SearchResult(
                document_id=hit.document_id,
                url=source.get("url", ""),
                title=source.get("title", "")[:100],
                score=max(0.0, min(1.0, normalized_score)),
                published_at=None,
                provenance=provenance,
                version=hit.version,
            ))

        return results

    def _candidates_to_results(
        self,
        candidates: list[tuple[str, int, float]],
    ) -> list[SearchResult]:
        """Convert raw candidates to SearchResult objects with provenance.

        Looks up document metadata and provenance scores from the indexer.

        Args:
            candidates: List of (document_id, version, score) tuples.

        Returns:
            List of SearchResult objects.
        """
        results = []
        for doc_id, version, score in candidates:
            doc_version = self._get_document_version(doc_id, version)
            if doc_version is None:
                continue

            # Only include visible documents (scored by Provenance_Scorer)
            if not doc_version.visible:
                continue

            # Build provenance info
            provenance = self._build_provenance(doc_version)

            results.append(SearchResult(
                document_id=doc_id,
                url=doc_version.source_url,
                title=self._extract_title(doc_version),
                score=max(0.0, min(1.0, score)),  # Clamp to [0.0, 1.0]
                published_at=None,  # Would come from metadata in production
                provenance=provenance,
                version=version,
            ))

        return results

    def _get_document_version(
        self, document_id: str, version: int
    ) -> DocumentVersion | None:
        """Look up a specific document version from the indexer."""
        versions = self._indexer.get_document(document_id)
        if not versions:
            return None

        for v in versions:
            if v.version == version:
                return v

        # Fall back to latest version
        return versions[-1] if versions else None

    def _build_provenance(self, doc_version: DocumentVersion) -> ProvenanceInfo:
        """Build ProvenanceInfo from a document version's provenance scores."""
        if doc_version.provenance:
            return ProvenanceInfo(
                credibility_score=doc_version.provenance.credibility_score,
                ai_generated_likelihood=doc_version.provenance.ai_generated_likelihood,
                scored_at=doc_version.provenance.scored_at,
            )

        # Default provenance for documents without scores
        return ProvenanceInfo(
            credibility_score=0.5,
            ai_generated_likelihood=0.5,
            scored_at=datetime.now(timezone.utc),
        )

    def _extract_title(self, doc_version: DocumentVersion) -> str:
        """Extract a title from the document content.

        In production, this would come from metadata. For the stub,
        we use the first line or first 100 chars of cleaned text.
        """
        text = doc_version.cleaned_text
        if not text:
            return ""

        # Use first line as title
        first_line = text.split("\n")[0].strip()
        if len(first_line) > 100:
            return first_line[:100]
        return first_line


class DocumentNotFoundError(Exception):
    """Raised when a document/URL is not found in the index."""

    pass
