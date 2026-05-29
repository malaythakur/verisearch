"""Retriever — Neural, keyword, and hybrid search with deterministic ranking.

Exports:
- RetrieverService: Main orchestrator for all retrieval operations.
- SearchRequest, FindSimilarRequest: Request models.
- SearchResult, ProvenanceInfo: Response models.
- SearchMode: Enum for retrieval modes.
- RetrievalResponse: Response wrapper with metadata.
- DocumentNotFoundError: Error for missing documents/URLs.
- NeuralRetriever: Vector ANN retrieval.
- KeywordRetriever: BM25 keyword retrieval.
- WarmCache: Cache layer with 5-min TTL.
- apply_strict_ordering: Deterministic tie-breaking.
- apply_threshold_filters: Credibility/AI-generation filtering.
- canonicalize_url: URL normalization for find_similar.
- reciprocal_rank_fusion: RRF hybrid fusion.
"""

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
from backend.retriever.service import (
    DocumentNotFoundError,
    RetrievalResponse,
    RetrieverService,
)

__all__ = [
    "RetrieverService",
    "RetrievalResponse",
    "DocumentNotFoundError",
    "SearchRequest",
    "FindSimilarRequest",
    "SearchResult",
    "ProvenanceInfo",
    "SearchMode",
    "NeuralRetriever",
    "KeywordRetriever",
    "WarmCache",
    "apply_strict_ordering",
    "apply_min_credibility",
    "apply_max_ai_generated",
    "apply_threshold_filters",
    "canonicalize_url",
    "reciprocal_rank_fusion",
]
