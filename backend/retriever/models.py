"""Retriever data models (Task 11, R3, R4, R10).

Defines request/response types for the Retriever subsystem:
- SearchRequest: Parameters for neural/keyword/hybrid search.
- FindSimilarRequest: Parameters for find-similar queries.
- SearchResult: A single ranked result with provenance.
- ProvenanceInfo: Credibility and AI-generation scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class SearchMode(str, Enum):
    """Supported retrieval modes (R3.1)."""

    NEURAL = "neural"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


@dataclass(frozen=True, slots=True)
class ProvenanceInfo:
    """Provenance metadata for a search result (R3.3, R10.2).

    Attributes:
        credibility_score: Score in [0.0, 1.0] indicating source credibility.
        ai_generated_likelihood: Score in [0.0, 1.0] indicating AI generation likelihood.
        scored_at: When the scoring was performed (ISO 8601 UTC).
    """

    credibility_score: float
    ai_generated_likelihood: float
    scored_at: datetime


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A single ranked search result (R3.3).

    Attributes:
        document_id: Stable document identifier.
        url: Canonical URL of the document.
        title: Document title.
        score: Relevance score in [0.0, 1.0], non-increasing order.
        published_at: Publication timestamp (ISO 8601 UTC) or None.
        provenance: Credibility and AI-generation scores.
        version: Document version (used internally for ordering).
    """

    document_id: str
    url: str
    title: str
    score: float
    published_at: datetime | None
    provenance: ProvenanceInfo
    version: int = 1


@dataclass
class SearchRequest:
    """Parameters for a search query (R3.1).

    Attributes:
        query: Search query string (1–2048 Unicode code points after trim).
        mode: Retrieval mode (neural, keyword, hybrid).
        num_results: Max results to return (default 10, max 100).
        filters: Optional filter expression (Filter_AST or dict).
        pipeline_id: Optional pipeline to apply.
        min_credibility: Minimum credibility threshold (R10.3).
        max_ai_generated_likelihood: Maximum AI-generation threshold (R10.4).
        tenant_id: The requesting tenant's ID.
    """

    query: str
    mode: SearchMode = SearchMode.HYBRID
    num_results: int = 10
    filters: Any | None = None
    pipeline_id: str | None = None
    min_credibility: float | None = None
    max_ai_generated_likelihood: float | None = None
    tenant_id: str = ""


@dataclass
class FindSimilarRequest:
    """Parameters for a find-similar query (R4).

    Attributes:
        url: The URL to find similar documents for.
        num_results: Max results to return (default 10, max 100).
        filters: Optional filter expression.
        min_credibility: Minimum credibility threshold (R10.3).
        max_ai_generated_likelihood: Maximum AI-generation threshold (R10.4).
        tenant_id: The requesting tenant's ID.
    """

    url: str
    num_results: int = 10
    filters: Any | None = None
    min_credibility: float | None = None
    max_ai_generated_likelihood: float | None = None
    tenant_id: str = ""
