"""Hybrid retrieval — Reciprocal-Rank Fusion with fixed k (Task 11.3).

Implements RRF to combine neural and keyword retrieval results into a
single ranked list. The fixed k parameter (k=60) ensures deterministic
fusion for the same inputs (R3.4).

RRF formula: score(d) = sum(1 / (k + rank_i(d))) for each ranker i
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.retriever.keyword import KeywordCandidate
from backend.retriever.neural import NeuralCandidate


# Fixed RRF k parameter for deterministic fusion (R3.4)
RRF_K = 60


@dataclass(frozen=True, slots=True)
class HybridCandidate:
    """A candidate from hybrid retrieval (RRF fusion).

    Attributes:
        document_id: The document's stable ID.
        version: The document version.
        score: RRF fusion score normalized to [0.0, 1.0].
    """

    document_id: str
    version: int
    score: float


def reciprocal_rank_fusion(
    neural_results: list[NeuralCandidate],
    keyword_results: list[KeywordCandidate],
    num_results: int = 10,
    k: int = RRF_K,
) -> list[HybridCandidate]:
    """Fuse neural and keyword results using Reciprocal-Rank Fusion.

    RRF assigns each document a score based on its rank in each result list:
        rrf_score(d) = sum(1 / (k + rank_i(d))) for each ranker i

    The fixed k=60 ensures stable, deterministic fusion (R3.4).

    Args:
        neural_results: Ranked results from neural retrieval.
        keyword_results: Ranked results from keyword retrieval.
        num_results: Maximum number of results to return.
        k: RRF constant (default 60).

    Returns:
        List of HybridCandidate sorted by RRF score descending.
    """
    # Accumulate RRF scores per document
    rrf_scores: dict[str, float] = {}
    doc_versions: dict[str, int] = {}

    # Neural contributions
    for rank, candidate in enumerate(neural_results, start=1):
        doc_id = candidate.document_id
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        doc_versions[doc_id] = candidate.version

    # Keyword contributions
    for rank, candidate in enumerate(keyword_results, start=1):
        doc_id = candidate.document_id
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        # Keep the version from whichever source we saw first
        if doc_id not in doc_versions:
            doc_versions[doc_id] = candidate.version

    if not rrf_scores:
        return []

    # Normalize to [0.0, 1.0]
    max_rrf = max(rrf_scores.values())
    candidates = []
    for doc_id, raw_score in rrf_scores.items():
        normalized = raw_score / max_rrf if max_rrf > 0 else 0.0
        candidates.append(HybridCandidate(
            document_id=doc_id,
            version=doc_versions[doc_id],
            score=normalized,
        ))

    # Sort by score descending with deterministic tie-breaking
    candidates.sort(key=lambda c: (-c.score, c.document_id, c.version))

    return candidates[:num_results]
