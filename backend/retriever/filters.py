"""Threshold filtering for provenance scores (Task 11.7, 11.8, R10.3, R10.4).

Implements:
- min_credibility: documents with credibility_score < threshold are EXCLUDED;
  documents with credibility_score == threshold are INCLUDED (R10.3).
- max_ai_generated_likelihood: documents with ai_generated_likelihood > threshold
  are EXCLUDED; documents with ai_generated_likelihood == threshold are INCLUDED (R10.4).
"""

from __future__ import annotations

from backend.retriever.models import SearchResult


def apply_min_credibility(
    results: list[SearchResult],
    min_credibility: float,
) -> list[SearchResult]:
    """Filter results by minimum credibility threshold (R10.3).

    Exclusion rule: strict-less-than excluded, equality included.
    - credibility_score < min_credibility → EXCLUDED
    - credibility_score >= min_credibility → INCLUDED

    Args:
        results: Search results to filter.
        min_credibility: Minimum credibility threshold in [0.0, 1.0].

    Returns:
        Filtered list with only results meeting the threshold.
    """
    return [
        r for r in results
        if r.provenance.credibility_score >= min_credibility
    ]


def apply_max_ai_generated(
    results: list[SearchResult],
    max_ai_generated_likelihood: float,
) -> list[SearchResult]:
    """Filter results by maximum AI-generated likelihood threshold (R10.4).

    Exclusion rule: strict-greater-than excluded, equality included.
    - ai_generated_likelihood > max_ai_generated_likelihood → EXCLUDED
    - ai_generated_likelihood <= max_ai_generated_likelihood → INCLUDED

    Args:
        results: Search results to filter.
        max_ai_generated_likelihood: Maximum AI-generation threshold in [0.0, 1.0].

    Returns:
        Filtered list with only results meeting the threshold.
    """
    return [
        r for r in results
        if r.provenance.ai_generated_likelihood <= max_ai_generated_likelihood
    ]


def apply_threshold_filters(
    results: list[SearchResult],
    min_credibility: float | None = None,
    max_ai_generated_likelihood: float | None = None,
) -> list[SearchResult]:
    """Apply all threshold filters to results.

    Convenience function that applies both filters when specified.

    Args:
        results: Search results to filter.
        min_credibility: Optional minimum credibility threshold.
        max_ai_generated_likelihood: Optional maximum AI-generation threshold.

    Returns:
        Filtered list.
    """
    filtered = results

    if min_credibility is not None:
        filtered = apply_min_credibility(filtered, min_credibility)

    if max_ai_generated_likelihood is not None:
        filtered = apply_max_ai_generated(filtered, max_ai_generated_likelihood)

    return filtered
