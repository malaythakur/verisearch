"""Strict total ordering for search results (Task 11.4).

Implements deterministic tie-breaking: score DESC, document_id ASC, version ASC.
This ensures that identical queries against an unchanged index version always
produce results in the same order (R3.4, R4.4).
"""

from __future__ import annotations

from backend.retriever.models import SearchResult


def apply_strict_ordering(results: list[SearchResult]) -> list[SearchResult]:
    """Sort results with strict total ordering for deterministic ranking.

    Ordering rules:
    1. score DESC (highest relevance first)
    2. document_id ASC (lexicographic tie-break)
    3. version ASC (numeric tie-break)

    This guarantees a unique position for every result, ensuring
    deterministic ranking for the same query against an unchanged
    index version (R3.4).

    Args:
        results: Unsorted or partially sorted search results.

    Returns:
        New list sorted with strict total ordering.
    """
    return sorted(
        results,
        key=lambda r: (-r.score, r.document_id, r.version),
    )
