"""Highlight extraction for /v1/contents endpoint.

Implements:
- Extraction of up to 5 highlight spans per document (R5.2).
- Half-open offset ranges: [start, end) satisfying 0 <= start < end <= len(cleaned_text).
- Query-based relevance matching for highlight selection.
"""

from __future__ import annotations

from backend.answer_engine.models import HighlightSpan

# Maximum number of highlight spans per document (R5.2)
MAX_HIGHLIGHTS_PER_DOCUMENT = 5


def extract_highlights(
    query: str,
    cleaned_text: str,
    max_spans: int = MAX_HIGHLIGHTS_PER_DOCUMENT,
) -> list[HighlightSpan]:
    """Extract highlight spans from a document based on a query.

    Returns between 0 and max_spans highlight spans, each as a half-open
    [start, end) range of Unicode code-point offsets into the cleaned text
    satisfying 0 <= start < end <= len(cleaned_text) (R5.2).

    Args:
        query: The search query to match against.
        cleaned_text: The document's cleaned text content.
        max_spans: Maximum number of spans to return (default 5, R5.2).

    Returns:
        List of HighlightSpan objects, at most max_spans.
    """
    if not query or not cleaned_text:
        return []

    if max_spans <= 0:
        return []

    # Cap max_spans at the requirement limit
    max_spans = min(max_spans, MAX_HIGHLIGHTS_PER_DOCUMENT)

    spans: list[HighlightSpan] = []

    # Simple keyword-based highlighting: find occurrences of query terms
    # in the cleaned text. A production implementation would use more
    # sophisticated NLP-based relevance matching.
    query_terms = _tokenize_query(query)

    if not query_terms:
        return []

    text_lower = cleaned_text.lower()

    # Find all matching positions for each query term
    matches: list[tuple[int, int]] = []
    for term in query_terms:
        term_lower = term.lower()
        start = 0
        while start < len(text_lower):
            pos = text_lower.find(term_lower, start)
            if pos == -1:
                break
            end = pos + len(term)
            # Validate half-open range constraints
            if 0 <= pos < end <= len(cleaned_text):
                matches.append((pos, end))
            start = pos + 1

    if not matches:
        # If no exact matches, try to find a relevant window
        # based on the first query term appearing as a substring
        return _fallback_highlights(query, cleaned_text, max_spans)

    # Sort by position and deduplicate overlapping spans
    matches.sort()
    merged = _merge_overlapping(matches)

    # Expand spans to sentence boundaries for better context
    expanded = _expand_to_context(merged, cleaned_text, context_chars=50)

    # Take the top max_spans spans
    for start, end in expanded[:max_spans]:
        # Ensure constraints: 0 <= start < end <= len(cleaned_text)
        start = max(0, start)
        end = min(end, len(cleaned_text))
        if start < end:
            spans.append(HighlightSpan(start=start, end=end))

    return spans


def validate_highlight_spans(
    spans: list[HighlightSpan], text_length: int
) -> bool:
    """Validate that all highlight spans satisfy the half-open range constraints.

    Checks: 0 <= start < end <= text_length for each span.
    Also checks: at most MAX_HIGHLIGHTS_PER_DOCUMENT spans.

    Args:
        spans: List of highlight spans to validate.
        text_length: Length of the cleaned text.

    Returns:
        True if all spans are valid, False otherwise.
    """
    if len(spans) > MAX_HIGHLIGHTS_PER_DOCUMENT:
        return False

    for span in spans:
        if not (0 <= span.start < span.end <= text_length):
            return False

    return True


def _tokenize_query(query: str) -> list[str]:
    """Split query into individual terms for matching."""
    # Simple whitespace tokenization; production would use proper NLP
    terms = query.strip().split()
    # Filter out very short terms (likely stopwords)
    return [t for t in terms if len(t) >= 2]


def _merge_overlapping(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping or adjacent spans."""
    if not spans:
        return []

    merged: list[tuple[int, int]] = [spans[0]]
    for start, end in spans[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            # Overlapping or adjacent — merge
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    return merged


def _expand_to_context(
    spans: list[tuple[int, int]], text: str, context_chars: int = 50
) -> list[tuple[int, int]]:
    """Expand spans to include surrounding context characters."""
    expanded: list[tuple[int, int]] = []
    for start, end in spans:
        # Expand backwards to word boundary
        new_start = max(0, start - context_chars)
        # Find word boundary
        while new_start > 0 and text[new_start] not in (" ", "\n", "\t"):
            new_start -= 1
        if new_start > 0:
            new_start += 1  # Skip the whitespace

        # Expand forwards to word boundary
        new_end = min(len(text), end + context_chars)
        while new_end < len(text) and text[new_end] not in (" ", "\n", "\t"):
            new_end += 1

        # Ensure valid half-open range
        new_start = max(0, new_start)
        new_end = min(new_end, len(text))
        if new_start < new_end:
            expanded.append((new_start, new_end))

    return expanded


def _fallback_highlights(
    query: str, cleaned_text: str, max_spans: int
) -> list[HighlightSpan]:
    """Generate fallback highlights when no exact matches are found.

    Uses a simple windowing approach over the beginning of the document.
    """
    if len(cleaned_text) == 0:
        return []

    # Return a single span covering the beginning of the document
    end = min(len(cleaned_text), 200)
    if end > 0:
        return [HighlightSpan(start=0, end=end)]
    return []
