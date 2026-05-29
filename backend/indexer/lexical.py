"""Lexical index write — BM25 with fixed analyzer (Task 10.11).

Stub implementation that records lexical index writes. In production,
this would write to OpenSearch/Elasticsearch with a fixed analysis
pipeline for deterministic BM25 scoring.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone


# Simple tokenizer: split on non-alphanumeric, lowercase, remove stopwords
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Minimal English stopwords for the fixed analyzer
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
    "neither", "each", "every", "all", "any", "few", "more", "most", "other",
    "some", "such", "no", "only", "own", "same", "than", "too", "very",
    "just", "because", "if", "when", "where", "how", "what", "which", "who",
    "whom", "this", "that", "these", "those", "it", "its", "i", "me", "my",
    "we", "our", "you", "your", "he", "him", "his", "she", "her", "they",
    "them", "their",
})


@dataclass(frozen=True, slots=True)
class LexicalEntry:
    """A lexical index entry for BM25 retrieval.

    Attributes:
        document_id: The document's stable ID.
        version: The document version.
        tokens: The analyzed token list.
        written_at: When the entry was written.
    """

    document_id: str
    version: int
    tokens: list[str]
    written_at: datetime


def analyze_text(text: str) -> list[str]:
    """Apply the fixed analyzer pipeline to text.

    Steps:
    1. Lowercase the text.
    2. Tokenize on word boundaries (alphanumeric sequences).
    3. Remove stopwords.

    This is a fixed analyzer to ensure deterministic BM25 scoring (R3.4).

    Args:
        text: The cleaned text to analyze.

    Returns:
        List of analyzed tokens.
    """
    lowered = text.lower()
    tokens = _TOKEN_RE.findall(lowered)
    return [t for t in tokens if t not in _STOPWORDS]


class LexicalIndex:
    """In-memory lexical index stub.

    Records lexical index writes for testing. In production, this would
    be backed by OpenSearch with a fixed analysis pipeline.
    """

    def __init__(self) -> None:
        self._entries: dict[tuple[str, int], LexicalEntry] = {}

    @property
    def entries(self) -> dict[tuple[str, int], LexicalEntry]:
        """Return all stored lexical entries."""
        return dict(self._entries)

    def write(self, document_id: str, version: int, text: str) -> LexicalEntry:
        """Analyze text and write to the lexical index.

        Args:
            document_id: The document's stable ID.
            version: The document version.
            text: The cleaned text to index.

        Returns:
            The LexicalEntry that was written.
        """
        tokens = analyze_text(text)
        entry = LexicalEntry(
            document_id=document_id,
            version=version,
            tokens=tokens,
            written_at=datetime.now(timezone.utc),
        )
        self._entries[(document_id, version)] = entry
        return entry

    def get(self, document_id: str, version: int) -> LexicalEntry | None:
        """Retrieve a lexical entry by document_id and version."""
        return self._entries.get((document_id, version))

    def delete(self, document_id: str, version: int) -> bool:
        """Delete a lexical entry. Returns True if it existed."""
        key = (document_id, version)
        if key in self._entries:
            del self._entries[key]
            return True
        return False
