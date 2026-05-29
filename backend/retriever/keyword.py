"""Keyword retrieval — BM25 with fixed analyzer pipeline (Task 11.2).

Implements BM25 scoring over the lexical index using the fixed analyzer
from the Indexer. The fixed analyzer pipeline ensures deterministic
scoring for the same query against an unchanged index (R3.4).

In production, this would delegate to OpenSearch/Elasticsearch with
custom analyzers. This stub operates over the in-memory LexicalIndex.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from backend.indexer.lexical import LexicalIndex, analyze_text


# BM25 parameters (fixed for deterministic scoring)
BM25_K1 = 1.2
BM25_B = 0.75


@dataclass(frozen=True, slots=True)
class KeywordCandidate:
    """A candidate from keyword retrieval.

    Attributes:
        document_id: The document's stable ID.
        version: The document version.
        score: BM25 score normalized to [0.0, 1.0].
    """

    document_id: str
    version: int
    score: float


class KeywordRetriever:
    """Keyword retrieval using BM25 scoring.

    Uses the fixed analyzer pipeline from the Indexer for deterministic
    tokenization and scoring. BM25 parameters are fixed (k1=1.2, b=0.75).
    """

    def __init__(self, lexical_index: LexicalIndex) -> None:
        self._lexical_index = lexical_index

    def search(
        self,
        query: str,
        num_results: int = 10,
        *,
        exclude_doc_ids: set[str] | None = None,
    ) -> list[KeywordCandidate]:
        """Perform keyword retrieval using BM25.

        Analyzes the query with the same fixed pipeline used for indexing,
        then computes BM25 scores against all indexed documents.

        Args:
            query: The search query text.
            num_results: Maximum number of results to return.
            exclude_doc_ids: Document IDs to exclude from results.

        Returns:
            List of KeywordCandidate sorted by score descending.
        """
        if not query.strip():
            return []

        exclude = exclude_doc_ids or set()

        # Analyze query with the same fixed pipeline
        query_tokens = analyze_text(query)
        if not query_tokens:
            return []

        # Gather all entries and compute corpus statistics
        entries = self._lexical_index.entries
        if not entries:
            return []

        # Only keep latest version per document_id
        latest_entries: dict[str, tuple[int, list[str]]] = {}
        for (doc_id, version), entry in entries.items():
            if doc_id in exclude:
                continue
            if doc_id not in latest_entries or version > latest_entries[doc_id][0]:
                latest_entries[doc_id] = (version, entry.tokens)

        if not latest_entries:
            return []

        # Corpus statistics for BM25
        num_docs = len(latest_entries)
        avg_dl = sum(len(tokens) for _, tokens in latest_entries.values()) / num_docs

        # Document frequency for each query token
        df: dict[str, int] = {}
        for token in set(query_tokens):
            count = sum(
                1 for _, tokens in latest_entries.values()
                if token in tokens
            )
            df[token] = count

        # Compute BM25 scores
        raw_scores: list[tuple[str, int, float]] = []
        for doc_id, (version, doc_tokens) in latest_entries.items():
            score = self._bm25_score(
                query_tokens, doc_tokens, num_docs, avg_dl, df
            )
            if score > 0:
                raw_scores.append((doc_id, version, score))

        if not raw_scores:
            return []

        # Normalize scores to [0.0, 1.0]
        max_score = max(s for _, _, s in raw_scores)
        candidates = []
        for doc_id, version, raw_score in raw_scores:
            normalized = raw_score / max_score if max_score > 0 else 0.0
            candidates.append(KeywordCandidate(
                document_id=doc_id,
                version=version,
                score=normalized,
            ))

        # Sort by score descending with deterministic tie-breaking
        candidates.sort(key=lambda c: (-c.score, c.document_id, c.version))

        return candidates[:num_results]

    def _bm25_score(
        self,
        query_tokens: list[str],
        doc_tokens: list[str],
        num_docs: int,
        avg_dl: float,
        df: dict[str, int],
    ) -> float:
        """Compute BM25 score for a document against query tokens.

        Uses fixed parameters k1=1.2, b=0.75 for deterministic scoring.
        """
        dl = len(doc_tokens)
        score = 0.0

        # Build term frequency map for the document
        tf_map: dict[str, int] = {}
        for token in doc_tokens:
            tf_map[token] = tf_map.get(token, 0) + 1

        for token in query_tokens:
            if token not in tf_map:
                continue

            tf = tf_map[token]
            doc_freq = df.get(token, 0)

            # IDF component
            idf = math.log((num_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)

            # TF component with length normalization
            tf_norm = (tf * (BM25_K1 + 1)) / (
                tf + BM25_K1 * (1 - BM25_B + BM25_B * dl / avg_dl)
            )

            score += idf * tf_norm

        return score
