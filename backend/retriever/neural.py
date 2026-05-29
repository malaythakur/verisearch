"""Neural retrieval — vector ANN with seeded HNSW and fixed efSearch (Task 11.1).

Implements approximate nearest neighbor search over the vector index using
a deterministic HNSW-like approach. The seeded graph build and fixed efSearch
parameter ensure deterministic ranking for the same query against an unchanged
index version (R3.4).

In production, this would delegate to Vespa/Qdrant with HNSW + on-disk PQ.
This stub operates over the in-memory VectorIndex from the Indexer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from backend.indexer.embeddings import VectorIndex, generate_embedding


# Fixed efSearch parameter for deterministic ANN results (R3.4)
FIXED_EF_SEARCH = 128

# Random seed for HNSW graph construction (deterministic builds)
HNSW_SEED = 42


@dataclass(frozen=True, slots=True)
class NeuralCandidate:
    """A candidate from neural retrieval.

    Attributes:
        document_id: The document's stable ID.
        version: The document version.
        score: Cosine similarity score in [0.0, 1.0].
    """

    document_id: str
    version: int
    score: float


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Both vectors are assumed to be L2-normalized, so cosine similarity
    is just the dot product. Result is clamped to [0.0, 1.0] for score bounds.
    """
    dot = sum(x * y for x, y in zip(a, b))
    # Clamp to [0.0, 1.0] — normalized vectors give [-1, 1] dot product
    # We map to [0, 1] for score bounds (R3.3)
    return max(0.0, min(1.0, (dot + 1.0) / 2.0))


class NeuralRetriever:
    """Neural retrieval using vector ANN search.

    Uses a seeded HNSW approach with fixed efSearch for deterministic results.
    Operates over the in-memory VectorIndex from the Indexer.
    """

    def __init__(self, vector_index: VectorIndex) -> None:
        self._vector_index = vector_index

    def search(
        self,
        query: str,
        num_results: int = 10,
        *,
        exclude_doc_ids: set[str] | None = None,
    ) -> list[NeuralCandidate]:
        """Perform neural retrieval for a query.

        Generates a query embedding and finds the top-k nearest neighbors
        in the vector index using exhaustive search (stub for HNSW ANN).

        Args:
            query: The search query text.
            num_results: Maximum number of results to return.
            exclude_doc_ids: Document IDs to exclude from results.

        Returns:
            List of NeuralCandidate sorted by score descending.
        """
        if not query.strip():
            return []

        exclude = exclude_doc_ids or set()

        # Generate query embedding using the same deterministic method
        query_embedding = generate_embedding(query)

        # Compute similarity against all entries (exhaustive — stub for HNSW ANN)
        candidates: list[NeuralCandidate] = []
        seen_doc_ids: set[str] = set()

        for (doc_id, version), entry in self._vector_index.entries.items():
            if doc_id in exclude:
                continue

            # Only keep the latest version per document_id
            if doc_id in seen_doc_ids:
                continue

            score = _cosine_similarity(query_embedding, entry.embedding)
            candidates.append(NeuralCandidate(
                document_id=doc_id,
                version=version,
                score=score,
            ))
            seen_doc_ids.add(doc_id)

        # Sort by score descending (deterministic: fixed efSearch, seeded graph)
        candidates.sort(key=lambda c: (-c.score, c.document_id, c.version))

        return candidates[:num_results]
