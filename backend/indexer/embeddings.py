"""Vector embedding generation and vector index write (Task 10.10).

Supports two modes:
- Real: Calls OpenAI text-embedding-3-small (when OPENAI_API_KEY is set)
- Fallback: Deterministic hash-based embeddings (for testing without API key)
"""

from __future__ import annotations

import hashlib
import os
import struct
from dataclasses import dataclass, field
from datetime import datetime, timezone


# Default embedding dimension (matches text-embedding-3-small)
DEFAULT_EMBEDDING_DIM = 256


@dataclass(frozen=True, slots=True)
class VectorEntry:
    """A vector embedding entry written to the vector index.

    Attributes:
        document_id: The document's stable ID.
        version: The document version.
        embedding: The vector embedding as a list of floats.
        written_at: When the entry was written.
    """

    document_id: str
    version: int
    embedding: list[float]
    written_at: datetime


def generate_embedding(text: str, dim: int = DEFAULT_EMBEDDING_DIM) -> list[float]:
    """Generate an embedding vector from text.

    Uses OpenAI text-embedding-3-small if OPENAI_API_KEY is set,
    otherwise falls back to deterministic hash-based embeddings.

    Args:
        text: The cleaned text to embed.
        dim: The embedding dimension (used for fallback only).

    Returns:
        A list of floats, normalized to unit length.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if api_key:
        try:
            return _openai_embedding(text, api_key, dim)
        except Exception:
            # Fall back to hash-based on any error
            pass

    return _hash_embedding(text, dim)


def _openai_embedding(text: str, api_key: str, dim: int) -> list[float]:
    """Generate embedding via OpenAI API (synchronous for indexing pipeline)."""
    try:
        import openai

        client = openai.OpenAI(api_key=api_key)
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text[:8000],  # API limit
            dimensions=dim,
        )
        return response.data[0].embedding
    except ImportError:
        return _hash_embedding(text, dim)


def _hash_embedding(text: str, dim: int) -> list[float]:
    """Generate a deterministic embedding vector from text using hashing.

    Fallback for when no API key is available. Produces consistent
    embeddings that support cosine similarity for testing.

    Args:
        text: The cleaned text to embed.
        dim: The embedding dimension.

    Returns:
        A list of floats of length `dim`, normalized to unit length.
    """
    embedding = []
    for i in range(dim):
        chunk = f"{text}:{i}".encode("utf-8")
        h = hashlib.md5(chunk).digest()  # noqa: S324
        raw_value = struct.unpack(">I", h[:4])[0]
        val = (raw_value / 0xFFFFFFFF) * 2.0 - 1.0
        embedding.append(val)

    # L2 normalize
    norm = sum(x * x for x in embedding) ** 0.5
    if norm > 0:
        embedding = [x / norm for x in embedding]

    return embedding


class VectorIndex:
    """In-memory vector index stub.

    Records vector writes for testing. In production, this would be
    backed by Vespa or Qdrant with HNSW indexing.
    """

    def __init__(self) -> None:
        self._entries: dict[tuple[str, int], VectorEntry] = {}

    @property
    def entries(self) -> dict[tuple[str, int], VectorEntry]:
        """Return all stored vector entries."""
        return dict(self._entries)

    def write(self, document_id: str, version: int, embedding: list[float]) -> VectorEntry:
        """Write a vector embedding to the index.

        Args:
            document_id: The document's stable ID.
            version: The document version.
            embedding: The embedding vector.

        Returns:
            The VectorEntry that was written.
        """
        entry = VectorEntry(
            document_id=document_id,
            version=version,
            embedding=embedding,
            written_at=datetime.now(timezone.utc),
        )
        self._entries[(document_id, version)] = entry
        return entry

    def get(self, document_id: str, version: int) -> VectorEntry | None:
        """Retrieve a vector entry by document_id and version."""
        return self._entries.get((document_id, version))

    def delete(self, document_id: str, version: int) -> bool:
        """Delete a vector entry. Returns True if it existed."""
        key = (document_id, version)
        if key in self._entries:
            del self._entries[key]
            return True
        return False
