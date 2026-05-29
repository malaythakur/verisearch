"""Warm-cache layer for search results (Task 11.6, R3.2).

Implements an in-memory cache keyed by (tenant_id, query, mode, filters, pipeline_id, num_results)
with a 5-minute TTL. Warm cache hits respond within 800ms p95 (R3.2).

In production, this would be backed by Redis with TTL-based expiration.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any

from backend.retriever.models import SearchResult


# Cache TTL in seconds (5 minutes per R3.2)
CACHE_TTL_SECONDS = 300


@dataclass
class CacheEntry:
    """A cached search result set.

    Attributes:
        results: The cached search results.
        index_version: The index version when results were cached.
        created_at: Unix timestamp when the entry was created.
    """

    results: list[SearchResult]
    index_version: int
    created_at: float


class WarmCache:
    """In-memory warm cache for search results.

    Keyed by (tenant_id, query, mode, filters, pipeline_id, num_results).
    Entries expire after 5 minutes (CACHE_TTL_SECONDS).
    """

    def __init__(self, ttl_seconds: float = CACHE_TTL_SECONDS) -> None:
        self._ttl = ttl_seconds
        self._store: dict[str, CacheEntry] = {}

    @property
    def size(self) -> int:
        """Return the number of entries in the cache."""
        return len(self._store)

    def get(
        self,
        tenant_id: str,
        query: str,
        mode: str,
        filters: Any | None,
        pipeline_id: str | None,
        num_results: int,
    ) -> list[SearchResult] | None:
        """Look up cached results for the given parameters.

        Returns None on cache miss or expired entry.

        Args:
            tenant_id: The requesting tenant's ID.
            query: The search query.
            mode: The retrieval mode.
            filters: Optional filter expression.
            pipeline_id: Optional pipeline ID.
            num_results: Number of results requested.

        Returns:
            Cached results if hit and not expired, None otherwise.
        """
        key = self._make_key(tenant_id, query, mode, filters, pipeline_id, num_results)
        entry = self._store.get(key)

        if entry is None:
            return None

        # Check TTL
        if time.time() - entry.created_at > self._ttl:
            del self._store[key]
            return None

        return entry.results

    def put(
        self,
        tenant_id: str,
        query: str,
        mode: str,
        filters: Any | None,
        pipeline_id: str | None,
        num_results: int,
        results: list[SearchResult],
        index_version: int,
    ) -> None:
        """Store results in the cache.

        Args:
            tenant_id: The requesting tenant's ID.
            query: The search query.
            mode: The retrieval mode.
            filters: Optional filter expression.
            pipeline_id: Optional pipeline ID.
            num_results: Number of results requested.
            results: The search results to cache.
            index_version: The current index version.
        """
        key = self._make_key(tenant_id, query, mode, filters, pipeline_id, num_results)
        self._store[key] = CacheEntry(
            results=results,
            index_version=index_version,
            created_at=time.time(),
        )

    def invalidate(self, tenant_id: str | None = None) -> int:
        """Invalidate cache entries.

        Args:
            tenant_id: If provided, only invalidate entries for this tenant.
                       If None, invalidate all entries.

        Returns:
            Number of entries invalidated.
        """
        if tenant_id is None:
            count = len(self._store)
            self._store.clear()
            return count

        # Invalidate entries for a specific tenant
        keys_to_remove = [
            k for k in self._store if k.startswith(f"{tenant_id}:")
        ]
        for k in keys_to_remove:
            del self._store[k]
        return len(keys_to_remove)

    def _make_key(
        self,
        tenant_id: str,
        query: str,
        mode: str,
        filters: Any | None,
        pipeline_id: str | None,
        num_results: int,
    ) -> str:
        """Create a deterministic cache key from the search parameters."""
        # Serialize filters deterministically
        filters_str = json.dumps(filters, sort_keys=True, default=str) if filters else ""

        # Combine all components
        raw = f"{tenant_id}:{query}:{mode}:{filters_str}:{pipeline_id or ''}:{num_results}"

        # Hash for fixed-length key
        key_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return f"{tenant_id}:{key_hash}"
