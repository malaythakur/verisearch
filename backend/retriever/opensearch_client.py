"""OpenSearch client for neural and keyword retrieval.

Connects to OpenSearch for:
- BM25 keyword search
- kNN vector search (neural)
- Hybrid search via RRF

Falls back to in-memory retrieval if OPENSEARCH_URL is not set or
the cluster is unreachable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass
class OpenSearchHit:
    """A single hit from OpenSearch."""

    document_id: str
    version: int
    score: float
    source: dict[str, Any]


class OpenSearchClient:
    """Client for OpenSearch vector and keyword search.

    Lazily initializes the connection. Falls back gracefully if
    opensearchpy is not installed or the cluster is unreachable.
    """

    INDEX_NAME = "documents"

    def __init__(self, url: str | None = None):
        self._url = url or os.environ.get("OPENSEARCH_URL", "")
        self._client = None
        self._available = None

    def _get_client(self):
        """Lazily initialize the OpenSearch client."""
        if self._client is not None:
            return self._client

        if not self._url:
            self._available = False
            return None

        try:
            from opensearchpy import OpenSearch

            self._client = OpenSearch(
                hosts=[self._url],
                use_ssl=self._url.startswith("https"),
                verify_certs=False,
                timeout=10,
            )
            # Test connectivity
            self._client.info()
            self._available = True
            return self._client
        except Exception:
            self._available = False
            return None

    @property
    def is_available(self) -> bool:
        """Check if OpenSearch is available."""
        if self._available is None:
            self._get_client()
        return self._available or False

    def ensure_index(self) -> bool:
        """Create the documents index if it doesn't exist."""
        client = self._get_client()
        if not client:
            return False

        try:
            if not client.indices.exists(index=self.INDEX_NAME):
                client.indices.create(
                    index=self.INDEX_NAME,
                    body={
                        "settings": {
                            "index": {
                                "knn": True,
                                "number_of_shards": 1,
                                "number_of_replicas": 0,
                            }
                        },
                        "mappings": {
                            "properties": {
                                "document_id": {"type": "keyword"},
                                "version": {"type": "integer"},
                                "tenant_id": {"type": "keyword"},
                                "url": {"type": "keyword"},
                                "title": {"type": "text"},
                                "cleaned_text": {"type": "text", "analyzer": "standard"},
                                "embedding": {
                                    "type": "knn_vector",
                                    "dimension": 256,
                                    "method": {
                                        "name": "hnsw",
                                        "space_type": "cosinesimil",
                                        "engine": "nmslib",
                                    },
                                },
                                "credibility_score": {"type": "float"},
                                "ai_generated_likelihood": {"type": "float"},
                                "visible": {"type": "boolean"},
                                "indexed_at": {"type": "date"},
                            }
                        },
                    },
                )
            return True
        except Exception:
            return False

    def index_document(
        self,
        document_id: str,
        version: int,
        tenant_id: str,
        url: str,
        title: str,
        cleaned_text: str,
        embedding: list[float],
        credibility_score: float = 0.5,
        ai_generated_likelihood: float = 0.5,
    ) -> bool:
        """Index a document into OpenSearch."""
        client = self._get_client()
        if not client:
            return False

        try:
            from datetime import datetime, timezone

            client.index(
                index=self.INDEX_NAME,
                id=f"{document_id}_{version}",
                body={
                    "document_id": document_id,
                    "version": version,
                    "tenant_id": tenant_id,
                    "url": url,
                    "title": title,
                    "cleaned_text": cleaned_text,
                    "embedding": embedding,
                    "credibility_score": credibility_score,
                    "ai_generated_likelihood": ai_generated_likelihood,
                    "visible": True,
                    "indexed_at": datetime.now(timezone.utc).isoformat(),
                },
                refresh="wait_for",
            )
            return True
        except Exception:
            return False

    def keyword_search(
        self, query: str, num_results: int = 10, tenant_id: str | None = None
    ) -> list[OpenSearchHit]:
        """BM25 keyword search."""
        client = self._get_client()
        if not client:
            return []

        try:
            body: dict[str, Any] = {
                "size": num_results,
                "query": {
                    "bool": {
                        "must": [
                            {"multi_match": {"query": query, "fields": ["title^2", "cleaned_text"]}},
                        ],
                        "filter": [{"term": {"visible": True}}],
                    }
                },
            }

            if tenant_id:
                body["query"]["bool"]["filter"].append({"term": {"tenant_id": tenant_id}})

            response = client.search(index=self.INDEX_NAME, body=body)
            return self._parse_hits(response)
        except Exception:
            return []

    def vector_search(
        self, embedding: list[float], num_results: int = 10, tenant_id: str | None = None
    ) -> list[OpenSearchHit]:
        """kNN vector search."""
        client = self._get_client()
        if not client:
            return []

        try:
            body: dict[str, Any] = {
                "size": num_results,
                "query": {
                    "bool": {
                        "must": [
                            {
                                "knn": {
                                    "embedding": {
                                        "vector": embedding,
                                        "k": num_results,
                                    }
                                }
                            }
                        ],
                        "filter": [{"term": {"visible": True}}],
                    }
                },
            }

            if tenant_id:
                body["query"]["bool"]["filter"].append({"term": {"tenant_id": tenant_id}})

            response = client.search(index=self.INDEX_NAME, body=body)
            return self._parse_hits(response)
        except Exception:
            return []

    def hybrid_search(
        self, query: str, embedding: list[float], num_results: int = 10, tenant_id: str | None = None
    ) -> list[OpenSearchHit]:
        """Hybrid search combining BM25 and kNN via score combination."""
        keyword_hits = self.keyword_search(query, num_results * 2, tenant_id)
        vector_hits = self.vector_search(embedding, num_results * 2, tenant_id)

        # RRF-style fusion
        scores: dict[str, float] = {}
        sources: dict[str, OpenSearchHit] = {}
        k = 60  # RRF constant

        for rank, hit in enumerate(keyword_hits):
            key = f"{hit.document_id}_{hit.version}"
            scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)
            sources[key] = hit

        for rank, hit in enumerate(vector_hits):
            key = f"{hit.document_id}_{hit.version}"
            scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)
            sources[key] = hit

        # Sort by fused score
        sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        results = []
        for key in sorted_keys[:num_results]:
            hit = sources[key]
            results.append(OpenSearchHit(
                document_id=hit.document_id,
                version=hit.version,
                score=scores[key],
                source=hit.source,
            ))

        return results

    def _parse_hits(self, response: dict) -> list[OpenSearchHit]:
        """Parse OpenSearch response into hits."""
        hits = []
        for hit in response.get("hits", {}).get("hits", []):
            source = hit.get("_source", {})
            hits.append(OpenSearchHit(
                document_id=source.get("document_id", ""),
                version=source.get("version", 1),
                score=hit.get("_score", 0.0),
                source=source,
            ))
        return hits
