"""Indexer - Document ingestion, versioning, content hashing, and index management.

Exports:
- clean_html: HTML → cleaned text pipeline
- compute_content_hash: SHA-256 content hashing
- IndexerService: Main indexing orchestrator
- DeadLetterQueue: DLQ routing for failed operations
- PriorityScheduler: Priority-source re-crawl scheduling
- VectorIndex: Vector embedding index (stub)
- LexicalIndex: Lexical/BM25 index (stub)
"""

from backend.indexer.cleaner import clean_html
from backend.indexer.dlq import DeadLetterQueue, DLQEntry, RetryState
from backend.indexer.embeddings import VectorIndex, generate_embedding
from backend.indexer.hasher import compute_content_hash
from backend.indexer.lexical import LexicalIndex, analyze_text
from backend.indexer.scheduler import PriorityScheduler, PrioritySource
from backend.indexer.service import DocumentVersion, IndexerService, IndexResult

__all__ = [
    "IndexerService",
    "IndexResult",
    "DocumentVersion",
    "clean_html",
    "compute_content_hash",
    "DeadLetterQueue",
    "DLQEntry",
    "RetryState",
    "PriorityScheduler",
    "PrioritySource",
    "VectorIndex",
    "generate_embedding",
    "LexicalIndex",
    "analyze_text",
]
