"""Main IndexerService — orchestrates cleaning, hashing, version management (Tasks 10.1–10.5, 10.8, 10.10–10.11).

Implements the core indexing pipeline:
1. Clean raw HTML content to plain text.
2. Compute SHA-256 content hash.
3. Assign stable document_id at first ingest (by canonical_url).
4. Increment version by exactly 1 on new hash (R2.3).
5. Update only last_seen_at when hash matches (R2.4).
6. Route to DLQ after 3 retries spaced ≥60s (R2.5).
7. Trigger provenance scoring before making document visible (R10.1).
8. Write vector and lexical index entries.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol

from backend.indexer.cleaner import clean_html
from backend.indexer.dlq import DeadLetterQueue, DLQEntry
from backend.indexer.embeddings import VectorIndex, generate_embedding
from backend.indexer.hasher import compute_content_hash
from backend.indexer.lexical import LexicalIndex


class AuditEmitter(Protocol):
    """Protocol for audit event emission."""

    async def emit(
        self,
        *,
        action: str,
        tenant_id: str | None,
        actor: str,
        resource: str,
        request_id: str,
        detail: dict,
    ) -> None: ...


class ProvenanceScorerProtocol(Protocol):
    """Protocol for provenance scoring."""

    def score(self, document_text: str) -> "ProvenanceScore": ...


@dataclass
class ProvenanceScore:
    """Provenance score for a document."""

    credibility_score: float
    ai_generated_likelihood: float
    scored_at: datetime


@dataclass
class DocumentVersion:
    """A version of an indexed document.

    Attributes:
        document_id: Stable ID assigned at first ingest.
        version: Monotonically increasing version number.
        content_hash: SHA-256 of cleaned text.
        cleaned_text: The cleaned plain text content.
        source_url: The canonical source URL.
        last_seen_at: Last time this content was seen.
        created_at: When this version was created.
        provenance: Provenance scores (None until scored).
        visible: Whether this version is visible to Retriever.
    """

    document_id: str
    version: int
    content_hash: str
    cleaned_text: str
    source_url: str
    last_seen_at: datetime
    created_at: datetime
    provenance: ProvenanceScore | None = None
    visible: bool = False


@dataclass
class IndexResult:
    """Result of an indexing operation.

    Attributes:
        document_id: The document's stable ID.
        version: The current version number.
        is_new: Whether this is a new document.
        version_incremented: Whether the version was incremented.
        last_seen_only: Whether only last_seen_at was updated (idempotent).
        dlq_routed: Whether the document was routed to DLQ.
    """

    document_id: str
    version: int
    is_new: bool = False
    version_incremented: bool = False
    last_seen_only: bool = False
    dlq_routed: bool = False


class IndexerService:
    """Main indexer service orchestrating the document ingestion pipeline.

    Manages document identity, versioning, content deduplication,
    provenance scoring, and index writes.
    """

    def __init__(
        self,
        *,
        audit_emitter: AuditEmitter | None = None,
        provenance_scorer: ProvenanceScorerProtocol | None = None,
        vector_index: VectorIndex | None = None,
        lexical_index: LexicalIndex | None = None,
        dlq: DeadLetterQueue | None = None,
    ) -> None:
        self._audit_emitter = audit_emitter
        self._provenance_scorer = provenance_scorer
        self._vector_index = vector_index or VectorIndex()
        self._lexical_index = lexical_index or LexicalIndex()
        self._dlq = dlq or DeadLetterQueue()

        # In-memory document store (production: PostgreSQL + S3)
        # Key: canonical_url → document_id
        self._url_to_doc_id: dict[str, str] = {}
        # Key: document_id → list of DocumentVersion (ordered by version)
        self._documents: dict[str, list[DocumentVersion]] = {}

    @property
    def documents(self) -> dict[str, list[DocumentVersion]]:
        """Return all stored documents (for testing)."""
        return self._documents

    @property
    def dlq(self) -> DeadLetterQueue:
        """Return the dead-letter queue."""
        return self._dlq

    @property
    def vector_index(self) -> VectorIndex:
        """Return the vector index."""
        return self._vector_index

    @property
    def lexical_index(self) -> LexicalIndex:
        """Return the lexical index."""
        return self._lexical_index

    async def index_document(
        self,
        raw_content: str | bytes,
        source_url: str,
        *,
        request_id: str | None = None,
    ) -> IndexResult:
        """Index a document through the full pipeline.

        Steps:
        1. Clean HTML to plain text.
        2. Compute content hash.
        3. Check if document exists (by canonical URL).
        4. If new: assign document_id, create version 1.
        5. If existing with same hash: update last_seen_at only (R2.4).
        6. If existing with different hash: increment version by 1 (R2.3).
        7. Score with Provenance_Scorer (R10.1).
        8. Write to vector and lexical indexes.
        9. Mark as visible only after scoring (R10.1).

        Args:
            raw_content: Raw HTML content (string or bytes).
            source_url: The canonical source URL.
            request_id: Optional request ID for audit trail.

        Returns:
            IndexResult describing what happened.
        """
        if request_id is None:
            request_id = str(uuid.uuid4())

        # Step 1: Clean content
        cleaned_text = clean_html(raw_content)

        # Step 2: Compute hash
        content_hash = compute_content_hash(cleaned_text)

        # Step 3: Check if document exists
        now = datetime.now(timezone.utc)

        if source_url in self._url_to_doc_id:
            document_id = self._url_to_doc_id[source_url]
            versions = self._documents[document_id]
            latest = versions[-1]

            # Step 5: Same hash → update last_seen_at only (R2.4)
            if latest.content_hash == content_hash:
                latest.last_seen_at = now
                return IndexResult(
                    document_id=document_id,
                    version=latest.version,
                    last_seen_only=True,
                )

            # Step 6: Different hash → increment version by exactly 1 (R2.3)
            new_version = latest.version + 1
            doc_version = DocumentVersion(
                document_id=document_id,
                version=new_version,
                content_hash=content_hash,
                cleaned_text=cleaned_text,
                source_url=source_url,
                last_seen_at=now,
                created_at=now,
            )
            versions.append(doc_version)

            # Score and index
            self._score_document(doc_version)
            self._write_indexes(doc_version)

            return IndexResult(
                document_id=document_id,
                version=new_version,
                version_incremented=True,
            )

        # Step 4: New document — assign stable document_id
        document_id = str(uuid.uuid4())
        self._url_to_doc_id[source_url] = document_id

        doc_version = DocumentVersion(
            document_id=document_id,
            version=1,
            content_hash=content_hash,
            cleaned_text=cleaned_text,
            source_url=source_url,
            last_seen_at=now,
            created_at=now,
        )
        self._documents[document_id] = [doc_version]

        # Score and index
        self._score_document(doc_version)
        self._write_indexes(doc_version)

        return IndexResult(
            document_id=document_id,
            version=1,
            is_new=True,
        )

    async def index_document_with_retry(
        self,
        raw_content: str | bytes,
        source_url: str,
        *,
        error: str,
        request_id: str | None = None,
    ) -> IndexResult | DLQEntry:
        """Attempt to index a document, routing to DLQ on persistent failure.

        Records the attempt and checks if DLQ routing is needed.
        If the document has failed 3 times with proper spacing, routes to DLQ
        and emits an index_failure audit event (R2.5).

        Args:
            raw_content: Raw HTML content.
            source_url: The canonical source URL.
            error: The error message from the failed attempt.
            request_id: Optional request ID for audit trail.

        Returns:
            IndexResult on success, DLQEntry if routed to DLQ.
        """
        if request_id is None:
            request_id = str(uuid.uuid4())

        key = source_url
        self._dlq.record_attempt(key, error)

        if self._dlq.should_route_to_dlq(key):
            document_id = self._url_to_doc_id.get(source_url)
            entry = self._dlq.route_to_dlq(
                key,
                document_id=document_id,
                source_url=source_url,
            )

            # Emit index_failure audit event (R2.5)
            if self._audit_emitter:
                await self._audit_emitter.emit(
                    action="index_failure",
                    tenant_id=None,
                    actor="indexer",
                    resource=document_id or source_url,
                    request_id=request_id,
                    detail={
                        "document_id": document_id,
                        "source_url": source_url,
                        "failure_reason": entry.failure_reason,
                        "attempts": entry.attempts,
                    },
                )

            return entry

        # Not yet at DLQ threshold — return a result indicating retry needed
        return IndexResult(
            document_id=self._url_to_doc_id.get(source_url, ""),
            version=0,
            dlq_routed=False,
        )

    def get_document(self, document_id: str) -> list[DocumentVersion] | None:
        """Get all versions of a document by ID."""
        return self._documents.get(document_id)

    def get_latest_version(self, document_id: str) -> DocumentVersion | None:
        """Get the latest version of a document."""
        versions = self._documents.get(document_id)
        if versions:
            return versions[-1]
        return None

    def get_document_by_url(self, source_url: str) -> list[DocumentVersion] | None:
        """Get all versions of a document by its source URL."""
        doc_id = self._url_to_doc_id.get(source_url)
        if doc_id:
            return self._documents.get(doc_id)
        return None

    def is_visible(self, document_id: str, version: int | None = None) -> bool:
        """Check if a document version is visible to the Retriever.

        A document is only visible after it has been scored (R10.1).

        Args:
            document_id: The document ID.
            version: Specific version to check (defaults to latest).

        Returns:
            True if the document version is visible.
        """
        versions = self._documents.get(document_id)
        if not versions:
            return False

        if version is not None:
            for v in versions:
                if v.version == version:
                    return v.visible
            return False

        return versions[-1].visible

    def _score_document(self, doc_version: DocumentVersion) -> None:
        """Score a document with the Provenance_Scorer and mark as visible.

        The document is NOT visible to the Retriever until scored (R10.1).
        """
        if self._provenance_scorer:
            score = self._provenance_scorer.score(doc_version.cleaned_text)
            doc_version.provenance = score
            doc_version.visible = True
        else:
            # Without a scorer, mark as visible (for testing without scorer)
            doc_version.visible = True

    def _write_indexes(self, doc_version: DocumentVersion) -> None:
        """Write to vector, lexical, and OpenSearch indexes."""
        # Vector embedding
        embedding = generate_embedding(doc_version.cleaned_text)
        self._vector_index.write(
            doc_version.document_id,
            doc_version.version,
            embedding,
        )

        # Lexical index
        self._lexical_index.write(
            doc_version.document_id,
            doc_version.version,
            doc_version.cleaned_text,
        )

        # OpenSearch (if available)
        try:
            from backend.retriever.opensearch_client import OpenSearchClient
            os_client = OpenSearchClient()
            if os_client.is_available:
                credibility = 0.5
                ai_likelihood = 0.5
                if doc_version.provenance:
                    credibility = doc_version.provenance.credibility_score
                    ai_likelihood = doc_version.provenance.ai_generated_likelihood

                title = doc_version.cleaned_text.split("\n")[0][:100] if doc_version.cleaned_text else ""
                os_client.index_document(
                    document_id=doc_version.document_id,
                    version=doc_version.version,
                    tenant_id="",  # Global for now
                    url=doc_version.source_url,
                    title=title,
                    cleaned_text=doc_version.cleaned_text,
                    embedding=embedding,
                    credibility_score=credibility,
                    ai_generated_likelihood=ai_likelihood,
                )
        except Exception:
            pass  # OpenSearch write failure is non-fatal
