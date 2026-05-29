"""Content Retrieval API — POST /v1/contents endpoint (Task 15, R5).

Implements:
- Batch fetch 1–100 document_ids, preserving request order (R5.1).
- Per-document error handling: document_not_found for missing docs (R5.7).
- Version field on each returned document matching indexed version (R5.4).
- Validation: count bounds (R5.5), highlights without query (R5.6).
- Highlights (≤5 spans) and summaries (1–512 tokens) via Answer_Engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from backend.answer_engine.highlights import extract_highlights
from backend.answer_engine.models import HighlightSpan


# ---------------------------------------------------------------------------
# Protocols for dependency injection
# ---------------------------------------------------------------------------


class DocumentStore(Protocol):
    """Protocol for fetching documents by ID."""

    def get_latest_version(self, document_id: str) -> Any | None:
        """Get the latest version of a document by ID.

        Returns a DocumentVersion-like object with attributes:
            document_id, version, cleaned_text, source_url, provenance, visible
        Or None if not found.
        """
        ...


class SummaryGenerator(Protocol):
    """Protocol for generating document summaries."""

    def summarize(self, text: str, max_tokens: int) -> str:
        """Generate a summary of the given text within token bounds."""
        ...


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


@dataclass
class ContentError:
    """Error object for a document that could not be retrieved."""

    code: str
    message: str


@dataclass
class ProvenanceData:
    """Provenance metadata for a content result."""

    credibility_score: float
    ai_generated_likelihood: float
    scored_at: str  # ISO 8601 UTC


@dataclass
class ContentResult:
    """Successful content retrieval result for a single document."""

    document_id: str
    version: int
    url: str
    cleaned_text: str
    highlights: list[HighlightSpan] | None = None
    summary: str | None = None
    provenance: ProvenanceData | None = None


@dataclass
class ContentEntry:
    """A single entry in the contents response — either success or error."""

    document_id: str
    result: ContentResult | None = None
    error: ContentError | None = None


@dataclass
class ContentsResponse:
    """Response from POST /v1/contents."""

    results: list[ContentEntry]


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class ContentsService:
    """Service for batch content retrieval (R5).

    Fetches documents by ID, preserves request order, returns per-document
    errors for missing docs, includes version field, and supports highlights
    and summaries.
    """

    def __init__(
        self,
        *,
        document_store: DocumentStore,
        summary_generator: SummaryGenerator | None = None,
    ) -> None:
        """Initialize the ContentsService.

        Args:
            document_store: Backend for fetching documents by ID.
            summary_generator: Optional generator for document summaries.
        """
        self._document_store = document_store
        self._summary_generator = summary_generator

    def validate_request(
        self,
        document_ids: list[str],
        highlights: bool | None,
        query: str | None,
    ) -> tuple[str, str] | None:
        """Validate the contents request parameters.

        Returns:
            None if valid, or (error_code, error_message) tuple if invalid.
        """
        # R5.5: document_ids count must be 1–100
        if len(document_ids) < 1 or len(document_ids) > 100:
            return (
                "invalid_document_id_count",
                "document_ids must contain between 1 and 100 items",
            )

        # R5.6: highlights=true requires non-empty query
        if highlights is True:
            if query is None or len(query.strip()) == 0:
                return (
                    "missing_highlight_query",
                    "highlights requires a non-empty query field",
                )

        return None

    def fetch_contents(
        self,
        document_ids: list[str],
        *,
        highlights: bool | None = None,
        query: str | None = None,
        summary: bool | None = None,
    ) -> ContentsResponse:
        """Batch fetch documents by ID, preserving request order (R5.1).

        For each requested document_id:
        - If found: returns cleaned text, version, and optional highlights/summary.
        - If not found: returns error object with code 'document_not_found' (R5.7).

        Args:
            document_ids: List of 1–100 document IDs to fetch.
            highlights: Whether to include highlight spans (requires query).
            query: Query string for highlight extraction.
            summary: Whether to include document summaries.

        Returns:
            ContentsResponse with one entry per requested ID in request order.
        """
        entries: list[ContentEntry] = []

        for doc_id in document_ids:
            doc_version = self._document_store.get_latest_version(doc_id)

            if doc_version is None:
                # R5.7: document_not_found for missing docs
                entries.append(ContentEntry(
                    document_id=doc_id,
                    error=ContentError(
                        code="document_not_found",
                        message=f"Document '{doc_id}' was not found in the index",
                    ),
                ))
                continue

            # Build provenance data
            provenance_data = None
            if doc_version.provenance is not None:
                provenance_data = ProvenanceData(
                    credibility_score=doc_version.provenance.credibility_score,
                    ai_generated_likelihood=doc_version.provenance.ai_generated_likelihood,
                    scored_at=doc_version.provenance.scored_at.isoformat()
                    if hasattr(doc_version.provenance.scored_at, "isoformat")
                    else str(doc_version.provenance.scored_at),
                )

            # Extract highlights if requested (R5.2)
            highlight_spans = None
            if highlights is True and query:
                highlight_spans = extract_highlights(query, doc_version.cleaned_text)

            # Generate summary if requested (R5.3)
            doc_summary = None
            if summary is True and self._summary_generator is not None:
                doc_summary = self._summary_generator.summarize(
                    doc_version.cleaned_text, max_tokens=512
                )

            # R5.4: version field matches indexed version
            result = ContentResult(
                document_id=doc_id,
                version=doc_version.version,
                url=doc_version.source_url,
                cleaned_text=doc_version.cleaned_text,
                highlights=highlight_spans,
                summary=doc_summary,
                provenance=provenance_data,
            )

            entries.append(ContentEntry(
                document_id=doc_id,
                result=result,
            ))

        return ContentsResponse(results=entries)

    def to_response_dict(self, response: ContentsResponse) -> list[dict]:
        """Convert ContentsResponse to a list of dicts for JSON serialization."""
        results = []
        for entry in response.results:
            if entry.error is not None:
                results.append({
                    "document_id": entry.document_id,
                    "error": {
                        "code": entry.error.code,
                        "message": entry.error.message,
                    },
                })
            elif entry.result is not None:
                item: dict[str, Any] = {
                    "document_id": entry.result.document_id,
                    "version": entry.result.version,
                    "url": entry.result.url,
                    "cleaned_text": entry.result.cleaned_text,
                }
                if entry.result.highlights is not None:
                    item["highlights"] = [
                        {"start": h.start, "end": h.end}
                        for h in entry.result.highlights
                    ]
                if entry.result.summary is not None:
                    item["summary"] = entry.result.summary
                if entry.result.provenance is not None:
                    item["provenance"] = {
                        "credibility_score": entry.result.provenance.credibility_score,
                        "ai_generated_likelihood": entry.result.provenance.ai_generated_likelihood,
                        "scored_at": entry.result.provenance.scored_at,
                    }
                results.append(item)
        return results
