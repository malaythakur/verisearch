"""Unit tests for the Content Retrieval API (Task 15, R5).

Tests cover:
- Batch fetch 1–100 document_ids, preserving request order (R5.1).
- Per-document error handling: document_not_found for missing docs (R5.7).
- Version field on each returned document matching indexed version (R5.4).
- Validation: count bounds (R5.5), highlights without query (R5.6).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pytest

from backend.api_gateway.contents import (
    ContentEntry,
    ContentError,
    ContentResult,
    ContentsResponse,
    ContentsService,
)


# ---------------------------------------------------------------------------
# Mock document store
# ---------------------------------------------------------------------------


@dataclass
class MockProvenance:
    credibility_score: float = 0.8
    ai_generated_likelihood: float = 0.2
    scored_at: datetime = datetime(2024, 1, 1, tzinfo=timezone.utc)


@dataclass
class MockDocumentVersion:
    document_id: str
    version: int
    cleaned_text: str
    source_url: str
    provenance: MockProvenance | None = None
    visible: bool = True


class MockDocumentStore:
    """In-memory document store for testing."""

    def __init__(self, documents: dict[str, MockDocumentVersion] | None = None):
        self._documents = documents or {}

    def add_document(self, doc: MockDocumentVersion) -> None:
        self._documents[doc.document_id] = doc

    def get_latest_version(self, document_id: str) -> MockDocumentVersion | None:
        return self._documents.get(document_id)


class MockSummaryGenerator:
    """Mock summary generator for testing."""

    def summarize(self, text: str, max_tokens: int = 512) -> str:
        # Return first 50 chars as a mock summary
        return text[:50] if len(text) > 50 else text


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def document_store():
    """Create a mock document store with sample documents."""
    store = MockDocumentStore()
    store.add_document(MockDocumentVersion(
        document_id="doc-1",
        version=3,
        cleaned_text="Machine learning algorithms for natural language processing are widely used.",
        source_url="https://example.com/ml-nlp",
        provenance=MockProvenance(),
    ))
    store.add_document(MockDocumentVersion(
        document_id="doc-2",
        version=1,
        cleaned_text="Deep learning neural networks and transformers architecture.",
        source_url="https://example.com/deep-learning",
        provenance=MockProvenance(credibility_score=0.9, ai_generated_likelihood=0.1),
    ))
    store.add_document(MockDocumentVersion(
        document_id="doc-3",
        version=5,
        cleaned_text="Python programming best practices and design patterns for software engineers.",
        source_url="https://example.com/python",
        provenance=MockProvenance(credibility_score=0.7, ai_generated_likelihood=0.3),
    ))
    return store


@pytest.fixture
def contents_service(document_store):
    """Create a ContentsService with mock dependencies."""
    return ContentsService(
        document_store=document_store,
        summary_generator=MockSummaryGenerator(),
    )


# ---------------------------------------------------------------------------
# Task 15.1: Batch fetch preserving request order
# ---------------------------------------------------------------------------


class TestBatchFetchOrder:
    """Tests for R5.1: batch fetch 1–100 document_ids, preserve request order."""

    def test_single_document_fetch(self, contents_service):
        """Fetching a single document returns one result."""
        response = contents_service.fetch_contents(["doc-1"])
        assert len(response.results) == 1
        assert response.results[0].document_id == "doc-1"
        assert response.results[0].result is not None
        assert response.results[0].error is None

    def test_preserves_request_order(self, contents_service):
        """Results are returned in the same order as requested."""
        response = contents_service.fetch_contents(["doc-3", "doc-1", "doc-2"])
        assert len(response.results) == 3
        assert response.results[0].document_id == "doc-3"
        assert response.results[1].document_id == "doc-1"
        assert response.results[2].document_id == "doc-2"

    def test_preserves_order_with_missing_docs(self, contents_service):
        """Order is preserved even when some documents are missing."""
        response = contents_service.fetch_contents(["doc-2", "missing-1", "doc-1", "missing-2"])
        assert len(response.results) == 4
        assert response.results[0].document_id == "doc-2"
        assert response.results[0].result is not None
        assert response.results[1].document_id == "missing-1"
        assert response.results[1].error is not None
        assert response.results[2].document_id == "doc-1"
        assert response.results[2].result is not None
        assert response.results[3].document_id == "missing-2"
        assert response.results[3].error is not None

    def test_duplicate_ids_returned_in_order(self, contents_service):
        """Duplicate IDs are each returned as separate entries."""
        response = contents_service.fetch_contents(["doc-1", "doc-1", "doc-2"])
        assert len(response.results) == 3
        assert response.results[0].document_id == "doc-1"
        assert response.results[1].document_id == "doc-1"
        assert response.results[2].document_id == "doc-2"


# ---------------------------------------------------------------------------
# Task 15.2: Per-document error handling
# ---------------------------------------------------------------------------


class TestPerDocumentErrors:
    """Tests for R5.7: document_not_found for missing docs, success for rest."""

    def test_missing_document_returns_error(self, contents_service):
        """Missing document returns error with code 'document_not_found'."""
        response = contents_service.fetch_contents(["nonexistent-doc"])
        assert len(response.results) == 1
        entry = response.results[0]
        assert entry.error is not None
        assert entry.error.code == "document_not_found"
        assert entry.result is None

    def test_mix_of_found_and_missing(self, contents_service):
        """Mix of found and missing docs returns success for found, error for missing."""
        response = contents_service.fetch_contents(["doc-1", "missing", "doc-2"])
        assert len(response.results) == 3

        # doc-1: success
        assert response.results[0].result is not None
        assert response.results[0].error is None

        # missing: error
        assert response.results[1].result is None
        assert response.results[1].error is not None
        assert response.results[1].error.code == "document_not_found"

        # doc-2: success
        assert response.results[2].result is not None
        assert response.results[2].error is None

    def test_all_missing_returns_all_errors(self, contents_service):
        """All missing documents returns all errors."""
        response = contents_service.fetch_contents(["missing-1", "missing-2", "missing-3"])
        assert len(response.results) == 3
        for entry in response.results:
            assert entry.error is not None
            assert entry.error.code == "document_not_found"
            assert entry.result is None

    def test_error_does_not_prevent_other_successes(self, contents_service):
        """Presence of errors does not prevent successful entries."""
        response = contents_service.fetch_contents(["missing", "doc-1", "doc-2", "doc-3"])
        successes = [e for e in response.results if e.result is not None]
        errors = [e for e in response.results if e.error is not None]
        assert len(successes) == 3
        assert len(errors) == 1


# ---------------------------------------------------------------------------
# Task 15.3: Version field
# ---------------------------------------------------------------------------


class TestVersionField:
    """Tests for R5.4: version field matches indexed version."""

    def test_version_matches_indexed_version(self, contents_service):
        """Version field matches the document's indexed version."""
        response = contents_service.fetch_contents(["doc-1"])
        result = response.results[0].result
        assert result is not None
        assert result.version == 3  # doc-1 has version 3

    def test_different_versions_for_different_docs(self, contents_service):
        """Each document returns its own version."""
        response = contents_service.fetch_contents(["doc-1", "doc-2", "doc-3"])
        assert response.results[0].result.version == 3
        assert response.results[1].result.version == 1
        assert response.results[2].result.version == 5


# ---------------------------------------------------------------------------
# Task 15.4: Validation
# ---------------------------------------------------------------------------


class TestValidation:
    """Tests for R5.5 (count bounds) and R5.6 (highlights without query)."""

    def test_empty_document_ids_rejected(self, contents_service):
        """Empty document_ids list is rejected."""
        error = contents_service.validate_request([], None, None)
        assert error is not None
        assert error[0] == "invalid_document_id_count"

    def test_over_100_document_ids_rejected(self, contents_service):
        """More than 100 document_ids is rejected."""
        ids = [f"doc-{i}" for i in range(101)]
        error = contents_service.validate_request(ids, None, None)
        assert error is not None
        assert error[0] == "invalid_document_id_count"

    def test_exactly_100_document_ids_accepted(self, contents_service):
        """Exactly 100 document_ids is accepted."""
        ids = [f"doc-{i}" for i in range(100)]
        error = contents_service.validate_request(ids, None, None)
        assert error is None

    def test_exactly_1_document_id_accepted(self, contents_service):
        """Exactly 1 document_id is accepted."""
        error = contents_service.validate_request(["doc-1"], None, None)
        assert error is None

    def test_highlights_without_query_rejected(self, contents_service):
        """highlights=true without query is rejected."""
        error = contents_service.validate_request(["doc-1"], True, None)
        assert error is not None
        assert error[0] == "missing_highlight_query"

    def test_highlights_with_empty_query_rejected(self, contents_service):
        """highlights=true with empty query is rejected."""
        error = contents_service.validate_request(["doc-1"], True, "   ")
        assert error is not None
        assert error[0] == "missing_highlight_query"

    def test_highlights_with_valid_query_accepted(self, contents_service):
        """highlights=true with non-empty query is accepted."""
        error = contents_service.validate_request(["doc-1"], True, "machine learning")
        assert error is None

    def test_no_highlights_without_query_accepted(self, contents_service):
        """highlights=false or None without query is accepted."""
        error = contents_service.validate_request(["doc-1"], False, None)
        assert error is None
        error = contents_service.validate_request(["doc-1"], None, None)
        assert error is None


# ---------------------------------------------------------------------------
# Additional integration-style tests
# ---------------------------------------------------------------------------


class TestContentsIntegration:
    """Integration tests for the full contents flow."""

    def test_highlights_returned_when_requested(self, contents_service):
        """Highlights are returned when highlights=true and query is provided."""
        response = contents_service.fetch_contents(
            ["doc-1"],
            highlights=True,
            query="machine learning",
        )
        result = response.results[0].result
        assert result is not None
        assert result.highlights is not None
        # Highlights should be valid spans
        for span in result.highlights:
            assert span.start >= 0
            assert span.end > span.start
            assert span.end <= len(result.cleaned_text)

    def test_summary_returned_when_requested(self, contents_service):
        """Summary is returned when summary=true."""
        response = contents_service.fetch_contents(
            ["doc-1"],
            summary=True,
        )
        result = response.results[0].result
        assert result is not None
        assert result.summary is not None
        assert len(result.summary) > 0

    def test_no_highlights_when_not_requested(self, contents_service):
        """Highlights are None when not requested."""
        response = contents_service.fetch_contents(["doc-1"])
        result = response.results[0].result
        assert result is not None
        assert result.highlights is None

    def test_no_summary_when_not_requested(self, contents_service):
        """Summary is None when not requested."""
        response = contents_service.fetch_contents(["doc-1"])
        result = response.results[0].result
        assert result is not None
        assert result.summary is None

    def test_to_response_dict_success(self, contents_service):
        """to_response_dict serializes successful results correctly."""
        response = contents_service.fetch_contents(["doc-1"])
        dicts = contents_service.to_response_dict(response)
        assert len(dicts) == 1
        assert dicts[0]["document_id"] == "doc-1"
        assert dicts[0]["version"] == 3
        assert "cleaned_text" in dicts[0]
        assert "url" in dicts[0]

    def test_to_response_dict_error(self, contents_service):
        """to_response_dict serializes error results correctly."""
        response = contents_service.fetch_contents(["missing"])
        dicts = contents_service.to_response_dict(response)
        assert len(dicts) == 1
        assert dicts[0]["document_id"] == "missing"
        assert dicts[0]["error"]["code"] == "document_not_found"

    def test_provenance_included_in_result(self, contents_service):
        """Provenance data is included in successful results."""
        response = contents_service.fetch_contents(["doc-1"])
        result = response.results[0].result
        assert result is not None
        assert result.provenance is not None
        assert result.provenance.credibility_score == 0.8
        assert result.provenance.ai_generated_likelihood == 0.2
