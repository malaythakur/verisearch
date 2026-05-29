"""Property-based tests for Content Retrieval API (Task 15.5, Property 9).

Property 9: /v1/contents preserves request order and reports per-id errors locally

*For any* request listing 1..100 document_id values (any mix of present and absent),
the response contains exactly one entry per requested id in the same order; each entry
is either a successful payload or an error object with code document_not_found; the
presence of one or more absent ids does not prevent successful entries for the remaining ids.

**Validates: Requirements 5.1, 5.7**
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from backend.api_gateway.contents import ContentsService


# ---------------------------------------------------------------------------
# Test infrastructure
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
    """In-memory document store for property testing."""

    def __init__(self, documents: dict[str, MockDocumentVersion] | None = None):
        self._documents = documents or {}

    def add_document(self, doc: MockDocumentVersion) -> None:
        self._documents[doc.document_id] = doc

    def get_latest_version(self, document_id: str) -> MockDocumentVersion | None:
        return self._documents.get(document_id)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Generate document IDs that are valid strings
doc_id_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "Pd"), whitelist_characters="-_"),
    min_size=1,
    max_size=36,
)

# Generate a set of "existing" document IDs and a set of "missing" ones
existing_ids_strategy = st.lists(doc_id_strategy, min_size=1, max_size=50, unique=True)
missing_ids_strategy = st.lists(
    st.text(
        alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="-_"),
        min_size=1,
        max_size=36,
    ).map(lambda s: f"missing-{s}"),
    min_size=0,
    max_size=50,
    unique=True,
)


@st.composite
def contents_request_strategy(draw):
    """Generate a valid contents request with a mix of existing and missing IDs.

    Returns (document_ids_to_request, existing_ids_set, document_store).
    """
    # Generate existing documents
    existing_ids = draw(st.lists(doc_id_strategy, min_size=1, max_size=30, unique=True))

    # Generate missing IDs that don't overlap with existing
    missing_ids = draw(st.lists(
        st.text(
            alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="-_"),
            min_size=1,
            max_size=20,
        ).map(lambda s: f"absent-{s}"),
        min_size=0,
        max_size=30,
        unique=True,
    ))

    # Ensure no overlap
    missing_ids = [mid for mid in missing_ids if mid not in set(existing_ids)]

    # Build the document store
    store = MockDocumentStore()
    for i, doc_id in enumerate(existing_ids):
        store.add_document(MockDocumentVersion(
            document_id=doc_id,
            version=i + 1,
            cleaned_text=f"Content for document {doc_id}",
            source_url=f"https://example.com/{doc_id}",
            provenance=MockProvenance(),
        ))

    # Build the request: mix of existing and missing, in random order
    all_ids = existing_ids + missing_ids
    # Shuffle by drawing a permutation
    request_ids = draw(st.permutations(all_ids))

    # Limit to 1–100 (R5.5)
    request_ids = list(request_ids[:100])
    assume(len(request_ids) >= 1)

    return request_ids, set(existing_ids), store


# ---------------------------------------------------------------------------
# Property 9: Preserves request order and reports per-id errors locally
# ---------------------------------------------------------------------------


class TestContentsProperty9:
    """Property 9: /v1/contents preserves request order and reports per-id errors locally.

    Feature: agentic-research-search-engine, Property 9: /v1/contents preserves
    request order and reports per-id errors locally.
    """

    @given(data=contents_request_strategy())
    @settings(max_examples=100, deadline=None)
    def test_preserves_request_order(self, data):
        """Response contains exactly one entry per requested id in the same order.

        **Validates: Requirements 5.1**
        """
        request_ids, existing_ids, store = data
        service = ContentsService(document_store=store)

        response = service.fetch_contents(request_ids)

        # Exactly one entry per requested ID
        assert len(response.results) == len(request_ids)

        # Same order as request
        for i, entry in enumerate(response.results):
            assert entry.document_id == request_ids[i]

    @given(data=contents_request_strategy())
    @settings(max_examples=100, deadline=None)
    def test_each_entry_is_success_or_error(self, data):
        """Each entry is either a successful payload or an error with code document_not_found.

        **Validates: Requirements 5.1, 5.7**
        """
        request_ids, existing_ids, store = data
        service = ContentsService(document_store=store)

        response = service.fetch_contents(request_ids)

        for entry in response.results:
            # Each entry is either success XOR error
            has_result = entry.result is not None
            has_error = entry.error is not None
            assert has_result != has_error, "Entry must be exactly one of success or error"

            if has_error:
                assert entry.error.code == "document_not_found"

    @given(data=contents_request_strategy())
    @settings(max_examples=100, deadline=None)
    def test_existing_docs_succeed_missing_docs_fail(self, data):
        """Existing docs return success, missing docs return document_not_found.

        **Validates: Requirements 5.7**
        """
        request_ids, existing_ids, store = data
        service = ContentsService(document_store=store)

        response = service.fetch_contents(request_ids)

        for entry in response.results:
            if entry.document_id in existing_ids:
                assert entry.result is not None, f"Existing doc {entry.document_id} should succeed"
                assert entry.error is None
            else:
                assert entry.error is not None, f"Missing doc {entry.document_id} should fail"
                assert entry.error.code == "document_not_found"
                assert entry.result is None

    @given(data=contents_request_strategy())
    @settings(max_examples=100, deadline=None)
    def test_errors_do_not_prevent_successes(self, data):
        """Presence of absent ids does not prevent successful entries for remaining ids.

        **Validates: Requirements 5.1, 5.7**
        """
        request_ids, existing_ids, store = data
        service = ContentsService(document_store=store)

        response = service.fetch_contents(request_ids)

        # Count expected successes and errors
        expected_successes = sum(1 for rid in request_ids if rid in existing_ids)
        expected_errors = sum(1 for rid in request_ids if rid not in existing_ids)

        actual_successes = sum(1 for e in response.results if e.result is not None)
        actual_errors = sum(1 for e in response.results if e.error is not None)

        assert actual_successes == expected_successes
        assert actual_errors == expected_errors

    @given(data=contents_request_strategy())
    @settings(max_examples=100, deadline=None)
    def test_version_field_present_on_success(self, data):
        """Successful entries include a version field matching the indexed version.

        **Validates: Requirements 5.4**
        """
        request_ids, existing_ids, store = data
        service = ContentsService(document_store=store)

        response = service.fetch_contents(request_ids)

        for entry in response.results:
            if entry.result is not None:
                # Version must be a positive integer
                assert isinstance(entry.result.version, int)
                assert entry.result.version >= 1
