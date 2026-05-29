"""Property-based tests for Indexer and Provenance Scorer (Tasks 10.12–10.15).

Uses Hypothesis to verify universal properties across all inputs.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from backend.indexer.cleaner import clean_html
from backend.indexer.hasher import compute_content_hash
from backend.indexer.service import IndexerService
from backend.provenance_scorer.scorer import ProvenanceScorer


# ============================================================
# Strategies
# ============================================================

# Strategy for generating HTML-like content
html_content = st.text(
    alphabet=st.characters(categories=("L", "N", "P", "Z", "S")),
    min_size=1,
    max_size=500,
).map(lambda t: f"<p>{t}</p>")

# Strategy for generating varied HTML content that produces non-empty cleaned text
non_empty_html = st.text(
    alphabet=st.characters(categories=("L", "N")),
    min_size=1,
    max_size=200,
).map(lambda t: f"<div>{t}</div>")

# Strategy for source URLs
source_urls = st.text(
    alphabet=st.characters(categories=("L", "N")),
    min_size=5,
    max_size=50,
).map(lambda t: f"https://example.com/{t}")

# Strategy for document text (for provenance scoring)
document_text = st.text(
    alphabet=st.characters(categories=("L", "N", "P", "Z")),
    min_size=1,
    max_size=1000,
)


# ============================================================
# Task 10.12: Property test — re-index identical content is idempotent (Property 1)
# ============================================================


class TestIdempotentReindexProperty:
    """**Validates: Requirements 2.4**

    Property 1: Re-indexing identical content is idempotent.
    When the same content is indexed twice for the same URL:
    - document_id remains unchanged
    - version remains unchanged
    - Only last_seen_at is updated
    """

    @given(content=non_empty_html, url=source_urls)
    @settings(max_examples=100, deadline=None)
    @pytest.mark.asyncio
    async def test_reindex_identical_content_is_idempotent(self, content: str, url: str):
        """Re-indexing the same content for the same URL must be idempotent.

        The document_id and version must not change; only last_seen_at updates.
        """
        indexer = IndexerService()

        # First index
        r1 = await indexer.index_document(content, url)

        # Second index with identical content
        r2 = await indexer.index_document(content, url)

        # Idempotency assertions (R2.4)
        assert r2.document_id == r1.document_id, "document_id must be stable"
        assert r2.version == r1.version, "version must not change on same hash"
        assert r2.last_seen_only is True, "should only update last_seen_at"
        assert r2.version_incremented is False, "version must not increment"

        # Verify only one version exists
        versions = indexer.get_document(r1.document_id)
        assert len(versions) == 1, "should have exactly one version"


# ============================================================
# Task 10.13: Property test — re-index changed content increments version by exactly 1 (Property 2)
# ============================================================


class TestVersionIncrementProperty:
    """**Validates: Requirements 2.3**

    Property 2: Re-indexing changed content increments version by exactly 1.
    When content changes for the same URL:
    - document_id is preserved
    - version increments by exactly 1
    """

    @given(
        content1=non_empty_html,
        content2=non_empty_html,
        url=source_urls,
    )
    @settings(max_examples=100, deadline=None)
    @pytest.mark.asyncio
    async def test_changed_content_increments_version_by_one(
        self, content1: str, content2: str, url: str
    ):
        """Changed content must increment version by exactly 1, preserving document_id."""
        # Ensure content actually differs after cleaning
        cleaned1 = clean_html(content1)
        cleaned2 = clean_html(content2)
        assume(compute_content_hash(cleaned1) != compute_content_hash(cleaned2))

        indexer = IndexerService()

        # First index
        r1 = await indexer.index_document(content1, url)
        assert r1.version == 1

        # Second index with different content
        r2 = await indexer.index_document(content2, url)

        # Version increment assertions (R2.3)
        assert r2.document_id == r1.document_id, "document_id must be preserved"
        assert r2.version == r1.version + 1, "version must increment by exactly 1"
        assert r2.version_incremented is True


# ============================================================
# Task 10.14: Property test — provenance scores in [0.0, 1.0] (Property 20 range)
# ============================================================


class TestProvenanceScoreRangeProperty:
    """**Validates: Requirements 10.1**

    Property 20 (range): Provenance scores must always be in [0.0, 1.0].
    For any document text, both credibility_score and ai_generated_likelihood
    must be within the closed interval [0.0, 1.0].
    """

    @given(text=document_text)
    @settings(max_examples=200, deadline=None)
    def test_provenance_scores_in_valid_range(self, text: str):
        """Both provenance scores must be in [0.0, 1.0] for any input text."""
        scorer = ProvenanceScorer()
        score = scorer.score(text)

        assert 0.0 <= score.credibility_score <= 1.0, (
            f"credibility_score {score.credibility_score} out of [0.0, 1.0]"
        )
        assert 0.0 <= score.ai_generated_likelihood <= 1.0, (
            f"ai_generated_likelihood {score.ai_generated_likelihood} out of [0.0, 1.0]"
        )


# ============================================================
# Task 10.15: Property test — rescore preserves document_id/version (Property 20 frame)
# ============================================================


class TestRescoreFrameProperty:
    """**Validates: Requirements 10.6**

    Property 20 (frame): Rescore preserves document_id/version, mutates only score fields.
    When a document is rescored:
    - document_id is unchanged
    - version is unchanged
    - content_hash is unchanged
    - Only credibility_score, ai_generated_likelihood, and scored_at may change
    """

    @given(
        doc_id=st.uuids().map(str),
        version=st.integers(min_value=1, max_value=100),
        content_hash=st.text(alphabet="0123456789abcdef", min_size=64, max_size=64),
        original_text=document_text,
        rescore_text=document_text,
    )
    @settings(max_examples=100, deadline=None)
    def test_rescore_preserves_identity_fields(
        self,
        doc_id: str,
        version: int,
        content_hash: str,
        original_text: str,
        rescore_text: str,
    ):
        """Rescore must preserve document_id, version, and content_hash."""
        scorer = ProvenanceScorer()

        # Initial scoring
        scorer.score_document(doc_id, version, content_hash, original_text)

        # Capture state before rescore
        doc_before = scorer.scored_documents[(doc_id, version)]
        assert doc_before.document_id == doc_id
        assert doc_before.version == version
        assert doc_before.content_hash == content_hash

        # Rescore
        new_score = scorer.rescore(doc_id, version, rescore_text)

        # Verify identity fields are preserved (R10.6)
        doc_after = scorer.scored_documents[(doc_id, version)]
        assert doc_after.document_id == doc_id, "document_id must not change on rescore"
        assert doc_after.version == version, "version must not change on rescore"
        assert doc_after.content_hash == content_hash, "content_hash must not change on rescore"
        assert doc_after.visible is True, "visibility must remain True after rescore"

        # Verify score fields are valid
        assert 0.0 <= new_score.credibility_score <= 1.0
        assert 0.0 <= new_score.ai_generated_likelihood <= 1.0
        assert new_score.scored_at is not None
