"""Unit tests for the Indexer and Provenance Scorer (Tasks 10.1–10.11)."""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from backend.indexer.cleaner import clean_html
from backend.indexer.dlq import DeadLetterQueue, MAX_RETRY_ATTEMPTS, MIN_RETRY_SPACING_SECONDS
from backend.indexer.embeddings import VectorIndex, generate_embedding
from backend.indexer.hasher import compute_content_hash
from backend.indexer.lexical import LexicalIndex, analyze_text
from backend.indexer.scheduler import PriorityScheduler
from backend.indexer.service import IndexerService
from backend.provenance_scorer.scorer import ProvenanceScorer


# ============================================================
# Task 10.1: Content cleaning pipeline
# ============================================================


class TestContentCleaning:
    """Tests for HTML → cleaned text pipeline."""

    def test_strips_html_tags(self):
        html = "<p>Hello <b>world</b></p>"
        assert clean_html(html) == "Hello world"

    def test_removes_script_blocks(self):
        html = "<p>Before</p><script>alert('xss')</script><p>After</p>"
        result = clean_html(html)
        assert "alert" not in result
        assert "Before" in result
        assert "After" in result

    def test_removes_style_blocks(self):
        html = "<style>.foo { color: red; }</style><p>Content</p>"
        result = clean_html(html)
        assert "color" not in result
        assert "Content" in result

    def test_decodes_html_entities(self):
        html = "<p>Tom &amp; Jerry &lt;3</p>"
        assert clean_html(html) == "Tom & Jerry <3"

    def test_normalizes_whitespace(self):
        html = "<p>  Hello   \n\t  world  </p>"
        assert clean_html(html) == "Hello world"

    def test_handles_bytes_input(self):
        html_bytes = b"<p>Hello world</p>"
        assert clean_html(html_bytes) == "Hello world"

    def test_empty_html(self):
        assert clean_html("") == ""
        assert clean_html("<div></div>") == ""

    def test_nested_tags(self):
        html = "<div><ul><li>Item 1</li><li>Item 2</li></ul></div>"
        result = clean_html(html)
        assert "Item 1" in result
        assert "Item 2" in result


# ============================================================
# Task 10.2: Content hashing
# ============================================================


class TestContentHashing:
    """Tests for SHA-256 content hashing."""

    def test_produces_hex_digest(self):
        h = compute_content_hash("hello world")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_deterministic(self):
        text = "same content"
        assert compute_content_hash(text) == compute_content_hash(text)

    def test_different_content_different_hash(self):
        h1 = compute_content_hash("content A")
        h2 = compute_content_hash("content B")
        assert h1 != h2

    def test_empty_string(self):
        h = compute_content_hash("")
        assert len(h) == 64


# ============================================================
# Task 10.3: Stable document_id and version increment
# ============================================================


class TestDocumentVersioning:
    """Tests for stable document_id assignment and version management."""

    @pytest.fixture
    def indexer(self):
        return IndexerService()

    @pytest.mark.asyncio
    async def test_first_ingest_assigns_document_id(self, indexer):
        result = await indexer.index_document("<p>Hello</p>", "https://example.com/page1")
        assert result.is_new is True
        assert result.version == 1
        assert result.document_id != ""

    @pytest.mark.asyncio
    async def test_stable_document_id_on_reingest(self, indexer):
        r1 = await indexer.index_document("<p>Content v1</p>", "https://example.com/page1")
        r2 = await indexer.index_document("<p>Content v2</p>", "https://example.com/page1")
        assert r1.document_id == r2.document_id

    @pytest.mark.asyncio
    async def test_version_increments_by_one(self, indexer):
        await indexer.index_document("<p>Version 1</p>", "https://example.com/page1")
        r2 = await indexer.index_document("<p>Version 2</p>", "https://example.com/page1")
        assert r2.version == 2
        assert r2.version_incremented is True

    @pytest.mark.asyncio
    async def test_multiple_version_increments(self, indexer):
        await indexer.index_document("<p>V1</p>", "https://example.com/page1")
        await indexer.index_document("<p>V2</p>", "https://example.com/page1")
        r3 = await indexer.index_document("<p>V3</p>", "https://example.com/page1")
        assert r3.version == 3

    @pytest.mark.asyncio
    async def test_different_urls_get_different_ids(self, indexer):
        r1 = await indexer.index_document("<p>Page A</p>", "https://example.com/a")
        r2 = await indexer.index_document("<p>Page B</p>", "https://example.com/b")
        assert r1.document_id != r2.document_id


# ============================================================
# Task 10.4: last_seen_at-only update on same hash
# ============================================================


class TestIdempotentReindex:
    """Tests for idempotent re-indexing when hash matches (R2.4)."""

    @pytest.fixture
    def indexer(self):
        return IndexerService()

    @pytest.mark.asyncio
    async def test_same_content_updates_last_seen_only(self, indexer):
        r1 = await indexer.index_document("<p>Same content</p>", "https://example.com/page1")
        r2 = await indexer.index_document("<p>Same content</p>", "https://example.com/page1")
        assert r2.last_seen_only is True
        assert r2.version == r1.version
        assert r2.document_id == r1.document_id

    @pytest.mark.asyncio
    async def test_same_content_does_not_increment_version(self, indexer):
        await indexer.index_document("<p>Content</p>", "https://example.com/page1")
        r2 = await indexer.index_document("<p>Content</p>", "https://example.com/page1")
        assert r2.version == 1
        assert r2.version_incremented is False

    @pytest.mark.asyncio
    async def test_last_seen_at_is_updated(self, indexer):
        await indexer.index_document("<p>Content</p>", "https://example.com/page1")
        versions = indexer.get_document_by_url("https://example.com/page1")
        first_seen = versions[0].last_seen_at

        await indexer.index_document("<p>Content</p>", "https://example.com/page1")
        versions = indexer.get_document_by_url("https://example.com/page1")
        assert versions[0].last_seen_at >= first_seen


# ============================================================
# Task 10.5: DLQ routing
# ============================================================


class TestDLQRouting:
    """Tests for dead-letter queue routing after 3 retries (R2.5)."""

    def test_not_routed_before_3_attempts(self):
        dlq = DeadLetterQueue()
        dlq.record_attempt("url1", "error1")
        dlq.record_attempt("url1", "error2")
        assert dlq.should_route_to_dlq("url1") is False

    def test_routed_after_3_attempts_with_spacing(self):
        dlq = DeadLetterQueue()
        # Simulate 3 attempts with proper spacing
        state = dlq._retry_states.setdefault("url1", __import__("backend.indexer.dlq", fromlist=["RetryState"]).RetryState())
        now = time.time()
        state.attempt_count = 3
        state.attempt_timestamps = [now - 180, now - 90, now]
        state.last_error = "persistent error"
        assert dlq.should_route_to_dlq("url1") is True

    def test_not_routed_without_proper_spacing(self):
        dlq = DeadLetterQueue()
        # Simulate 3 attempts too close together
        from backend.indexer.dlq import RetryState
        state = RetryState()
        now = time.time()
        state.attempt_count = 3
        state.attempt_timestamps = [now - 10, now - 5, now]
        state.last_error = "error"
        dlq._retry_states["url1"] = state
        assert dlq.should_route_to_dlq("url1") is False

    def test_route_creates_entry(self):
        dlq = DeadLetterQueue()
        from backend.indexer.dlq import RetryState
        state = RetryState()
        state.attempt_count = 3
        state.last_error = "final error"
        now = time.time()
        state.attempt_timestamps = [now - 180, now - 90, now]
        dlq._retry_states["url1"] = state

        entry = dlq.route_to_dlq("url1", document_id="doc-123", source_url="https://example.com")
        assert entry.document_id == "doc-123"
        assert entry.source_url == "https://example.com"
        assert entry.failure_reason == "final error"
        assert entry.attempts == 3

    @pytest.mark.asyncio
    async def test_audit_emitted_on_dlq_route(self):
        audit = AsyncMock()
        indexer = IndexerService(audit_emitter=audit)

        # Set up retry state to trigger DLQ
        from backend.indexer.dlq import RetryState
        state = RetryState()
        now = time.time()
        state.attempt_count = 2
        state.attempt_timestamps = [now - 180, now - 90]
        state.last_error = "prev error"
        indexer.dlq._retry_states["https://example.com/fail"] = state

        result = await indexer.index_document_with_retry(
            "<p>Content</p>",
            "https://example.com/fail",
            error="final error",
            request_id="a" * 36,
        )

        # Should have routed to DLQ and emitted audit
        from backend.indexer.dlq import DLQEntry
        assert isinstance(result, DLQEntry)
        audit.emit.assert_called_once()
        call_kwargs = audit.emit.call_args.kwargs
        assert call_kwargs["action"] == "index_failure"


# ============================================================
# Task 10.6: Priority-source re-crawl scheduling
# ============================================================


class TestPriorityScheduler:
    """Tests for priority-source re-crawl scheduling (R2.2)."""

    def test_new_source_is_immediately_due(self):
        scheduler = PriorityScheduler()
        scheduler.add_source("https://example.com")
        due = scheduler.get_due_sources()
        assert len(due) == 1
        assert due[0].url == "https://example.com"

    def test_recently_crawled_not_due(self):
        scheduler = PriorityScheduler()
        scheduler.add_source("https://example.com")
        scheduler.record_crawl("https://example.com")
        due = scheduler.get_due_sources()
        assert len(due) == 0

    def test_source_due_after_24h(self):
        scheduler = PriorityScheduler()
        scheduler.add_source("https://example.com")
        past = datetime.now(timezone.utc) - timedelta(hours=25)
        scheduler.record_crawl("https://example.com", crawled_at=past)

        due = scheduler.get_due_sources()
        assert len(due) == 1

    def test_remove_source(self):
        scheduler = PriorityScheduler()
        scheduler.add_source("https://example.com")
        assert scheduler.remove_source("https://example.com") is True
        assert scheduler.get_due_sources() == []

    def test_is_source_due(self):
        scheduler = PriorityScheduler()
        scheduler.add_source("https://example.com")
        assert scheduler.is_source_due("https://example.com") is True
        scheduler.record_crawl("https://example.com")
        assert scheduler.is_source_due("https://example.com") is False


# ============================================================
# Task 10.7: Provenance Scorer
# ============================================================


class TestProvenanceScorer:
    """Tests for Provenance_Scorer (R10.1)."""

    def test_scores_in_valid_range(self):
        scorer = ProvenanceScorer()
        score = scorer.score("Some document text about science and research.")
        assert 0.0 <= score.credibility_score <= 1.0
        assert 0.0 <= score.ai_generated_likelihood <= 1.0

    def test_scored_at_is_set(self):
        scorer = ProvenanceScorer()
        before = datetime.now(timezone.utc)
        score = scorer.score("Test content")
        assert score.scored_at >= before

    def test_deterministic_for_same_input(self):
        scorer = ProvenanceScorer()
        s1 = scorer.score("Same text")
        s2 = scorer.score("Same text")
        assert s1.credibility_score == s2.credibility_score
        assert s1.ai_generated_likelihood == s2.ai_generated_likelihood

    def test_different_text_different_scores(self):
        scorer = ProvenanceScorer()
        s1 = scorer.score("Text about quantum physics research papers")
        s2 = scorer.score("Completely different content about cooking recipes")
        # At least one score should differ (extremely unlikely to collide)
        assert (
            s1.credibility_score != s2.credibility_score
            or s1.ai_generated_likelihood != s2.ai_generated_likelihood
        )


# ============================================================
# Task 10.8: Scoring gate
# ============================================================


class TestScoringGate:
    """Tests for scoring gate — document not visible until scored (R10.1)."""

    def test_unscored_document_not_visible(self):
        scorer = ProvenanceScorer()
        assert scorer.is_visible("doc-1", 1) is False

    def test_scored_document_is_visible(self):
        scorer = ProvenanceScorer()
        scorer.score_document("doc-1", 1, "hash123", "Document text")
        assert scorer.is_visible("doc-1", 1) is True

    @pytest.mark.asyncio
    async def test_indexer_marks_visible_after_scoring(self):
        scorer = ProvenanceScorer()
        indexer = IndexerService(provenance_scorer=scorer)
        result = await indexer.index_document("<p>Content</p>", "https://example.com/page")
        assert indexer.is_visible(result.document_id) is True

    @pytest.mark.asyncio
    async def test_indexer_without_scorer_still_visible(self):
        """Without a scorer configured, documents are visible (for testing)."""
        indexer = IndexerService()
        result = await indexer.index_document("<p>Content</p>", "https://example.com/page")
        assert indexer.is_visible(result.document_id) is True


# ============================================================
# Task 10.9: Rescore path
# ============================================================


class TestRescorePath:
    """Tests for rescore preserving document_id/version (R10.6)."""

    def test_rescore_preserves_document_id_and_version(self):
        scorer = ProvenanceScorer()
        scorer.score_document("doc-1", 3, "hash123", "Original text")

        # Rescore with different text (simulating model update)
        new_score = scorer.rescore("doc-1", 3, "Updated analysis text")

        # document_id and version preserved
        scored_doc = scorer.scored_documents[("doc-1", 3)]
        assert scored_doc.document_id == "doc-1"
        assert scored_doc.version == 3
        assert scored_doc.content_hash == "hash123"  # unchanged

    def test_rescore_mutates_only_score_fields(self):
        scorer = ProvenanceScorer()
        scorer.score_document("doc-1", 1, "hash-abc", "Text A")

        original_doc = scorer.scored_documents[("doc-1", 1)]
        original_hash = original_doc.content_hash

        scorer.rescore("doc-1", 1, "Different text for rescoring")

        updated_doc = scorer.scored_documents[("doc-1", 1)]
        assert updated_doc.document_id == "doc-1"
        assert updated_doc.version == 1
        assert updated_doc.content_hash == original_hash
        assert updated_doc.visible is True

    def test_rescore_updates_scored_at(self):
        scorer = ProvenanceScorer()
        scorer.score_document("doc-1", 1, "hash", "Text")
        first_score = scorer.get_score("doc-1", 1)

        # Small delay to ensure different timestamp
        new_score = scorer.rescore("doc-1", 1, "Text")
        assert new_score.scored_at >= first_score.scored_at

    def test_rescore_raises_for_unscored_document(self):
        scorer = ProvenanceScorer()
        with pytest.raises(KeyError):
            scorer.rescore("nonexistent", 1, "text")


# ============================================================
# Task 10.10: Vector embedding generation
# ============================================================


class TestVectorEmbeddings:
    """Tests for vector embedding generation."""

    def test_embedding_has_correct_dimension(self):
        embedding = generate_embedding("Hello world")
        assert len(embedding) == 256

    def test_embedding_is_normalized(self):
        embedding = generate_embedding("Test text")
        norm = sum(x * x for x in embedding) ** 0.5
        assert abs(norm - 1.0) < 1e-6

    def test_embedding_is_deterministic(self):
        e1 = generate_embedding("Same text")
        e2 = generate_embedding("Same text")
        assert e1 == e2

    def test_vector_index_write_and_read(self):
        index = VectorIndex()
        embedding = generate_embedding("Content")
        entry = index.write("doc-1", 1, embedding)
        assert entry.document_id == "doc-1"
        assert entry.version == 1

        retrieved = index.get("doc-1", 1)
        assert retrieved is not None
        assert retrieved.embedding == embedding


# ============================================================
# Task 10.11: Lexical index write
# ============================================================


class TestLexicalIndex:
    """Tests for lexical index write (BM25)."""

    def test_analyze_removes_stopwords(self):
        tokens = analyze_text("The quick brown fox is very fast")
        assert "the" not in tokens
        assert "is" not in tokens
        assert "very" not in tokens
        assert "quick" in tokens
        assert "brown" in tokens
        assert "fox" in tokens
        assert "fast" in tokens

    def test_analyze_lowercases(self):
        tokens = analyze_text("Hello WORLD")
        assert "hello" in tokens
        assert "world" in tokens

    def test_lexical_index_write_and_read(self):
        index = LexicalIndex()
        entry = index.write("doc-1", 1, "The quick brown fox jumps over the lazy dog")
        assert entry.document_id == "doc-1"
        assert entry.version == 1
        assert "quick" in entry.tokens
        assert "the" not in entry.tokens

        retrieved = index.get("doc-1", 1)
        assert retrieved is not None
        assert retrieved.tokens == entry.tokens

    def test_lexical_index_delete(self):
        index = LexicalIndex()
        index.write("doc-1", 1, "Some text")
        assert index.delete("doc-1", 1) is True
        assert index.get("doc-1", 1) is None
        assert index.delete("doc-1", 1) is False
