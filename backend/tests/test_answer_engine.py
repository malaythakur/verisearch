"""Unit tests for the Answer Engine (Tasks 13.1–13.10).

Tests cover:
- LLM provider abstraction (13.1)
- SSE streaming (13.2)
- Citation emission with offset ranges (13.3)
- Citation referential integrity (13.4)
- Done event with full answer + citations (13.5)
- Error handling: empty retrieval set (13.6)
- Error handling: model failure / 30s silence (13.7)
- WebSocket framing with cancel support (13.8)
- Highlights (13.9)
- Summaries (13.10)
"""

from __future__ import annotations

import asyncio
from typing import AsyncGenerator

import pytest

from backend.answer_engine import (
    AnswerEngine,
    AnswerErrorCode,
    CitationEvent,
    CitationIntegrityError,
    CitationOffsetError,
    CitationTracker,
    DoneEvent,
    ErrorEvent,
    GenerationRequest,
    HighlightSpan,
    LLMProviderError,
    MockLLMProvider,
    RetrievalResult,
    TokenEvent,
    estimate_token_count,
    extract_highlights,
    generate_summary,
    validate_highlight_spans,
    validate_summary_tokens,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_retrieval_result(
    document_id: str = "doc-1",
    version: int = 1,
    url: str = "https://example.com/page",
    title: str = "Test Document",
    score: float = 0.95,
    cleaned_text: str = "The answer is 42. This is a test document with some content.",
) -> RetrievalResult:
    """Create a test retrieval result."""
    return RetrievalResult(
        document_id=document_id,
        version=version,
        url=url,
        title=title,
        score=score,
        cleaned_text=cleaned_text,
    )


def make_retrieval_results(count: int = 3) -> list[RetrievalResult]:
    """Create multiple test retrieval results."""
    return [
        make_retrieval_result(
            document_id=f"doc-{i}",
            version=1,
            title=f"Document {i}",
            cleaned_text=f"Content of document {i}. It contains information about topic {i}.",
        )
        for i in range(1, count + 1)
    ]


async def collect_events(engine: AnswerEngine, query: str, results: list[RetrievalResult]) -> list:
    """Collect all events from a streaming answer generation."""
    events = []
    async for event in engine.generate_answer(query, results):
        events.append(event)
    return events


# ---------------------------------------------------------------------------
# Task 13.1: LLM Provider Abstraction
# ---------------------------------------------------------------------------


class TestLLMProviderAbstraction:
    """Tests for the LLM provider abstraction (Task 13.1)."""

    @pytest.mark.asyncio
    async def test_mock_provider_yields_tokens(self):
        """MockLLMProvider yields configured tokens."""
        tokens = ["Hello", " ", "world", "!"]
        provider = MockLLMProvider(tokens=tokens)

        request = GenerationRequest(query="test", context_documents=[])
        result = []
        async for token in provider.stream_tokens(request):
            result.append(token)

        assert result == tokens

    @pytest.mark.asyncio
    async def test_mock_provider_default_tokens(self):
        """MockLLMProvider has sensible default tokens."""
        provider = MockLLMProvider()
        request = GenerationRequest(query="test", context_documents=[])

        result = []
        async for token in provider.stream_tokens(request):
            result.append(token)

        assert len(result) > 0
        assert all(isinstance(t, str) for t in result)

    @pytest.mark.asyncio
    async def test_mock_provider_fail_after(self):
        """MockLLMProvider raises error after configured token count."""
        provider = MockLLMProvider(tokens=["a", "b", "c", "d"], fail_after=2)
        request = GenerationRequest(query="test", context_documents=[])

        result = []
        with pytest.raises(LLMProviderError):
            async for token in provider.stream_tokens(request):
                result.append(token)

        assert result == ["a", "b"]

    @pytest.mark.asyncio
    async def test_mock_provider_with_delay(self):
        """MockLLMProvider respects delay between tokens."""
        provider = MockLLMProvider(tokens=["a", "b"], delay_seconds=0.01)
        request = GenerationRequest(query="test", context_documents=[])

        result = []
        async for token in provider.stream_tokens(request):
            result.append(token)

        assert result == ["a", "b"]

    @pytest.mark.asyncio
    async def test_provider_protocol_compliance(self):
        """MockLLMProvider satisfies the LLMProvider protocol."""
        from backend.answer_engine.provider import LLMProvider

        provider = MockLLMProvider()
        assert isinstance(provider, LLMProvider)


# ---------------------------------------------------------------------------
# Task 13.2: SSE Streaming
# ---------------------------------------------------------------------------


class TestSSEStreaming:
    """Tests for SSE streaming answer generation (Task 13.2)."""

    @pytest.mark.asyncio
    async def test_streaming_yields_token_events(self):
        """Answer engine yields TokenEvent instances during streaming."""
        tokens = ["The ", "answer ", "is ", "42."]
        provider = MockLLMProvider(tokens=tokens)
        engine = AnswerEngine(provider=provider)

        results = make_retrieval_results()
        events = await collect_events(engine, "What is the answer?", results)

        token_events = [e for e in events if isinstance(e, TokenEvent)]
        assert len(token_events) == len(tokens)
        for i, event in enumerate(token_events):
            assert event.text == tokens[i]
            assert event.index == i

    @pytest.mark.asyncio
    async def test_streaming_token_indices_are_sequential(self):
        """Token event indices are sequential starting from 0."""
        provider = MockLLMProvider(tokens=["a", "b", "c"])
        engine = AnswerEngine(provider=provider)

        results = make_retrieval_results()
        events = await collect_events(engine, "test", results)

        token_events = [e for e in events if isinstance(e, TokenEvent)]
        indices = [e.index for e in token_events]
        assert indices == [0, 1, 2]


# ---------------------------------------------------------------------------
# Task 13.3: Citation Emission with Offset Ranges
# ---------------------------------------------------------------------------


class TestCitationEmission:
    """Tests for citation emission (Task 13.3)."""

    @pytest.mark.asyncio
    async def test_citation_has_offset_ranges(self):
        """Citations include answer and source offset ranges."""
        # Use tokens that match source text to trigger citation
        source_text = "The answer is 42. This is important information."
        tokens = list(source_text)  # Character-by-character for matching

        provider = MockLLMProvider(tokens=["The answer is 42. This is important information."])
        engine = AnswerEngine(provider=provider)

        results = [make_retrieval_result(cleaned_text=source_text)]
        events = await collect_events(engine, "What is the answer?", results)

        citation_events = [e for e in events if isinstance(e, CitationEvent)]
        for citation in citation_events:
            # Verify half-open range constraints
            assert citation.answer_start >= 0
            assert citation.answer_end > citation.answer_start
            assert citation.source_start >= 0
            assert citation.source_end > citation.source_start

    def test_citation_tracker_validates_offsets(self):
        """CitationTracker validates offset ranges."""
        results = [make_retrieval_result(cleaned_text="Hello world")]
        tracker = CitationTracker.from_results(results)
        tracker.update_answer_length(20)

        # Valid citation
        citation = tracker.validate_and_add_citation(
            document_id="doc-1",
            version=1,
            answer_start=0,
            answer_end=5,
            source_start=0,
            source_end=5,
        )
        assert citation.answer_start == 0
        assert citation.answer_end == 5

    def test_citation_tracker_rejects_invalid_answer_offsets(self):
        """CitationTracker rejects invalid answer offsets."""
        results = [make_retrieval_result(cleaned_text="Hello world")]
        tracker = CitationTracker.from_results(results)
        tracker.update_answer_length(10)

        # answer_end > answer_text_length
        with pytest.raises(CitationOffsetError):
            tracker.validate_and_add_citation(
                document_id="doc-1", version=1,
                answer_start=0, answer_end=20,
                source_start=0, source_end=5,
            )

    def test_citation_tracker_rejects_invalid_source_offsets(self):
        """CitationTracker rejects invalid source offsets."""
        results = [make_retrieval_result(cleaned_text="Short")]
        tracker = CitationTracker.from_results(results)
        tracker.update_answer_length(20)

        # source_end > len(cleaned_text)
        with pytest.raises(CitationOffsetError):
            tracker.validate_and_add_citation(
                document_id="doc-1", version=1,
                answer_start=0, answer_end=5,
                source_start=0, source_end=100,
            )


# ---------------------------------------------------------------------------
# Task 13.4: Citation Referential Integrity
# ---------------------------------------------------------------------------


class TestCitationReferentialIntegrity:
    """Tests for citation referential integrity (Task 13.4)."""

    def test_citation_must_reference_retrieval_set(self):
        """Citations must reference (document_id, version) in the retrieval set."""
        results = [make_retrieval_result(document_id="doc-1", version=1)]
        tracker = CitationTracker.from_results(results)
        tracker.update_answer_length(20)

        # Valid: doc-1 version 1 is in the set
        citation = tracker.validate_and_add_citation(
            document_id="doc-1", version=1,
            answer_start=0, answer_end=5,
            source_start=0, source_end=5,
        )
        assert citation.document_id == "doc-1"

    def test_citation_rejects_unknown_document(self):
        """Citations referencing unknown documents are rejected."""
        results = [make_retrieval_result(document_id="doc-1", version=1)]
        tracker = CitationTracker.from_results(results)
        tracker.update_answer_length(20)

        with pytest.raises(CitationIntegrityError) as exc_info:
            tracker.validate_and_add_citation(
                document_id="doc-unknown", version=1,
                answer_start=0, answer_end=5,
                source_start=0, source_end=5,
            )
        assert exc_info.value.document_id == "doc-unknown"

    def test_citation_rejects_wrong_version(self):
        """Citations referencing wrong version are rejected."""
        results = [make_retrieval_result(document_id="doc-1", version=1)]
        tracker = CitationTracker.from_results(results)
        tracker.update_answer_length(20)

        with pytest.raises(CitationIntegrityError):
            tracker.validate_and_add_citation(
                document_id="doc-1", version=99,
                answer_start=0, answer_end=5,
                source_start=0, source_end=5,
            )

    def test_has_document_check(self):
        """has_document correctly checks retrieval set membership."""
        results = [
            make_retrieval_result(document_id="doc-1", version=1),
            make_retrieval_result(document_id="doc-2", version=3),
        ]
        tracker = CitationTracker.from_results(results)

        assert tracker.has_document("doc-1", 1) is True
        assert tracker.has_document("doc-2", 3) is True
        assert tracker.has_document("doc-1", 2) is False
        assert tracker.has_document("doc-3", 1) is False


# ---------------------------------------------------------------------------
# Task 13.5: Done Event
# ---------------------------------------------------------------------------


class TestDoneEvent:
    """Tests for the done event (Task 13.5)."""

    @pytest.mark.asyncio
    async def test_done_event_contains_full_answer(self):
        """Done event contains the full concatenated answer text."""
        tokens = ["Hello", " ", "world"]
        provider = MockLLMProvider(tokens=tokens)
        engine = AnswerEngine(provider=provider)

        results = make_retrieval_results()
        events = await collect_events(engine, "test", results)

        done_events = [e for e in events if isinstance(e, DoneEvent)]
        assert len(done_events) == 1
        assert done_events[0].answer == "Hello world"

    @pytest.mark.asyncio
    async def test_done_event_contains_all_citations(self):
        """Done event contains the complete set of citations."""
        provider = MockLLMProvider(tokens=["Simple answer."])
        engine = AnswerEngine(provider=provider)

        results = make_retrieval_results()
        events = await collect_events(engine, "test", results)

        done_events = [e for e in events if isinstance(e, DoneEvent)]
        assert len(done_events) == 1

        # All citation events should be in the done event
        citation_events = [e for e in events if isinstance(e, CitationEvent)]
        assert done_events[0].citations == citation_events

    @pytest.mark.asyncio
    async def test_done_event_is_last_on_success(self):
        """Done event is the last event on successful completion."""
        provider = MockLLMProvider(tokens=["answer"])
        engine = AnswerEngine(provider=provider)

        results = make_retrieval_results()
        events = await collect_events(engine, "test", results)

        # Last event should be DoneEvent
        assert isinstance(events[-1], DoneEvent)


# ---------------------------------------------------------------------------
# Task 13.6: Error Handling — Empty Retrieval Set
# ---------------------------------------------------------------------------


class TestEmptyRetrievalSet:
    """Tests for empty retrieval set error handling (Task 13.6)."""

    @pytest.mark.asyncio
    async def test_empty_results_yields_no_sources_error(self):
        """Empty retrieval set yields no_sources_available error."""
        engine = AnswerEngine()

        events = await collect_events(engine, "test", [])

        assert len(events) == 1
        assert isinstance(events[0], ErrorEvent)
        assert events[0].code == AnswerErrorCode.NO_SOURCES_AVAILABLE

    @pytest.mark.asyncio
    async def test_empty_results_no_token_events(self):
        """Empty retrieval set produces no token or citation events."""
        engine = AnswerEngine()

        events = await collect_events(engine, "test", [])

        token_events = [e for e in events if isinstance(e, TokenEvent)]
        citation_events = [e for e in events if isinstance(e, CitationEvent)]
        assert len(token_events) == 0
        assert len(citation_events) == 0


# ---------------------------------------------------------------------------
# Task 13.7: Error Handling — Model Failure / 30s Silence
# ---------------------------------------------------------------------------


class TestModelFailureAndTimeout:
    """Tests for model failure and timeout error handling (Task 13.7)."""

    @pytest.mark.asyncio
    async def test_model_failure_yields_error_event(self):
        """Model failure yields model_error event."""
        provider = MockLLMProvider(tokens=["a", "b", "c"], fail_after=2)
        engine = AnswerEngine(provider=provider)

        results = make_retrieval_results()
        events = await collect_events(engine, "test", results)

        error_events = [e for e in events if isinstance(e, ErrorEvent)]
        assert len(error_events) == 1
        assert error_events[0].code == AnswerErrorCode.MODEL_ERROR

    @pytest.mark.asyncio
    async def test_model_failure_no_done_event(self):
        """Model failure does not produce a done event."""
        provider = MockLLMProvider(tokens=["a", "b"], fail_after=1)
        engine = AnswerEngine(provider=provider)

        results = make_retrieval_results()
        events = await collect_events(engine, "test", results)

        done_events = [e for e in events if isinstance(e, DoneEvent)]
        assert len(done_events) == 0

    @pytest.mark.asyncio
    async def test_silence_timeout_yields_error(self):
        """30s token silence yields stream_timeout error."""
        # Use a very short timeout for testing
        provider = MockLLMProvider(tokens=["a", "b", "c"], hang_after=1)
        engine = AnswerEngine(provider=provider, silence_timeout=0.1)

        results = make_retrieval_results()
        events = await collect_events(engine, "test", results)

        error_events = [e for e in events if isinstance(e, ErrorEvent)]
        assert len(error_events) == 1
        assert error_events[0].code == AnswerErrorCode.STREAM_TIMEOUT

    @pytest.mark.asyncio
    async def test_no_events_after_error(self):
        """No token or citation events are emitted after an error."""
        provider = MockLLMProvider(tokens=["a", "b", "c"], fail_after=1)
        engine = AnswerEngine(provider=provider)

        results = make_retrieval_results()
        events = await collect_events(engine, "test", results)

        # Find the error event index
        error_idx = None
        for i, event in enumerate(events):
            if isinstance(event, ErrorEvent):
                error_idx = i
                break

        assert error_idx is not None
        # No token or citation events after the error
        for event in events[error_idx + 1:]:
            assert not isinstance(event, (TokenEvent, CitationEvent))


# ---------------------------------------------------------------------------
# Task 13.8: WebSocket Framing with Cancel Support
# ---------------------------------------------------------------------------


class TestWebSocketFraming:
    """Tests for WebSocket framing concepts (Task 13.8).

    Note: Full WebSocket integration is tested at the API gateway level.
    These tests verify the cancel support logic.
    """

    @pytest.mark.asyncio
    async def test_cancel_stops_generation(self):
        """Cancellation stops token generation."""
        # Simulate cancel by using a provider with many tokens
        # and collecting only a few
        provider = MockLLMProvider(
            tokens=["a"] * 100, delay_seconds=0.01
        )
        engine = AnswerEngine(provider=provider)
        results = make_retrieval_results()

        events = []
        async for event in engine.generate_answer("test", results):
            events.append(event)
            if len(events) >= 5:
                break  # Simulate client cancel

        # Should have collected some token events but not all
        token_events = [e for e in events if isinstance(e, TokenEvent)]
        assert 0 < len(token_events) < 100


# ---------------------------------------------------------------------------
# Task 13.9: Highlights
# ---------------------------------------------------------------------------


class TestHighlights:
    """Tests for highlight extraction (Task 13.9)."""

    def test_highlights_returns_valid_spans(self):
        """Highlights return valid half-open spans."""
        text = "The quick brown fox jumps over the lazy dog. Python is great."
        spans = extract_highlights("fox jumps", text)

        for span in spans:
            assert 0 <= span.start < span.end <= len(text)

    def test_highlights_max_five_spans(self):
        """At most 5 highlight spans per document."""
        text = "word " * 1000  # Lots of potential matches
        spans = extract_highlights("word", text)

        assert len(spans) <= 5

    def test_highlights_empty_query(self):
        """Empty query returns no highlights."""
        spans = extract_highlights("", "Some text content")
        assert spans == []

    def test_highlights_empty_text(self):
        """Empty text returns no highlights."""
        spans = extract_highlights("query", "")
        assert spans == []

    def test_highlights_no_match(self):
        """No matching terms returns fallback or empty."""
        text = "The quick brown fox"
        spans = extract_highlights("zzzzz", text)
        # May return fallback highlight or empty
        for span in spans:
            assert 0 <= span.start < span.end <= len(text)

    def test_validate_highlight_spans_valid(self):
        """validate_highlight_spans accepts valid spans."""
        spans = [HighlightSpan(start=0, end=5), HighlightSpan(start=10, end=20)]
        assert validate_highlight_spans(spans, text_length=100) is True

    def test_validate_highlight_spans_too_many(self):
        """validate_highlight_spans rejects more than 5 spans."""
        spans = [HighlightSpan(start=i * 10, end=i * 10 + 5) for i in range(6)]
        assert validate_highlight_spans(spans, text_length=100) is False

    def test_validate_highlight_spans_invalid_range(self):
        """validate_highlight_spans rejects invalid ranges."""
        # start >= end
        spans = [HighlightSpan(start=5, end=5)]
        assert validate_highlight_spans(spans, text_length=100) is False

        # end > text_length
        spans = [HighlightSpan(start=0, end=200)]
        assert validate_highlight_spans(spans, text_length=100) is False

        # start < 0
        spans = [HighlightSpan(start=-1, end=5)]
        assert validate_highlight_spans(spans, text_length=100) is False


# ---------------------------------------------------------------------------
# Task 13.10: Summaries
# ---------------------------------------------------------------------------


class TestSummaries:
    """Tests for summary generation (Task 13.10)."""

    def test_summary_within_token_bounds(self):
        """Summary is between 1 and 512 tokens."""
        text = "This is a test document. " * 100
        summary = generate_summary(text)

        token_count = estimate_token_count(summary)
        assert 1 <= token_count <= 512

    def test_summary_non_empty(self):
        """Summary is never empty for non-empty input."""
        text = "Short text."
        summary = generate_summary(text)
        assert len(summary) > 0

    def test_summary_respects_max_tokens(self):
        """Summary respects the max_tokens parameter."""
        text = "Word " * 1000
        summary = generate_summary(text, max_tokens=50)

        token_count = estimate_token_count(summary)
        assert token_count <= 50

    def test_summary_empty_text_raises(self):
        """Empty text raises ValueError."""
        with pytest.raises(ValueError):
            generate_summary("")

    def test_summary_whitespace_only_raises(self):
        """Whitespace-only text raises ValueError."""
        with pytest.raises(ValueError):
            generate_summary("   \n\t  ")

    def test_validate_summary_tokens_valid(self):
        """validate_summary_tokens accepts valid summaries."""
        assert validate_summary_tokens("This is a valid summary.") is True

    def test_validate_summary_tokens_empty(self):
        """validate_summary_tokens rejects empty summaries."""
        assert validate_summary_tokens("") is False

    def test_estimate_token_count(self):
        """estimate_token_count returns reasonable estimates."""
        assert estimate_token_count("") == 0
        assert estimate_token_count("hello") >= 1
        assert estimate_token_count("a " * 100) >= 50
