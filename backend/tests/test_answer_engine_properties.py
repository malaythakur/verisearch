"""Property-based tests for the Answer Engine (Tasks 13.11–13.15).

Properties tested:
- Property 10: Highlight spans are valid half-open ranges (R5.2).
- Property 11: Summaries within token bounds (R5.3).
- Property 12: Done event reflects full stream (R6.3).
- Property 13: Citations reference retrieval result set (R6.2, R6.4).
- Property 14: Failure modes emit exactly one terminal error (R6.5, R6.6).
"""

from __future__ import annotations

import asyncio
from typing import AsyncGenerator

import pytest
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from backend.answer_engine import (
    AnswerEngine,
    AnswerErrorCode,
    CitationEvent,
    CitationTracker,
    DoneEvent,
    ErrorEvent,
    HighlightSpan,
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
# Strategies
# ---------------------------------------------------------------------------

# Strategy for generating cleaned text content
cleaned_text_strategy = st.text(
    min_size=1,
    max_size=2000,
    alphabet=st.characters(
        whitelist_categories=("L", "N", "P", "Z"),
        whitelist_characters=" .,!?;:-\n",
    ),
).filter(lambda t: len(t.strip()) > 0)

# Strategy for generating query strings
query_strategy = st.text(
    min_size=1,
    max_size=200,
    alphabet=st.characters(whitelist_categories=("L", "N", "Z")),
).filter(lambda t: len(t.strip()) >= 2)

# Strategy for document IDs
document_id_strategy = st.uuids().map(str)

# Strategy for document versions
version_strategy = st.integers(min_value=1, max_value=100)

# Strategy for generating retrieval results
retrieval_result_strategy = st.builds(
    RetrievalResult,
    document_id=document_id_strategy,
    version=version_strategy,
    url=st.from_regex(r"https://[a-z]{3,10}\.[a-z]{2,4}/[a-z]{1,10}", fullmatch=True),
    title=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("L", "N", "Z"))),
    score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    cleaned_text=cleaned_text_strategy,
)

# Strategy for non-empty retrieval result lists
retrieval_results_strategy = st.lists(
    retrieval_result_strategy, min_size=1, max_size=5
)

# Strategy for token lists (simulating LLM output)
token_strategy = st.lists(
    st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L", "N", "Z", "P"))),
    min_size=1,
    max_size=20,
)

# Strategy for max_tokens parameter
max_tokens_strategy = st.integers(min_value=1, max_value=512)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def collect_events(engine: AnswerEngine, query: str, results: list[RetrievalResult]) -> list:
    """Collect all events from a streaming answer generation."""
    events = []
    async for event in engine.generate_answer(query, results):
        events.append(event)
    return events


# ---------------------------------------------------------------------------
# Property 10: Highlight spans are valid half-open ranges
# ---------------------------------------------------------------------------


class TestProperty10HighlightSpans:
    """Property 10: Highlight spans are valid half-open ranges within bounds.

    **Validates: Requirements 5.2**

    For any /v1/contents response with highlights: true and a valid query,
    every highlight span satisfies 0 <= start < end <= length(cleaned_text)
    and the per-document highlight count is in [0, 5].
    """

    @given(
        query=query_strategy,
        cleaned_text=cleaned_text_strategy,
    )
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=100)
    def test_highlight_spans_valid_half_open_ranges(self, query: str, cleaned_text: str):
        """Every highlight span satisfies 0 <= start < end <= len(cleaned_text)."""
        spans = extract_highlights(query, cleaned_text)

        for span in spans:
            assert 0 <= span.start, f"start ({span.start}) must be >= 0"
            assert span.start < span.end, f"start ({span.start}) must be < end ({span.end})"
            assert span.end <= len(cleaned_text), (
                f"end ({span.end}) must be <= text length ({len(cleaned_text)})"
            )

    @given(
        query=query_strategy,
        cleaned_text=cleaned_text_strategy,
    )
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=100)
    def test_highlight_count_within_bounds(self, query: str, cleaned_text: str):
        """Per-document highlight count is in [0, 5]."""
        spans = extract_highlights(query, cleaned_text)
        assert 0 <= len(spans) <= 5

    @given(
        query=query_strategy,
        cleaned_text=cleaned_text_strategy,
    )
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=100)
    def test_highlight_validation_passes(self, query: str, cleaned_text: str):
        """All generated highlights pass validation."""
        spans = extract_highlights(query, cleaned_text)
        assert validate_highlight_spans(spans, len(cleaned_text)) is True

    @given(
        cleaned_text=cleaned_text_strategy,
        num_spans=st.integers(min_value=0, max_value=5),
    )
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=100)
    def test_generated_spans_always_valid(self, cleaned_text: str, num_spans: int):
        """Randomly generated valid spans pass validation."""
        assume(len(cleaned_text) >= 2)

        spans = []
        for _ in range(num_spans):
            start = 0
            end = min(len(cleaned_text), start + 10)
            if start < end <= len(cleaned_text):
                spans.append(HighlightSpan(start=start, end=end))

        assert validate_highlight_spans(spans, len(cleaned_text)) is True


# ---------------------------------------------------------------------------
# Property 11: Summaries within token bounds
# ---------------------------------------------------------------------------


class TestProperty11SummaryTokenBounds:
    """Property 11: Summaries are within token bounds.

    **Validates: Requirements 5.3**

    For any /v1/contents response with summary: true, the per-document
    summary length is in [1, 512] model tokens.
    """

    @given(cleaned_text=cleaned_text_strategy)
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=100)
    def test_summary_token_count_in_bounds(self, cleaned_text: str):
        """Summary token count is in [1, 512]."""
        assume(len(cleaned_text.strip()) > 0)

        summary = generate_summary(cleaned_text)
        token_count = estimate_token_count(summary)

        assert token_count >= 1, f"Summary must have at least 1 token, got {token_count}"
        assert token_count <= 512, f"Summary must have at most 512 tokens, got {token_count}"

    @given(
        cleaned_text=cleaned_text_strategy,
        max_tokens=max_tokens_strategy,
    )
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=100)
    def test_summary_respects_max_tokens(self, cleaned_text: str, max_tokens: int):
        """Summary respects the max_tokens parameter."""
        assume(len(cleaned_text.strip()) > 0)

        summary = generate_summary(cleaned_text, max_tokens=max_tokens)
        token_count = estimate_token_count(summary)

        assert token_count >= 1, "Summary must have at least 1 token"
        assert token_count <= max_tokens, (
            f"Summary ({token_count} tokens) exceeds max_tokens ({max_tokens})"
        )

    @given(cleaned_text=cleaned_text_strategy)
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=100)
    def test_summary_is_non_empty(self, cleaned_text: str):
        """Summary is never empty for non-empty input."""
        assume(len(cleaned_text.strip()) > 0)

        summary = generate_summary(cleaned_text)
        assert len(summary) > 0

    @given(cleaned_text=cleaned_text_strategy)
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=100)
    def test_summary_validates_successfully(self, cleaned_text: str):
        """Generated summaries pass token validation."""
        assume(len(cleaned_text.strip()) > 0)

        summary = generate_summary(cleaned_text)
        assert validate_summary_tokens(summary) is True


# ---------------------------------------------------------------------------
# Property 12: Done event reflects full stream
# ---------------------------------------------------------------------------


class TestProperty12DoneEventReflectsStream:
    """Property 12: /v1/answer done event reflects the full stream.

    **Validates: Requirements 6.3**

    For any successful streaming /v1/answer execution, the final done event's
    answer field equals the concatenation of all token events in emission order,
    and its citations field as a set equals the set of citation events emitted
    during the stream.
    """

    @given(tokens=token_strategy)
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=50, deadline=None)
    @pytest.mark.asyncio
    async def test_done_answer_equals_concatenated_tokens(self, tokens: list[str]):
        """Done event answer equals concatenation of all token events."""
        provider = MockLLMProvider(tokens=tokens)
        engine = AnswerEngine(provider=provider)

        results = [
            RetrievalResult(
                document_id="doc-1",
                version=1,
                url="https://example.com",
                title="Test",
                score=0.9,
                cleaned_text="x" * 1000,  # Long text unlikely to match tokens
            )
        ]

        events = await collect_events(engine, "test query", results)

        # Find token events and done event
        token_events = [e for e in events if isinstance(e, TokenEvent)]
        done_events = [e for e in events if isinstance(e, DoneEvent)]

        assert len(done_events) == 1, "Exactly one done event expected"

        # Concatenation of token texts should equal done.answer
        concatenated = "".join(e.text for e in token_events)
        assert done_events[0].answer == concatenated, (
            f"Done answer '{done_events[0].answer}' != concatenated tokens '{concatenated}'"
        )

    @given(tokens=token_strategy)
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=50, deadline=None)
    @pytest.mark.asyncio
    async def test_done_citations_equal_stream_citations(self, tokens: list[str]):
        """Done event citations set equals the set of citation events in stream."""
        provider = MockLLMProvider(tokens=tokens)
        engine = AnswerEngine(provider=provider)

        results = [
            RetrievalResult(
                document_id="doc-1",
                version=1,
                url="https://example.com",
                title="Test",
                score=0.9,
                cleaned_text="x" * 1000,
            )
        ]

        events = await collect_events(engine, "test query", results)

        citation_events = [e for e in events if isinstance(e, CitationEvent)]
        done_events = [e for e in events if isinstance(e, DoneEvent)]

        assert len(done_events) == 1
        assert done_events[0].citations == citation_events

    @given(tokens=token_strategy)
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=50, deadline=None)
    @pytest.mark.asyncio
    async def test_exactly_one_terminal_event_on_success(self, tokens: list[str]):
        """Successful stream has exactly one done event and no error events."""
        provider = MockLLMProvider(tokens=tokens)
        engine = AnswerEngine(provider=provider)

        results = [
            RetrievalResult(
                document_id="doc-1",
                version=1,
                url="https://example.com",
                title="Test",
                score=0.9,
                cleaned_text="x" * 1000,
            )
        ]

        events = await collect_events(engine, "test query", results)

        done_events = [e for e in events if isinstance(e, DoneEvent)]
        error_events = [e for e in events if isinstance(e, ErrorEvent)]

        assert len(done_events) == 1
        assert len(error_events) == 0


# ---------------------------------------------------------------------------
# Property 13: Citations reference retrieval result set
# ---------------------------------------------------------------------------


class TestProperty13CitationsReferenceRetrievalSet:
    """Property 13: Citations on /v1/answer reference the request's retrieval result set.

    **Validates: Requirements 6.2, 6.4**

    For any streaming /v1/answer execution with retrieval result set R,
    every emitted citation event's (document_id, version) pair is in R;
    and every emitted citation has valid offset ranges.
    """

    @given(
        tokens=token_strategy,
        results=retrieval_results_strategy,
    )
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=50, deadline=None)
    @pytest.mark.asyncio
    async def test_all_citations_reference_retrieval_set(
        self, tokens: list[str], results: list[RetrievalResult]
    ):
        """Every citation's (document_id, version) is in the retrieval set."""
        provider = MockLLMProvider(tokens=tokens)
        engine = AnswerEngine(provider=provider)

        events = await collect_events(engine, "test query", results)

        # Build the retrieval set for validation
        retrieval_set = {(r.document_id, r.version) for r in results}

        citation_events = [e for e in events if isinstance(e, CitationEvent)]
        for citation in citation_events:
            assert (citation.document_id, citation.version) in retrieval_set, (
                f"Citation ({citation.document_id}, {citation.version}) "
                f"not in retrieval set {retrieval_set}"
            )

    @given(
        tokens=token_strategy,
        results=retrieval_results_strategy,
    )
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=50, deadline=None)
    @pytest.mark.asyncio
    async def test_citation_answer_offsets_valid(
        self, tokens: list[str], results: list[RetrievalResult]
    ):
        """Every citation has valid answer offset ranges."""
        provider = MockLLMProvider(tokens=tokens)
        engine = AnswerEngine(provider=provider)

        events = await collect_events(engine, "test query", results)

        # Get the full answer text
        done_events = [e for e in events if isinstance(e, DoneEvent)]
        if not done_events:
            return  # Error case, no done event

        answer_text = done_events[0].answer

        citation_events = [e for e in events if isinstance(e, CitationEvent)]
        for citation in citation_events:
            assert 0 <= citation.answer_start, (
                f"answer_start ({citation.answer_start}) must be >= 0"
            )
            assert citation.answer_start < citation.answer_end, (
                f"answer_start ({citation.answer_start}) must be < answer_end ({citation.answer_end})"
            )
            assert citation.answer_end <= len(answer_text), (
                f"answer_end ({citation.answer_end}) must be <= answer length ({len(answer_text)})"
            )

    @given(
        tokens=token_strategy,
        results=retrieval_results_strategy,
    )
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=50, deadline=None)
    @pytest.mark.asyncio
    async def test_citation_source_offsets_valid(
        self, tokens: list[str], results: list[RetrievalResult]
    ):
        """Every citation has valid source offset ranges."""
        provider = MockLLMProvider(tokens=tokens)
        engine = AnswerEngine(provider=provider)

        events = await collect_events(engine, "test query", results)

        # Build lookup for source text lengths
        source_lengths = {
            (r.document_id, r.version): len(r.cleaned_text) for r in results
        }

        citation_events = [e for e in events if isinstance(e, CitationEvent)]
        for citation in citation_events:
            source_length = source_lengths.get(
                (citation.document_id, citation.version), 0
            )
            assert 0 <= citation.source_start, (
                f"source_start ({citation.source_start}) must be >= 0"
            )
            assert citation.source_start < citation.source_end, (
                f"source_start ({citation.source_start}) must be < source_end ({citation.source_end})"
            )
            assert citation.source_end <= source_length, (
                f"source_end ({citation.source_end}) must be <= source length ({source_length})"
            )

    @given(results=retrieval_results_strategy)
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=50)
    def test_citation_tracker_enforces_integrity(self, results: list[RetrievalResult]):
        """CitationTracker rejects citations not in the retrieval set."""
        tracker = CitationTracker.from_results(results)
        tracker.update_answer_length(100)

        # Try to add a citation for a document NOT in the set
        from backend.answer_engine import CitationIntegrityError

        try:
            tracker.validate_and_add_citation(
                document_id="nonexistent-doc",
                version=999,
                answer_start=0,
                answer_end=10,
                source_start=0,
                source_end=10,
            )
            # Should not reach here
            assert False, "Should have raised CitationIntegrityError"
        except CitationIntegrityError:
            pass  # Expected


# ---------------------------------------------------------------------------
# Property 14: Failure modes emit exactly one terminal error
# ---------------------------------------------------------------------------


class TestProperty14FailureModesTerminalError:
    """Property 14: /v1/answer failure modes emit exactly one terminal error.

    **Validates: Requirements 6.5, 6.6**

    For any streaming /v1/answer execution that experiences:
    (a) an empty retrieval set,
    (b) an upstream model failure, or
    (c) 30 consecutive seconds of token silence,
    the engine emits exactly one error event with a code drawn from the
    documented enum, emits no further token or citation events after the error,
    and closes the stream within 2 seconds of the error.
    """

    @given(st.data())
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=50, deadline=None)
    @pytest.mark.asyncio
    async def test_empty_retrieval_set_single_error(self, data):
        """Empty retrieval set emits exactly one no_sources_available error."""
        engine = AnswerEngine()

        events = await collect_events(engine, "any query", [])

        error_events = [e for e in events if isinstance(e, ErrorEvent)]
        token_events = [e for e in events if isinstance(e, TokenEvent)]
        citation_events = [e for e in events if isinstance(e, CitationEvent)]
        done_events = [e for e in events if isinstance(e, DoneEvent)]

        # Exactly one error event
        assert len(error_events) == 1
        assert error_events[0].code == AnswerErrorCode.NO_SOURCES_AVAILABLE

        # No token, citation, or done events
        assert len(token_events) == 0
        assert len(citation_events) == 0
        assert len(done_events) == 0

    @given(
        tokens=st.lists(
            st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=("L",))),
            min_size=2,
            max_size=10,
        ),
        fail_after=st.integers(min_value=1, max_value=5),
    )
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=50, deadline=None)
    @pytest.mark.asyncio
    async def test_model_failure_single_error(self, tokens: list[str], fail_after: int):
        """Model failure emits exactly one model_error event."""
        assume(fail_after < len(tokens))

        provider = MockLLMProvider(tokens=tokens, fail_after=fail_after)
        engine = AnswerEngine(provider=provider)

        results = [
            RetrievalResult(
                document_id="doc-1",
                version=1,
                url="https://example.com",
                title="Test",
                score=0.9,
                cleaned_text="x" * 1000,
            )
        ]

        events = await collect_events(engine, "test", results)

        error_events = [e for e in events if isinstance(e, ErrorEvent)]
        done_events = [e for e in events if isinstance(e, DoneEvent)]

        # Exactly one error event
        assert len(error_events) == 1
        assert error_events[0].code == AnswerErrorCode.MODEL_ERROR

        # No done event
        assert len(done_events) == 0

    @given(
        tokens=st.lists(
            st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=("L",))),
            min_size=2,
            max_size=10,
        ),
        hang_after=st.integers(min_value=0, max_value=3),
    )
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=20, deadline=None)
    @pytest.mark.asyncio
    async def test_silence_timeout_single_error(self, tokens: list[str], hang_after: int):
        """Token silence emits exactly one stream_timeout error."""
        assume(hang_after < len(tokens))

        provider = MockLLMProvider(tokens=tokens, hang_after=hang_after)
        engine = AnswerEngine(provider=provider, silence_timeout=0.1)

        results = [
            RetrievalResult(
                document_id="doc-1",
                version=1,
                url="https://example.com",
                title="Test",
                score=0.9,
                cleaned_text="x" * 1000,
            )
        ]

        events = await collect_events(engine, "test", results)

        error_events = [e for e in events if isinstance(e, ErrorEvent)]
        done_events = [e for e in events if isinstance(e, DoneEvent)]

        # Exactly one error event
        assert len(error_events) == 1
        assert error_events[0].code == AnswerErrorCode.STREAM_TIMEOUT

        # No done event
        assert len(done_events) == 0

    @given(
        tokens=st.lists(
            st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=("L",))),
            min_size=2,
            max_size=10,
        ),
        fail_after=st.integers(min_value=1, max_value=5),
    )
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=50, deadline=None)
    @pytest.mark.asyncio
    async def test_no_events_after_error(self, tokens: list[str], fail_after: int):
        """No token or citation events are emitted after an error event."""
        assume(fail_after < len(tokens))

        provider = MockLLMProvider(tokens=tokens, fail_after=fail_after)
        engine = AnswerEngine(provider=provider)

        results = [
            RetrievalResult(
                document_id="doc-1",
                version=1,
                url="https://example.com",
                title="Test",
                score=0.9,
                cleaned_text="x" * 1000,
            )
        ]

        events = await collect_events(engine, "test", results)

        # Find the error event index
        error_idx = None
        for i, event in enumerate(events):
            if isinstance(event, ErrorEvent):
                error_idx = i
                break

        assert error_idx is not None, "Expected an error event"

        # No token or citation events after the error
        for event in events[error_idx + 1:]:
            assert not isinstance(event, TokenEvent), (
                "TokenEvent found after ErrorEvent"
            )
            assert not isinstance(event, CitationEvent), (
                "CitationEvent found after ErrorEvent"
            )

    @given(
        tokens=st.lists(
            st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=("L",))),
            min_size=2,
            max_size=10,
        ),
        fail_after=st.integers(min_value=1, max_value=5),
    )
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=50, deadline=None)
    @pytest.mark.asyncio
    async def test_error_code_from_documented_enum(self, tokens: list[str], fail_after: int):
        """Error codes are drawn from the documented enum."""
        assume(fail_after < len(tokens))

        provider = MockLLMProvider(tokens=tokens, fail_after=fail_after)
        engine = AnswerEngine(provider=provider)

        results = [
            RetrievalResult(
                document_id="doc-1",
                version=1,
                url="https://example.com",
                title="Test",
                score=0.9,
                cleaned_text="x" * 1000,
            )
        ]

        events = await collect_events(engine, "test", results)

        error_events = [e for e in events if isinstance(e, ErrorEvent)]
        for error in error_events:
            assert error.code in AnswerErrorCode, (
                f"Error code '{error.code}' not in documented enum"
            )
