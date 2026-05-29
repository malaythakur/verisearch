"""Main Answer Engine service — streaming answer generation with citations.

Implements:
- generate_answer(): AsyncGenerator yielding TokenEvent, CitationEvent, DoneEvent, ErrorEvent.
- Empty retrieval set handling → no_sources_available (R6.6).
- Model failure or 30s silence → error event, close within 2s (R6.5).
- Citation emission within 500ms of supported span (R6.2).
- Done event with full answer text + complete citation set (R6.3).
"""

from __future__ import annotations

import asyncio
import time
from typing import AsyncGenerator

from backend.answer_engine.citations import CitationTracker
from backend.answer_engine.models import (
    AnswerErrorCode,
    AnswerEvent,
    CitationEvent,
    DoneEvent,
    ErrorEvent,
    RetrievalResult,
    TokenEvent,
)
from backend.answer_engine.provider import (
    GenerationRequest,
    LLMProvider,
    LLMProviderError,
    MockLLMProvider,
)

# Timeout for token silence (R6.5)
TOKEN_SILENCE_TIMEOUT_SECONDS = 30.0

# Maximum time for stream close after error (R6.5, R6.6)
ERROR_CLOSE_TIMEOUT_SECONDS = 2.0

# Target for first token latency (R6.1)
FIRST_TOKEN_TARGET_SECONDS = 3.0


class AnswerEngine:
    """Main service for streaming answer generation with citations.

    Orchestrates LLM provider calls, citation tracking, and event emission.
    """

    def __init__(
        self,
        provider: LLMProvider | None = None,
        silence_timeout: float = TOKEN_SILENCE_TIMEOUT_SECONDS,
    ):
        """Initialize the Answer Engine.

        Args:
            provider: LLM provider for token generation. Defaults to MockLLMProvider.
            silence_timeout: Seconds of token silence before timeout error (default 30s).
        """
        self.provider = provider or MockLLMProvider()
        self.silence_timeout = silence_timeout

    async def generate_answer(
        self,
        query: str,
        retrieval_results: list[RetrievalResult],
        session_id: str | None = None,
    ) -> AsyncGenerator[AnswerEvent, None]:
        """Generate a streaming answer with citations.

        Yields events in order: token events, citation events (interleaved),
        and finally a done event or error event.

        Args:
            query: The user's query.
            retrieval_results: Documents from the retrieval set.
            session_id: Optional session ID for context.

        Yields:
            AnswerEvent instances (TokenEvent, CitationEvent, DoneEvent, or ErrorEvent).
        """
        # R6.6: Empty retrieval set → no_sources_available
        if not retrieval_results:
            yield ErrorEvent(
                code=AnswerErrorCode.NO_SOURCES_AVAILABLE,
                message="No sources available to generate an answer.",
            )
            return

        # Initialize citation tracker with the retrieval set
        citation_tracker = CitationTracker.from_results(retrieval_results)

        # Build the generation request
        context_docs = [
            {
                "document_id": r.document_id,
                "title": r.title,
                "text": r.cleaned_text,
            }
            for r in retrieval_results
        ]

        request = GenerationRequest(
            query=query,
            context_documents=context_docs,
            system_prompt=_build_system_prompt(retrieval_results),
        )

        # Stream tokens from the provider with timeout handling
        answer_text = ""
        token_index = 0
        all_citations: list[CitationEvent] = []
        error_emitted = False

        try:
            token_stream = self.provider.stream_tokens(request)

            async for token in self._stream_with_timeout(token_stream):
                # Emit token event
                token_event = TokenEvent(text=token, index=token_index)
                yield token_event

                answer_text += token
                token_index += 1
                citation_tracker.update_answer_length(len(answer_text))

                # Check for citation opportunities after each token
                new_citations = self._detect_citations(
                    answer_text, retrieval_results, citation_tracker
                )
                for citation in new_citations:
                    all_citations.append(citation)
                    yield citation

        except asyncio.TimeoutError:
            # R6.5: 30s silence → stream_timeout error
            yield ErrorEvent(
                code=AnswerErrorCode.STREAM_TIMEOUT,
                message="No tokens received for 30 seconds.",
            )
            error_emitted = True

        except LLMProviderError as e:
            # R6.5: Model failure → model_error
            yield ErrorEvent(
                code=AnswerErrorCode.MODEL_ERROR,
                message=f"Model error: {str(e)}",
            )
            error_emitted = True

        except Exception as e:
            # Unexpected error → internal_error
            yield ErrorEvent(
                code=AnswerErrorCode.INTERNAL_ERROR,
                message=f"Internal error: {str(e)}",
            )
            error_emitted = True

        # R6.3: On success, emit done event with full answer + all citations
        if not error_emitted:
            yield DoneEvent(answer=answer_text, citations=all_citations)

    async def _stream_with_timeout(
        self, token_stream: AsyncGenerator[str, None]
    ) -> AsyncGenerator[str, None]:
        """Wrap a token stream with silence timeout detection.

        Raises asyncio.TimeoutError if no token is received within
        self.silence_timeout seconds.
        """
        async for token in _timeout_aiter(token_stream, self.silence_timeout):
            yield token

    def _detect_citations(
        self,
        answer_text: str,
        retrieval_results: list[RetrievalResult],
        tracker: CitationTracker,
    ) -> list[CitationEvent]:
        """Detect citation opportunities in the current answer text.

        Looks for spans in the answer that match source document content.
        Citations are emitted within 500ms of the supported span (R6.2).

        Returns a list of new citations detected.
        """
        citations: list[CitationEvent] = []

        # Simple substring matching for citation detection
        # A production implementation would use more sophisticated NLP
        for result in retrieval_results:
            new_citation = self._find_citation_match(
                answer_text, result, tracker
            )
            if new_citation:
                citations.append(new_citation)

        return citations

    def _find_citation_match(
        self,
        answer_text: str,
        result: RetrievalResult,
        tracker: CitationTracker,
    ) -> CitationEvent | None:
        """Find a citation match between answer text and a source document.

        Uses a sliding window approach to find matching spans.
        """
        if len(answer_text) < 10:
            return None

        source_text = result.cleaned_text
        if not source_text:
            return None

        # Look for the most recent chunk of answer text in the source
        # Check the last 50 characters of the answer
        window_size = min(30, len(answer_text))
        recent_text = answer_text[-window_size:]

        # Find this text in the source
        source_pos = source_text.find(recent_text)
        if source_pos == -1:
            return None

        # Calculate offsets
        answer_start = len(answer_text) - window_size
        answer_end = len(answer_text)
        source_start = source_pos
        source_end = source_pos + window_size

        # Validate offsets before creating citation
        if answer_start < 0 or answer_end > len(answer_text):
            return None
        if source_start < 0 or source_end > len(source_text):
            return None
        if answer_start >= answer_end or source_start >= source_end:
            return None

        # Check if we already have a citation for this exact span
        for existing in tracker.citations:
            if (
                existing.document_id == result.document_id
                and existing.answer_start == answer_start
                and existing.answer_end == answer_end
            ):
                return None

        try:
            citation = tracker.validate_and_add_citation(
                document_id=result.document_id,
                version=result.version,
                answer_start=answer_start,
                answer_end=answer_end,
                source_start=source_start,
                source_end=source_end,
            )
            return citation
        except Exception:
            return None


async def _timeout_aiter(
    aiter: AsyncGenerator[str, None], timeout: float
) -> AsyncGenerator[str, None]:
    """Wrap an async generator with per-item timeout.

    Raises asyncio.TimeoutError if no item is yielded within `timeout` seconds.
    """
    aiter_obj = aiter.__aiter__()
    while True:
        try:
            token = await asyncio.wait_for(aiter_obj.__anext__(), timeout=timeout)
            yield token
        except StopAsyncIteration:
            break
        except asyncio.TimeoutError:
            raise


def _build_system_prompt(retrieval_results: list[RetrievalResult]) -> str:
    """Build a system prompt for the LLM with source context."""
    sources = []
    for i, result in enumerate(retrieval_results, 1):
        sources.append(
            f"[{i}] {result.title} ({result.url})\n"
            f"    {result.cleaned_text[:200]}..."
        )

    return (
        "You are a research assistant. Answer the user's question based on "
        "the provided sources. Cite sources inline when making claims.\n\n"
        "Sources:\n" + "\n".join(sources)
    )
