"""Answer Engine — Streaming answer generation with inline citations.

Exports:
- AnswerEngine: Main service for streaming answer generation.
- LLMProvider: Protocol for LLM provider abstraction.
- MockLLMProvider: Test implementation of LLMProvider.
- OpenAIProvider, AnthropicProvider: Placeholder real provider implementations.
- CitationTracker: Citation tracking with referential integrity.
- TokenEvent, CitationEvent, DoneEvent, ErrorEvent: Stream event types.
- AnswerErrorCode: Documented error code enum.
- RetrievalResult: Input document from retrieval set.
- HighlightSpan: Highlight span model.
- extract_highlights: Highlight extraction for /v1/contents.
- generate_summary: Summary generation for /v1/contents.
- validate_highlight_spans: Highlight validation utility.
- validate_summary_tokens: Summary token validation utility.
- estimate_token_count: Token count estimation utility.
"""

from backend.answer_engine.citations import (
    CitationIntegrityError,
    CitationOffsetError,
    CitationTracker,
)
from backend.answer_engine.highlights import (
    MAX_HIGHLIGHTS_PER_DOCUMENT,
    extract_highlights,
    validate_highlight_spans,
)
from backend.answer_engine.models import (
    AnswerErrorCode,
    AnswerEvent,
    CitationEvent,
    DoneEvent,
    ErrorEvent,
    HighlightSpan,
    RetrievalResult,
    TokenEvent,
)
from backend.answer_engine.provider import (
    AnthropicProvider,
    GenerationRequest,
    LLMProvider,
    LLMProviderError,
    MockLLMProvider,
    OpenAIProvider,
    ProviderConfig,
)
from backend.answer_engine.service import (
    AnswerEngine,
    TOKEN_SILENCE_TIMEOUT_SECONDS,
)
from backend.answer_engine.summaries import (
    MAX_SUMMARY_TOKENS,
    MIN_SUMMARY_TOKENS,
    estimate_token_count,
    generate_summary,
    validate_summary_tokens,
)

__all__ = [
    # Service
    "AnswerEngine",
    "TOKEN_SILENCE_TIMEOUT_SECONDS",
    # Provider
    "LLMProvider",
    "MockLLMProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "ProviderConfig",
    "GenerationRequest",
    "LLMProviderError",
    # Citations
    "CitationTracker",
    "CitationIntegrityError",
    "CitationOffsetError",
    # Models
    "AnswerEvent",
    "TokenEvent",
    "CitationEvent",
    "DoneEvent",
    "ErrorEvent",
    "AnswerErrorCode",
    "RetrievalResult",
    "HighlightSpan",
    # Highlights
    "extract_highlights",
    "validate_highlight_spans",
    "MAX_HIGHLIGHTS_PER_DOCUMENT",
    # Summaries
    "generate_summary",
    "validate_summary_tokens",
    "estimate_token_count",
    "MIN_SUMMARY_TOKENS",
    "MAX_SUMMARY_TOKENS",
]
