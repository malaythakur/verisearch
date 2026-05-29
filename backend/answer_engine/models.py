"""Answer Engine data models — event types for streaming answer generation.

Event types:
- TokenEvent: A single token emitted during streaming.
- CitationEvent: A citation linking answer text to a source document.
- DoneEvent: Final event with full answer text and complete citation set.
- ErrorEvent: Terminal error event with a stable error code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Union


class AnswerErrorCode(str, Enum):
    """Documented error codes for /v1/answer streams (R6.5, R6.6)."""

    NO_SOURCES_AVAILABLE = "no_sources_available"
    STREAM_TIMEOUT = "stream_timeout"
    MODEL_ERROR = "model_error"
    INTERNAL_ERROR = "internal_error"
    CLIENT_CANCELLED = "client_cancelled"


@dataclass(frozen=True)
class TokenEvent:
    """A single token emitted during streaming generation."""

    text: str
    index: int


@dataclass(frozen=True)
class CitationEvent:
    """A citation linking a span of answer text to a source document.

    Offset ranges are half-open: [start, end).
    Referential integrity: (document_id, version) must be in the retrieval set (R6.4).
    """

    document_id: str
    version: int
    answer_start: int  # half-open [answer_start, answer_end) into answer text
    answer_end: int
    source_start: int  # half-open [source_start, source_end) into source cleaned text
    source_end: int


@dataclass(frozen=True)
class DoneEvent:
    """Final event emitted on successful completion (R6.3).

    Contains the full answer text (concatenation of all tokens) and
    the complete set of citations emitted during the stream.
    """

    answer: str
    citations: list[CitationEvent] = field(default_factory=list)


@dataclass(frozen=True)
class ErrorEvent:
    """Terminal error event (R6.5, R6.6).

    After this event, no further token or citation events are emitted,
    and the stream closes within 2 seconds.
    """

    code: AnswerErrorCode
    message: str


# Union type for all answer events
AnswerEvent = Union[TokenEvent, CitationEvent, DoneEvent, ErrorEvent]


@dataclass(frozen=True)
class RetrievalResult:
    """A document from the retrieval set used for answer generation."""

    document_id: str
    version: int
    url: str
    title: str
    score: float
    cleaned_text: str


@dataclass(frozen=True)
class HighlightSpan:
    """A highlight span within a document's cleaned text.

    Half-open range: [start, end) satisfying 0 <= start < end <= len(cleaned_text).
    """

    start: int
    end: int
