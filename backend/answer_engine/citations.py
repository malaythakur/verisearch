"""Citation tracking and referential integrity for the Answer Engine.

Implements:
- Referential integrity check: (document_id, version) must be in retrieval set (R6.4).
- Offset range tracking: answer_start, answer_end, source_start, source_end (R6.2).
- Citation emission within 500ms of supported span.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from backend.answer_engine.models import CitationEvent, RetrievalResult


class CitationIntegrityError(Exception):
    """Raised when a citation references a document not in the retrieval set."""

    def __init__(self, document_id: str, version: int):
        self.document_id = document_id
        self.version = version
        super().__init__(
            f"Citation references (document_id={document_id}, version={version}) "
            f"not in retrieval result set"
        )


class CitationOffsetError(Exception):
    """Raised when citation offsets are invalid."""

    def __init__(self, message: str):
        super().__init__(message)


@dataclass
class CitationTracker:
    """Tracks citations during answer generation and enforces integrity.

    Ensures:
    - Every cited (document_id, version) is in the retrieval result set (R6.4).
    - Offset ranges are valid half-open ranges (R6.2).
    - Answer offsets are within the answer text emitted so far.
    - Source offsets are within the source document's cleaned text.
    """

    retrieval_set: dict[tuple[str, int], RetrievalResult] = field(default_factory=dict)
    citations: list[CitationEvent] = field(default_factory=list)
    answer_text_length: int = 0

    @classmethod
    def from_results(cls, results: list[RetrievalResult]) -> "CitationTracker":
        """Create a tracker from a list of retrieval results."""
        retrieval_set = {
            (r.document_id, r.version): r for r in results
        }
        return cls(retrieval_set=retrieval_set)

    def update_answer_length(self, new_length: int) -> None:
        """Update the current answer text length as tokens are emitted."""
        self.answer_text_length = new_length

    def validate_and_add_citation(
        self,
        document_id: str,
        version: int,
        answer_start: int,
        answer_end: int,
        source_start: int,
        source_end: int,
    ) -> CitationEvent:
        """Validate a citation and add it to the tracked set.

        Args:
            document_id: The cited document's ID.
            version: The cited document's version.
            answer_start: Start offset in answer text (inclusive).
            answer_end: End offset in answer text (exclusive).
            source_start: Start offset in source cleaned text (inclusive).
            source_end: End offset in source cleaned text (exclusive).

        Returns:
            The validated CitationEvent.

        Raises:
            CitationIntegrityError: If (document_id, version) not in retrieval set.
            CitationOffsetError: If offsets are invalid.
        """
        # Check referential integrity (R6.4)
        key = (document_id, version)
        if key not in self.retrieval_set:
            raise CitationIntegrityError(document_id, version)

        result = self.retrieval_set[key]

        # Validate answer offsets: 0 <= answer_start < answer_end <= answer_text_length
        if answer_start < 0:
            raise CitationOffsetError(
                f"answer_start ({answer_start}) must be >= 0"
            )
        if answer_end <= answer_start:
            raise CitationOffsetError(
                f"answer_end ({answer_end}) must be > answer_start ({answer_start})"
            )
        if answer_end > self.answer_text_length:
            raise CitationOffsetError(
                f"answer_end ({answer_end}) must be <= answer text length ({self.answer_text_length})"
            )

        # Validate source offsets: 0 <= source_start < source_end <= len(cleaned_text)
        source_length = len(result.cleaned_text)
        if source_start < 0:
            raise CitationOffsetError(
                f"source_start ({source_start}) must be >= 0"
            )
        if source_end <= source_start:
            raise CitationOffsetError(
                f"source_end ({source_end}) must be > source_start ({source_start})"
            )
        if source_end > source_length:
            raise CitationOffsetError(
                f"source_end ({source_end}) must be <= source text length ({source_length})"
            )

        citation = CitationEvent(
            document_id=document_id,
            version=version,
            answer_start=answer_start,
            answer_end=answer_end,
            source_start=source_start,
            source_end=source_end,
        )
        self.citations.append(citation)
        return citation

    def get_all_citations(self) -> list[CitationEvent]:
        """Return all tracked citations."""
        return list(self.citations)

    def has_document(self, document_id: str, version: int) -> bool:
        """Check if a (document_id, version) pair is in the retrieval set."""
        return (document_id, version) in self.retrieval_set
