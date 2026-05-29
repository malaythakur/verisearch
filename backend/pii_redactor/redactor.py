"""PII redaction: replaces detected PII with type-specific placeholders.

Implements R15.2: PII_Redactor detects and redacts PII from queries;
original never crosses to audit/analytics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from backend.pii_redactor.patterns import PIIMatch, detect_pii


# ---------------------------------------------------------------------------
# Type-to-placeholder mapping
# ---------------------------------------------------------------------------

_TYPE_PLACEHOLDER_MAP: dict[str, str] = {
    "email": "[REDACTED_EMAIL]",
    "phone": "[REDACTED_PHONE]",
    "us_ssn": "[REDACTED_SSN]",
    "eu_national_id": "[REDACTED_EU_ID]",
    "credit_card": "[REDACTED_CARD]",
}


def _placeholder_for(pii_type: str) -> str:
    """Return the placeholder string for a given PII type."""
    return _TYPE_PLACEHOLDER_MAP.get(pii_type, f"[REDACTED_{pii_type.upper()}]")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RedactionInfo:
    """Metadata about a single redaction performed on the text.

    Attributes:
        type: The PII type that was redacted.
        start: Original start offset in the source text (inclusive).
        end: Original end offset in the source text (exclusive).
        placeholder: The placeholder string that replaced the PII.
    """

    type: str
    start: int
    end: int
    placeholder: str


@dataclass(frozen=True, slots=True)
class RedactionResult:
    """Result of a redaction operation.

    Attributes:
        redacted_text: The text with all PII replaced by placeholders.
        redactions: Metadata about each redaction performed.
        had_pii: Whether any PII was detected in the original text.
    """

    redacted_text: str
    redactions: list[RedactionInfo] = field(default_factory=list)
    had_pii: bool = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def redact(text: str) -> str:
    """Replace all detected PII in text with type-specific placeholders.

    Calls detect_pii(text) to find all PII matches, then replaces each
    match with a placeholder like [REDACTED_EMAIL], [REDACTED_PHONE], etc.

    Handles overlapping matches by processing from end to start to
    preserve character offsets.

    Args:
        text: The input text that may contain PII.

    Returns:
        The text with all PII replaced by placeholders. If no PII is
        detected, the original text is returned unchanged.
    """
    matches = detect_pii(text)
    if not matches:
        return text

    # Process matches from end to start to preserve offsets
    result = text
    for match in reversed(matches):
        placeholder = _placeholder_for(match.type)
        result = result[: match.start] + placeholder + result[match.end :]

    return result


def redact_with_info(text: str) -> RedactionResult:
    """Redact PII and return both the redacted text and metadata.

    This function is useful for logging/metrics: it provides information
    about what types of PII were found and where, without exposing the
    original PII values.

    Args:
        text: The input text that may contain PII.

    Returns:
        A RedactionResult containing the redacted text, metadata about
        each redaction, and whether any PII was detected.
    """
    matches = detect_pii(text)
    if not matches:
        return RedactionResult(redacted_text=text, redactions=[], had_pii=False)

    # Build redaction info list (preserving original offsets)
    redactions = [
        RedactionInfo(
            type=match.type,
            start=match.start,
            end=match.end,
            placeholder=_placeholder_for(match.type),
        )
        for match in matches
    ]

    # Process matches from end to start to preserve offsets
    result = text
    for match in reversed(matches):
        placeholder = _placeholder_for(match.type)
        result = result[: match.start] + placeholder + result[match.end :]

    return RedactionResult(redacted_text=result, redactions=redactions, had_pii=True)
