"""PII pattern detection for email, phone, US SSN, EU national ID, and credit card.

Implements R15.2: Detect email_address, phone_number (E.164), us_ssn,
eu_national_id, credit_card_number (Luhn-passing PAN) from query text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PIIMatch:
    """A detected PII occurrence in text.

    Attributes:
        type: One of "email", "phone", "us_ssn", "eu_national_id", "credit_card".
        start: Start offset in the text (inclusive).
        end: End offset in the text (exclusive).
        value: The matched text.
    """

    type: str
    start: int
    end: int
    value: str


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Email (simplified RFC 5322): local@domain.tld
# Allows common characters in local part, requires at least one dot in domain.
_EMAIL_PATTERN = re.compile(
    r"[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)+",
)

# Phone (E.164 with common formatting): + followed by 1-15 digits,
# optionally with spaces, dashes, or parentheses for readability.
_PHONE_PATTERN = re.compile(
    r"\+(?:\d[\s\-]?){1,15}\d",
)

# US SSN: XXX-XX-XXXX or XXXXXXXXX (9 digits with optional dashes)
_US_SSN_PATTERN = re.compile(
    r"\b\d{3}-\d{2}-\d{4}\b|\b\d{9}\b",
)

# EU National ID: Simplified patterns for common formats
# - German Personalausweis: L followed by 8 alphanumeric chars (e.g., L1234ABCD)
# - French NIR (social security): 1 or 2 followed by 12 digits (with optional spaces)
# - Spanish DNI: 8 digits followed by a letter
# - Italian Codice Fiscale: 16 alphanumeric chars (6 letters + 2 digits + 1 letter + 2 digits + 1 letter + 3 digits + 1 letter)
# We use a combined pattern that catches common formats.
_EU_NATIONAL_ID_PATTERNS = [
    # French NIR: 1 or 2 + 12 digits (with optional spaces/dashes)
    re.compile(r"\b[12]\s?\d{2}\s?\d{2}\s?\d{2}\s?\d{3}\s?\d{3}\s?\d{2}\b"),
    # Spanish DNI: 8 digits + letter
    re.compile(r"\b\d{8}[A-Z]\b"),
    # German Personalausweis: letter + 8 alphanumeric
    re.compile(r"\b[A-Z][0-9A-Z]{8}\b"),
    # Italian Codice Fiscale: 16 chars (letters and digits in specific pattern)
    re.compile(r"\b[A-Z]{6}\d{2}[A-Z]\d{2}[A-Z]\d{3}[A-Z]\b"),
]

# Credit card: 13-19 digit sequences, optionally separated by spaces or dashes.
# Groups of 4 digits separated by spaces/dashes, or a continuous run of digits.
_CREDIT_CARD_PATTERN = re.compile(
    r"\b(?:\d[\s\-]?){12,18}\d\b",
)


# ---------------------------------------------------------------------------
# Luhn algorithm
# ---------------------------------------------------------------------------


def luhn_check(digits: str) -> bool:
    """Validate a digit string using the Luhn algorithm.

    Args:
        digits: A string containing only digits (no spaces/dashes).

    Returns:
        True if the digit string passes the Luhn checksum, False otherwise.
    """
    if not digits or not digits.isdigit():
        return False

    total = 0
    reverse_digits = digits[::-1]
    for i, ch in enumerate(reverse_digits):
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def _strip_separators(value: str) -> str:
    """Remove spaces and dashes from a string."""
    return value.replace(" ", "").replace("-", "")


def detect_pii(text: str) -> list[PIIMatch]:
    """Scan text for PII patterns and return all matches.

    Detects: email addresses, phone numbers (E.164), US SSNs,
    EU national IDs, and credit card numbers (Luhn-validated).

    Args:
        text: The input text to scan.

    Returns:
        A list of PIIMatch objects sorted by start offset.
    """
    matches: list[PIIMatch] = []

    # Email detection
    for m in _EMAIL_PATTERN.finditer(text):
        matches.append(PIIMatch(type="email", start=m.start(), end=m.end(), value=m.group()))

    # Phone detection
    for m in _PHONE_PATTERN.finditer(text):
        # Verify we have at least 7 digits (minimum for a real phone number)
        digits_only = _strip_separators(m.group()).lstrip("+")
        if len(digits_only) >= 7:
            matches.append(PIIMatch(type="phone", start=m.start(), end=m.end(), value=m.group()))

    # US SSN detection
    for m in _US_SSN_PATTERN.finditer(text):
        # Exclude all-zeros in any group and known invalid SSNs
        digits = _strip_separators(m.group())
        # SSN cannot start with 000, 666, or 9xx; middle cannot be 00; last cannot be 0000
        if (
            len(digits) == 9
            and digits[:3] not in ("000", "666")
            and not digits.startswith("9")
            and digits[3:5] != "00"
            and digits[5:] != "0000"
        ):
            matches.append(PIIMatch(type="us_ssn", start=m.start(), end=m.end(), value=m.group()))

    # EU National ID detection
    for pattern in _EU_NATIONAL_ID_PATTERNS:
        for m in pattern.finditer(text):
            matches.append(PIIMatch(type="eu_national_id", start=m.start(), end=m.end(), value=m.group()))

    # Credit card detection (with Luhn validation)
    for m in _CREDIT_CARD_PATTERN.finditer(text):
        digits = _strip_separators(m.group())
        if 13 <= len(digits) <= 19 and luhn_check(digits):
            matches.append(PIIMatch(type="credit_card", start=m.start(), end=m.end(), value=m.group()))

    # Sort by start offset, then by end offset for stability
    matches.sort(key=lambda x: (x.start, x.end))

    # Remove duplicates (same span detected by multiple patterns)
    seen: set[tuple[int, int]] = set()
    deduplicated: list[PIIMatch] = []
    for match in matches:
        key = (match.start, match.end)
        if key not in seen:
            seen.add(key)
            deduplicated.append(match)

    return deduplicated
