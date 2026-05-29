"""Property-based tests for PII Redactor — PII patterns absent from redacted output (Property 38).

**Validates: Requirements 15.2**

Property 38: PII is redacted from queries before they cross to audit/analytics.
For any text containing one or more PII values (email, phone, US SSN, EU national ID,
credit card), after redaction:
  1. Re-scanning with detect_pii() finds NO matches.
  2. None of the original PII values appear in the redacted output.
  3. The redacted output length accounts for placeholder replacements (no data loss).

Uses Hypothesis to generate valid PII values embedded in random surrounding text.
"""

from __future__ import annotations

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from backend.pii_redactor.patterns import detect_pii, luhn_check
from backend.pii_redactor.redactor import redact


# ---------------------------------------------------------------------------
# Hypothesis strategies for generating valid PII
# ---------------------------------------------------------------------------


@st.composite
def email_addresses(draw: st.DrawFn) -> str:
    """Generate random email addresses in user@domain.tld format."""
    local_chars = st.text(
        alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz0123456789._"),
        min_size=1,
        max_size=20,
    )
    local_part = draw(local_chars)
    # Ensure local part doesn't start/end with a dot
    local_part = local_part.strip(".")
    assume(len(local_part) >= 1)

    domain_label = draw(
        st.text(
            alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz0123456789"),
            min_size=2,
            max_size=10,
        )
    )
    tld = draw(st.sampled_from(["com", "org", "net", "io", "co", "uk"]))
    return f"{local_part}@{domain_label}.{tld}"


@st.composite
def e164_phone_numbers(draw: st.DrawFn) -> str:
    """Generate random E.164 phone numbers (+1XXXXXXXXXX format)."""
    country_code = draw(st.sampled_from(["1", "44", "49", "33", "61"]))
    # Generate enough digits to have at least 7 total (country + subscriber)
    subscriber_len = draw(st.integers(min_value=7, max_value=12))
    subscriber_digits = draw(
        st.text(
            alphabet=st.sampled_from("0123456789"),
            min_size=subscriber_len,
            max_size=subscriber_len,
        )
    )
    return f"+{country_code}{subscriber_digits}"


@st.composite
def us_ssns(draw: st.DrawFn) -> str:
    """Generate random US SSNs in XXX-XX-XXXX format with valid ranges."""
    # Area number: 001-665 or 667-899 (not 000, 666, or 9xx)
    area = draw(st.integers(min_value=1, max_value=899))
    assume(area != 666 and area < 900)

    # Group number: 01-99 (not 00)
    group = draw(st.integers(min_value=1, max_value=99))

    # Serial number: 0001-9999 (not 0000)
    serial = draw(st.integers(min_value=1, max_value=9999))

    return f"{area:03d}-{group:02d}-{serial:04d}"


@st.composite
def credit_card_numbers(draw: st.DrawFn) -> str:
    """Generate random credit card numbers that pass Luhn check."""
    # Generate a 15-digit prefix, then compute the Luhn check digit
    length = draw(st.sampled_from([13, 15, 16, 19]))
    prefix_len = length - 1

    prefix_digits = draw(
        st.text(
            alphabet=st.sampled_from("0123456789"),
            min_size=prefix_len,
            max_size=prefix_len,
        )
    )

    # Compute Luhn check digit
    # The check digit is the digit that makes the full number pass Luhn
    check_digit = _compute_luhn_check_digit(prefix_digits)
    card_number = prefix_digits + str(check_digit)

    # Verify it actually passes Luhn
    assume(luhn_check(card_number))
    return card_number


def _compute_luhn_check_digit(prefix: str) -> int:
    """Compute the Luhn check digit for a given prefix."""
    # Append a 0 as placeholder for check digit, then compute
    digits = prefix + "0"
    total = 0
    reverse_digits = digits[::-1]
    for i, ch in enumerate(reverse_digits):
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    check = (10 - (total % 10)) % 10
    return check


@st.composite
def surrounding_text(draw: st.DrawFn) -> str:
    """Generate random surrounding text that doesn't contain PII-like patterns."""
    # Use simple words to avoid accidentally generating PII
    words = draw(
        st.lists(
            st.text(
                alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz "),
                min_size=1,
                max_size=15,
            ),
            min_size=0,
            max_size=5,
        )
    )
    return " ".join(words)


@st.composite
def text_with_email(draw: st.DrawFn) -> tuple[str, str]:
    """Generate text containing an embedded email address."""
    prefix = draw(surrounding_text())
    email = draw(email_addresses())
    suffix = draw(surrounding_text())
    text = f"{prefix} {email} {suffix}".strip()
    return text, email


@st.composite
def text_with_phone(draw: st.DrawFn) -> tuple[str, str]:
    """Generate text containing an embedded E.164 phone number."""
    prefix = draw(surrounding_text())
    phone = draw(e164_phone_numbers())
    suffix = draw(surrounding_text())
    text = f"{prefix} {phone} {suffix}".strip()
    return text, phone


@st.composite
def text_with_ssn(draw: st.DrawFn) -> tuple[str, str]:
    """Generate text containing an embedded US SSN."""
    prefix = draw(surrounding_text())
    ssn = draw(us_ssns())
    suffix = draw(surrounding_text())
    text = f"{prefix} {ssn} {suffix}".strip()
    return text, ssn


@st.composite
def text_with_credit_card(draw: st.DrawFn) -> tuple[str, str]:
    """Generate text containing an embedded credit card number."""
    prefix = draw(surrounding_text())
    card = draw(credit_card_numbers())
    suffix = draw(surrounding_text())
    text = f"{prefix} {card} {suffix}".strip()
    return text, card


@st.composite
def text_with_mixed_pii(draw: st.DrawFn) -> tuple[str, list[str]]:
    """Generate text containing multiple types of PII."""
    pii_values: list[str] = []
    parts: list[str] = []

    # Always include at least one PII type
    parts.append(draw(surrounding_text()))

    # Include 1-4 PII types
    pii_types = draw(
        st.lists(
            st.sampled_from(["email", "phone", "ssn", "credit_card"]),
            min_size=1,
            max_size=4,
            unique=True,
        )
    )

    for pii_type in pii_types:
        if pii_type == "email":
            val = draw(email_addresses())
        elif pii_type == "phone":
            val = draw(e164_phone_numbers())
        elif pii_type == "ssn":
            val = draw(us_ssns())
        else:
            val = draw(credit_card_numbers())
        pii_values.append(val)
        parts.append(val)
        parts.append(draw(surrounding_text()))

    text = " ".join(parts).strip()
    return text, pii_values


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


class TestProperty38PIIAbsentFromRedactedOutput:
    """Property 38: PII patterns are absent from redacted output.

    **Validates: Requirements 15.2**
    """

    @given(data=text_with_email())
    def test_no_pii_detected_after_redacting_email(self, data: tuple[str, str]):
        """After redacting text with an email, detect_pii finds no matches."""
        text, email = data
        # Confirm PII is actually detected in the original
        original_matches = detect_pii(text)
        assume(len(original_matches) > 0)

        redacted = redact(text)
        rescan_matches = detect_pii(redacted)
        assert rescan_matches == [], (
            f"PII still detected after redaction: {rescan_matches}"
        )

    @given(data=text_with_phone())
    def test_no_pii_detected_after_redacting_phone(self, data: tuple[str, str]):
        """After redacting text with a phone number, detect_pii finds no matches."""
        text, phone = data
        original_matches = detect_pii(text)
        assume(len(original_matches) > 0)

        redacted = redact(text)
        rescan_matches = detect_pii(redacted)
        assert rescan_matches == [], (
            f"PII still detected after redaction: {rescan_matches}"
        )

    @given(data=text_with_ssn())
    def test_no_pii_detected_after_redacting_ssn(self, data: tuple[str, str]):
        """After redacting text with a US SSN, detect_pii finds no matches."""
        text, ssn = data
        original_matches = detect_pii(text)
        assume(len(original_matches) > 0)

        redacted = redact(text)
        rescan_matches = detect_pii(redacted)
        assert rescan_matches == [], (
            f"PII still detected after redaction: {rescan_matches}"
        )

    @given(data=text_with_credit_card())
    def test_no_pii_detected_after_redacting_credit_card(self, data: tuple[str, str]):
        """After redacting text with a credit card, detect_pii finds no matches."""
        text, card = data
        original_matches = detect_pii(text)
        assume(len(original_matches) > 0)

        redacted = redact(text)
        rescan_matches = detect_pii(redacted)
        assert rescan_matches == [], (
            f"PII still detected after redaction: {rescan_matches}"
        )

    @given(data=text_with_mixed_pii())
    def test_no_pii_detected_after_redacting_mixed(self, data: tuple[str, list[str]]):
        """After redacting text with multiple PII types, detect_pii finds no matches."""
        text, pii_values = data
        original_matches = detect_pii(text)
        assume(len(original_matches) > 0)

        redacted = redact(text)
        rescan_matches = detect_pii(redacted)
        assert rescan_matches == [], (
            f"PII still detected after redaction: {rescan_matches}"
        )

    @given(data=text_with_email())
    def test_original_email_absent_from_redacted(self, data: tuple[str, str]):
        """The original email value does not appear in the redacted output."""
        text, email = data
        original_matches = detect_pii(text)
        assume(len(original_matches) > 0)

        redacted = redact(text)
        for match in original_matches:
            assert match.value not in redacted, (
                f"Original PII value '{match.value}' still present in redacted output"
            )

    @given(data=text_with_phone())
    def test_original_phone_absent_from_redacted(self, data: tuple[str, str]):
        """The original phone value does not appear in the redacted output."""
        text, phone = data
        original_matches = detect_pii(text)
        assume(len(original_matches) > 0)

        redacted = redact(text)
        for match in original_matches:
            assert match.value not in redacted, (
                f"Original PII value '{match.value}' still present in redacted output"
            )

    @given(data=text_with_ssn())
    def test_original_ssn_absent_from_redacted(self, data: tuple[str, str]):
        """The original SSN value does not appear in the redacted output."""
        text, ssn = data
        original_matches = detect_pii(text)
        assume(len(original_matches) > 0)

        redacted = redact(text)
        for match in original_matches:
            assert match.value not in redacted, (
                f"Original PII value '{match.value}' still present in redacted output"
            )

    @given(data=text_with_credit_card())
    def test_original_card_absent_from_redacted(self, data: tuple[str, str]):
        """The original credit card value does not appear in the redacted output."""
        text, card = data
        original_matches = detect_pii(text)
        assume(len(original_matches) > 0)

        redacted = redact(text)
        for match in original_matches:
            assert match.value not in redacted, (
                f"Original PII value '{match.value}' still present in redacted output"
            )

    @given(data=text_with_mixed_pii())
    def test_original_values_absent_from_redacted_mixed(self, data: tuple[str, list[str]]):
        """All original PII values are absent from the redacted output (mixed types)."""
        text, pii_values = data
        original_matches = detect_pii(text)
        assume(len(original_matches) > 0)

        redacted = redact(text)
        for match in original_matches:
            assert match.value not in redacted, (
                f"Original PII value '{match.value}' still present in redacted output"
            )

    @given(data=text_with_mixed_pii())
    def test_redacted_length_accounts_for_placeholders(self, data: tuple[str, list[str]]):
        """Redacted output length is consistent with placeholder replacements.

        The redacted output length should equal:
        original_length - sum(len(pii_value)) + sum(len(placeholder))

        This verifies no data is silently lost during redaction.
        """
        text, pii_values = data
        original_matches = detect_pii(text)
        assume(len(original_matches) > 0)

        redacted = redact(text)

        # Compute expected length change
        # Each match is replaced by its placeholder
        from backend.pii_redactor.redactor import _placeholder_for

        total_removed = sum(len(m.value) for m in original_matches)
        total_added = sum(len(_placeholder_for(m.type)) for m in original_matches)
        expected_length = len(text) - total_removed + total_added

        assert len(redacted) == expected_length, (
            f"Length mismatch: expected {expected_length}, got {len(redacted)}. "
            f"Original length: {len(text)}, removed: {total_removed}, added: {total_added}"
        )
