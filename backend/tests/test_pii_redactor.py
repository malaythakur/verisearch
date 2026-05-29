"""Tests for PII redaction (R15.2).

Validates that redact() replaces PII with type-specific placeholders
and that original PII values never appear in the redacted output.
"""

import pytest

from backend.pii_redactor.redactor import (
    RedactionInfo,
    RedactionResult,
    redact,
    redact_with_info,
)


class TestRedactNoPII:
    """Text with no PII returns unchanged."""

    def test_plain_text_unchanged(self):
        text = "This is a normal search query about python programming"
        assert redact(text) == text

    def test_empty_string(self):
        assert redact("") == ""

    def test_whitespace_only(self):
        text = "   \t\n  "
        assert redact(text) == text


class TestRedactEmail:
    """Email is replaced with [REDACTED_EMAIL]."""

    def test_simple_email(self):
        result = redact("contact user@example.com for info")
        assert "[REDACTED_EMAIL]" in result
        assert "user@example.com" not in result

    def test_email_at_start(self):
        result = redact("admin@test.org is the admin")
        assert result.startswith("[REDACTED_EMAIL]")
        assert "admin@test.org" not in result

    def test_email_at_end(self):
        result = redact("send to hello@world.io")
        assert result.endswith("[REDACTED_EMAIL]")


class TestRedactPhone:
    """Phone is replaced with [REDACTED_PHONE]."""

    def test_e164_phone(self):
        result = redact("call +14155551234 now")
        assert "[REDACTED_PHONE]" in result
        assert "+14155551234" not in result

    def test_phone_with_spaces(self):
        result = redact("call +1 415 555 1234 now")
        assert "[REDACTED_PHONE]" in result
        assert "+1 415 555 1234" not in result

    def test_phone_with_dashes(self):
        result = redact("ring +44-207-123-4567")
        assert "[REDACTED_PHONE]" in result


class TestRedactSSN:
    """SSN is replaced with [REDACTED_SSN]."""

    def test_ssn_with_dashes(self):
        result = redact("ssn is 123-45-6789")
        assert "[REDACTED_SSN]" in result
        assert "123-45-6789" not in result

    def test_ssn_without_dashes(self):
        result = redact("ssn is 123456789")
        assert "[REDACTED_SSN]" in result
        assert "123456789" not in result


class TestRedactEUID:
    """EU ID is replaced with [REDACTED_EU_ID]."""

    def test_spanish_dni(self):
        result = redact("DNI: 12345678Z")
        assert "[REDACTED_EU_ID]" in result
        assert "12345678Z" not in result

    def test_french_nir(self):
        result = redact("NIR: 1 85 05 78 006 084 42")
        assert "[REDACTED_EU_ID]" in result

    def test_italian_codice_fiscale(self):
        result = redact("CF: RSSMRA85M01H501Z")
        assert "[REDACTED_EU_ID]" in result
        assert "RSSMRA85M01H501Z" not in result


class TestRedactCreditCard:
    """Credit card is replaced with [REDACTED_CARD]."""

    def test_visa_plain(self):
        result = redact("card: 4111111111111111")
        assert "[REDACTED_CARD]" in result
        assert "4111111111111111" not in result

    def test_visa_with_spaces(self):
        result = redact("card: 4111 1111 1111 1111")
        assert "[REDACTED_CARD]" in result
        assert "4111 1111 1111 1111" not in result

    def test_visa_with_dashes(self):
        result = redact("card: 4111-1111-1111-1111")
        assert "[REDACTED_CARD]" in result


class TestRedactMultiplePII:
    """Multiple PII types in one text are all redacted."""

    def test_email_and_phone(self):
        text = "Contact user@example.com or +14155551234"
        result = redact(text)
        assert "[REDACTED_EMAIL]" in result
        assert "[REDACTED_PHONE]" in result
        assert "user@example.com" not in result
        assert "+14155551234" not in result

    def test_all_types(self):
        text = (
            "Email: user@test.com "
            "Phone: +14155551234 "
            "SSN: 123-45-6789 "
            "DNI: 12345678Z "
            "Card: 4111111111111111"
        )
        result = redact(text)
        assert "[REDACTED_EMAIL]" in result
        assert "[REDACTED_PHONE]" in result
        assert "[REDACTED_SSN]" in result
        assert "[REDACTED_EU_ID]" in result
        assert "[REDACTED_CARD]" in result


class TestOriginalNeverInOutput:
    """Original PII values do NOT appear in the redacted output."""

    def test_email_value_absent(self):
        result = redact("secret: admin@corp.com")
        assert "admin@corp.com" not in result

    def test_phone_value_absent(self):
        result = redact("ph: +491234567890")
        assert "+491234567890" not in result

    def test_ssn_value_absent(self):
        result = redact("id: 234-56-7890")
        assert "234-56-7890" not in result

    def test_card_value_absent(self):
        result = redact("pay: 5500000000000004")
        assert "5500000000000004" not in result

    def test_multiple_values_all_absent(self):
        text = "a@b.co +14155551234 123-45-6789"
        result = redact(text)
        assert "a@b.co" not in result
        assert "+14155551234" not in result
        assert "123-45-6789" not in result


class TestRedactionResult:
    """RedactionResult has correct metadata."""

    def test_no_pii_result(self):
        result = redact_with_info("hello world")
        assert isinstance(result, RedactionResult)
        assert result.redacted_text == "hello world"
        assert result.redactions == []
        assert result.had_pii is False

    def test_single_email_result(self):
        result = redact_with_info("hi user@test.com bye")
        assert result.had_pii is True
        assert len(result.redactions) == 1
        assert result.redactions[0].type == "email"
        assert result.redactions[0].placeholder == "[REDACTED_EMAIL]"
        assert result.redactions[0].start == 3
        assert result.redactions[0].end == 16
        assert "user@test.com" not in result.redacted_text
        assert "[REDACTED_EMAIL]" in result.redacted_text

    def test_multiple_pii_result(self):
        text = "email: a@b.co phone: +14155551234"
        result = redact_with_info(text)
        assert result.had_pii is True
        assert len(result.redactions) == 2
        types = {r.type for r in result.redactions}
        assert "email" in types
        assert "phone" in types

    def test_redaction_info_fields(self):
        result = redact_with_info("ssn: 123-45-6789")
        assert result.had_pii is True
        info = result.redactions[0]
        assert isinstance(info, RedactionInfo)
        assert info.type == "us_ssn"
        assert info.placeholder == "[REDACTED_SSN]"
        assert info.start == 5
        assert info.end == 16

    def test_redacted_text_matches_redact_function(self):
        text = "Contact user@example.com or +14155551234 about SSN 234-56-7890"
        result = redact_with_info(text)
        assert result.redacted_text == redact(text)
