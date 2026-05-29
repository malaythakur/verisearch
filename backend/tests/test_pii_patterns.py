"""Tests for PII pattern detection.

Validates R15.2: Detect and redact email_address, phone_number (E.164),
us_ssn, eu_national_id, credit_card_number (Luhn-passing PAN).
"""

from __future__ import annotations

import pytest

from backend.pii_redactor.patterns import PIIMatch, detect_pii, luhn_check


# ---------------------------------------------------------------------------
# Luhn algorithm tests
# ---------------------------------------------------------------------------


class TestLuhnCheck:
    """Tests for the Luhn checksum validator."""

    def test_valid_visa(self):
        assert luhn_check("4111111111111111") is True

    def test_valid_mastercard(self):
        assert luhn_check("5500000000000004") is True

    def test_valid_amex(self):
        assert luhn_check("378282246310005") is True

    def test_valid_discover(self):
        assert luhn_check("6011111111111117") is True

    def test_invalid_single_digit_change(self):
        # Change last digit of valid Visa
        assert luhn_check("4111111111111112") is False

    def test_invalid_random_digits(self):
        assert luhn_check("1234567890123456") is False

    def test_empty_string(self):
        assert luhn_check("") is False

    def test_non_digit_string(self):
        assert luhn_check("abcdefghijklmnop") is False

    def test_mixed_content(self):
        assert luhn_check("4111-1111-1111") is False  # contains dashes

    def test_single_zero(self):
        assert luhn_check("0") is True  # 0 mod 10 == 0


# ---------------------------------------------------------------------------
# Email detection tests
# ---------------------------------------------------------------------------


class TestEmailDetection:
    """Tests for email pattern detection."""

    def test_simple_email(self):
        matches = detect_pii("contact user@example.com for info")
        email_matches = [m for m in matches if m.type == "email"]
        assert len(email_matches) == 1
        assert email_matches[0].value == "user@example.com"

    def test_email_with_plus(self):
        matches = detect_pii("send to user+tag@example.com")
        email_matches = [m for m in matches if m.type == "email"]
        assert len(email_matches) == 1
        assert email_matches[0].value == "user+tag@example.com"

    def test_email_with_dots_in_local(self):
        matches = detect_pii("first.last@company.co.uk")
        email_matches = [m for m in matches if m.type == "email"]
        assert len(email_matches) == 1
        assert email_matches[0].value == "first.last@company.co.uk"

    def test_email_with_subdomain(self):
        matches = detect_pii("admin@mail.server.example.org")
        email_matches = [m for m in matches if m.type == "email"]
        assert len(email_matches) == 1
        assert email_matches[0].value == "admin@mail.server.example.org"

    def test_no_match_without_domain(self):
        matches = detect_pii("user@ is not valid")
        email_matches = [m for m in matches if m.type == "email"]
        assert len(email_matches) == 0

    def test_no_match_without_at(self):
        matches = detect_pii("just a plain word")
        email_matches = [m for m in matches if m.type == "email"]
        assert len(email_matches) == 0

    def test_offsets_correct(self):
        text = "hi user@test.com bye"
        matches = detect_pii(text)
        email_matches = [m for m in matches if m.type == "email"]
        assert len(email_matches) == 1
        assert text[email_matches[0].start : email_matches[0].end] == "user@test.com"


# ---------------------------------------------------------------------------
# Phone detection tests
# ---------------------------------------------------------------------------


class TestPhoneDetection:
    """Tests for phone number (E.164) detection."""

    def test_e164_basic(self):
        matches = detect_pii("call +14155551234 now")
        phone_matches = [m for m in matches if m.type == "phone"]
        assert len(phone_matches) == 1
        assert phone_matches[0].value == "+14155551234"

    def test_e164_with_spaces(self):
        matches = detect_pii("call +1 415 555 1234 now")
        phone_matches = [m for m in matches if m.type == "phone"]
        assert len(phone_matches) == 1
        assert phone_matches[0].value == "+1 415 555 1234"

    def test_e164_with_dashes(self):
        matches = detect_pii("call +1-415-555-1234 now")
        phone_matches = [m for m in matches if m.type == "phone"]
        assert len(phone_matches) == 1
        assert phone_matches[0].value == "+1-415-555-1234"

    def test_international_uk(self):
        matches = detect_pii("ring +442071234567")
        phone_matches = [m for m in matches if m.type == "phone"]
        assert len(phone_matches) == 1
        assert phone_matches[0].value == "+442071234567"

    def test_short_number_rejected(self):
        # Less than 7 digits should not match
        matches = detect_pii("code +12345 here")
        phone_matches = [m for m in matches if m.type == "phone"]
        assert len(phone_matches) == 0

    def test_no_plus_no_match(self):
        # Without + prefix, not E.164
        matches = detect_pii("call 14155551234 now")
        phone_matches = [m for m in matches if m.type == "phone"]
        assert len(phone_matches) == 0

    def test_offsets_correct(self):
        text = "ph: +491234567890 end"
        matches = detect_pii(text)
        phone_matches = [m for m in matches if m.type == "phone"]
        assert len(phone_matches) == 1
        assert text[phone_matches[0].start : phone_matches[0].end] == "+491234567890"


# ---------------------------------------------------------------------------
# US SSN detection tests
# ---------------------------------------------------------------------------


class TestUSSSNDetection:
    """Tests for US Social Security Number detection."""

    def test_ssn_with_dashes(self):
        matches = detect_pii("ssn is 123-45-6789")
        ssn_matches = [m for m in matches if m.type == "us_ssn"]
        assert len(ssn_matches) == 1
        assert ssn_matches[0].value == "123-45-6789"

    def test_ssn_without_dashes(self):
        matches = detect_pii("ssn is 123456789")
        ssn_matches = [m for m in matches if m.type == "us_ssn"]
        assert len(ssn_matches) == 1
        assert ssn_matches[0].value == "123456789"

    def test_invalid_ssn_starts_with_000(self):
        matches = detect_pii("ssn 000-12-3456")
        ssn_matches = [m for m in matches if m.type == "us_ssn"]
        assert len(ssn_matches) == 0

    def test_invalid_ssn_starts_with_666(self):
        matches = detect_pii("ssn 666-12-3456")
        ssn_matches = [m for m in matches if m.type == "us_ssn"]
        assert len(ssn_matches) == 0

    def test_invalid_ssn_starts_with_9(self):
        matches = detect_pii("ssn 900-12-3456")
        ssn_matches = [m for m in matches if m.type == "us_ssn"]
        assert len(ssn_matches) == 0

    def test_invalid_ssn_middle_00(self):
        matches = detect_pii("ssn 123-00-6789")
        ssn_matches = [m for m in matches if m.type == "us_ssn"]
        assert len(ssn_matches) == 0

    def test_invalid_ssn_last_0000(self):
        matches = detect_pii("ssn 123-45-0000")
        ssn_matches = [m for m in matches if m.type == "us_ssn"]
        assert len(ssn_matches) == 0

    def test_offsets_correct(self):
        text = "my ssn: 234-56-7890 ok"
        matches = detect_pii(text)
        ssn_matches = [m for m in matches if m.type == "us_ssn"]
        assert len(ssn_matches) == 1
        assert text[ssn_matches[0].start : ssn_matches[0].end] == "234-56-7890"


# ---------------------------------------------------------------------------
# EU National ID detection tests
# ---------------------------------------------------------------------------


class TestEUNationalIDDetection:
    """Tests for EU national ID detection."""

    def test_french_nir(self):
        # French NIR: 1 85 05 78 006 084 42
        matches = detect_pii("NIR: 1 85 05 78 006 084 42")
        eu_matches = [m for m in matches if m.type == "eu_national_id"]
        assert len(eu_matches) == 1

    def test_spanish_dni(self):
        matches = detect_pii("DNI: 12345678Z")
        eu_matches = [m for m in matches if m.type == "eu_national_id"]
        assert len(eu_matches) == 1
        assert eu_matches[0].value == "12345678Z"

    def test_italian_codice_fiscale(self):
        matches = detect_pii("CF: RSSMRA85M01H501Z")
        eu_matches = [m for m in matches if m.type == "eu_national_id"]
        assert len(eu_matches) == 1
        assert eu_matches[0].value == "RSSMRA85M01H501Z"

    def test_german_personalausweis(self):
        # German ID: letter + 8 alphanumeric
        matches = detect_pii("ID: L1234AB8C")
        eu_matches = [m for m in matches if m.type == "eu_national_id"]
        assert len(eu_matches) == 1
        assert eu_matches[0].value == "L1234AB8C"

    def test_no_match_short_string(self):
        matches = detect_pii("ID: ABC")
        eu_matches = [m for m in matches if m.type == "eu_national_id"]
        assert len(eu_matches) == 0


# ---------------------------------------------------------------------------
# Credit card detection tests
# ---------------------------------------------------------------------------


class TestCreditCardDetection:
    """Tests for credit card number detection (Luhn-validated)."""

    def test_visa_plain(self):
        matches = detect_pii("card: 4111111111111111")
        cc_matches = [m for m in matches if m.type == "credit_card"]
        assert len(cc_matches) == 1
        assert cc_matches[0].value == "4111111111111111"

    def test_visa_with_spaces(self):
        matches = detect_pii("card: 4111 1111 1111 1111")
        cc_matches = [m for m in matches if m.type == "credit_card"]
        assert len(cc_matches) == 1
        assert cc_matches[0].value == "4111 1111 1111 1111"

    def test_visa_with_dashes(self):
        matches = detect_pii("card: 4111-1111-1111-1111")
        cc_matches = [m for m in matches if m.type == "credit_card"]
        assert len(cc_matches) == 1
        assert cc_matches[0].value == "4111-1111-1111-1111"

    def test_mastercard(self):
        matches = detect_pii("mc: 5500000000000004")
        cc_matches = [m for m in matches if m.type == "credit_card"]
        assert len(cc_matches) == 1
        assert cc_matches[0].value == "5500000000000004"

    def test_amex_15_digits(self):
        matches = detect_pii("amex: 378282246310005")
        cc_matches = [m for m in matches if m.type == "credit_card"]
        assert len(cc_matches) == 1
        assert cc_matches[0].value == "378282246310005"

    def test_invalid_luhn_rejected(self):
        # This is NOT a valid Luhn number
        matches = detect_pii("card: 4111111111111112")
        cc_matches = [m for m in matches if m.type == "credit_card"]
        assert len(cc_matches) == 0

    def test_too_short_rejected(self):
        # 12 digits - too short
        matches = detect_pii("num: 411111111111")
        cc_matches = [m for m in matches if m.type == "credit_card"]
        assert len(cc_matches) == 0

    def test_offsets_correct(self):
        text = "pay with 4111111111111111 please"
        matches = detect_pii(text)
        cc_matches = [m for m in matches if m.type == "credit_card"]
        assert len(cc_matches) == 1
        assert text[cc_matches[0].start : cc_matches[0].end] == "4111111111111111"


# ---------------------------------------------------------------------------
# Multiple PII types in one text
# ---------------------------------------------------------------------------


class TestMultiplePIIDetection:
    """Tests for detecting multiple PII types in a single text."""

    def test_email_and_phone(self):
        text = "Contact user@example.com or +14155551234"
        matches = detect_pii(text)
        types = {m.type for m in matches}
        assert "email" in types
        assert "phone" in types

    def test_all_types_in_one_text(self):
        text = (
            "Email: test@example.com, "
            "Phone: +442071234567, "
            "SSN: 123-45-6789, "
            "DNI: 12345678Z, "
            "Card: 4111111111111111"
        )
        matches = detect_pii(text)
        types = {m.type for m in matches}
        assert "email" in types
        assert "phone" in types
        assert "us_ssn" in types
        assert "eu_national_id" in types
        assert "credit_card" in types
        assert len(matches) >= 5

    def test_results_sorted_by_offset(self):
        text = "SSN: 123-45-6789 email: a@b.co card: 4111111111111111"
        matches = detect_pii(text)
        offsets = [m.start for m in matches]
        assert offsets == sorted(offsets)

    def test_no_pii_returns_empty(self):
        text = "This is a normal search query about python programming"
        matches = detect_pii(text)
        assert matches == []


# ---------------------------------------------------------------------------
# Offset correctness
# ---------------------------------------------------------------------------


class TestOffsetCorrectness:
    """Tests that start/end offsets correctly index into the original text."""

    def test_all_matches_have_correct_offsets(self):
        text = "hi user@test.com and +14155551234 and 234-56-7890"
        matches = detect_pii(text)
        for match in matches:
            assert text[match.start : match.end] == match.value
            assert match.start >= 0
            assert match.end <= len(text)
            assert match.start < match.end

    def test_unicode_text_offsets(self):
        # Ensure offsets work with unicode characters before PII
        text = "café user@test.com résumé"
        matches = detect_pii(text)
        email_matches = [m for m in matches if m.type == "email"]
        assert len(email_matches) == 1
        assert text[email_matches[0].start : email_matches[0].end] == "user@test.com"
