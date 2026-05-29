"""Tests for the S3-compatible object store layout and key management."""

import pytest

from backend.storage.object_store import DocumentObjectStore


class TestBuildKey:
    """Tests for DocumentObjectStore.build_key()."""

    def test_tenant_scoped_key(self):
        """Tenant-scoped documents use tenant_id as the first path segment."""
        key = DocumentObjectStore.build_key("tenant-abc", "doc-123", 1)
        assert key == "tenant-abc/doc-123/1/cleaned.txt"

    def test_global_key_uses_global_prefix(self):
        """Global documents (None tenant_id) use _global as the prefix."""
        key = DocumentObjectStore.build_key(None, "doc-456", 3)
        assert key == "_global/doc-456/3/cleaned.txt"

    def test_version_in_path(self):
        """Version number appears as a path segment."""
        key = DocumentObjectStore.build_key("t1", "d1", 42)
        assert "/42/" in key

    def test_uuid_style_ids(self):
        """UUID-style IDs are handled correctly."""
        tenant = "550e8400-e29b-41d4-a716-446655440000"
        doc = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
        key = DocumentObjectStore.build_key(tenant, doc, 1)
        assert key == f"{tenant}/{doc}/1/cleaned.txt"

    def test_empty_document_id_raises(self):
        """Empty document_id raises ValueError."""
        with pytest.raises(ValueError, match="document_id must not be empty"):
            DocumentObjectStore.build_key("tenant", "", 1)

    def test_version_zero_raises(self):
        """Version 0 raises ValueError (versions start at 1)."""
        with pytest.raises(ValueError, match="version must be >= 1"):
            DocumentObjectStore.build_key("tenant", "doc", 0)

    def test_negative_version_raises(self):
        """Negative version raises ValueError."""
        with pytest.raises(ValueError, match="version must be >= 1"):
            DocumentObjectStore.build_key("tenant", "doc", -1)

    def test_special_characters_in_tenant_id(self):
        """Special characters in tenant_id are preserved (S3 allows most chars)."""
        key = DocumentObjectStore.build_key("org_test-123", "doc-1", 1)
        assert key == "org_test-123/doc-1/1/cleaned.txt"

    def test_special_characters_in_document_id(self):
        """Special characters in document_id are preserved."""
        key = DocumentObjectStore.build_key("t1", "doc_with-special.chars", 1)
        assert key == "t1/doc_with-special.chars/1/cleaned.txt"

    def test_large_version_number(self):
        """Large version numbers are handled correctly."""
        key = DocumentObjectStore.build_key("t1", "d1", 999999)
        assert key == "t1/d1/999999/cleaned.txt"


class TestParseKey:
    """Tests for DocumentObjectStore.parse_key()."""

    def test_parse_tenant_scoped_key(self):
        """Parsing a tenant-scoped key returns the correct components."""
        tenant_id, doc_id, version = DocumentObjectStore.parse_key(
            "tenant-abc/doc-123/1/cleaned.txt"
        )
        assert tenant_id == "tenant-abc"
        assert doc_id == "doc-123"
        assert version == 1

    def test_parse_global_key(self):
        """Parsing a global key returns None for tenant_id."""
        tenant_id, doc_id, version = DocumentObjectStore.parse_key(
            "_global/doc-456/3/cleaned.txt"
        )
        assert tenant_id is None
        assert doc_id == "doc-456"
        assert version == 3

    def test_parse_uuid_style_key(self):
        """UUID-style IDs are parsed correctly."""
        tenant = "550e8400-e29b-41d4-a716-446655440000"
        doc = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
        key = f"{tenant}/{doc}/7/cleaned.txt"
        tenant_id, doc_id, version = DocumentObjectStore.parse_key(key)
        assert tenant_id == tenant
        assert doc_id == doc
        assert version == 7

    def test_parse_invalid_segment_count_raises(self):
        """Keys with wrong number of segments raise ValueError."""
        with pytest.raises(ValueError, match="expected 4 path segments"):
            DocumentObjectStore.parse_key("too/few/segments")

    def test_parse_too_many_segments_raises(self):
        """Keys with too many segments raise ValueError."""
        with pytest.raises(ValueError, match="expected 4 path segments"):
            DocumentObjectStore.parse_key("a/b/c/d/e")

    def test_parse_wrong_filename_raises(self):
        """Keys with wrong filename raise ValueError."""
        with pytest.raises(ValueError, match="expected filename 'cleaned.txt'"):
            DocumentObjectStore.parse_key("tenant/doc/1/wrong.txt")

    def test_parse_non_integer_version_raises(self):
        """Keys with non-integer version raise ValueError."""
        with pytest.raises(ValueError, match="not a valid integer"):
            DocumentObjectStore.parse_key("tenant/doc/abc/cleaned.txt")

    def test_parse_zero_version_raises(self):
        """Keys with version 0 raise ValueError."""
        with pytest.raises(ValueError, match="version must be >= 1"):
            DocumentObjectStore.parse_key("tenant/doc/0/cleaned.txt")

    def test_parse_negative_version_raises(self):
        """Keys with negative version raise ValueError."""
        with pytest.raises(ValueError, match="version must be >= 1"):
            DocumentObjectStore.parse_key("tenant/doc/-1/cleaned.txt")

    def test_parse_empty_document_id_raises(self):
        """Keys with empty document_id segment raise ValueError."""
        with pytest.raises(ValueError, match="document_id segment is empty"):
            DocumentObjectStore.parse_key("tenant//1/cleaned.txt")


class TestRoundTrip:
    """Tests verifying build_key and parse_key are inverses."""

    def test_round_trip_tenant_scoped(self):
        """build_key → parse_key round-trip preserves tenant-scoped components."""
        original = ("my-tenant", "my-doc-id", 5)
        key = DocumentObjectStore.build_key(*original)
        result = DocumentObjectStore.parse_key(key)
        assert result == original

    def test_round_trip_global(self):
        """build_key → parse_key round-trip preserves global (None tenant) components."""
        original = (None, "global-doc", 12)
        key = DocumentObjectStore.build_key(*original)
        result = DocumentObjectStore.parse_key(key)
        assert result == original

    def test_round_trip_uuid_ids(self):
        """Round-trip works with UUID-style identifiers."""
        original = (
            "550e8400-e29b-41d4-a716-446655440000",
            "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
            99,
        )
        key = DocumentObjectStore.build_key(*original)
        result = DocumentObjectStore.parse_key(key)
        assert result == original

    def test_round_trip_special_characters(self):
        """Round-trip works with special characters in IDs."""
        original = ("org_test-123.corp", "doc_v2-final.rev", 1)
        key = DocumentObjectStore.build_key(*original)
        result = DocumentObjectStore.parse_key(key)
        assert result == original

    def test_round_trip_large_version(self):
        """Round-trip works with large version numbers."""
        original = ("t", "d", 1000000)
        key = DocumentObjectStore.build_key(*original)
        result = DocumentObjectStore.parse_key(key)
        assert result == original


class TestKeyFormat:
    """Tests verifying the exact key format conventions."""

    def test_key_ends_with_cleaned_txt(self):
        """All keys end with /cleaned.txt."""
        key = DocumentObjectStore.build_key("t", "d", 1)
        assert key.endswith("/cleaned.txt")

    def test_key_has_exactly_three_slashes(self):
        """Keys have exactly 3 forward slashes (4 segments)."""
        key = DocumentObjectStore.build_key("tenant", "doc", 1)
        assert key.count("/") == 3

    def test_global_prefix_is_underscore_global(self):
        """The global prefix is literally '_global'."""
        key = DocumentObjectStore.build_key(None, "doc", 1)
        assert key.startswith("_global/")

    def test_tenant_prefix_is_tenant_id(self):
        """Tenant-scoped keys start with the tenant_id."""
        key = DocumentObjectStore.build_key("my-tenant", "doc", 1)
        assert key.startswith("my-tenant/")


class TestUriBuilding:
    """Tests for URI construction and parsing (requires Settings)."""

    def test_build_uri_format(self):
        """The URI follows s3://bucket/key format."""
        store = DocumentObjectStore()
        uri = store._build_uri("tenant/doc/1/cleaned.txt")
        assert uri == "s3://agentic-research-docs/tenant/doc/1/cleaned.txt"

    def test_parse_uri_extracts_key(self):
        """Parsing a valid URI extracts the key."""
        store = DocumentObjectStore()
        key = store._parse_uri("s3://agentic-research-docs/tenant/doc/1/cleaned.txt")
        assert key == "tenant/doc/1/cleaned.txt"

    def test_parse_uri_invalid_prefix_raises(self):
        """Parsing a URI with wrong bucket raises ValueError."""
        store = DocumentObjectStore()
        with pytest.raises(ValueError, match="Invalid URI"):
            store._parse_uri("s3://wrong-bucket/tenant/doc/1/cleaned.txt")

    def test_parse_uri_non_s3_scheme_raises(self):
        """Parsing a non-s3 URI raises ValueError."""
        store = DocumentObjectStore()
        with pytest.raises(ValueError, match="Invalid URI"):
            store._parse_uri("https://example.com/tenant/doc/1/cleaned.txt")
