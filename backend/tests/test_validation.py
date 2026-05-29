"""Tests for API Gateway request validation models.

Validates bounds checks for all endpoints per:
  - R3.5–R3.7: POST /v1/search
  - R4.5–R4.6: POST /v1/find_similar
  - R5.5–R5.6: POST /v1/contents
  - R7.8: POST /v1/research
  - R10.5: search thresholds
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.api_gateway.validation import (
    ERROR_INVALID_DOCUMENT_ID_COUNT,
    ERROR_INVALID_MODE,
    ERROR_INVALID_NUM_RESULTS,
    ERROR_INVALID_QUERY,
    ERROR_INVALID_RESEARCH_REQUEST,
    ERROR_INVALID_THRESHOLD,
    ERROR_INVALID_URL,
    ERROR_MISSING_HIGHLIGHT_QUERY,
    ContentsRequest,
    FindSimilarRequest,
    RequestValidationError,
    ResearchRequest,
    SearchRequest,
    map_pydantic_error_to_code,
    validate_contents_request,
)


# ---------------------------------------------------------------------------
# SearchRequest Tests (R3.5–R3.7, R10.5)
# ---------------------------------------------------------------------------


class TestSearchRequest:
    """Tests for SearchRequest validation."""

    def test_valid_request(self) -> None:
        """A well-formed search request passes validation."""
        req = SearchRequest(query="hello world", mode="neural", num_results=10)
        assert req.query == "hello world"
        assert req.mode == "neural"
        assert req.num_results == 10

    def test_valid_request_all_modes(self) -> None:
        """All valid modes are accepted."""
        for mode in ("neural", "keyword", "hybrid"):
            req = SearchRequest(query="test", mode=mode)
            assert req.mode == mode

    def test_valid_request_defaults(self) -> None:
        """Default num_results is 10."""
        req = SearchRequest(query="test", mode="neural")
        assert req.num_results == 10

    def test_valid_request_with_thresholds(self) -> None:
        """Thresholds at boundary values are accepted."""
        req = SearchRequest(
            query="test",
            mode="neural",
            min_credibility=0.0,
            max_ai_generated_likelihood=1.0,
        )
        assert req.min_credibility == 0.0
        assert req.max_ai_generated_likelihood == 1.0

    # --- R3.7: invalid_query ---

    def test_empty_query_raises(self) -> None:
        """R3.7: Empty query after trimming raises invalid_query."""
        with pytest.raises(RequestValidationError) as exc_info:
            SearchRequest(query="   ", mode="neural")
        assert exc_info.value.code == ERROR_INVALID_QUERY

    def test_query_exceeds_max_code_points(self) -> None:
        """R3.7: Query exceeding 2048 code points raises invalid_query."""
        long_query = "a" * 2049
        with pytest.raises(RequestValidationError) as exc_info:
            SearchRequest(query=long_query, mode="neural")
        assert exc_info.value.code == ERROR_INVALID_QUERY

    def test_query_at_max_code_points_accepted(self) -> None:
        """R3.7: Query at exactly 2048 code points is accepted."""
        query = "a" * 2048
        req = SearchRequest(query=query, mode="neural")
        assert len(req.query) == 2048

    def test_query_unicode_code_points(self) -> None:
        """R3.7: Unicode characters count as code points (not bytes)."""
        # Each emoji is 1 code point but multiple bytes
        query = "🔍" * 2048
        req = SearchRequest(query=query, mode="neural")
        assert len(req.query) == 2048

    def test_query_unicode_exceeds_limit(self) -> None:
        """R3.7: Unicode query exceeding 2048 code points is rejected."""
        query = "🔍" * 2049
        with pytest.raises(RequestValidationError) as exc_info:
            SearchRequest(query=query, mode="neural")
        assert exc_info.value.code == ERROR_INVALID_QUERY

    # --- R3.6: invalid_mode ---

    def test_invalid_mode_raises(self) -> None:
        """R3.6: Invalid mode raises invalid_mode."""
        with pytest.raises(RequestValidationError) as exc_info:
            SearchRequest(query="test", mode="semantic")
        assert exc_info.value.code == ERROR_INVALID_MODE

    def test_mode_case_sensitive(self) -> None:
        """R3.6: Mode is case-sensitive — 'Neural' is invalid."""
        with pytest.raises(RequestValidationError) as exc_info:
            SearchRequest(query="test", mode="Neural")
        assert exc_info.value.code == ERROR_INVALID_MODE

    # --- R3.5: invalid_num_results ---

    def test_num_results_negative_raises(self) -> None:
        """R3.5: Negative num_results raises invalid_num_results."""
        with pytest.raises(RequestValidationError) as exc_info:
            SearchRequest(query="test", mode="neural", num_results=-1)
        assert exc_info.value.code == ERROR_INVALID_NUM_RESULTS

    def test_num_results_exceeds_100_raises(self) -> None:
        """R3.5: num_results > 100 raises invalid_num_results."""
        with pytest.raises(RequestValidationError) as exc_info:
            SearchRequest(query="test", mode="neural", num_results=101)
        assert exc_info.value.code == ERROR_INVALID_NUM_RESULTS

    def test_num_results_zero_accepted(self) -> None:
        """R3.5: num_results=0 is accepted."""
        req = SearchRequest(query="test", mode="neural", num_results=0)
        assert req.num_results == 0

    def test_num_results_100_accepted(self) -> None:
        """R3.5: num_results=100 is accepted."""
        req = SearchRequest(query="test", mode="neural", num_results=100)
        assert req.num_results == 100

    # --- R10.5: invalid_threshold ---

    def test_min_credibility_below_zero_raises(self) -> None:
        """R10.5: min_credibility < 0 raises invalid_threshold."""
        with pytest.raises(RequestValidationError) as exc_info:
            SearchRequest(query="test", mode="neural", min_credibility=-0.1)
        assert exc_info.value.code == ERROR_INVALID_THRESHOLD

    def test_min_credibility_above_one_raises(self) -> None:
        """R10.5: min_credibility > 1.0 raises invalid_threshold."""
        with pytest.raises(RequestValidationError) as exc_info:
            SearchRequest(query="test", mode="neural", min_credibility=1.1)
        assert exc_info.value.code == ERROR_INVALID_THRESHOLD

    def test_max_ai_generated_below_zero_raises(self) -> None:
        """R10.5: max_ai_generated_likelihood < 0 raises invalid_threshold."""
        with pytest.raises(RequestValidationError) as exc_info:
            SearchRequest(query="test", mode="neural", max_ai_generated_likelihood=-0.01)
        assert exc_info.value.code == ERROR_INVALID_THRESHOLD

    def test_max_ai_generated_above_one_raises(self) -> None:
        """R10.5: max_ai_generated_likelihood > 1.0 raises invalid_threshold."""
        with pytest.raises(RequestValidationError) as exc_info:
            SearchRequest(query="test", mode="neural", max_ai_generated_likelihood=1.5)
        assert exc_info.value.code == ERROR_INVALID_THRESHOLD

    def test_threshold_boundary_zero_accepted(self) -> None:
        """R10.5: Threshold at 0.0 is accepted."""
        req = SearchRequest(query="test", mode="neural", min_credibility=0.0)
        assert req.min_credibility == 0.0

    def test_threshold_boundary_one_accepted(self) -> None:
        """R10.5: Threshold at 1.0 is accepted."""
        req = SearchRequest(query="test", mode="neural", max_ai_generated_likelihood=1.0)
        assert req.max_ai_generated_likelihood == 1.0

    def test_threshold_none_accepted(self) -> None:
        """R10.5: None thresholds are accepted (optional fields)."""
        req = SearchRequest(query="test", mode="neural")
        assert req.min_credibility is None
        assert req.max_ai_generated_likelihood is None


# ---------------------------------------------------------------------------
# FindSimilarRequest Tests (R4.5–R4.6)
# ---------------------------------------------------------------------------


class TestFindSimilarRequest:
    """Tests for FindSimilarRequest validation."""

    def test_valid_request(self) -> None:
        """A well-formed find_similar request passes validation."""
        req = FindSimilarRequest(url="https://example.com/page", num_results=5)
        assert req.url == "https://example.com/page"
        assert req.num_results == 5

    def test_valid_request_defaults(self) -> None:
        """Default num_results is 10."""
        req = FindSimilarRequest(url="https://example.com")
        assert req.num_results == 10

    # --- R4.5: invalid_url ---

    def test_empty_url_raises(self) -> None:
        """R4.5: Empty URL raises invalid_url."""
        with pytest.raises(RequestValidationError) as exc_info:
            FindSimilarRequest(url="")
        assert exc_info.value.code == ERROR_INVALID_URL

    def test_whitespace_only_url_raises(self) -> None:
        """R4.5: Whitespace-only URL raises invalid_url."""
        with pytest.raises(RequestValidationError) as exc_info:
            FindSimilarRequest(url="   ")
        assert exc_info.value.code == ERROR_INVALID_URL

    def test_url_exceeds_max_code_points(self) -> None:
        """R4.5: URL exceeding 2048 code points raises invalid_url."""
        long_url = "https://example.com/" + "a" * 2030
        with pytest.raises(RequestValidationError) as exc_info:
            FindSimilarRequest(url=long_url)
        assert exc_info.value.code == ERROR_INVALID_URL

    def test_url_at_max_code_points_accepted(self) -> None:
        """R4.5: URL at exactly 2048 code points is accepted."""
        # https://example.com/ is 20 chars, pad to 2048
        url = "https://example.com/" + "a" * 2028
        req = FindSimilarRequest(url=url)
        assert len(req.url) == 2048

    def test_syntactically_invalid_url_raises(self) -> None:
        """R4.5: Syntactically invalid URL raises invalid_url."""
        with pytest.raises(RequestValidationError) as exc_info:
            FindSimilarRequest(url="not-a-url")
        assert exc_info.value.code == ERROR_INVALID_URL

    def test_url_missing_scheme_raises(self) -> None:
        """R4.5: URL without scheme raises invalid_url."""
        with pytest.raises(RequestValidationError) as exc_info:
            FindSimilarRequest(url="example.com/page")
        assert exc_info.value.code == ERROR_INVALID_URL

    def test_url_missing_host_raises(self) -> None:
        """R4.5: URL without host raises invalid_url."""
        with pytest.raises(RequestValidationError) as exc_info:
            FindSimilarRequest(url="https://")
        assert exc_info.value.code == ERROR_INVALID_URL

    # --- R4.6: invalid_num_results ---

    def test_num_results_negative_raises(self) -> None:
        """R4.6: Negative num_results raises invalid_num_results."""
        with pytest.raises(RequestValidationError) as exc_info:
            FindSimilarRequest(url="https://example.com", num_results=-1)
        assert exc_info.value.code == ERROR_INVALID_NUM_RESULTS

    def test_num_results_exceeds_100_raises(self) -> None:
        """R4.6: num_results > 100 raises invalid_num_results."""
        with pytest.raises(RequestValidationError) as exc_info:
            FindSimilarRequest(url="https://example.com", num_results=101)
        assert exc_info.value.code == ERROR_INVALID_NUM_RESULTS

    def test_num_results_zero_accepted(self) -> None:
        """R4.6: num_results=0 is accepted."""
        req = FindSimilarRequest(url="https://example.com", num_results=0)
        assert req.num_results == 0

    def test_num_results_100_accepted(self) -> None:
        """R4.6: num_results=100 is accepted."""
        req = FindSimilarRequest(url="https://example.com", num_results=100)
        assert req.num_results == 100


# ---------------------------------------------------------------------------
# ContentsRequest Tests (R5.5–R5.6)
# ---------------------------------------------------------------------------


class TestContentsRequest:
    """Tests for ContentsRequest validation."""

    def test_valid_request(self) -> None:
        """A well-formed contents request passes validation."""
        req = ContentsRequest(document_ids=["doc-1", "doc-2"])
        assert req.document_ids == ["doc-1", "doc-2"]

    def test_valid_request_with_highlights(self) -> None:
        """Contents request with highlights and query passes."""
        req = ContentsRequest(document_ids=["doc-1"], highlights=True, query="search term")
        assert req.highlights is True
        assert req.query == "search term"

    # --- R5.5: invalid_document_id_count ---

    def test_empty_document_ids_raises(self) -> None:
        """R5.5: Empty document_ids list raises invalid_document_id_count."""
        with pytest.raises(RequestValidationError) as exc_info:
            ContentsRequest(document_ids=[])
        assert exc_info.value.code == ERROR_INVALID_DOCUMENT_ID_COUNT

    def test_document_ids_exceeds_100_raises(self) -> None:
        """R5.5: More than 100 document_ids raises invalid_document_id_count."""
        ids = [f"doc-{i}" for i in range(101)]
        with pytest.raises(RequestValidationError) as exc_info:
            ContentsRequest(document_ids=ids)
        assert exc_info.value.code == ERROR_INVALID_DOCUMENT_ID_COUNT

    def test_document_ids_at_100_accepted(self) -> None:
        """R5.5: Exactly 100 document_ids is accepted."""
        ids = [f"doc-{i}" for i in range(100)]
        req = ContentsRequest(document_ids=ids)
        assert len(req.document_ids) == 100

    def test_document_ids_single_accepted(self) -> None:
        """R5.5: A single document_id is accepted."""
        req = ContentsRequest(document_ids=["doc-1"])
        assert len(req.document_ids) == 1

    # --- R5.6: missing_highlight_query (cross-field) ---

    def test_highlights_without_query_raises(self) -> None:
        """R5.6: highlights=true without query raises missing_highlight_query."""
        req = ContentsRequest(document_ids=["doc-1"], highlights=True)
        with pytest.raises(RequestValidationError) as exc_info:
            validate_contents_request(req)
        assert exc_info.value.code == ERROR_MISSING_HIGHLIGHT_QUERY

    def test_highlights_with_empty_query_raises(self) -> None:
        """R5.6: highlights=true with empty query raises missing_highlight_query."""
        req = ContentsRequest(document_ids=["doc-1"], highlights=True, query="   ")
        with pytest.raises(RequestValidationError) as exc_info:
            validate_contents_request(req)
        assert exc_info.value.code == ERROR_MISSING_HIGHLIGHT_QUERY

    def test_highlights_with_valid_query_passes(self) -> None:
        """R5.6: highlights=true with non-empty query passes."""
        req = ContentsRequest(document_ids=["doc-1"], highlights=True, query="search")
        validate_contents_request(req)  # Should not raise

    def test_highlights_false_without_query_passes(self) -> None:
        """R5.6: highlights=false without query passes."""
        req = ContentsRequest(document_ids=["doc-1"], highlights=False)
        validate_contents_request(req)  # Should not raise

    def test_highlights_none_without_query_passes(self) -> None:
        """R5.6: highlights=None without query passes."""
        req = ContentsRequest(document_ids=["doc-1"])
        validate_contents_request(req)  # Should not raise


# ---------------------------------------------------------------------------
# ResearchRequest Tests (R7.8)
# ---------------------------------------------------------------------------


class TestResearchRequest:
    """Tests for ResearchRequest validation."""

    def test_valid_request(self) -> None:
        """A well-formed research request passes validation."""
        req = ResearchRequest(research_goal="Find information about quantum computing")
        assert req.research_goal == "Find information about quantum computing"

    def test_valid_request_with_schema(self) -> None:
        """Research request with valid output_schema passes."""
        schema = {"type": "object", "properties": {"summary": {"type": "string"}}}
        req = ResearchRequest(research_goal="Research topic", output_schema=schema)
        assert req.output_schema == schema

    # --- R7.8: invalid_research_request (goal) ---

    def test_empty_research_goal_raises(self) -> None:
        """R7.8: Empty research_goal raises invalid_research_request."""
        with pytest.raises(RequestValidationError) as exc_info:
            ResearchRequest(research_goal="")
        assert exc_info.value.code == ERROR_INVALID_RESEARCH_REQUEST

    def test_whitespace_only_goal_raises(self) -> None:
        """R7.8: Whitespace-only research_goal raises invalid_research_request."""
        with pytest.raises(RequestValidationError) as exc_info:
            ResearchRequest(research_goal="   ")
        assert exc_info.value.code == ERROR_INVALID_RESEARCH_REQUEST

    def test_goal_exceeds_4096_code_points_raises(self) -> None:
        """R7.8: research_goal exceeding 4096 code points raises invalid_research_request."""
        long_goal = "a" * 4097
        with pytest.raises(RequestValidationError) as exc_info:
            ResearchRequest(research_goal=long_goal)
        assert exc_info.value.code == ERROR_INVALID_RESEARCH_REQUEST

    def test_goal_at_4096_code_points_accepted(self) -> None:
        """R7.8: research_goal at exactly 4096 code points is accepted."""
        goal = "a" * 4096
        req = ResearchRequest(research_goal=goal)
        assert len(req.research_goal) == 4096

    def test_goal_unicode_code_points(self) -> None:
        """R7.8: Unicode characters count as code points."""
        goal = "🔬" * 4096
        req = ResearchRequest(research_goal=goal)
        assert len(req.research_goal) == 4096

    def test_goal_unicode_exceeds_limit(self) -> None:
        """R7.8: Unicode goal exceeding 4096 code points is rejected."""
        goal = "🔬" * 4097
        with pytest.raises(RequestValidationError) as exc_info:
            ResearchRequest(research_goal=goal)
        assert exc_info.value.code == ERROR_INVALID_RESEARCH_REQUEST

    # --- R7.8: invalid_research_request (output_schema) ---

    def test_valid_output_schema_accepted(self) -> None:
        """R7.8: A valid JSON Schema object is accepted."""
        schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "findings": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["title"],
        }
        req = ResearchRequest(research_goal="test", output_schema=schema)
        assert req.output_schema == schema

    def test_none_output_schema_accepted(self) -> None:
        """R7.8: None output_schema is accepted (optional)."""
        req = ResearchRequest(research_goal="test")
        assert req.output_schema is None


# ---------------------------------------------------------------------------
# Error mapping tests
# ---------------------------------------------------------------------------


class TestErrorMapping:
    """Tests for Pydantic error → spec error code mapping."""

    def test_map_query_field_error(self) -> None:
        """Maps query field errors to invalid_query."""
        errors = [{"loc": ("body", "query"), "msg": "Field required", "type": "missing"}]
        result = map_pydantic_error_to_code(errors)
        assert result.code == ERROR_INVALID_QUERY

    def test_map_mode_field_error(self) -> None:
        """Maps mode field errors to invalid_mode."""
        errors = [{"loc": ("body", "mode"), "msg": "Field required", "type": "missing"}]
        result = map_pydantic_error_to_code(errors)
        assert result.code == ERROR_INVALID_MODE

    def test_map_num_results_field_error(self) -> None:
        """Maps num_results field errors to invalid_num_results."""
        errors = [{"loc": ("body", "num_results"), "msg": "Invalid value", "type": "value_error"}]
        result = map_pydantic_error_to_code(errors)
        assert result.code == ERROR_INVALID_NUM_RESULTS

    def test_map_url_field_error(self) -> None:
        """Maps url field errors to invalid_url."""
        errors = [{"loc": ("body", "url"), "msg": "Field required", "type": "missing"}]
        result = map_pydantic_error_to_code(errors)
        assert result.code == ERROR_INVALID_URL

    def test_map_document_ids_field_error(self) -> None:
        """Maps document_ids field errors to invalid_document_id_count."""
        errors = [{"loc": ("body", "document_ids"), "msg": "Field required", "type": "missing"}]
        result = map_pydantic_error_to_code(errors)
        assert result.code == ERROR_INVALID_DOCUMENT_ID_COUNT

    def test_map_research_goal_field_error(self) -> None:
        """Maps research_goal field errors to invalid_research_request."""
        errors = [{"loc": ("body", "research_goal"), "msg": "Field required", "type": "missing"}]
        result = map_pydantic_error_to_code(errors)
        assert result.code == ERROR_INVALID_RESEARCH_REQUEST

    def test_map_threshold_field_error(self) -> None:
        """Maps threshold field errors to invalid_threshold."""
        errors = [{"loc": ("body", "min_credibility"), "msg": "Invalid value", "type": "value_error"}]
        result = map_pydantic_error_to_code(errors)
        assert result.code == ERROR_INVALID_THRESHOLD

    def test_map_empty_errors_fallback(self) -> None:
        """Empty error list falls back to generic validation_error."""
        result = map_pydantic_error_to_code([])
        assert result.code == "validation_error"

    def test_map_unknown_field_fallback(self) -> None:
        """Unknown field falls back to generic validation_error."""
        errors = [{"loc": ("body", "unknown_field"), "msg": "Some error", "type": "value_error"}]
        result = map_pydantic_error_to_code(errors)
        assert result.code == "validation_error"


# ---------------------------------------------------------------------------
# RequestValidationError tests
# ---------------------------------------------------------------------------


class TestRequestValidationError:
    """Tests for the RequestValidationError class."""

    def test_to_response_body(self) -> None:
        """to_response_body returns the uniform error shape."""
        err = RequestValidationError(code="invalid_query", message="query is required")
        body = err.to_response_body()
        assert body == {"error": {"code": "invalid_query", "message": "query is required"}}

    def test_error_is_exception(self) -> None:
        """RequestValidationError is an Exception."""
        err = RequestValidationError(code="test", message="test message")
        assert isinstance(err, Exception)
        assert str(err) == "test message"
