"""Property-based tests for API Gateway validation — bounds-violation rejection (Property 8).

**Validates: Requirements 3.5, 3.6, 3.7, 4.5, 4.6, 5.5, 7.8, 10.5**

Property 8: Bounds-violation requests are rejected with correct codes, downstream not invoked.

For any randomly generated invalid input that violates the documented bounds of an endpoint,
the validation layer:
  1. Raises RequestValidationError (not a generic exception)
  2. The error code matches the expected stable code for that violation type
  3. No downstream service is invoked (validation happens at the model layer before any
     business logic — Pydantic field validators raise before the model is fully constructed)

Uses Hypothesis to generate values just outside the valid bounds (boundary testing).
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from backend.api_gateway.validation import (
    ERROR_INVALID_DOCUMENT_ID_COUNT,
    ERROR_INVALID_MODE,
    ERROR_INVALID_NUM_RESULTS,
    ERROR_INVALID_QUERY,
    ERROR_INVALID_RESEARCH_REQUEST,
    ERROR_INVALID_THRESHOLD,
    ERROR_INVALID_URL,
    ContentsRequest,
    FindSimilarRequest,
    RequestValidationError,
    ResearchRequest,
    SearchRequest,
)


# ---------------------------------------------------------------------------
# Strategies for generating invalid inputs
# ---------------------------------------------------------------------------

# num_results out of [0, 100] — negative or > 100
invalid_num_results_strategy = st.one_of(
    st.integers(max_value=-1),  # negative: -1, -100, etc.
    st.integers(min_value=101),  # too large: 101, 1000, etc.
)

# Query too long (> 2048 code points) — use a simple repeated character approach
# to avoid Hypothesis max-size limitations on st.text
query_too_long_strategy = st.integers(min_value=2049, max_value=10000).map(
    lambda n: "a" * n
)

# Invalid search mode — any string not in {"neural", "keyword", "hybrid"}
invalid_mode_strategy = st.text(min_size=1, max_size=50).filter(
    lambda s: s not in {"neural", "keyword", "hybrid"}
)

# Thresholds out of [0.0, 1.0]
invalid_threshold_below_strategy = st.floats(
    max_value=-0.001, min_value=-100.0, allow_nan=False, allow_infinity=False
)
invalid_threshold_above_strategy = st.floats(
    min_value=1.001, max_value=100.0, allow_nan=False, allow_infinity=False
)
invalid_threshold_strategy = st.one_of(
    invalid_threshold_below_strategy,
    invalid_threshold_above_strategy,
)

# URL too long (> 2048 code points)
url_too_long_strategy = st.integers(min_value=2049, max_value=5000).map(
    lambda n: "https://example.com/" + "a" * (n - len("https://example.com/"))
)

# Invalid URL format — missing scheme or host
invalid_url_format_strategy = st.one_of(
    # No scheme
    st.text(
        alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters=".-/"),
        min_size=3,
        max_size=100,
    ).map(lambda s: s if "://" not in s else s.replace("://", "")),
    # Just a path
    st.just("not-a-url"),
    st.just("/just/a/path"),
    st.just("example.com/page"),
)

# document_ids count out of [1, 100]
invalid_doc_ids_empty_strategy = st.just([])
invalid_doc_ids_too_many_strategy = st.integers(min_value=101, max_value=200).map(
    lambda n: [f"doc-{i}" for i in range(n)]
)

# Research goal too long (> 4096 code points) — use a simple repeated character approach
research_goal_too_long_strategy = st.integers(min_value=4097, max_value=10000).map(
    lambda n: "x" * n
)

# Research goal empty or whitespace-only
research_goal_empty_strategy = st.one_of(
    st.just(""),
    st.text(alphabet=st.just(" "), min_size=1, max_size=50),  # whitespace only
    st.text(alphabet=st.sampled_from([" ", "\t", "\n", "\r"]), min_size=1, max_size=20),
)


# ---------------------------------------------------------------------------
# Property 8: SearchRequest — query too long
# ---------------------------------------------------------------------------


class TestSearchRequestBoundsViolation:
    """Property tests for SearchRequest bounds violations.

    **Validates: Requirements 3.5, 3.6, 3.7, 10.5**
    """

    @given(query=query_too_long_strategy)
    @settings(suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example])
    def test_query_too_long_rejected_with_correct_code(self, query: str) -> None:
        """Property: Any query exceeding 2048 code points is rejected with invalid_query.

        **Validates: Requirements 3.7**

        No downstream service can be invoked because the Pydantic model raises
        RequestValidationError during field validation, before the model is constructed.
        """
        assert len(query) > 2048, "Strategy must generate queries > 2048 code points"

        with pytest.raises(RequestValidationError) as exc_info:
            SearchRequest(query=query, mode="neural", num_results=10)

        assert exc_info.value.code == ERROR_INVALID_QUERY

    @given(mode=invalid_mode_strategy)
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_invalid_mode_rejected_with_correct_code(self, mode: str) -> None:
        """Property: Any mode not in {neural, keyword, hybrid} is rejected with invalid_mode.

        **Validates: Requirements 3.6**
        """
        with pytest.raises(RequestValidationError) as exc_info:
            SearchRequest(query="valid query", mode=mode, num_results=10)

        assert exc_info.value.code == ERROR_INVALID_MODE

    @given(num_results=invalid_num_results_strategy)
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_num_results_out_of_range_rejected_with_correct_code(
        self, num_results: int
    ) -> None:
        """Property: Any num_results outside [0, 100] is rejected with invalid_num_results.

        **Validates: Requirements 3.5**
        """
        with pytest.raises(RequestValidationError) as exc_info:
            SearchRequest(query="valid query", mode="neural", num_results=num_results)

        assert exc_info.value.code == ERROR_INVALID_NUM_RESULTS

    @given(threshold=invalid_threshold_strategy)
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_min_credibility_out_of_range_rejected_with_correct_code(
        self, threshold: float
    ) -> None:
        """Property: Any min_credibility outside [0.0, 1.0] is rejected with invalid_threshold.

        **Validates: Requirements 10.5**
        """
        with pytest.raises(RequestValidationError) as exc_info:
            SearchRequest(
                query="valid query",
                mode="neural",
                num_results=10,
                min_credibility=threshold,
            )

        assert exc_info.value.code == ERROR_INVALID_THRESHOLD

    @given(threshold=invalid_threshold_strategy)
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_max_ai_generated_likelihood_out_of_range_rejected_with_correct_code(
        self, threshold: float
    ) -> None:
        """Property: Any max_ai_generated_likelihood outside [0.0, 1.0] is rejected with invalid_threshold.

        **Validates: Requirements 10.5**
        """
        with pytest.raises(RequestValidationError) as exc_info:
            SearchRequest(
                query="valid query",
                mode="neural",
                num_results=10,
                max_ai_generated_likelihood=threshold,
            )

        assert exc_info.value.code == ERROR_INVALID_THRESHOLD


# ---------------------------------------------------------------------------
# Property 8: FindSimilarRequest — URL and num_results bounds
# ---------------------------------------------------------------------------


class TestFindSimilarRequestBoundsViolation:
    """Property tests for FindSimilarRequest bounds violations.

    **Validates: Requirements 4.5, 4.6**
    """

    @given(url=url_too_long_strategy)
    @settings(suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example])
    def test_url_too_long_rejected_with_correct_code(self, url: str) -> None:
        """Property: Any URL exceeding 2048 code points is rejected with invalid_url.

        **Validates: Requirements 4.5**
        """
        assert len(url) > 2048, "Strategy must generate URLs > 2048 code points"

        with pytest.raises(RequestValidationError) as exc_info:
            FindSimilarRequest(url=url, num_results=10)

        assert exc_info.value.code == ERROR_INVALID_URL

    @given(url=invalid_url_format_strategy)
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_invalid_url_format_rejected_with_correct_code(self, url: str) -> None:
        """Property: Any URL without valid scheme+host is rejected with invalid_url.

        **Validates: Requirements 4.5**
        """
        with pytest.raises(RequestValidationError) as exc_info:
            FindSimilarRequest(url=url, num_results=10)

        assert exc_info.value.code == ERROR_INVALID_URL

    @given(num_results=invalid_num_results_strategy)
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_num_results_out_of_range_rejected_with_correct_code(
        self, num_results: int
    ) -> None:
        """Property: Any num_results outside [0, 100] is rejected with invalid_num_results.

        **Validates: Requirements 4.6**
        """
        with pytest.raises(RequestValidationError) as exc_info:
            FindSimilarRequest(
                url="https://example.com/page", num_results=num_results
            )

        assert exc_info.value.code == ERROR_INVALID_NUM_RESULTS


# ---------------------------------------------------------------------------
# Property 8: ContentsRequest — document_ids count bounds
# ---------------------------------------------------------------------------


class TestContentsRequestBoundsViolation:
    """Property tests for ContentsRequest bounds violations.

    **Validates: Requirements 5.5**
    """

    @given(doc_ids=invalid_doc_ids_too_many_strategy)
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_too_many_document_ids_rejected_with_correct_code(
        self, doc_ids: list[str]
    ) -> None:
        """Property: Any document_ids list with > 100 items is rejected with invalid_document_id_count.

        **Validates: Requirements 5.5**
        """
        assert len(doc_ids) > 100, "Strategy must generate lists > 100 items"

        with pytest.raises(RequestValidationError) as exc_info:
            ContentsRequest(document_ids=doc_ids)

        assert exc_info.value.code == ERROR_INVALID_DOCUMENT_ID_COUNT

    def test_empty_document_ids_rejected_with_correct_code(self) -> None:
        """Property: An empty document_ids list is rejected with invalid_document_id_count.

        **Validates: Requirements 5.5**
        """
        with pytest.raises(RequestValidationError) as exc_info:
            ContentsRequest(document_ids=[])

        assert exc_info.value.code == ERROR_INVALID_DOCUMENT_ID_COUNT


# ---------------------------------------------------------------------------
# Property 8: ResearchRequest — research_goal bounds
# ---------------------------------------------------------------------------


class TestResearchRequestBoundsViolation:
    """Property tests for ResearchRequest bounds violations.

    **Validates: Requirements 7.8**
    """

    @given(goal=research_goal_too_long_strategy)
    @settings(suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example])
    def test_research_goal_too_long_rejected_with_correct_code(
        self, goal: str
    ) -> None:
        """Property: Any research_goal exceeding 4096 code points is rejected with invalid_research_request.

        **Validates: Requirements 7.8**
        """
        assert len(goal) > 4096, "Strategy must generate goals > 4096 code points"

        with pytest.raises(RequestValidationError) as exc_info:
            ResearchRequest(research_goal=goal)

        assert exc_info.value.code == ERROR_INVALID_RESEARCH_REQUEST

    @given(goal=research_goal_empty_strategy)
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_research_goal_empty_rejected_with_correct_code(self, goal: str) -> None:
        """Property: Any empty or whitespace-only research_goal is rejected with invalid_research_request.

        **Validates: Requirements 7.8**
        """
        with pytest.raises(RequestValidationError) as exc_info:
            ResearchRequest(research_goal=goal)

        assert exc_info.value.code == ERROR_INVALID_RESEARCH_REQUEST


# ---------------------------------------------------------------------------
# Property 8: Downstream not invoked (structural argument)
# ---------------------------------------------------------------------------


class TestDownstreamNotInvoked:
    """Verify that validation rejects before any downstream logic can execute.

    **Validates: Requirements 3.5, 3.6, 3.7, 4.5, 4.6, 5.5, 7.8, 10.5**

    The Pydantic models use @field_validator decorators that raise RequestValidationError
    during model construction. Since the model is never fully constructed on validation
    failure, no downstream service method can be called. This is a structural property
    of the validation layer design.

    These tests confirm that:
    1. The exception type is always RequestValidationError (not ValueError, TypeError, etc.)
    2. The exception is raised during model construction (not after)
    """

    @given(
        num_results=invalid_num_results_strategy,
        threshold=invalid_threshold_strategy,
    )
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_validation_raises_only_request_validation_error(
        self, num_results: int, threshold: float
    ) -> None:
        """Property: Bounds violations always raise RequestValidationError, never a generic exception.

        **Validates: Requirements 3.5, 10.5**

        This confirms the validation layer produces typed, spec-defined errors that
        can be mapped to HTTP responses without leaking implementation details.
        """
        # Test num_results violation
        try:
            SearchRequest(query="valid", mode="neural", num_results=num_results)
            # If we get here, the model was constructed — this should not happen
            pytest.fail(
                f"SearchRequest accepted invalid num_results={num_results}"
            )
        except RequestValidationError:
            pass  # Expected
        except Exception as e:
            pytest.fail(
                f"Expected RequestValidationError but got {type(e).__name__}: {e}"
            )

        # Test threshold violation
        try:
            SearchRequest(
                query="valid",
                mode="neural",
                num_results=10,
                min_credibility=threshold,
            )
            pytest.fail(
                f"SearchRequest accepted invalid min_credibility={threshold}"
            )
        except RequestValidationError:
            pass  # Expected
        except Exception as e:
            pytest.fail(
                f"Expected RequestValidationError but got {type(e).__name__}: {e}"
            )

    @given(url=url_too_long_strategy)
    @settings(suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example])
    def test_find_similar_validation_raises_only_request_validation_error(
        self, url: str
    ) -> None:
        """Property: FindSimilarRequest bounds violations always raise RequestValidationError.

        **Validates: Requirements 4.5**
        """
        try:
            FindSimilarRequest(url=url, num_results=10)
            pytest.fail(f"FindSimilarRequest accepted invalid url of length {len(url)}")
        except RequestValidationError:
            pass  # Expected
        except Exception as e:
            pytest.fail(
                f"Expected RequestValidationError but got {type(e).__name__}: {e}"
            )

    @given(goal=research_goal_too_long_strategy)
    @settings(suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example])
    def test_research_validation_raises_only_request_validation_error(
        self, goal: str
    ) -> None:
        """Property: ResearchRequest bounds violations always raise RequestValidationError.

        **Validates: Requirements 7.8**
        """
        try:
            ResearchRequest(research_goal=goal)
            pytest.fail(
                f"ResearchRequest accepted invalid goal of length {len(goal)}"
            )
        except RequestValidationError:
            pass  # Expected
        except Exception as e:
            pytest.fail(
                f"Expected RequestValidationError but got {type(e).__name__}: {e}"
            )
