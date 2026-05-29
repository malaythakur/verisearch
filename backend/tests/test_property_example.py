"""Example property-based test using Hypothesis to verify the setup works."""

from hypothesis import given
from hypothesis import strategies as st


@given(st.integers(min_value=0, max_value=100))
def test_score_bounds_property(score_pct: int):
    """Property: normalized scores are always in [0.0, 1.0].

    This is a simple example demonstrating Hypothesis property-based testing
    works correctly in this project. It models the score normalization that
    will be used throughout the search engine.
    """
    normalized = score_pct / 100.0
    assert 0.0 <= normalized <= 1.0


@given(st.text(min_size=1, max_size=100))
def test_string_roundtrip_property(s: str):
    """Property: encoding and decoding a string is a round-trip.

    Demonstrates Hypothesis string generation and round-trip testing,
    which will be used extensively for the Query Filter DSL.
    """
    encoded = s.encode("utf-8")
    decoded = encoded.decode("utf-8")
    assert decoded == s


@given(st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False), min_size=1, max_size=50))
def test_sorted_scores_ordering_property(scores: list[float]):
    """Property: sorting scores in descending order maintains non-increasing invariant.

    Demonstrates testing the score ordering invariant that the Retriever
    must maintain (R3.3: results ordered by non-increasing score).
    """
    sorted_scores = sorted(scores, reverse=True)
    for i in range(len(sorted_scores) - 1):
        assert sorted_scores[i] >= sorted_scores[i + 1]
