"""Property-based tests for the Query Filter DSL Parser & Printer.

Uses Hypothesis to verify:
- Property 22: parse(print(ast)) ≡ ast for all well-formed ASTs
- Property 23: parse(print(parse(s))) ≡ parse(s) for all parseable strings
- Property 24: Oversized/over-nested/over-leaf inputs → filter_too_large
- Property 25: Invalid input → structured error, no partial AST
"""

from __future__ import annotations

import string

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from backend.query_filter import (
    FILTER_TOO_LARGE,
    And,
    Eq,
    FilterNode,
    In,
    Ne,
    Not,
    NumberLiteral,
    Or,
    ParseError,
    Range,
    StringLiteral,
    TimestampLiteral,
    parse,
    print_filter,
    structural_eq,
)


# ============================================================
# Strategies for generating well-formed ASTs
# ============================================================

# Valid top-level fields for the DSL
VALID_FIELDS = ["domain", "url", "published_at", "language", "category"]

# Valid metadata field patterns
METADATA_FIELDS = [
    "metadata.author",
    "metadata.tags",
    "metadata.source",
    "metadata.type",
    "metadata.region",
]

ALL_FIELDS = VALID_FIELDS + METADATA_FIELDS


@st.composite
def string_literals(draw: st.DrawFn) -> StringLiteral:
    """Generate valid string literals (no embedded quotes/backslashes for simplicity)."""
    # Use printable ASCII without quotes and backslashes to avoid escaping issues
    safe_chars = string.ascii_letters + string.digits + " .-_/:@"
    value = draw(st.text(alphabet=safe_chars, min_size=1, max_size=50))
    return StringLiteral(value=value)


@st.composite
def number_literals(draw: st.DrawFn) -> NumberLiteral:
    """Generate valid number literals."""
    value = draw(
        st.floats(
            min_value=-1e6,
            max_value=1e6,
            allow_nan=False,
            allow_infinity=False,
        ).filter(lambda x: x == x)  # exclude NaN
    )
    # Normalize to avoid floating point representation issues
    # Use values that round-trip cleanly through str/float
    value = float(f"{value:g}")
    return NumberLiteral(value=value)


@st.composite
def timestamp_literals(draw: st.DrawFn) -> TimestampLiteral:
    """Generate valid ISO 8601 timestamp literals."""
    year = draw(st.integers(min_value=2000, max_value=2030))
    month = draw(st.integers(min_value=1, max_value=12))
    day = draw(st.integers(min_value=1, max_value=28))  # safe for all months
    hour = draw(st.integers(min_value=0, max_value=23))
    minute = draw(st.integers(min_value=0, max_value=59))
    second = draw(st.integers(min_value=0, max_value=59))
    value = f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:02d}Z"
    return TimestampLiteral(value=value)


literals = st.one_of(string_literals(), number_literals(), timestamp_literals())

fields = st.sampled_from(ALL_FIELDS)


@st.composite
def eq_nodes(draw: st.DrawFn) -> Eq:
    """Generate Eq leaf nodes."""
    field = draw(fields)
    value = draw(literals)
    return Eq(field=field, value=value)


@st.composite
def ne_nodes(draw: st.DrawFn) -> Ne:
    """Generate Ne leaf nodes."""
    field = draw(fields)
    value = draw(literals)
    return Ne(field=field, value=value)


@st.composite
def in_nodes(draw: st.DrawFn) -> In:
    """Generate In leaf nodes with 1-10 values."""
    field = draw(fields)
    values = draw(st.lists(literals, min_size=1, max_size=10))
    return In(field=field, values=tuple(values))


@st.composite
def range_nodes(draw: st.DrawFn) -> Range:
    """Generate Range leaf nodes."""
    field = draw(fields)
    op = draw(st.sampled_from(["lt", "le", "gt", "ge"]))
    value = draw(literals)
    return Range(field=field, op=op, value=value)


leaf_nodes = st.one_of(eq_nodes(), ne_nodes(), in_nodes(), range_nodes())


@st.composite
def filter_ast(draw: st.DrawFn, max_depth: int = 4, parent_type: str | None = None) -> FilterNode:
    """Generate well-formed canonical Filter_AST trees with bounded depth.

    Canonical means no nested And-within-And or Or-within-Or, since the
    printer flattens these (And is associative). The parser always produces
    flat And/Or nodes, so round-trip only holds for canonical forms.
    """
    if max_depth <= 1:
        return draw(leaf_nodes)

    # Avoid generating same-type nested And/Or (not canonical)
    available_types = ["leaf", "and", "or", "not"]
    if parent_type == "and":
        available_types.remove("and")
    elif parent_type == "or":
        available_types.remove("or")

    node_type = draw(st.sampled_from(available_types))

    if node_type == "leaf":
        return draw(leaf_nodes)
    elif node_type == "not":
        child = draw(filter_ast(max_depth=max_depth - 1, parent_type="not"))
        return Not(child=child)
    elif node_type == "and":
        n_children = draw(st.integers(min_value=2, max_value=4))
        children = [draw(filter_ast(max_depth=max_depth - 1, parent_type="and")) for _ in range(n_children)]
        return And(children=tuple(children))
    else:  # or
        n_children = draw(st.integers(min_value=2, max_value=4))
        children = [draw(filter_ast(max_depth=max_depth - 1, parent_type="or")) for _ in range(n_children)]
        return Or(children=tuple(children))


# ============================================================
# Strategy for generating valid filter strings
# ============================================================


@st.composite
def valid_filter_strings(draw: st.DrawFn) -> str:
    """Generate valid filter DSL strings by building an AST and printing it."""
    ast = draw(filter_ast(max_depth=3, parent_type=None))
    return print_filter(ast)


# ============================================================
# Property 22: parse(print(ast)) ≡ ast
# Validates: Requirements 1.2
# ============================================================


class TestProperty22RoundTripAstPrintParse:
    """**Validates: Requirements 1.2**

    Property 22: For any well-formed Filter_AST, parse(print(ast)) ≡ ast.
    """

    @given(ast=filter_ast(max_depth=4, parent_type=None))
    @settings(max_examples=200, deadline=5000)
    def test_parse_print_roundtrip(self, ast: FilterNode):
        """For any well-formed AST, parse(print(ast)) must be structurally
        equivalent to the original AST."""
        printed = print_filter(ast)
        reparsed = parse(printed)

        # Must not be a parse error
        assert not isinstance(reparsed, ParseError), (
            f"print_filter produced unparseable output.\n"
            f"AST: {ast}\n"
            f"Printed: {printed}\n"
            f"Error: {reparsed}"
        )

        # Must be structurally equivalent
        assert structural_eq(ast, reparsed), (
            f"Round-trip failed: parse(print(ast)) ≢ ast\n"
            f"Original AST: {ast}\n"
            f"Printed: {printed}\n"
            f"Reparsed AST: {reparsed}"
        )


# ============================================================
# Property 23: parse(print(parse(s))) ≡ parse(s)
# Validates: Requirements 1.2
# ============================================================


class TestProperty23RoundTripStringParsePrintParse:
    """**Validates: Requirements 1.2**

    Property 23: For any parseable string s, parse(print(parse(s))) ≡ parse(s).
    """

    @given(s=valid_filter_strings())
    @settings(max_examples=200, deadline=5000)
    def test_parse_print_parse_idempotent(self, s: str):
        """For any parseable string, parsing, printing, and re-parsing
        must yield the same AST as the first parse."""
        first_parse = parse(s)
        assume(not isinstance(first_parse, ParseError))

        printed = print_filter(first_parse)
        second_parse = parse(printed)

        # Second parse must not fail
        assert not isinstance(second_parse, ParseError), (
            f"print_filter produced unparseable output on second pass.\n"
            f"Input: {s}\n"
            f"First parse: {first_parse}\n"
            f"Printed: {printed}\n"
            f"Error: {second_parse}"
        )

        # Must be structurally equivalent
        assert structural_eq(first_parse, second_parse), (
            f"Idempotency failed: parse(print(parse(s))) ≢ parse(s)\n"
            f"Input: {s}\n"
            f"First parse: {first_parse}\n"
            f"Printed: {printed}\n"
            f"Second parse: {second_parse}"
        )


# ============================================================
# Property 24: Oversized/over-nested/over-leaf → filter_too_large
# Validates: Requirements 1.2
# ============================================================


class TestProperty24OversizedInputsRejected:
    """**Validates: Requirements 1.2**

    Property 24: Inputs exceeding size/nesting/leaf bounds are rejected
    with filter_too_large error code.
    """

    @given(extra_len=st.integers(min_value=1, max_value=1000))
    @settings(max_examples=50, deadline=5000)
    def test_oversized_input_rejected(self, extra_len: int):
        """Inputs >16384 code points must produce filter_too_large."""
        # Build an input that exceeds 16384 code points
        # Use a valid-looking prefix to ensure it's the length check that triggers
        padding = "a" * (16384 + extra_len)
        result = parse(padding)

        assert isinstance(result, ParseError), (
            f"Expected ParseError for input of {len(padding)} code points, got: {type(result)}"
        )
        assert result.code == FILTER_TOO_LARGE, (
            f"Expected filter_too_large, got: {result.code}"
        )

    @given(depth=st.integers(min_value=33, max_value=50))
    @settings(max_examples=30, deadline=10000)
    def test_over_nested_input_rejected(self, depth: int):
        """Inputs with >32 nesting levels must produce filter_too_large."""
        # Build deeply nested expression with parentheses
        input_str = "(" * depth + 'domain = "x"' + ")" * depth
        result = parse(input_str)

        assert isinstance(result, ParseError), (
            f"Expected ParseError for nesting depth {depth}, got: {type(result)}"
        )
        assert result.code == FILTER_TOO_LARGE, (
            f"Expected filter_too_large for depth {depth}, got: {result.code}"
        )

    @given(leaf_count=st.integers(min_value=1025, max_value=1050))
    @settings(max_examples=10, deadline=30000)
    def test_over_leaf_input_rejected(self, leaf_count: int):
        """Inputs with >1024 leaf comparisons must produce filter_too_large."""
        # Build expression with many leaf comparisons joined by 'or'
        leaves = " or ".join(f'domain = "{i}"' for i in range(leaf_count))
        result = parse(leaves)

        assert isinstance(result, ParseError), (
            f"Expected ParseError for {leaf_count} leaves, got: {type(result)}"
        )
        assert result.code == FILTER_TOO_LARGE, (
            f"Expected filter_too_large for {leaf_count} leaves, got: {result.code}"
        )


# ============================================================
# Property 25: Invalid input → structured error, no partial AST
# Validates: Requirements 1.2
# ============================================================


class TestProperty25InvalidInputStructuredError:
    """**Validates: Requirements 1.2**

    Property 25: Invalid input produces a structured ParseError with valid
    line (>=1), column (>=1), message (1-256 chars), and code. No partial
    AST is ever returned.
    """

    @given(
        data=st.one_of(
            # Random binary/garbage strings
            st.text(
                alphabet=st.characters(
                    blacklist_categories=("Cs",),  # exclude surrogates
                ),
                min_size=1,
                max_size=200,
            ),
            # Strings with invalid operators
            st.from_regex(r"[a-z]+ [@#$%^&*!]{1,3} [a-z]+", fullmatch=True),
            # Incomplete expressions
            st.sampled_from([
                "domain =",
                "domain !=",
                "and",
                "or or",
                "not",
                "((",
                "))",
                'domain = "unterminated',
                "invalid_field = 42",
                "domain in ()",
                "domain in",
                "= = =",
                "domain domain domain",
                '""" triple quotes',
                "123 + 456",
                "SELECT * FROM users",
            ]),
        )
    )
    @settings(max_examples=200, deadline=5000)
    def test_invalid_input_produces_structured_error(self, data: str):
        """Invalid inputs must produce a ParseError (never a FilterNode)
        with valid structured fields."""
        result = parse(data)

        # If it happens to be valid DSL, skip this test case
        if not isinstance(result, ParseError):
            # The input was actually valid DSL - skip
            assume(False)

        # Verify structured error fields
        assert isinstance(result, ParseError)
        assert result.line >= 1, f"line must be >= 1, got: {result.line}"
        assert result.column >= 1, f"column must be >= 1, got: {result.column}"
        assert 1 <= len(result.message) <= 256, (
            f"message length must be 1-256, got: {len(result.message)}"
        )
        assert result.code in ("empty_input", "filter_too_large", "syntax_error"), (
            f"Unknown error code: {result.code}"
        )

    @given(
        s=st.text(
            alphabet=st.characters(blacklist_categories=("Cs",)),
            min_size=1,
            max_size=500,
        )
    )
    @settings(max_examples=200, deadline=5000)
    def test_no_partial_ast_on_error(self, s: str):
        """The parser must return either a valid FilterNode OR a ParseError,
        never both. If the result is a ParseError, it must not be a FilterNode."""
        result = parse(s)

        # Result must be exactly one of FilterNode or ParseError
        is_filter = isinstance(result, (And, Or, Not, Eq, Ne, In, Range))
        is_error = isinstance(result, ParseError)

        assert is_filter or is_error, (
            f"Result is neither FilterNode nor ParseError: {type(result)}"
        )
        assert not (is_filter and is_error), (
            f"Result is both FilterNode and ParseError (impossible but checking): {result}"
        )

        # If it's an error, verify structure
        if is_error:
            assert result.line >= 1
            assert result.column >= 1
            assert 1 <= len(result.message) <= 256
            assert result.code in ("empty_input", "filter_too_large", "syntax_error")
