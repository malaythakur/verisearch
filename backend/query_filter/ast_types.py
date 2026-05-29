"""Filter_AST data types for the Query Filter DSL.

Defines the abstract syntax tree nodes for filter expressions:
- And, Or (commutative, order-insensitive for equivalence)
- Not (non-commutative, order-sensitive)
- Eq, Ne, In, Range (leaf comparisons)
- Literal types: StringLiteral, NumberLiteral, TimestampLiteral

Structural equivalence (R11.7):
- Commutative operators (And, Or): children compared as multisets
- Non-commutative operators (Not, Range): children compared in order
- Literals normalized: numbers without trailing zeros, timestamps in ISO 8601 UTC
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union


# --- Literal Types ---


@dataclass(frozen=True, slots=True)
class StringLiteral:
    """A string literal value."""

    value: str


@dataclass(frozen=True, slots=True)
class NumberLiteral:
    """A numeric literal value (stored as float)."""

    value: float


@dataclass(frozen=True, slots=True)
class TimestampLiteral:
    """An ISO 8601 timestamp literal value (stored as normalized string)."""

    value: str


Literal = Union[StringLiteral, NumberLiteral, TimestampLiteral]


# --- AST Node Types ---


@dataclass(frozen=True, slots=True)
class Eq:
    """Equality comparison: field = value."""

    field: str
    value: Literal


@dataclass(frozen=True, slots=True)
class Ne:
    """Inequality comparison: field != value."""

    field: str
    value: Literal


@dataclass(frozen=True, slots=True)
class In:
    """Set membership: field in (value1, value2, ...).

    Set cardinality must be between 1 and 256 (R11.8).
    """

    field: str
    values: tuple[Literal, ...]


@dataclass(frozen=True, slots=True)
class Range:
    """Range comparison: field op value.

    op is one of: lt, le, gt, ge.
    """

    field: str
    op: str  # "lt", "le", "gt", "ge"
    value: Literal


@dataclass(frozen=True, slots=True)
class Not:
    """Logical negation (non-commutative)."""

    child: FilterNode


@dataclass(frozen=True, slots=True)
class And:
    """Logical conjunction (commutative - order-insensitive for equivalence)."""

    children: tuple[FilterNode, ...]


@dataclass(frozen=True, slots=True)
class Or:
    """Logical disjunction (commutative - order-insensitive for equivalence)."""

    children: tuple[FilterNode, ...]


FilterNode = Union[And, Or, Not, Eq, Ne, In, Range]


# Update forward references for Not, And, Or
Not.__class_getitem__ = classmethod(lambda cls, item: cls)  # type: ignore[attr-defined]


# --- Structural Equivalence (R11.7) ---


def _normalize_literal(lit: Literal) -> Literal:
    """Normalize a literal for comparison.

    - Numbers: remove trailing zeros (normalize float representation)
    - Timestamps: already stored normalized
    - Strings: compared as-is (case-sensitive)
    """
    if isinstance(lit, NumberLiteral):
        # Normalize: 1.0 == 1.0, 2.10 == 2.1
        return NumberLiteral(float(lit.value))
    return lit


def _literal_eq(a: Literal, b: Literal) -> bool:
    """Compare two literals with normalization."""
    if type(a) is not type(b):
        return False
    a_norm = _normalize_literal(a)
    b_norm = _normalize_literal(b)
    if isinstance(a_norm, NumberLiteral) and isinstance(b_norm, NumberLiteral):
        return a_norm.value == b_norm.value
    return a_norm == b_norm


def _node_sort_key(node: FilterNode) -> str:
    """Generate a canonical sort key for a node (used for multiset comparison)."""
    from backend.query_filter.printer import print_filter

    return print_filter(node)


def structural_eq(a: FilterNode, b: FilterNode) -> bool:
    """Test structural equivalence of two Filter_AST nodes per R11.7.

    - Commutative operators (And, Or): children compared as multisets (order-insensitive)
    - Non-commutative operators (Not, Range): compared in order
    - Literals normalized before comparison
    """
    if type(a) is not type(b):
        return False

    if isinstance(a, Eq) and isinstance(b, Eq):
        return a.field == b.field and _literal_eq(a.value, b.value)

    if isinstance(a, Ne) and isinstance(b, Ne):
        return a.field == b.field and _literal_eq(a.value, b.value)

    if isinstance(a, In) and isinstance(b, In):
        if a.field != b.field:
            return False
        if len(a.values) != len(b.values):
            return False
        # In is set membership - compare as multisets
        a_sorted = sorted((_literal_sort_key(v) for v in a.values))
        b_sorted = sorted((_literal_sort_key(v) for v in b.values))
        return a_sorted == b_sorted

    if isinstance(a, Range) and isinstance(b, Range):
        return a.field == b.field and a.op == b.op and _literal_eq(a.value, b.value)

    if isinstance(a, Not) and isinstance(b, Not):
        return structural_eq(a.child, b.child)

    if isinstance(a, And) and isinstance(b, And):
        if len(a.children) != len(b.children):
            return False
        # Commutative: compare as multisets using canonical sort keys
        a_keys = sorted(_node_sort_key(c) for c in a.children)
        b_keys = sorted(_node_sort_key(c) for c in b.children)
        return a_keys == b_keys

    if isinstance(a, Or) and isinstance(b, Or):
        if len(a.children) != len(b.children):
            return False
        # Commutative: compare as multisets using canonical sort keys
        a_keys = sorted(_node_sort_key(c) for c in a.children)
        b_keys = sorted(_node_sort_key(c) for c in b.children)
        return a_keys == b_keys

    return False


def _literal_sort_key(lit: Literal) -> str:
    """Generate a sort key for a literal value."""
    if isinstance(lit, StringLiteral):
        return f"s:{lit.value}"
    elif isinstance(lit, NumberLiteral):
        return f"n:{lit.value}"
    elif isinstance(lit, TimestampLiteral):
        return f"t:{lit.value}"
    return ""
