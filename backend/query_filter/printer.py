"""Query_Filter_Printer — canonical serialization of Filter_AST to DSL string.

Canonical form rules:
- And/Or children sorted by their canonical string representation (commutative normalization)
- Literals normalized: numbers without trailing zeros, timestamps in ISO 8601 UTC
- Minimal parenthesization based on operator precedence
"""

from __future__ import annotations

from backend.query_filter.ast_types import (
    And,
    Eq,
    FilterNode,
    In,
    Literal,
    Ne,
    Not,
    NumberLiteral,
    Or,
    Range,
    StringLiteral,
    TimestampLiteral,
)


def print_filter(ast: FilterNode) -> str:
    """Serialize a Filter_AST to its canonical Query_Filter_DSL string.

    The output is deterministic and normalized:
    - And/Or children are sorted by their canonical representation
    - Numbers are printed without unnecessary trailing zeros
    - Timestamps are printed as-is (assumed already normalized)
    """
    return _print_node(ast, parent_precedence=0)


def _print_literal(lit: Literal) -> str:
    """Print a literal value in canonical form."""
    if isinstance(lit, StringLiteral):
        # Escape quotes and backslashes in string literals
        escaped = lit.value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    elif isinstance(lit, NumberLiteral):
        # Normalize: remove trailing zeros, but keep at least one decimal if float
        val = lit.value
        if val == int(val):
            return str(int(val))
        else:
            # Format without trailing zeros
            return f"{val:g}"
    elif isinstance(lit, TimestampLiteral):
        return lit.value
    return ""


def _print_node(node: FilterNode, parent_precedence: int) -> str:
    """Print a node with appropriate parenthesization.

    Precedence levels:
    0 - top level (no parens needed)
    1 - or
    2 - and
    3 - not
    4 - comparison (leaf)
    """
    if isinstance(node, Eq):
        return f"{node.field} = {_print_literal(node.value)}"

    elif isinstance(node, Ne):
        return f"{node.field} != {_print_literal(node.value)}"

    elif isinstance(node, In):
        values_str = ", ".join(_print_literal(v) for v in _sort_literals(node.values))
        return f"{node.field} in ({values_str})"

    elif isinstance(node, Range):
        op_str = _range_op_to_str(node.op)
        return f"{node.field} {op_str} {_print_literal(node.value)}"

    elif isinstance(node, Not):
        child_str = _print_node(node.child, parent_precedence=3)
        return f"not {child_str}"

    elif isinstance(node, And):
        # Sort children by canonical representation for commutative normalization
        child_strs = sorted(_print_node(c, parent_precedence=2) for c in node.children)
        result = " and ".join(child_strs)
        if parent_precedence > 2:
            result = f"({result})"
        return result

    elif isinstance(node, Or):
        # Sort children by canonical representation for commutative normalization
        child_strs = sorted(_print_node(c, parent_precedence=1) for c in node.children)
        result = " or ".join(child_strs)
        if parent_precedence > 1:
            result = f"({result})"
        return result

    return ""


def _range_op_to_str(op: str) -> str:
    """Convert range operator to its canonical string form."""
    return op  # lt, le, gt, ge are already canonical


def _sort_literals(values: tuple[Literal, ...]) -> list[Literal]:
    """Sort literals by their canonical string representation."""
    return sorted(values, key=_literal_sort_key)


def _literal_sort_key(lit: Literal) -> str:
    """Generate a sort key for ordering literals."""
    return _print_literal(lit)
