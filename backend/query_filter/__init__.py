"""Query Filter DSL - Parser, printer, and AST with round-trip guarantees.

Public API:
    parse(input: str) -> FilterNode | ParseError
    print_filter(ast: FilterNode) -> str
    structural_eq(a: FilterNode, b: FilterNode) -> bool

AST Types:
    FilterNode = And | Or | Not | Eq | Ne | In | Range
    Literal = StringLiteral | NumberLiteral | TimestampLiteral

Error Types:
    ParseError(code, message, line, column)
    Codes: empty_input, filter_too_large, syntax_error
"""

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
    structural_eq,
)
from backend.query_filter.errors import (
    EMPTY_INPUT,
    FILTER_TOO_LARGE,
    SYNTAX_ERROR,
    ParseError,
)
from backend.query_filter.parser import parse
from backend.query_filter.printer import print_filter

__all__ = [
    # Parser & Printer
    "parse",
    "print_filter",
    "structural_eq",
    # AST Types
    "And",
    "Or",
    "Not",
    "Eq",
    "Ne",
    "In",
    "Range",
    "FilterNode",
    # Literal Types
    "Literal",
    "StringLiteral",
    "NumberLiteral",
    "TimestampLiteral",
    # Error Types
    "ParseError",
    "EMPTY_INPUT",
    "FILTER_TOO_LARGE",
    "SYNTAX_ERROR",
]
