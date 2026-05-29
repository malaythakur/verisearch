"""Recursive-descent parser for the Query Filter DSL.

Grammar (EBNF):
    expr        → or_expr
    or_expr     → and_expr ('or' and_expr)*
    and_expr    → not_expr ('and' not_expr)*
    not_expr    → 'not' not_expr | primary
    primary     → comparison | '(' expr ')'
    comparison  → field op value | field 'in' '(' value_list ')'
    field       → identifier ('.' identifier)*
    op          → '=' | '!=' | 'lt' | 'le' | 'gt' | 'ge'
    value       → string_literal | number_literal | timestamp_literal
    value_list  → value (',' value)*
    identifier  → [a-zA-Z_][a-zA-Z0-9_]*

Fields: domain, url, published_at, language, category, metadata.*
Operators: =, !=, in, lt, le, gt, ge
Values: quoted strings, numbers, ISO 8601 timestamps

Bounds enforcement (R11.2, R11.4):
- Empty/whitespace → empty_input
- >16384 code points → filter_too_large
- >32 nesting levels → filter_too_large
- >1024 leaf comparisons → filter_too_large

Performance target: ≤100ms on single core for inputs up to 16384 code points (R11.1).
"""

from __future__ import annotations

import re
from typing import Union

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
from backend.query_filter.errors import (
    EMPTY_INPUT,
    FILTER_TOO_LARGE,
    SYNTAX_ERROR,
    ParseError,
)

# Bounds constants
MAX_INPUT_LENGTH = 16384  # code points
MAX_NESTING_DEPTH = 32
MAX_LEAF_COUNT = 1024
MAX_IN_SET_SIZE = 256

# Valid top-level fields
VALID_FIELDS = {"domain", "url", "published_at", "language", "category"}

# Timestamp pattern (ISO 8601)
_TIMESTAMP_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}"
    r"(?:T\d{2}:\d{2}:\d{2}"
    r"(?:\.\d+)?"
    r"(?:Z|[+-]\d{2}:\d{2})?)?"
)

# Keywords that cannot be field names
_KEYWORDS = {"and", "or", "not", "in", "lt", "le", "gt", "ge"}


def parse(input_str: str) -> Union[FilterNode, ParseError]:
    """Parse a Query_Filter_DSL string into a Filter_AST.

    Returns either a FilterNode on success or a ParseError on failure.
    Never returns a partial AST (R11.3).
    """
    # R11.2: Empty/whitespace check
    if not input_str or input_str.isspace():
        return ParseError(
            code=EMPTY_INPUT,
            message="Input is empty or contains only whitespace",
            line=1,
            column=1,
        )

    # R11.4: Length check (code points)
    if len(input_str) > MAX_INPUT_LENGTH:
        return ParseError(
            code=FILTER_TOO_LARGE,
            message=f"Input exceeds maximum length of {MAX_INPUT_LENGTH} code points",
            line=1,
            column=1,
        )

    parser = _Parser(input_str)
    try:
        result = parser.parse_expr()
        parser.skip_whitespace()
        if parser.pos < len(parser.source):
            line, col = parser.get_position()
            return ParseError(
                code=SYNTAX_ERROR,
                message=f"Unexpected character '{parser.source[parser.pos]}' after expression",
                line=line,
                column=col,
            )
        return result
    except _ParseException as e:
        return e.error


class _ParseException(Exception):
    """Internal exception for parser errors."""

    def __init__(self, error: ParseError) -> None:
        self.error = error
        super().__init__(error.message)


class _Parser:
    """Recursive-descent parser state."""

    def __init__(self, source: str) -> None:
        self.source = source
        self.pos = 0
        self.depth = 0
        self.leaf_count = 0

    def get_position(self) -> tuple[int, int]:
        """Get 1-indexed line and column for current position."""
        line = 1
        col = 1
        for i in range(min(self.pos, len(self.source))):
            if self.source[i] == "\n":
                line += 1
                col = 1
            else:
                col += 1
        return line, col

    def error(self, message: str) -> _ParseException:
        """Create a parse error at the current position."""
        line, col = self.get_position()
        return _ParseException(ParseError(code=SYNTAX_ERROR, message=message, line=line, column=col))

    def skip_whitespace(self) -> None:
        """Skip whitespace characters."""
        while self.pos < len(self.source) and self.source[self.pos] in " \t\r\n":
            self.pos += 1

    def peek(self) -> str | None:
        """Peek at the current character without consuming."""
        if self.pos < len(self.source):
            return self.source[self.pos]
        return None

    def match_keyword(self, keyword: str) -> bool:
        """Try to match a keyword (must be followed by non-identifier char)."""
        self.skip_whitespace()
        end = self.pos + len(keyword)
        if end > len(self.source):
            return False
        if self.source[self.pos:end].lower() != keyword:
            return False
        # Must be followed by non-identifier character or end of input
        if end < len(self.source) and (self.source[end].isalnum() or self.source[end] == "_"):
            return False
        self.pos = end
        return True

    def expect_char(self, ch: str) -> None:
        """Expect and consume a specific character."""
        self.skip_whitespace()
        if self.pos >= len(self.source) or self.source[self.pos] != ch:
            expected = repr(ch)
            got = repr(self.source[self.pos]) if self.pos < len(self.source) else "end of input"
            raise self.error(f"Expected {expected}, got {got}")
        self.pos += 1

    def parse_expr(self) -> FilterNode:
        """Parse: expr → or_expr."""
        self.depth += 1
        if self.depth > MAX_NESTING_DEPTH:
            raise _ParseException(
                ParseError(
                    code=FILTER_TOO_LARGE,
                    message=f"Filter exceeds maximum nesting depth of {MAX_NESTING_DEPTH}",
                    line=1,
                    column=1,
                )
            )
        try:
            result = self.parse_or_expr()
            return result
        finally:
            self.depth -= 1

    def parse_or_expr(self) -> FilterNode:
        """Parse: or_expr → and_expr ('or' and_expr)*."""
        children = [self.parse_and_expr()]
        while True:
            saved_pos = self.pos
            self.skip_whitespace()
            if self.match_keyword("or"):
                children.append(self.parse_and_expr())
            else:
                self.pos = saved_pos
                break

        if len(children) == 1:
            return children[0]
        return Or(children=tuple(children))

    def parse_and_expr(self) -> FilterNode:
        """Parse: and_expr → not_expr ('and' not_expr)*."""
        children = [self.parse_not_expr()]
        while True:
            saved_pos = self.pos
            self.skip_whitespace()
            if self.match_keyword("and"):
                children.append(self.parse_not_expr())
            else:
                self.pos = saved_pos
                break

        if len(children) == 1:
            return children[0]
        return And(children=tuple(children))

    def parse_not_expr(self) -> FilterNode:
        """Parse: not_expr → 'not' not_expr | primary."""
        self.skip_whitespace()
        saved_pos = self.pos
        if self.match_keyword("not"):
            child = self.parse_not_expr()
            return Not(child=child)
        self.pos = saved_pos
        return self.parse_primary()

    def parse_primary(self) -> FilterNode:
        """Parse: primary → comparison | '(' expr ')'."""
        self.skip_whitespace()
        if self.pos < len(self.source) and self.source[self.pos] == "(":
            self.pos += 1
            node = self.parse_expr()
            self.expect_char(")")
            return node
        return self.parse_comparison()

    def parse_comparison(self) -> FilterNode:
        """Parse: comparison → field op value | field 'in' '(' value_list ')'."""
        self.skip_whitespace()
        field = self.parse_field()

        self.skip_whitespace()

        # Check for 'in' operator
        saved_pos = self.pos
        if self.match_keyword("in"):
            return self._parse_in(field)
        self.pos = saved_pos

        # Check for operators
        op = self.parse_operator()
        value = self.parse_value()

        self.leaf_count += 1
        if self.leaf_count > MAX_LEAF_COUNT:
            raise _ParseException(
                ParseError(
                    code=FILTER_TOO_LARGE,
                    message=f"Filter exceeds maximum of {MAX_LEAF_COUNT} leaf comparisons",
                    line=1,
                    column=1,
                )
            )

        if op == "=":
            return Eq(field=field, value=value)
        elif op == "!=":
            return Ne(field=field, value=value)
        elif op in ("lt", "le", "gt", "ge"):
            return Range(field=field, op=op, value=value)
        else:
            raise self.error(f"Unknown operator '{op}'")

    def _parse_in(self, field: str) -> In:
        """Parse the 'in (value_list)' portion."""
        self.expect_char("(")
        values = self.parse_value_list()
        self.expect_char(")")

        if len(values) < 1:
            raise self.error("Set membership requires at least 1 value")
        if len(values) > MAX_IN_SET_SIZE:
            raise self.error(f"Set membership exceeds maximum of {MAX_IN_SET_SIZE} values")

        self.leaf_count += 1
        if self.leaf_count > MAX_LEAF_COUNT:
            raise _ParseException(
                ParseError(
                    code=FILTER_TOO_LARGE,
                    message=f"Filter exceeds maximum of {MAX_LEAF_COUNT} leaf comparisons",
                    line=1,
                    column=1,
                )
            )

        return In(field=field, values=tuple(values))

    def parse_field(self) -> str:
        """Parse: field → identifier ('.' identifier)*."""
        self.skip_whitespace()
        start = self.pos
        ident = self.parse_identifier()

        # Check for dotted field (metadata.*)
        parts = [ident]
        while self.pos < len(self.source) and self.source[self.pos] == ".":
            self.pos += 1
            parts.append(self.parse_identifier())

        field = ".".join(parts)

        # Validate field name
        top_level = parts[0]
        if top_level == "metadata":
            if len(parts) < 2:
                raise self.error("metadata field requires at least one key segment (e.g., metadata.key)")
            # Validate key segment lengths (1-128 code points each)
            for segment in parts[1:]:
                if len(segment) < 1 or len(segment) > 128:
                    raise self.error(f"Metadata key segment must be 1-128 code points, got {len(segment)}")
        elif top_level not in VALID_FIELDS:
            raise self.error(
                f"Unknown field '{top_level}'. Valid fields: domain, url, published_at, language, category, metadata.*"
            )

        return field

    def parse_identifier(self) -> str:
        """Parse an identifier: [a-zA-Z_][a-zA-Z0-9_]*."""
        self.skip_whitespace()
        if self.pos >= len(self.source):
            raise self.error("Expected identifier, got end of input")

        start = self.pos
        ch = self.source[self.pos]
        if not (ch.isalpha() or ch == "_"):
            raise self.error(f"Expected identifier, got '{ch}'")

        self.pos += 1
        while self.pos < len(self.source) and (self.source[self.pos].isalnum() or self.source[self.pos] == "_"):
            self.pos += 1

        ident = self.source[start : self.pos]

        # Keywords cannot be used as identifiers in field position
        if ident.lower() in _KEYWORDS and start == self.pos - len(ident):
            # But we allow them as part of dotted paths (metadata.in is unlikely but valid)
            pass

        return ident

    def parse_operator(self) -> str:
        """Parse an operator: =, !=, lt, le, gt, ge."""
        self.skip_whitespace()
        if self.pos >= len(self.source):
            raise self.error("Expected operator, got end of input")

        # Check two-char operators first
        if self.pos + 1 < len(self.source):
            two = self.source[self.pos : self.pos + 2]
            if two == "!=":
                self.pos += 2
                return "!="
            if two == "<=":
                self.pos += 2
                return "le"
            if two == ">=":
                self.pos += 2
                return "ge"

        # Single char operators
        if self.source[self.pos] == "=":
            self.pos += 1
            return "="
        if self.source[self.pos] == "<":
            self.pos += 1
            return "lt"
        if self.source[self.pos] == ">":
            self.pos += 1
            return "ge" if False else "gt"

        # Keyword operators: lt, le, gt, ge
        saved_pos = self.pos
        for kw in ("lt", "le", "gt", "ge"):
            if self.match_keyword(kw):
                return kw
            self.pos = saved_pos

        raise self.error(f"Expected operator (=, !=, lt, le, gt, ge), got '{self.source[self.pos]}'")

    def parse_value(self) -> Literal:
        """Parse a value: string_literal | number_literal | timestamp_literal."""
        self.skip_whitespace()
        if self.pos >= len(self.source):
            raise self.error("Expected value, got end of input")

        ch = self.source[self.pos]

        # String literal
        if ch == '"':
            return self.parse_string_literal()

        # Number or timestamp (timestamps start with digit like 2024-...)
        if ch == "-" or ch.isdigit():
            return self.parse_number_or_timestamp()

        raise self.error(f"Expected value (string, number, or timestamp), got '{ch}'")

    def parse_string_literal(self) -> StringLiteral:
        """Parse a quoted string literal."""
        self.pos += 1  # skip opening quote
        result: list[str] = []
        while self.pos < len(self.source):
            ch = self.source[self.pos]
            if ch == "\\":
                self.pos += 1
                if self.pos >= len(self.source):
                    raise self.error("Unterminated escape sequence in string")
                esc = self.source[self.pos]
                if esc == '"':
                    result.append('"')
                elif esc == "\\":
                    result.append("\\")
                elif esc == "n":
                    result.append("\n")
                elif esc == "t":
                    result.append("\t")
                elif esc == "r":
                    result.append("\r")
                else:
                    result.append(esc)
                self.pos += 1
            elif ch == '"':
                self.pos += 1
                value = "".join(result)
                if len(value) > 1024:
                    raise self.error("String literal exceeds maximum of 1024 code points")
                return StringLiteral(value=value)
            else:
                result.append(ch)
                self.pos += 1

        raise self.error("Unterminated string literal")

    def parse_number_or_timestamp(self) -> Literal:
        """Parse a number or ISO 8601 timestamp."""
        start = self.pos

        # Try to match a timestamp first (YYYY-MM-DD...)
        match = _TIMESTAMP_RE.match(self.source, self.pos)
        if match and len(match.group()) >= 10:  # At least YYYY-MM-DD
            # Verify it looks like a timestamp (has dashes in right places)
            candidate = match.group()
            if len(candidate) >= 10 and candidate[4] == "-" and candidate[7] == "-":
                self.pos = match.end()
                return TimestampLiteral(value=candidate)

        # Parse as number
        self.pos = start
        return self.parse_number()

    def parse_number(self) -> NumberLiteral:
        """Parse a numeric literal."""
        start = self.pos
        if self.pos < len(self.source) and self.source[self.pos] == "-":
            self.pos += 1

        if self.pos >= len(self.source) or not self.source[self.pos].isdigit():
            raise self.error("Expected number")

        while self.pos < len(self.source) and self.source[self.pos].isdigit():
            self.pos += 1

        # Decimal part
        if self.pos < len(self.source) and self.source[self.pos] == ".":
            self.pos += 1
            if self.pos >= len(self.source) or not self.source[self.pos].isdigit():
                raise self.error("Expected digit after decimal point")
            while self.pos < len(self.source) and self.source[self.pos].isdigit():
                self.pos += 1

        # Scientific notation
        if self.pos < len(self.source) and self.source[self.pos] in "eE":
            self.pos += 1
            if self.pos < len(self.source) and self.source[self.pos] in "+-":
                self.pos += 1
            if self.pos >= len(self.source) or not self.source[self.pos].isdigit():
                raise self.error("Expected digit in exponent")
            while self.pos < len(self.source) and self.source[self.pos].isdigit():
                self.pos += 1

        num_str = self.source[start : self.pos]
        try:
            value = float(num_str)
        except ValueError:
            raise self.error(f"Invalid number: {num_str}")

        return NumberLiteral(value=value)

    def parse_value_list(self) -> list[Literal]:
        """Parse: value_list → value (',' value)*."""
        values = [self.parse_value()]
        while True:
            self.skip_whitespace()
            if self.pos < len(self.source) and self.source[self.pos] == ",":
                self.pos += 1
                values.append(self.parse_value())
            else:
                break
        return values
