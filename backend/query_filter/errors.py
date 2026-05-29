"""Error types for the Query Filter DSL parser.

Provides structured error reporting with 1-indexed line/column positions
and human-readable descriptions (1-256 code points). No partial AST is
ever returned alongside an error (R11.3).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ParseError:
    """Structured parse error with location information.

    Attributes:
        code: Stable error code (empty_input, filter_too_large, syntax_error).
        message: Human-readable description, 1-256 code points.
        line: 1-indexed line number of the first offending character.
        column: 1-indexed column number of the first offending character.
    """

    code: str
    message: str
    line: int
    column: int

    def __post_init__(self) -> None:
        # Enforce message length constraint (1-256 code points)
        if len(self.message) < 1:
            object.__setattr__(self, "message", "Unknown error")
        elif len(self.message) > 256:
            object.__setattr__(self, "message", self.message[:256])


# Standard error codes
EMPTY_INPUT = "empty_input"
FILTER_TOO_LARGE = "filter_too_large"
SYNTAX_ERROR = "syntax_error"
