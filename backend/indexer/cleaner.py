"""Content cleaning pipeline — HTML → cleaned text (Task 10.1).

Strips HTML tags, decodes entities, normalizes whitespace, and produces
clean text suitable for hashing, embedding, and lexical indexing.
"""

from __future__ import annotations

import html
import re


# Pre-compiled patterns for performance
_SCRIPT_STYLE_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def clean_html(raw_content: str | bytes) -> str:
    """Convert raw HTML content to cleaned plain text.

    Steps:
    1. Decode bytes to string (UTF-8) if needed.
    2. Remove <script> and <style> blocks entirely.
    3. Strip all remaining HTML tags.
    4. Decode HTML entities (&amp; → &, etc.).
    5. Normalize whitespace (collapse runs, strip leading/trailing).

    Args:
        raw_content: Raw HTML as string or bytes.

    Returns:
        Cleaned plain text string.
    """
    if isinstance(raw_content, bytes):
        text = raw_content.decode("utf-8", errors="replace")
    else:
        text = raw_content

    # Remove script and style blocks
    text = _SCRIPT_STYLE_RE.sub("", text)

    # Strip HTML tags
    text = _TAG_RE.sub(" ", text)

    # Decode HTML entities
    text = html.unescape(text)

    # Normalize whitespace
    text = _WHITESPACE_RE.sub(" ", text).strip()

    return text
