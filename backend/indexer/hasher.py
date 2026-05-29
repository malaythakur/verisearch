"""Content hashing — SHA-256 over cleaned text (Task 10.2, R2.4).

Provides deterministic content hashing for idempotent re-indexing.
The hash is computed over the UTF-8 encoded cleaned text.
"""

from __future__ import annotations

import hashlib


def compute_content_hash(cleaned_text: str) -> str:
    """Compute SHA-256 hash of cleaned text content.

    The hash is computed over the UTF-8 encoding of the cleaned text,
    producing a hex-encoded digest string. This is used to detect
    content changes for version management (R2.3, R2.4).

    Args:
        cleaned_text: The cleaned plain text content.

    Returns:
        Hex-encoded SHA-256 digest string (64 characters).
    """
    return hashlib.sha256(cleaned_text.encode("utf-8")).hexdigest()
