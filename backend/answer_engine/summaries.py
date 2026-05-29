"""Summary generation for /v1/contents endpoint.

Implements:
- Summary generation with token bounds: 1-512 model tokens (R5.3).
- Simple extractive summarization for the MVP.
"""

from __future__ import annotations

# Token bounds per R5.3
MIN_SUMMARY_TOKENS = 1
MAX_SUMMARY_TOKENS = 512

# Approximate characters per token (conservative estimate for English text)
CHARS_PER_TOKEN_ESTIMATE = 4


def generate_summary(
    cleaned_text: str,
    max_tokens: int = MAX_SUMMARY_TOKENS,
) -> str:
    """Generate a summary of a document's cleaned text.

    The summary is between 1 and 512 model tokens (R5.3).
    Uses extractive summarization for the MVP — a production implementation
    would use an LLM for abstractive summarization.

    Args:
        cleaned_text: The document's cleaned text content.
        max_tokens: Maximum number of tokens for the summary (default 512).

    Returns:
        A summary string between 1 and max_tokens tokens.

    Raises:
        ValueError: If cleaned_text is empty.
    """
    if not cleaned_text or not cleaned_text.strip():
        raise ValueError("Cannot generate summary from empty text")

    # Clamp max_tokens to valid range
    max_tokens = max(MIN_SUMMARY_TOKENS, min(max_tokens, MAX_SUMMARY_TOKENS))

    # Estimate max characters based on token limit
    max_chars = max_tokens * CHARS_PER_TOKEN_ESTIMATE

    # Simple extractive approach: take the first N sentences that fit
    sentences = _split_sentences(cleaned_text)

    if not sentences:
        # Fallback: truncate the text directly
        truncated = cleaned_text.strip()[:max_chars]
        return truncated if truncated else cleaned_text.strip()[:CHARS_PER_TOKEN_ESTIMATE]

    summary_parts: list[str] = []
    total_chars = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Check if adding this sentence would exceed the limit
        if total_chars + len(sentence) + 1 > max_chars:
            # If we have nothing yet, take a truncated version of the first sentence
            if not summary_parts:
                truncated = sentence[:max_chars]
                summary_parts.append(truncated)
            break

        summary_parts.append(sentence)
        total_chars += len(sentence) + 1  # +1 for space

    result = " ".join(summary_parts)

    # Ensure we have at least 1 token worth of content
    if not result:
        result = cleaned_text.strip()[:CHARS_PER_TOKEN_ESTIMATE]

    return result


def estimate_token_count(text: str) -> int:
    """Estimate the number of model tokens in a text string.

    Uses a simple heuristic: ~4 characters per token for English text.
    A production implementation would use the actual tokenizer.

    Args:
        text: The text to estimate tokens for.

    Returns:
        Estimated token count (at least 1 for non-empty text).
    """
    if not text:
        return 0
    # Simple estimation: split on whitespace and punctuation
    # More accurate than pure character count
    words = text.split()
    # Rough estimate: most words are 1 token, long words may be 2+
    token_count = 0
    for word in words:
        if len(word) <= 6:
            token_count += 1
        else:
            token_count += max(1, len(word) // 4)

    return max(1, token_count)


def validate_summary_tokens(summary: str) -> bool:
    """Validate that a summary is within the token bounds [1, 512].

    Args:
        summary: The summary text to validate.

    Returns:
        True if the summary is within bounds, False otherwise.
    """
    if not summary:
        return False
    token_count = estimate_token_count(summary)
    return MIN_SUMMARY_TOKENS <= token_count <= MAX_SUMMARY_TOKENS


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using simple heuristics."""
    # Simple sentence splitting on common terminators
    sentences: list[str] = []
    current: list[str] = []

    for char in text:
        current.append(char)
        if char in ".!?" and len(current) > 1:
            sentence = "".join(current).strip()
            if sentence:
                sentences.append(sentence)
            current = []

    # Don't forget the last part
    if current:
        remainder = "".join(current).strip()
        if remainder:
            sentences.append(remainder)

    return sentences
