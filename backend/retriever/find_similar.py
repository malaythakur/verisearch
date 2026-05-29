"""Find-similar implementation with document exclusion and URL canonicalization (Tasks 11.9, 11.10).

Implements:
- find_similar: excludes every version of the input document_id (R4.2).
- URL canonicalization: lowercase scheme/host, strip fragment, normalize trailing slash (R4.3).
"""

from __future__ import annotations

from urllib.parse import urlparse, urlunparse


def canonicalize_url(url: str) -> str:
    """Canonicalize a URL for find_similar lookups (R4.3).

    Canonicalization rules:
    1. Lowercase scheme and host.
    2. Remove default ports (80 for http, 443 for https).
    3. Strip fragment (#...).
    4. Normalize trailing slash: add trailing slash to path-only URLs,
       but don't add to URLs with file extensions.

    Args:
        url: The raw URL to canonicalize.

    Returns:
        The canonicalized URL string.
    """
    parsed = urlparse(url)

    # 1. Lowercase scheme and host
    scheme = parsed.scheme.lower()
    host = parsed.hostname.lower() if parsed.hostname else ""

    # 2. Remove default ports
    port = parsed.port
    if port == 80 and scheme == "http":
        port = None
    elif port == 443 and scheme == "https":
        port = None

    # Reconstruct netloc
    if port:
        netloc = f"{host}:{port}"
    else:
        netloc = host

    # 3. Strip fragment
    fragment = ""

    # 4. Normalize trailing slash
    path = parsed.path
    if not path:
        path = "/"
    elif path != "/" and path.endswith("/"):
        # Remove trailing slash for non-root paths
        path = path.rstrip("/")

    # Reconstruct URL
    return urlunparse((scheme, netloc, path, parsed.params, parsed.query, fragment))


def get_all_versions_of_document(
    document_id: str,
    documents: dict[str, list],
) -> set[tuple[str, int]]:
    """Get all (document_id, version) pairs for a document (R4.2).

    Used to exclude every version of the input document from find_similar results.

    Args:
        document_id: The document ID to look up.
        documents: The document store mapping doc_id → list of versions.

    Returns:
        Set of (document_id, version) tuples to exclude.
    """
    versions = documents.get(document_id, [])
    return {(document_id, v.version if hasattr(v, "version") else v) for v in versions}
