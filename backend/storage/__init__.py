"""Storage - S3-compatible object store for cleaned document text.

Provides content-addressed storage with the key layout:
    {tenant_id}/{document_id}/{version}/cleaned.txt

For global (null tenant_id) documents:
    _global/{document_id}/{version}/cleaned.txt
"""

from backend.storage.object_store import DocumentObjectStore

__all__ = ["DocumentObjectStore"]
