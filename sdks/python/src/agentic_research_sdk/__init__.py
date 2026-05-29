"""Agentic Research Search Engine - Python SDK.

Provides typed client methods for all API endpoints with async iterators for streams.
"""

from agentic_research_sdk.client import (
    AgenticResearchClient,
    APIError,
    Citation,
    ConnectionError,
    ContentEntry,
    ContentsResponse,
    ParseError,
    Pipeline,
    ProvenanceInfo,
    ResearchJob,
    SDKError,
    SearchMode,
    SearchResponse,
    SearchResult,
    Session,
    StreamEvent,
    TimeoutError,
)

__all__ = [
    "AgenticResearchClient",
    "SDKError",
    "APIError",
    "TimeoutError",
    "ConnectionError",
    "ParseError",
    "SearchMode",
    "ProvenanceInfo",
    "SearchResult",
    "SearchResponse",
    "Citation",
    "ContentEntry",
    "ContentsResponse",
    "Session",
    "Pipeline",
    "ResearchJob",
    "StreamEvent",
]
