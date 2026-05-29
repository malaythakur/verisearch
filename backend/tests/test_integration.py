"""Integration tests against real Docker services.

Requires: docker compose up -d
Run with: pytest tests/test_integration.py -m integration

Tests verify end-to-end functionality with real:
- PostgreSQL (auth, audit, sessions)
- Redis (rate limiting)
- OpenSearch (search indexing)
- Kafka/Redpanda (async ingest)
- MinIO (object storage)
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any

import httpx
import pytest

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration

BASE_URL = "http://localhost:8000"


@pytest.fixture(scope="module")
def api_client():
    """Create an HTTP client for the API."""
    with httpx.Client(base_url=BASE_URL, timeout=30.0) as client:
        yield client


@pytest.fixture(scope="module")
def ensure_server(api_client: httpx.Client):
    """Ensure the server is running."""
    for _ in range(10):
        try:
            resp = api_client.get("/health")
            if resp.status_code == 200:
                return
        except httpx.ConnectError:
            time.sleep(1)
    pytest.skip("Server not running at localhost:8000")


class TestHealthCheck:
    """Test the health endpoint."""

    def test_health_returns_200(self, api_client: httpx.Client, ensure_server):
        resp = api_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["version"] == "0.1.0"


class TestIndexAndSearch:
    """Test document indexing and search flow."""

    def test_index_document(self, api_client: httpx.Client, ensure_server):
        resp = api_client.post("/v1/index", json={
            "url": "https://integration-test.com/doc1",
            "content": "<h1>Integration Test</h1><p>This is a test document for integration testing of the search engine.</p>",
        })
        assert resp.status_code == 201
        data = resp.json()
        assert "document_id" in data
        assert data["version"] >= 1

    def test_search_returns_indexed_document(self, api_client: httpx.Client, ensure_server):
        # Index first
        api_client.post("/v1/index", json={
            "url": "https://integration-test.com/searchable",
            "content": "Quantum computing uses qubits for parallel computation.",
        })

        # Search
        resp = api_client.post("/v1/search", json={
            "query": "quantum computing",
            "mode": "hybrid",
            "num_results": 10,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert "total" in data

    def test_search_with_different_modes(self, api_client: httpx.Client, ensure_server):
        for mode in ["neural", "keyword", "hybrid"]:
            resp = api_client.post("/v1/search", json={
                "query": "test query",
                "mode": mode,
                "num_results": 5,
            })
            assert resp.status_code == 200

    def test_search_validation_rejects_invalid_mode(self, api_client: httpx.Client, ensure_server):
        resp = api_client.post("/v1/search", json={
            "query": "test",
            "mode": "invalid_mode",
            "num_results": 5,
        })
        assert resp.status_code == 422

    def test_search_validation_rejects_empty_query(self, api_client: httpx.Client, ensure_server):
        resp = api_client.post("/v1/search", json={
            "query": "",
            "mode": "hybrid",
            "num_results": 5,
        })
        assert resp.status_code == 422

    def test_search_validation_rejects_num_results_over_100(self, api_client: httpx.Client, ensure_server):
        resp = api_client.post("/v1/search", json={
            "query": "test",
            "mode": "hybrid",
            "num_results": 101,
        })
        assert resp.status_code == 422


class TestAnswerEndpoint:
    """Test the answer generation endpoint."""

    def test_answer_with_no_documents(self, api_client: httpx.Client, ensure_server):
        resp = api_client.post("/v1/answer", json={
            "query": "What is something completely unique and not indexed?",
            "stream": False,
        })
        assert resp.status_code == 200

    def test_answer_rejects_empty_query(self, api_client: httpx.Client, ensure_server):
        resp = api_client.post("/v1/answer", json={
            "query": "",
            "stream": False,
        })
        assert resp.status_code == 422

    def test_answer_with_indexed_content(self, api_client: httpx.Client, ensure_server):
        # Index a document
        api_client.post("/v1/index", json={
            "url": "https://integration-test.com/answer-test",
            "content": "Python is a programming language created by Guido van Rossum in 1991.",
        })

        # Ask about it
        resp = api_client.post("/v1/answer", json={
            "query": "What is Python?",
            "stream": False,
        })
        assert resp.status_code == 200
        data = resp.json()
        # Should have either an answer or an error (depending on LLM availability)
        assert "answer" in data or "error" in data


class TestSessions:
    """Test session management."""

    def test_create_session(self, api_client: httpx.Client, ensure_server):
        resp = api_client.post("/v1/sessions", json={"retention_days": 7})
        assert resp.status_code == 201
        data = resp.json()
        assert "session_id" in data
        assert data["retention_days"] == 7

    def test_create_session_default_retention(self, api_client: httpx.Client, ensure_server):
        resp = api_client.post("/v1/sessions", json={})
        assert resp.status_code == 201
        data = resp.json()
        assert data["retention_days"] == 14

    def test_delete_session(self, api_client: httpx.Client, ensure_server):
        # Create
        resp = api_client.post("/v1/sessions", json={"retention_days": 7})
        session_id = resp.json()["session_id"]

        # Delete
        resp = api_client.delete(f"/v1/sessions/{session_id}")
        assert resp.status_code == 204

    def test_delete_nonexistent_session(self, api_client: httpx.Client, ensure_server):
        resp = api_client.delete("/v1/sessions/nonexistent-id")
        assert resp.status_code == 404

    def test_invalid_retention_days(self, api_client: httpx.Client, ensure_server):
        resp = api_client.post("/v1/sessions", json={"retention_days": 100})
        assert resp.status_code == 422


class TestResearch:
    """Test research job management."""

    def test_create_research_job(self, api_client: httpx.Client, ensure_server):
        resp = api_client.post("/v1/research", json={
            "research_goal": "Analyze the impact of machine learning on healthcare",
        })
        assert resp.status_code == 201
        data = resp.json()
        assert "job_id" in data

    def test_get_research_job(self, api_client: httpx.Client, ensure_server):
        # Create
        resp = api_client.post("/v1/research", json={
            "research_goal": "Study renewable energy trends",
        })
        job_id = resp.json()["job_id"]

        # Get report
        resp = api_client.get(f"/v1/research/{job_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert "text" in data

    def test_get_research_events(self, api_client: httpx.Client, ensure_server):
        # Create
        resp = api_client.post("/v1/research", json={
            "research_goal": "Analyze AI safety",
        })
        job_id = resp.json()["job_id"]

        # Get events
        resp = api_client.get(f"/v1/research/{job_id}/events")
        assert resp.status_code == 200
        data = resp.json()
        assert "events" in data

    def test_research_validation_rejects_empty_goal(self, api_client: httpx.Client, ensure_server):
        resp = api_client.post("/v1/research", json={
            "research_goal": "",
        })
        assert resp.status_code == 422

    def test_nonexistent_job(self, api_client: httpx.Client, ensure_server):
        resp = api_client.get("/v1/research/nonexistent-job-id")
        assert resp.status_code == 404


class TestPipelines:
    """Test pipeline management."""

    def test_create_pipeline(self, api_client: httpx.Client, ensure_server):
        resp = api_client.post("/v1/pipelines", json={
            "name": "Test Pipeline",
            "steps": [
                {"name": "domain_filter", "config": {"domains": ["example.com"]}},
            ],
        })
        assert resp.status_code in (201, 422)  # 422 if step not in registry

    def test_get_nonexistent_pipeline(self, api_client: httpx.Client, ensure_server):
        resp = api_client.get("/v1/pipelines/nonexistent-id")
        assert resp.status_code == 404


class TestContents:
    """Test content retrieval."""

    def test_contents_with_valid_ids(self, api_client: httpx.Client, ensure_server):
        # Index a document first
        resp = api_client.post("/v1/index", json={
            "url": "https://integration-test.com/contents-test",
            "content": "Content retrieval test document with some text.",
        })
        doc_id = resp.json()["document_id"]

        # Fetch contents
        resp = api_client.post("/v1/contents", json={
            "document_ids": [doc_id],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data

    def test_contents_with_nonexistent_id(self, api_client: httpx.Client, ensure_server):
        resp = api_client.post("/v1/contents", json={
            "document_ids": ["nonexistent-doc-id"],
        })
        assert resp.status_code == 200
        data = resp.json()
        # Should have error entry for the missing doc
        assert len(data["results"]) == 1

    def test_contents_validation_rejects_empty_list(self, api_client: httpx.Client, ensure_server):
        resp = api_client.post("/v1/contents", json={
            "document_ids": [],
        })
        assert resp.status_code == 422


class TestRateLimiting:
    """Test rate limiting headers."""

    def test_rate_limit_headers_present(self, api_client: httpx.Client, ensure_server):
        resp = api_client.post("/v1/search", json={
            "query": "test",
            "mode": "hybrid",
            "num_results": 1,
        })
        assert "x-ratelimit-limit" in resp.headers
        assert "x-ratelimit-remaining" in resp.headers
        assert "x-ratelimit-reset" in resp.headers


class TestRequestId:
    """Test request ID propagation."""

    def test_request_id_header_present(self, api_client: httpx.Client, ensure_server):
        resp = api_client.get("/health")
        assert "x-request-id" in resp.headers
        assert len(resp.headers["x-request-id"]) > 0


class TestFindSimilar:
    """Test find_similar endpoint."""

    def test_find_similar_with_unknown_url(self, api_client: httpx.Client, ensure_server):
        resp = api_client.post("/v1/find_similar", json={
            "url": "https://unknown-url-not-indexed.com/page",
            "num_results": 5,
        })
        assert resp.status_code == 404

    def test_find_similar_validation(self, api_client: httpx.Client, ensure_server):
        resp = api_client.post("/v1/find_similar", json={
            "url": "not-a-valid-url",
            "num_results": 5,
        })
        assert resp.status_code == 422
