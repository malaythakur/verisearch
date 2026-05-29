"""Smoke test: verify the backend package loads correctly."""


def test_backend_package_imports():
    """The backend package should be importable without errors."""
    import backend

    assert hasattr(backend, "__doc__")


def test_subpackages_importable():
    """All backend subpackages should be importable."""
    import backend.auth
    import backend.api_gateway
    import backend.crawler
    import backend.indexer
    import backend.retriever
    import backend.pipeline_engine
    import backend.answer_engine
    import backend.research_agent
    import backend.session_store
    import backend.provenance_scorer
    import backend.query_filter
    import backend.rate_limiter
    import backend.pii_redactor
    import backend.audit_log
    import backend.mcp_server

    # If we get here, all imports succeeded
    assert True
