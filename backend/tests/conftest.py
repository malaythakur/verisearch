"""Shared test fixtures and configuration for the Agentic Research backend."""

import os
import uuid

import pytest
from hypothesis import settings, HealthCheck

# ---------------------------------------------------------------------------
# Hypothesis profile configuration
# ---------------------------------------------------------------------------
# CI profile: more examples for thorough coverage
settings.register_profile(
    "ci",
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow],
)

# Dev profile: fewer examples for fast feedback
settings.register_profile(
    "dev",
    max_examples=20,
    suppress_health_check=[HealthCheck.too_slow],
)

# Load profile from environment variable, default to "dev"
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "dev"))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tenant_id() -> str:
    """Generate a random tenant ID for test isolation."""
    return str(uuid.uuid4())


@pytest.fixture
def second_tenant_id() -> str:
    """Generate a second tenant ID for cross-tenant isolation tests."""
    return str(uuid.uuid4())


@pytest.fixture
def request_id() -> str:
    """Generate a request ID matching the 16-64 code point requirement."""
    return f"req-{uuid.uuid4().hex}"
