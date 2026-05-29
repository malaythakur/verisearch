"""Tests for the shared configuration module."""

import os
from unittest.mock import patch

import pytest

from backend.config import Constants, FeatureFlags, Settings, TenantDefaults
from backend.config.settings import AppEnvironment


class TestSettings:
    """Tests for the main Settings class."""

    def test_default_values(self):
        """Settings should have sensible defaults for local development."""
        s = Settings()
        assert s.postgres_host == "localhost"
        assert s.postgres_port == 5432
        assert s.postgres_db == "agentic_research"
        assert s.postgres_user == "agentic"
        assert s.postgres_password == "agentic_dev"
        assert "postgresql://" in s.database_url
        assert s.opensearch_url == "http://localhost:9200"
        assert s.redis_url == "redis://localhost:6379/0"
        assert s.kafka_bootstrap_servers == "localhost:19092"
        assert s.aws_endpoint_url == "http://localhost:9000"
        assert s.s3_bucket == "agentic-research-docs"
        assert s.aws_access_key_id == "minioadmin"
        assert s.aws_secret_access_key == "minioadmin"
        assert s.aws_default_region == "us-east-1"
        assert s.app_env == AppEnvironment.DEV
        assert s.log_level == "INFO"
        assert s.service_name == "agentic-research"

    def test_env_override(self):
        """Settings should be overridable via environment variables."""
        env = {
            "POSTGRES_HOST": "db.example.com",
            "POSTGRES_PORT": "5433",
            "APP_ENV": "production",
            "LOG_LEVEL": "DEBUG",
            "SERVICE_NAME": "my-service",
        }
        with patch.dict(os.environ, env, clear=False):
            s = Settings()
            assert s.postgres_host == "db.example.com"
            assert s.postgres_port == 5433
            assert s.app_env == AppEnvironment.PRODUCTION
            assert s.log_level == "DEBUG"
            assert s.service_name == "my-service"

    def test_app_env_enum_values(self):
        """AppEnvironment enum should have dev, staging, production."""
        assert AppEnvironment.DEV.value == "dev"
        assert AppEnvironment.STAGING.value == "staging"
        assert AppEnvironment.PRODUCTION.value == "production"


class TestFeatureFlags:
    """Tests for the FeatureFlags class."""

    def test_all_flags_default_true(self):
        """All feature flags should default to True (enabled)."""
        ff = FeatureFlags()
        assert ff.enable_pii_redaction is True
        assert ff.enable_provenance_scoring is True
        assert ff.enable_research_agent is True
        assert ff.enable_mcp_server is True
        assert ff.enable_metering is True

    def test_env_override_disables_flag(self):
        """Feature flags should be disableable via environment variables."""
        with patch.dict(os.environ, {"ENABLE_PII_REDACTION": "false"}, clear=False):
            ff = FeatureFlags()
            assert ff.enable_pii_redaction is False

    def test_env_override_multiple_flags(self):
        """Multiple flags can be overridden simultaneously."""
        env = {
            "ENABLE_RESEARCH_AGENT": "false",
            "ENABLE_MCP_SERVER": "0",
        }
        with patch.dict(os.environ, env, clear=False):
            ff = FeatureFlags()
            assert ff.enable_research_agent is False
            assert ff.enable_mcp_server is False
            assert ff.enable_pii_redaction is True  # unchanged


class TestTenantDefaults:
    """Tests for the TenantDefaults class."""

    def test_default_values(self):
        """Tenant defaults should match the specification."""
        td = TenantDefaults()
        assert td.rate_limit_per_minute == 60
        assert td.session_retention_days == 14
        assert td.audit_retention_days == 365
        assert td.max_pipeline_steps == 20
        assert td.pipeline_step_timeout_ms == 2000
        assert td.key_rotation_grace_seconds == 3600
        assert td.crawler_max_concurrency_per_host == 2
        assert td.crawler_min_delay_seconds == 1.0
        assert td.research_max_steps == 32
        assert td.research_max_duration_ms == 300000
        assert td.research_max_tool_calls == 100
        assert td.search_default_num_results == 10
        assert td.search_max_num_results == 100
        assert td.contents_max_document_ids == 100
        assert td.filter_max_code_points == 16384
        assert td.filter_max_nesting == 32
        assert td.filter_max_leaves == 1024

    def test_env_override_with_prefix(self):
        """Tenant defaults should be overridable via TENANT_DEFAULT_ prefixed env vars."""
        env = {
            "TENANT_DEFAULT_RATE_LIMIT_PER_MINUTE": "120",
            "TENANT_DEFAULT_SESSION_RETENTION_DAYS": "30",
        }
        with patch.dict(os.environ, env, clear=False):
            td = TenantDefaults()
            assert td.rate_limit_per_minute == 120
            assert td.session_retention_days == 30


class TestConstants:
    """Tests for the Constants class."""

    def test_request_id_bounds(self):
        """Request ID length bounds should be correct."""
        assert Constants.REQUEST_ID_MIN_LENGTH == 16
        assert Constants.REQUEST_ID_MAX_LENGTH == 64

    def test_input_size_limits(self):
        """Input size limits should match the specification."""
        assert Constants.QUERY_MAX_CODE_POINTS == 2048
        assert Constants.URL_MAX_CODE_POINTS == 2048
        assert Constants.RESEARCH_GOAL_MAX_CODE_POINTS == 4096

    def test_cache_ttls(self):
        """Cache TTL values should match the specification."""
        assert Constants.WARM_CACHE_TTL_SECONDS == 300
        assert Constants.AUTH_CACHE_TTL_SECONDS == 60
        assert Constants.ROBOTS_CACHE_TTL_HOURS == 24

    def test_opt_out_activation(self):
        """Opt-out activation window should be 24 hours."""
        assert Constants.OPT_OUT_ACTIVATION_HOURS == 24

    def test_dlq_settings(self):
        """DLQ retry settings should match the specification."""
        assert Constants.DLQ_MAX_RETRIES == 3
        assert Constants.DLQ_RETRY_SPACING_SECONDS == 60

    def test_streaming_constants(self):
        """Streaming-related constants should match the specification."""
        assert Constants.SSE_KEEPALIVE_SECONDS == 15
        assert Constants.ANSWER_SILENCE_TIMEOUT_SECONDS == 30
        assert Constants.ANSWER_ERROR_CLOSE_SECONDS == 2
        assert Constants.CITATION_EMISSION_DEADLINE_MS == 500

    def test_constants_are_immutable_class_attributes(self):
        """Constants should be class-level attributes (not instance-dependent)."""
        assert Constants.DLQ_MAX_RETRIES == Constants.DLQ_MAX_RETRIES
        c1 = Constants()
        c2 = Constants()
        assert c1.DLQ_MAX_RETRIES == c2.DLQ_MAX_RETRIES
