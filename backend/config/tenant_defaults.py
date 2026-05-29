"""Default values for tenant-configurable settings.

These values are used when a tenant has not overridden a particular setting.
Each constant documents the requirement it satisfies and its valid range.
"""

from pydantic import Field
from pydantic_settings import BaseSettings


class TenantDefaults(BaseSettings):
    """Tenant-configurable defaults that can be overridden per-tenant at runtime.

    Environment variables can override these system-wide defaults.
    Per-tenant overrides are stored in the database.
    """

    # Rate limiting
    rate_limit_per_minute: int = Field(default=60, description="Default requests per minute per tenant per endpoint")

    # Session retention (R8.1: valid range [1, 90])
    session_retention_days: int = Field(default=14, description="Default session retention in days [1, 90]")

    # Audit retention (R15.4: valid range [365, 2555])
    audit_retention_days: int = Field(default=365, description="Default audit log retention in days [365, 2555]")

    # Pipeline engine (R9.1)
    max_pipeline_steps: int = Field(default=20, description="Maximum number of steps in a pipeline [1, 20]")

    # Pipeline step timeout (R9.6: valid range [100, 30000])
    pipeline_step_timeout_ms: int = Field(
        default=2000, description="Default per-step timeout in milliseconds [100, 30000]"
    )

    # Key rotation grace period (R13.5: valid range [1, 86400])
    key_rotation_grace_seconds: int = Field(
        default=3600, description="Default key rotation grace period in seconds [1, 86400]"
    )

    # Crawler settings (R1.4: valid range [1, 8])
    crawler_max_concurrency_per_host: int = Field(
        default=2, description="Max concurrent requests per host [1, 8]"
    )

    # Crawler delay (R1.5)
    crawler_min_delay_seconds: float = Field(
        default=1.0, description="Minimum delay between requests to the same host in seconds"
    )

    # Research agent budgets (R7.2)
    research_max_steps: int = Field(default=32, description="Maximum research plan steps [1, 32]")
    research_max_duration_ms: int = Field(default=300000, description="Maximum research duration (5 min)")
    research_max_tool_calls: int = Field(default=100, description="Maximum tool calls per research job")

    # Search defaults
    search_default_num_results: int = Field(default=10, description="Default number of search results")
    search_max_num_results: int = Field(default=100, description="Maximum number of search results")

    # Contents API (R5.5)
    contents_max_document_ids: int = Field(default=100, description="Maximum document IDs per /v1/contents request")

    # Filter DSL limits (R11.4)
    filter_max_code_points: int = Field(default=16384, description="Maximum code points in a filter expression")
    filter_max_nesting: int = Field(default=32, description="Maximum nesting depth in a filter expression")
    filter_max_leaves: int = Field(default=1024, description="Maximum leaf nodes in a filter expression")

    model_config = {"env_prefix": "TENANT_DEFAULT_", "env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}
