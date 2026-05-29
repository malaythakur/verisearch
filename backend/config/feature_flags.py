"""Feature flag configuration for enabling/disabling subsystems."""

from pydantic_settings import BaseSettings


class FeatureFlags(BaseSettings):
    """Feature flags controlling subsystem availability.

    All flags default to True (enabled). Override via environment variables
    prefixed with the flag name, e.g. ENABLE_PII_REDACTION=false.
    """

    enable_pii_redaction: bool = True
    enable_provenance_scoring: bool = True
    enable_research_agent: bool = True
    enable_mcp_server: bool = True
    enable_metering: bool = True

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}
