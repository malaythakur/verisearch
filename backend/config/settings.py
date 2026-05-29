"""Application settings loaded from environment variables using Pydantic v2 BaseSettings."""

from enum import Enum

from pydantic import Field
from pydantic_settings import BaseSettings


class AppEnvironment(str, Enum):
    """Application deployment environment."""

    DEV = "dev"
    STAGING = "staging"
    PRODUCTION = "production"


class Settings(BaseSettings):
    """Main application settings sourced from environment variables.

    All settings can be overridden via environment variables.
    Defaults are tuned for local development with Docker Compose.
    """

    # --- PostgreSQL ---
    postgres_host: str = Field(default="localhost", description="PostgreSQL host")
    postgres_port: int = Field(default=5432, description="PostgreSQL port")
    postgres_db: str = Field(default="agentic_research", description="PostgreSQL database name")
    postgres_user: str = Field(default="agentic", description="PostgreSQL user")
    postgres_password: str = Field(default="agentic_dev", description="PostgreSQL password")
    database_url: str = Field(
        default="postgresql://agentic:agentic_dev@localhost:5432/agentic_research",
        description="Full PostgreSQL connection URL",
    )

    # --- OpenSearch ---
    opensearch_url: str = Field(default="http://localhost:9200", description="OpenSearch cluster URL")

    # --- Redis ---
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis connection URL")

    # --- Kafka ---
    kafka_bootstrap_servers: str = Field(
        default="localhost:19092", description="Kafka bootstrap servers (comma-separated)"
    )

    # --- S3-compatible object store ---
    aws_endpoint_url: str = Field(default="http://localhost:9000", description="S3-compatible endpoint URL")
    s3_bucket: str = Field(default="agentic-research-docs", description="S3 bucket for document storage")
    aws_access_key_id: str = Field(default="minioadmin", description="AWS/S3 access key ID")
    aws_secret_access_key: str = Field(default="minioadmin", description="AWS/S3 secret access key")
    aws_default_region: str = Field(default="us-east-1", description="AWS/S3 default region")

    # --- Application ---
    app_env: AppEnvironment = Field(default=AppEnvironment.DEV, description="Application environment")
    log_level: str = Field(default="INFO", description="Logging level")
    service_name: str = Field(default="agentic-research", description="Service name for observability")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}
