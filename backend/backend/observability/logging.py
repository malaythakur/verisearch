"""
Structured JSON logging with tenant_id, request_id, endpoint (PII-redacted).

Uses structlog for structured logging with automatic context binding.
All log entries include tenant_id and request_id for correlation.
PII fields are automatically redacted before emission.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from typing import Any

import structlog


# PII patterns for automatic redaction
PII_PATTERNS = {
    "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "phone": re.compile(r"\+?1?\d{10,15}"),
    "ssn": re.compile(r"\b\d{3}-?\d{2}-?\d{4}\b"),
    "credit_card": re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"),
}

# Fields that should never appear in logs
SENSITIVE_FIELDS = frozenset({
    "api_key",
    "bearer_token",
    "password",
    "secret",
    "key_hash",
    "authorization",
})


def _redact_pii(value: str) -> str:
    """Redact PII patterns from a string value."""
    result = value
    for pattern_name, pattern in PII_PATTERNS.items():
        result = pattern.sub(f"[REDACTED_{pattern_name.upper()}]", result)
    return result


def _pii_redactor(logger: Any, method_name: str, event_dict: dict) -> dict:
    """Structlog processor that redacts PII from log events."""
    for key, value in list(event_dict.items()):
        # Remove sensitive fields entirely
        if key.lower() in SENSITIVE_FIELDS:
            event_dict[key] = "[REDACTED]"
            continue

        # Redact PII patterns in string values
        if isinstance(value, str):
            event_dict[key] = _redact_pii(value)

    return event_dict


def _add_service_context(logger: Any, method_name: str, event_dict: dict) -> dict:
    """Add service-level context to all log entries."""
    event_dict.setdefault("service", os.getenv("OTEL_SERVICE_NAME", "agentic-research-engine"))
    event_dict.setdefault("environment", os.getenv("DEPLOYMENT_ENV", "development"))
    event_dict.setdefault("version", os.getenv("SERVICE_VERSION", "0.1.0"))
    return event_dict


def setup_logging(
    level: str = "INFO",
    json_output: bool = True,
    log_file: str | None = None,
) -> None:
    """Configure structured logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        json_output: Whether to output JSON format (True for production).
        log_file: Optional file path for log output.
    """
    # Determine renderer based on environment
    if json_output:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        _add_service_context,
        _pii_redactor,
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        renderer,
    ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper(), logging.INFO),
    )

    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
        logging.getLogger().addHandler(file_handler)


def get_logger(
    name: str,
    tenant_id: str | None = None,
    request_id: str | None = None,
    endpoint: str | None = None,
) -> structlog.stdlib.BoundLogger:
    """Get a structured logger with bound context.

    Args:
        name: Logger name (typically the subsystem name).
        tenant_id: Tenant ID for multi-tenant log correlation.
        request_id: Request ID for end-to-end tracing.
        endpoint: API endpoint being served.

    Returns:
        Bound structlog logger with context.
    """
    logger = structlog.get_logger(name)

    bindings: dict[str, Any] = {"subsystem": name}
    if tenant_id:
        bindings["tenant_id"] = tenant_id
    if request_id:
        bindings["request_id"] = request_id
    if endpoint:
        bindings["endpoint"] = endpoint

    return logger.bind(**bindings)


def bind_request_context(
    tenant_id: str,
    request_id: str,
    endpoint: str,
) -> None:
    """Bind request context to all loggers in the current async context.

    Call this at the start of request handling to automatically include
    tenant_id, request_id, and endpoint in all subsequent log entries.

    Args:
        tenant_id: Authenticated tenant ID.
        request_id: Unique request identifier.
        endpoint: API endpoint path.
    """
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        tenant_id=tenant_id,
        request_id=request_id,
        endpoint=endpoint,
    )
