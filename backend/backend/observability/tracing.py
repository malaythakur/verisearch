"""
OpenTelemetry tracing setup with request_id propagation.

All subsystems are instrumented with distributed tracing to enable
end-to-end request tracking across the entire platform.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Generator, Optional

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import StatusCode, Span
from opentelemetry.trace.propagation import set_span_in_context


# Service metadata
SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "agentic-research-engine")
SERVICE_VERSION = os.getenv("SERVICE_VERSION", "0.1.0")
DEPLOYMENT_ENV = os.getenv("DEPLOYMENT_ENV", "development")

# OTLP exporter endpoint
OTLP_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")


def setup_tracing(
    service_name: str | None = None,
    exporter: str = "otlp",
) -> TracerProvider:
    """Initialize OpenTelemetry tracing for the service.

    Args:
        service_name: Override service name (defaults to env var).
        exporter: Exporter type - 'otlp', 'console', or 'none'.

    Returns:
        Configured TracerProvider.
    """
    resource = Resource.create(
        {
            "service.name": service_name or SERVICE_NAME,
            "service.version": SERVICE_VERSION,
            "deployment.environment": DEPLOYMENT_ENV,
            "service.namespace": "agentic-research",
        }
    )

    provider = TracerProvider(resource=resource)

    if exporter == "otlp":
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )

            otlp_exporter = OTLPSpanExporter(endpoint=OTLP_ENDPOINT)
            provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
        except ImportError:
            # Fall back to console if OTLP exporter not installed
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    elif exporter == "console":
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    # 'none' = no exporter (testing)

    trace.set_tracer_provider(provider)
    return provider


def get_tracer(name: str) -> trace.Tracer:
    """Get a tracer for the given subsystem.

    Args:
        name: Subsystem name (e.g., 'api_gateway', 'retriever', 'answer_engine').

    Returns:
        OpenTelemetry Tracer instance.
    """
    return trace.get_tracer(
        instrumenting_module_name=f"agentic_research.{name}",
        tracer_provider=trace.get_tracer_provider(),
    )


@contextmanager
def trace_span(
    tracer: trace.Tracer,
    name: str,
    attributes: dict[str, Any] | None = None,
    tenant_id: str | None = None,
    request_id: str | None = None,
) -> Generator[Span, None, None]:
    """Create a traced span with standard attributes.

    Automatically propagates tenant_id and request_id through the span context.

    Args:
        tracer: The tracer to use.
        name: Span name (e.g., 'search.execute', 'auth.resolve').
        attributes: Additional span attributes.
        tenant_id: Tenant ID for multi-tenant isolation tracking.
        request_id: Request ID for end-to-end correlation.

    Yields:
        The active span.
    """
    span_attributes = {}

    if tenant_id:
        span_attributes["tenant.id"] = tenant_id
    if request_id:
        span_attributes["request.id"] = request_id
    if attributes:
        span_attributes.update(attributes)

    with tracer.start_as_current_span(name, attributes=span_attributes) as span:
        try:
            yield span
        except Exception as exc:
            span.set_status(StatusCode.ERROR, str(exc))
            span.record_exception(exc)
            raise


# Subsystem-specific tracers (lazy initialization)
_tracers: dict[str, trace.Tracer] = {}


def get_subsystem_tracer(subsystem: str) -> trace.Tracer:
    """Get or create a tracer for a specific subsystem.

    Supported subsystems:
    - api_gateway
    - auth_service
    - retriever
    - pipeline_engine
    - answer_engine
    - research_agent
    - crawler
    - indexer
    - session_store
    - mcp_server
    - audit_log
    - pii_redactor
    """
    if subsystem not in _tracers:
        _tracers[subsystem] = get_tracer(subsystem)
    return _tracers[subsystem]
