"""
Observability module for the Agentic Research Search Engine.

Provides:
- OpenTelemetry tracing with request_id propagation
- Structured JSON logging with PII redaction
- Prometheus metrics for all SLO targets
"""

from backend.observability.tracing import setup_tracing, get_tracer, trace_span
from backend.observability.logging import setup_logging, get_logger
from backend.observability.metrics import setup_metrics, get_metrics

__all__ = [
    "setup_tracing",
    "get_tracer",
    "trace_span",
    "setup_logging",
    "get_logger",
    "setup_metrics",
    "get_metrics",
]
