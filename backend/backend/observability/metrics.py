"""
Prometheus metrics for all SLO targets.

Defines histograms, counters, and gauges for monitoring the platform's
performance against its SLO commitments.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator


# Histogram bucket definitions aligned with SLO targets
# Buckets are chosen to provide good resolution around SLO boundaries

# Auth resolution: SLO p95 ≤ 50ms
AUTH_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0)

# Search warm-cache: SLO p95 ≤ 800ms
SEARCH_LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 0.8, 1.0, 1.5, 2.0, 5.0)

# First answer token: SLO p95 ≤ 3s
ANSWER_LATENCY_BUCKETS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 10.0, 30.0)

# Research job return: SLO p95 ≤ 1s
RESEARCH_LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 5.0)

# Parser: SLO p95 ≤ 100ms
PARSER_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.25, 0.5)

# Ingest pipeline: SLO p95 ≤ 60min, p99 ≤ 4h
INGEST_LATENCY_BUCKETS = (60, 300, 600, 1800, 3600, 7200, 14400, 28800)

# Default HTTP request buckets
HTTP_LATENCY_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)


@dataclass
class MetricsRegistry:
    """Registry of all application metrics.

    In production, these map to Prometheus metrics via the prometheus_client library.
    This abstraction allows testing without requiring the full Prometheus stack.
    """

    # Histograms for latency SLOs
    search_request_duration: dict[str, list[float]] = field(default_factory=dict)
    auth_resolution_duration: list[float] = field(default_factory=list)
    answer_first_token_duration: list[float] = field(default_factory=list)
    research_job_creation_duration: list[float] = field(default_factory=list)
    parser_duration: list[float] = field(default_factory=list)
    ingest_pipeline_duration: list[float] = field(default_factory=list)
    http_request_duration: dict[str, list[float]] = field(default_factory=dict)

    # Counters
    http_requests_total: dict[str, int] = field(default_factory=dict)
    rate_limit_rejections_total: dict[str, int] = field(default_factory=dict)
    dlq_messages_total: int = 0
    auth_failures_total: int = 0
    search_errors_total: int = 0

    # Gauges
    active_research_jobs: int = 0
    active_connections: int = 0
    cache_hit_ratio: float = 0.0
    index_document_count: int = 0

    def record_search_latency(self, duration_s: float, cache: str = "warm") -> None:
        """Record a search request latency measurement."""
        key = f"cache={cache}"
        if key not in self.search_request_duration:
            self.search_request_duration[key] = []
        self.search_request_duration[key].append(duration_s)

    def record_auth_latency(self, duration_s: float) -> None:
        """Record an auth resolution latency measurement."""
        self.auth_resolution_duration.append(duration_s)

    def record_first_token_latency(self, duration_s: float) -> None:
        """Record time to first answer token."""
        self.answer_first_token_duration.append(duration_s)

    def record_research_job_latency(self, duration_s: float) -> None:
        """Record research job creation latency."""
        self.research_job_creation_duration.append(duration_s)

    def record_parser_latency(self, duration_s: float) -> None:
        """Record parser execution latency."""
        self.parser_duration.append(duration_s)

    def record_ingest_latency(self, duration_s: float) -> None:
        """Record ingest pipeline latency."""
        self.ingest_pipeline_duration.append(duration_s)

    def record_http_request(
        self, method: str, endpoint: str, status: int, duration_s: float
    ) -> None:
        """Record an HTTP request with method, endpoint, status, and duration."""
        key = f"{method}:{endpoint}:{status}"
        if key not in self.http_request_duration:
            self.http_request_duration[key] = []
        self.http_request_duration[key].append(duration_s)

        status_key = f"{method}:{endpoint}:{status}"
        self.http_requests_total[status_key] = (
            self.http_requests_total.get(status_key, 0) + 1
        )

    def increment_rate_limit_rejections(self, tenant_id: str) -> None:
        """Increment rate limit rejection counter for a tenant."""
        self.rate_limit_rejections_total[tenant_id] = (
            self.rate_limit_rejections_total.get(tenant_id, 0) + 1
        )

    def increment_dlq(self) -> None:
        """Increment dead letter queue message counter."""
        self.dlq_messages_total += 1

    @contextmanager
    def timer(self, metric_name: str) -> Generator[None, None, None]:
        """Context manager to time an operation and record it.

        Args:
            metric_name: One of 'search', 'auth', 'first_token',
                        'research_job', 'parser', 'ingest'.
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            recorder = getattr(self, f"record_{metric_name}_latency", None)
            if recorder:
                recorder(duration)


# Global metrics registry
_metrics: MetricsRegistry | None = None


def setup_metrics(
    enable_prometheus: bool = True,
    port: int = 9090,
) -> MetricsRegistry:
    """Initialize the metrics registry.

    Args:
        enable_prometheus: Whether to start a Prometheus HTTP server.
        port: Port for the Prometheus metrics endpoint.

    Returns:
        The global MetricsRegistry instance.
    """
    global _metrics
    _metrics = MetricsRegistry()

    if enable_prometheus:
        try:
            # In production, start prometheus_client HTTP server
            # from prometheus_client import start_http_server
            # start_http_server(port)
            pass
        except ImportError:
            pass

    return _metrics


def get_metrics() -> MetricsRegistry:
    """Get the global metrics registry.

    Returns:
        The MetricsRegistry instance (creates one if not initialized).
    """
    global _metrics
    if _metrics is None:
        _metrics = MetricsRegistry()
    return _metrics


# Prometheus metrics exposition format (for /metrics endpoint)
PROMETHEUS_METRICS_CONFIG = """
# HELP search_request_duration_seconds Search request latency histogram
# TYPE search_request_duration_seconds histogram
search_request_duration_seconds_bucket{cache="warm",le="0.05"} 0
search_request_duration_seconds_bucket{cache="warm",le="0.1"} 0
search_request_duration_seconds_bucket{cache="warm",le="0.25"} 0
search_request_duration_seconds_bucket{cache="warm",le="0.5"} 0
search_request_duration_seconds_bucket{cache="warm",le="0.8"} 0
search_request_duration_seconds_bucket{cache="warm",le="1.0"} 0
search_request_duration_seconds_bucket{cache="warm",le="+Inf"} 0

# HELP auth_resolution_duration_seconds Auth resolution latency histogram
# TYPE auth_resolution_duration_seconds histogram
auth_resolution_duration_seconds_bucket{le="0.005"} 0
auth_resolution_duration_seconds_bucket{le="0.01"} 0
auth_resolution_duration_seconds_bucket{le="0.025"} 0
auth_resolution_duration_seconds_bucket{le="0.05"} 0
auth_resolution_duration_seconds_bucket{le="+Inf"} 0

# HELP answer_first_token_duration_seconds Time to first answer token histogram
# TYPE answer_first_token_duration_seconds histogram
answer_first_token_duration_seconds_bucket{le="0.5"} 0
answer_first_token_duration_seconds_bucket{le="1.0"} 0
answer_first_token_duration_seconds_bucket{le="2.0"} 0
answer_first_token_duration_seconds_bucket{le="3.0"} 0
answer_first_token_duration_seconds_bucket{le="+Inf"} 0

# HELP research_job_creation_duration_seconds Research job creation latency histogram
# TYPE research_job_creation_duration_seconds histogram
research_job_creation_duration_seconds_bucket{le="0.1"} 0
research_job_creation_duration_seconds_bucket{le="0.5"} 0
research_job_creation_duration_seconds_bucket{le="1.0"} 0
research_job_creation_duration_seconds_bucket{le="+Inf"} 0

# HELP parser_duration_seconds Filter parser latency histogram
# TYPE parser_duration_seconds histogram
parser_duration_seconds_bucket{le="0.01"} 0
parser_duration_seconds_bucket{le="0.05"} 0
parser_duration_seconds_bucket{le="0.1"} 0
parser_duration_seconds_bucket{le="+Inf"} 0

# HELP ingest_pipeline_duration_seconds Ingest pipeline latency histogram
# TYPE ingest_pipeline_duration_seconds histogram
ingest_pipeline_duration_seconds_bucket{le="60"} 0
ingest_pipeline_duration_seconds_bucket{le="300"} 0
ingest_pipeline_duration_seconds_bucket{le="3600"} 0
ingest_pipeline_duration_seconds_bucket{le="14400"} 0
ingest_pipeline_duration_seconds_bucket{le="+Inf"} 0

# HELP http_requests_total Total HTTP requests counter
# TYPE http_requests_total counter
http_requests_total{method="GET",endpoint="/v1/search",status="200"} 0

# HELP rate_limit_rejections_total Rate limit rejections counter
# TYPE rate_limit_rejections_total counter
rate_limit_rejections_total{tenant_id=""} 0

# HELP dlq_messages_total Dead letter queue messages counter
# TYPE dlq_messages_total counter
dlq_messages_total 0

# HELP active_research_jobs Current active research jobs gauge
# TYPE active_research_jobs gauge
active_research_jobs 0

# HELP active_connections Current active connections gauge
# TYPE active_connections gauge
active_connections 0
"""
