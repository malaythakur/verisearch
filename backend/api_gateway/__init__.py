"""API Gateway - REST, SSE, and WebSocket endpoint termination.

Provides the create_app() factory function that returns a configured FastAPI
application with the middleware chain: request_id → auth → rate_limit → pii_redact → route.
"""

from backend.api_gateway.app import create_app

__all__ = ["create_app"]
