# Agentic Research Search Engine - Backend

Python backend services for the Agentic Research Search Engine.

## Package Structure

```
backend/
├── api_gateway/       # REST + SSE + WebSocket endpoints
├── auth/              # API key authentication and tenant isolation
├── crawler/           # Ethical web crawling with robots.txt compliance
├── indexer/           # Document ingestion and index management
├── retriever/         # Neural, keyword, and hybrid search
├── pipeline_engine/   # Programmable retrieval pipeline execution
├── answer_engine/     # Streaming answer generation with citations
├── research_agent/    # Multi-hop agentic research orchestration
├── session_store/     # Persistent research session memory
├── provenance_scorer/ # Credibility and AI-content scoring
├── query_filter/      # Query Filter DSL parser and printer
├── pii_redactor/      # PII detection and redaction
├── audit_log/         # Append-only audit ledger
├── rate_limiter/      # Per-tenant rate limiting and metering
└── mcp_server/        # Model Context Protocol interface
```

## Development

```bash
poetry install
poetry run pytest
poetry run ruff check .
```
