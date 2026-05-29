# Agentic Research Search Engine

A multi-tenant SaaS platform that combines neural search, streaming answers with citations, programmable retrieval pipelines, and agentic deep research — all accessible via REST, SSE, WebSocket, and MCP interfaces.

## Key Features

- **Agentic Deep Research** — Multi-hop reasoning with inspectable plans, tool-use loops, and structured reports
- **Streaming Answers with Live Citations** — Token-by-token answers with verifiable source references and offset ranges
- **Programmable Retrieval Pipelines** — User-defined filters, rerankers, and transforms composed via a typed DSL
- **Provenance & Credibility Scoring** — AI-generated content detection and credibility signals on every result
- **Persistent Research Sessions** — Context carried across queries within a tenant
- **Query Filter DSL** — Textual filter language with provable parse/print round-trip guarantees

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Clients                                  │
│   Python SDK  │  TypeScript SDK  │  MCP Agent  │  REST/SSE/WS   │
└───────┬───────┴────────┬─────────┴──────┬──────┴───────┬────────┘
        │                │                │              │
┌───────▼────────────────▼────────────────▼──────────────▼────────┐
│                    Edge / Control Plane                           │
│  API Gateway  │  MCP Server  │  Auth Service  │  Rate Limiter   │
└───────┬────────────────┬────────────────┬──────────────┬────────┘
        │                │                │              │
┌───────▼────────────────▼────────────────▼──────────────▼────────┐
│                       Query Plane                                 │
│  Retriever  │  Pipeline Engine  │  Answer Engine  │  Research    │
└───────┬─────────────────────────────────────────────────────────┘
        │
┌───────▼─────────────────────────────────────────────────────────┐
│                       Ingest Plane                                │
│  Crawler  │  Indexer  │  Provenance Scorer  │  Embedding Gen     │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 20+ with pnpm 9+
- Docker & Docker Compose
- Poetry (Python package manager)

### 1. Clone and Install

```bash
git clone <repository-url>
cd search

# Backend (Python)
cd backend
poetry install
cd ..

# TypeScript SDK
pnpm install
```

### 2. Start Infrastructure

```bash
docker compose up -d
```

This starts PostgreSQL, OpenSearch, Redis, Kafka (Redpanda), and MinIO for local development.

### 3. Run Migrations

```bash
# Apply all database migrations
cd backend
poetry run python -m scripts.migrate
```

### 4. Run the Backend

```bash
cd backend
poetry run uvicorn backend.api_gateway.app:create_app --factory --reload --port 8000
```

The API is now available at `http://localhost:8000`. Health check: `GET /health`.

### 5. Run Tests

```bash
# Backend tests
cd backend
poetry run pytest

# With coverage
poetry run pytest --cov=backend --cov-report=html

# Property-based tests with CI profile (100 examples)
HYPOTHESIS_PROFILE=ci poetry run pytest -k "property or Property"

# TypeScript SDK tests
pnpm --filter @agentic-research/sdk test
```

## Project Structure

```
search/
├── backend/                    # Python backend (FastAPI)
│   ├── backend/
│   │   ├── api_gateway/        # HTTP framework, middleware, routes
│   │   ├── auth/               # API key auth, tenant isolation
│   │   ├── audit_log/          # Append-only audit ledger
│   │   ├── rate_limiter/       # Token-bucket rate limiting, metering
│   │   ├── pii_redactor/       # PII detection and redaction
│   │   ├── query_filter/       # Filter DSL parser & printer
│   │   ├── crawler/            # Web crawling with robots.txt
│   │   ├── indexer/            # Document ingestion & versioning
│   │   ├── provenance_scorer/  # Credibility & AI-detection scoring
│   │   ├── retriever/          # Neural, keyword, hybrid search
│   │   ├── pipeline_engine/    # Programmable retrieval pipelines
│   │   ├── answer_engine/      # Streaming answers with citations
│   │   ├── research_agent/     # Multi-hop agentic research
│   │   ├── session_store/      # Persistent research sessions
│   │   ├── mcp_server/         # Model Context Protocol interface
│   │   ├── storage/            # S3-compatible object store
│   │   ├── config/             # Settings, feature flags, constants
│   │   └── observability/      # Tracing, logging, metrics
│   ├── tests/                  # Unit, property, and load tests
│   └── pyproject.toml
├── sdks/
│   ├── python/                 # Python SDK
│   └── typescript/             # TypeScript SDK
├── migrations/                 # PostgreSQL migrations (000-009)
├── deploy/
│   ├── k8s/                    # Kubernetes manifests
│   ├── helm/                   # Helm charts
│   ├── kafka/                  # Kafka topic configuration
│   └── secrets/                # Secrets management (ExternalSecrets)
├── docker-compose.yml          # Local dev infrastructure
├── .github/workflows/          # CI/CD pipelines
└── .pre-commit-config.yaml     # Pre-commit hooks
```

## API Reference

All endpoints are prefixed with `/v1`. Authentication is via `Authorization: Bearer <api_key>`.

### Search

```bash
# Neural, keyword, or hybrid search
POST /v1/search
{
  "query": "machine learning algorithms",
  "mode": "hybrid",
  "num_results": 10,
  "min_credibility": 0.5
}
```

### Find Similar

```bash
# Find documents similar to a URL
POST /v1/find_similar
{
  "url": "https://example.com/article",
  "num_results": 10
}
```

### Content Retrieval

```bash
# Batch fetch cleaned text, highlights, summaries
POST /v1/contents
{
  "document_ids": ["doc-1", "doc-2"],
  "highlights": true,
  "query": "machine learning",
  "summary": true
}
```

### Streaming Answers

```bash
# SSE stream with token-by-token answers and citations
POST /v1/answer
{
  "query": "What are the latest advances in AI?",
  "stream": true,
  "session_id": "optional-session-id"
}
```

Events: `token`, `citation`, `done`, `error`

### Research Jobs

```bash
# Launch a multi-hop research job
POST /v1/research
{
  "research_goal": "Analyze the impact of transformer architectures on NLP",
  "max_steps": 10,
  "output_schema": {"type": "object", "properties": {"summary": {"type": "string"}}}
}

# Get job report
GET /v1/research/{job_id}

# Stream job events (supports Last-Event-ID for reconnection)
GET /v1/research/{job_id}/events
```

### Sessions

```bash
# Create a persistent research session
POST /v1/sessions
{"retention_days": 14}

# Delete a session
DELETE /v1/sessions/{session_id}
```

### Pipelines

```bash
# Create a retrieval pipeline
POST /v1/pipelines
{
  "name": "My Pipeline",
  "steps": [
    {"name": "domain_filter", "config": {"domains": ["example.com"]}},
    {"name": "credibility_reranker", "config": {"credibility_weight": 0.3}},
    {"name": "dedup_transform"}
  ]
}
```

### MCP Tools

The MCP server exposes `search`, `find_similar`, `contents`, `answer`, and `research` as tool calls with JSON Schema validation.

## SDK Usage

### Python

```python
from agentic_research_sdk import AgenticResearchClient, SearchMode

client = AgenticResearchClient(
    base_url="https://api.example.com/v1",
    api_key="your-api-key"
)

# Search
results = await client.search("quantum computing", mode=SearchMode.HYBRID)

# Streaming answer
async for event in client.answer(query="What is quantum entanglement?"):
    if event.event_type == "token":
        print(event.data["text"], end="")
    elif event.event_type == "done":
        print("\n\nCitations:", event.data["citations"])

# Research
job_id = await client.create_research(
    research_goal="Analyze recent breakthroughs in quantum computing"
)
job = await client.get_research_job(job_id)
```

### TypeScript

```typescript
import { AgenticResearchClient } from '@agentic-research/sdk';

const client = new AgenticResearchClient({
  baseUrl: 'https://api.example.com/v1',
  apiKey: 'your-api-key',
});

// Search
const results = await client.search({ query: 'quantum computing', mode: 'hybrid' });

// Streaming answer
for await (const event of client.answer({ query: 'What is quantum entanglement?' })) {
  if (event.event_type === 'token') process.stdout.write(event.data.text);
  if (event.event_type === 'done') console.log('\nDone:', event.data.answer);
}
```

## Configuration

Configuration is via environment variables. Copy `.env.example` to `.env` for local development:

```bash
cp .env.example .env
```

Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql://agentic:agentic_dev@localhost:5432/agentic_research` | PostgreSQL connection |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis for rate limiting |
| `OPENSEARCH_URL` | `http://localhost:9200` | OpenSearch cluster |
| `KAFKA_BOOTSTRAP_SERVERS` | `localhost:19092` | Kafka brokers |
| `AWS_ENDPOINT_URL` | `http://localhost:9000` | S3-compatible store (MinIO) |
| `APP_ENV` | `dev` | Environment (dev/staging/production) |
| `LOG_LEVEL` | `INFO` | Logging level |

### Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `ENABLE_PII_REDACTION` | `true` | PII detection and redaction |
| `ENABLE_PROVENANCE_SCORING` | `true` | Credibility/AI scoring |
| `ENABLE_RESEARCH_AGENT` | `true` | Multi-hop research |
| `ENABLE_MCP_SERVER` | `true` | MCP tool interface |
| `ENABLE_METERING` | `true` | Usage metering |

### Tenant Defaults

Override with `TENANT_DEFAULT_` prefix:

| Setting | Default | Range |
|---------|---------|-------|
| `RATE_LIMIT_PER_MINUTE` | 60 | — |
| `SESSION_RETENTION_DAYS` | 14 | [1, 90] |
| `AUDIT_RETENTION_DAYS` | 365 | [365, 2555] |
| `KEY_ROTATION_GRACE_SECONDS` | 3600 | [1, 86400] |
| `CRAWLER_MAX_CONCURRENCY_PER_HOST` | 2 | [1, 8] |
| `RESEARCH_MAX_STEPS` | 32 | [1, 32] |

## Testing

### Test Categories

- **Unit tests** — Fast, no external dependencies. Run with `pytest`.
- **Property-based tests** — Hypothesis-powered invariant verification. 46 properties covering all correctness guarantees.
- **Load tests** — SLO validation under load. Located in `tests/load/`.
- **Integration tests** — Require Docker infrastructure. Marked with `@pytest.mark.integration`.

### Running Specific Test Groups

```bash
# All tests
poetry run pytest

# Property tests only
poetry run pytest -k "property or Property"

# Load tests
poetry run pytest tests/load/

# Specific subsystem
poetry run pytest tests/test_retriever.py tests/test_retriever_properties.py

# With verbose output
poetry run pytest -v --tb=short
```

### Property-Based Testing

The system uses 46 formal correctness properties verified via Hypothesis:

| Property | Requirement | Description |
|----------|-------------|-------------|
| 1-2 | R2.3, R2.4 | Indexing idempotence and version monotonicity |
| 3-4 | R1.4, R1.6 | Crawler concurrency and opt-out enforcement |
| 5-7 | R3.3, R3.4, R4.2 | Search ordering, determinism, find_similar exclusion |
| 8 | R3.5-R3.7 | Bounds-violation rejection |
| 9-14 | R5, R6 | Content retrieval order, answer streaming invariants |
| 15-18 | R7 | Research event monotonicity, budget enforcement |
| 19 | R13.3 | Tenant isolation uniformity |
| 20-21 | R10 | Provenance score bounds and threshold semantics |
| 22-25 | R11 | Filter DSL round-trip and error reporting |
| 26-28 | R9 | Pipeline composition and timeout behavior |
| 29-30 | R8 | Session memory bounds and expiry |
| 31-34 | R13 | Auth revocation, rotation, audit shape |
| 35-37 | R14 | Rate limit headers, metering dedup, outage resilience |
| 38 | R15.2 | PII absence from redacted output |
| 39-41 | R15 | Audit append-only, blocking, legal hold |
| 42 | R12 | MCP schema validation |
| 43-46 | R16 | SDK streaming, error mapping, surface equivalence |

## Deployment

### Staging

Staging deploys automatically on merge to `main`:

```bash
git push origin main  # Triggers .github/workflows/deploy-staging.yml
```

### Production

Production uses canary rollout (5% → 25% → 50% → 100%):

```bash
# Trigger via GitHub Actions workflow_dispatch
# Requires the SHA from a successful staging deployment
```

### Infrastructure

- **Kubernetes** — Manifests in `deploy/k8s/` with HPA, PDB, and topology spread
- **Helm** — Chart in `deploy/helm/agentic-research/` with environment-specific values
- **Kafka** — Topic configuration in `deploy/kafka/topics.yaml` (Strimzi CRDs)
- **Secrets** — AWS Secrets Manager + ExternalSecrets Operator in `deploy/secrets/`

## Observability

### Tracing

OpenTelemetry traces propagate `request_id` across all subsystems. Configure the OTLP endpoint:

```bash
OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317
```

### Logging

Structured JSON logs include `tenant_id`, `request_id`, and `endpoint` on every line. PII is automatically redacted before logging.

### Metrics

Prometheus metrics are exposed on `:9090/metrics` with histogram buckets aligned to SLO targets:

- `auth_resolution_duration_seconds` (target: p95 ≤ 50ms)
- `search_request_duration_seconds` (target: warm-cache p95 ≤ 800ms)
- `answer_first_token_duration_seconds` (target: p95 ≤ 3s)
- `research_job_creation_duration_seconds` (target: p95 ≤ 1s)
- `parser_duration_seconds` (target: p95 ≤ 100ms)
- `ingest_pipeline_duration_seconds` (target: p95 ≤ 60min)

### Alerting

Prometheus alerting rules in `backend/tests/load/slo_alerts.yml` cover all SLO targets with warning and critical thresholds.

## Security

- **Tenant Isolation** — Row-Level Security (RLS) at the database level; uniform 404 responses prevent resource enumeration
- **API Key Management** — Argon2id hashing, prefix-indexed lookup, rotation with configurable grace period
- **PII Redaction** — Automatic detection and redaction of emails, phones, SSNs, EU IDs, and credit cards before audit/analytics
- **Audit Log** — Append-only ledger with immutability triggers; privileged actions blocked on audit failure
- **Rate Limiting** — Per-tenant token-bucket with Redis; durable local buffer during outages

## Contributing

1. Install pre-commit hooks: `pre-commit install`
2. Run linting: `poetry run ruff check .` and `pnpm lint`
3. Run tests before pushing: `poetry run pytest`
4. Property tests must pass with 100 examples: `HYPOTHESIS_PROFILE=ci poetry run pytest -k property`

## License

Proprietary. All rights reserved.
