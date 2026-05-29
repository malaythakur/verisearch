# Database Migrations

PostgreSQL schema migrations for the Agentic Research Search Engine.

## Structure

Migrations are numbered sequentially and applied in order:

```
migrations/
├── 001_tenants.sql
├── 002_api_keys.sql
├── 003_documents.sql
├── ...
```

## Running Migrations

```bash
# Apply all pending migrations
cd backend
python -m backend.migrations.apply

# Or using alembic (if configured)
alembic upgrade head
```

## Conventions

- Each migration file is idempotent where possible
- Down migrations are provided for reversible changes
- RLS (Row-Level Security) policies are applied per-table for tenant isolation
- All timestamps are stored in UTC
