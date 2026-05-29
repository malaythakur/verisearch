-- Migration: 004_create_sessions.sql
-- Description: Create sessions table with tenant-scoped isolation, retention_days,
--              and memory ring buffers for citations and document IDs.
-- Requirements: R8.1 (retention_days [1..90] default 14), R8.2 (memory ring buffers),
--              R8.3 (tenant isolation), R8.4 (expiry and deletion), R13.3 (RLS)
-- Depends on: 001_create_tenants.sql

-- UP
-- =============================================================================

-- Create the sessions table
CREATE TABLE IF NOT EXISTS sessions (
    session_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id           UUID NOT NULL
                        REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    retention_days      INTEGER NOT NULL DEFAULT 14
                        CHECK (retention_days BETWEEN 1 AND 90),
    expires_at          TIMESTAMPTZ NOT NULL,
    state               VARCHAR(20) NOT NULL DEFAULT 'active'
                        CHECK (state IN ('active', 'expired', 'deleted')),

    -- Ring buffer of up to 50 most recent citation references (R8.2)
    -- Application layer enforces the ≤50 element cap on insert/update
    memory_citations    JSONB NOT NULL DEFAULT '[]'::jsonb,

    -- Ring buffer of up to 20 most recent distinct document IDs (R8.2)
    -- Application layer enforces the ≤20 element cap on insert/update
    memory_doc_ids      JSONB NOT NULL DEFAULT '[]'::jsonb
);

-- =============================================================================
-- Trigger to compute expires_at from created_at + retention_days on INSERT
-- This ensures expires_at is always consistent with the retention policy.
-- =============================================================================

CREATE OR REPLACE FUNCTION compute_session_expires_at()
RETURNS TRIGGER AS $$
BEGIN
    -- Compute expires_at as created_at + retention_days interval
    NEW.expires_at := NEW.created_at + (NEW.retention_days || ' days')::INTERVAL;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_compute_session_expires_at
    BEFORE INSERT ON sessions
    FOR EACH ROW
    EXECUTE FUNCTION compute_session_expires_at();

-- =============================================================================
-- Check constraints for ring buffer sizes (defense in depth)
-- The application layer is the primary enforcer, but these constraints
-- provide a safety net at the database level.
-- =============================================================================

ALTER TABLE sessions
    ADD CONSTRAINT chk_memory_citations_max_size
    CHECK (jsonb_array_length(memory_citations) <= 50);

ALTER TABLE sessions
    ADD CONSTRAINT chk_memory_doc_ids_max_size
    CHECK (jsonb_array_length(memory_doc_ids) <= 20);

-- =============================================================================
-- Indexes
-- =============================================================================

-- Index on tenant_id for tenant-scoped queries
CREATE INDEX IF NOT EXISTS idx_sessions_tenant_id
    ON sessions (tenant_id);

-- Index on expires_at for expiry sweep (R8.4)
CREATE INDEX IF NOT EXISTS idx_sessions_expires_at
    ON sessions (expires_at);

-- Index on state for filtering active sessions
CREATE INDEX IF NOT EXISTS idx_sessions_state
    ON sessions (state);

-- Composite index on (tenant_id, state) for common query pattern
-- e.g., "find all active sessions for this tenant"
CREATE INDEX IF NOT EXISTS idx_sessions_tenant_id_state
    ON sessions (tenant_id, state);

-- =============================================================================
-- Row-Level Security
-- Sessions are strictly tenant-scoped; no global/shared sessions.
-- =============================================================================

ALTER TABLE sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE sessions FORCE ROW LEVEL SECURITY;

-- RLS Policy: app_user can only access sessions belonging to their tenant
CREATE POLICY sessions_tenant_isolation_policy
    ON sessions
    FOR ALL
    TO app_user
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID)
    WITH CHECK (tenant_id = current_setting('app.current_tenant_id')::UUID);

-- Grant table permissions to app_user
GRANT SELECT, INSERT, UPDATE, DELETE ON sessions TO app_user;

-- DOWN
-- =============================================================================

-- Revoke permissions
REVOKE SELECT, INSERT, UPDATE, DELETE ON sessions FROM app_user;

-- Drop RLS policy
DROP POLICY IF EXISTS sessions_tenant_isolation_policy ON sessions;

-- Disable RLS
ALTER TABLE sessions DISABLE ROW LEVEL SECURITY;

-- Drop indexes
DROP INDEX IF EXISTS idx_sessions_tenant_id_state;
DROP INDEX IF EXISTS idx_sessions_state;
DROP INDEX IF EXISTS idx_sessions_expires_at;
DROP INDEX IF EXISTS idx_sessions_tenant_id;

-- Drop constraints
ALTER TABLE sessions DROP CONSTRAINT IF EXISTS chk_memory_doc_ids_max_size;
ALTER TABLE sessions DROP CONSTRAINT IF EXISTS chk_memory_citations_max_size;

-- Drop trigger and function
DROP TRIGGER IF EXISTS trg_compute_session_expires_at ON sessions;
DROP FUNCTION IF EXISTS compute_session_expires_at();

-- Drop table
DROP TABLE IF EXISTS sessions;
