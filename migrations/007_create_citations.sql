-- Migration: 007_create_citations.sql
-- Description: Create citations table with offset ranges for answer and source text,
--              foreign key to document_versions, and optional links to research jobs
--              and sessions.
-- Requirements: R6.2 (answer_start < answer_end), R6.4 (source_start < source_end),
--              R6 (citation tracking), R7 (research job citations), R13.3 (tenant isolation)
-- Depends on: 001_create_tenants.sql, 003_create_documents.sql, 004_create_sessions.sql,
--             006_create_research.sql

-- UP
-- =============================================================================

-- Create the citations table
CREATE TABLE IF NOT EXISTS citations (
    citation_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id       UUID NOT NULL
                    REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    document_id     UUID NOT NULL,
    version         INTEGER NOT NULL,
    answer_start    INTEGER NOT NULL
                    CHECK (answer_start >= 0),
    answer_end      INTEGER NOT NULL
                    CHECK (answer_end > answer_start),
    source_start    INTEGER NOT NULL
                    CHECK (source_start >= 0),
    source_end      INTEGER NOT NULL
                    CHECK (source_end > source_start),
    emitted_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    job_id          UUID
                    REFERENCES research_jobs(job_id) ON DELETE SET NULL,
    session_id      UUID
                    REFERENCES sessions(session_id) ON DELETE SET NULL,

    -- Foreign key to document_versions composite key
    CONSTRAINT fk_citations_document_version
        FOREIGN KEY (document_id, version)
        REFERENCES document_versions(document_id, version)
);

-- =============================================================================
-- Indexes
-- =============================================================================

-- Index on tenant_id for tenant-scoped queries
CREATE INDEX IF NOT EXISTS idx_citations_tenant_id
    ON citations (tenant_id);

-- Index on (document_id, version) for citation lookups by document
CREATE INDEX IF NOT EXISTS idx_citations_document_version
    ON citations (document_id, version);

-- Index on job_id for research job citations
CREATE INDEX IF NOT EXISTS idx_citations_job_id
    ON citations (job_id)
    WHERE job_id IS NOT NULL;

-- Index on session_id for session memory citations
CREATE INDEX IF NOT EXISTS idx_citations_session_id
    ON citations (session_id)
    WHERE session_id IS NOT NULL;

-- Index on emitted_at for recency queries
CREATE INDEX IF NOT EXISTS idx_citations_emitted_at
    ON citations (emitted_at);

-- =============================================================================
-- Row-Level Security
-- Citations are strictly tenant-scoped.
-- =============================================================================

ALTER TABLE citations ENABLE ROW LEVEL SECURITY;
ALTER TABLE citations FORCE ROW LEVEL SECURITY;

-- RLS Policy: app_user can only access citations belonging to their tenant
CREATE POLICY citations_tenant_isolation_policy
    ON citations
    FOR ALL
    TO app_user
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID)
    WITH CHECK (tenant_id = current_setting('app.current_tenant_id')::UUID);

-- Grant table permissions to app_user
GRANT SELECT, INSERT, UPDATE, DELETE ON citations TO app_user;

-- DOWN
-- =============================================================================

-- Revoke permissions
REVOKE SELECT, INSERT, UPDATE, DELETE ON citations FROM app_user;

-- Drop RLS policy
DROP POLICY IF EXISTS citations_tenant_isolation_policy ON citations;

-- Disable RLS
ALTER TABLE citations DISABLE ROW LEVEL SECURITY;

-- Drop indexes
DROP INDEX IF EXISTS idx_citations_emitted_at;
DROP INDEX IF EXISTS idx_citations_session_id;
DROP INDEX IF EXISTS idx_citations_job_id;
DROP INDEX IF EXISTS idx_citations_document_version;
DROP INDEX IF EXISTS idx_citations_tenant_id;

-- Drop table
DROP TABLE IF EXISTS citations;
