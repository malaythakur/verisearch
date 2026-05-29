-- Migration: 003_create_documents.sql
-- Description: Create documents and document_versions tables with stable document_id,
--              version monotonicity enforcement, content_hash, and provenance fields.
-- Requirements: R2.3 (stable document_id, version increment), R2.4 (idempotent re-index),
--              R10 (provenance scoring), R13.3 (tenant isolation)
-- Depends on: 001_create_tenants.sql

-- UP
-- =============================================================================

-- Create the documents table (one row per unique canonical URL)
CREATE TABLE IF NOT EXISTS documents (
    document_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id       UUID
                    REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    canonical_url   TEXT NOT NULL,
    first_seen_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- One document per canonical URL (global uniqueness)
    CONSTRAINT uq_documents_canonical_url UNIQUE (canonical_url)
);

-- Index on tenant_id for tenant-scoped queries (nullable — null means global crawl doc)
CREATE INDEX IF NOT EXISTS idx_documents_tenant_id
    ON documents (tenant_id)
    WHERE tenant_id IS NOT NULL;

-- Create the document_versions table (versioned content per document)
CREATE TABLE IF NOT EXISTS document_versions (
    document_id             UUID NOT NULL
                            REFERENCES documents(document_id) ON DELETE CASCADE,
    version                 INTEGER NOT NULL
                            CHECK (version >= 1),
    content_hash            CHAR(64) NOT NULL,
    cleaned_text_uri        TEXT NOT NULL,
    fetch_timestamp_utc     TIMESTAMPTZ NOT NULL,
    http_status             SMALLINT NOT NULL,
    content_type            VARCHAR(255) NOT NULL,
    source_url              TEXT NOT NULL,
    last_seen_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    published_at            TIMESTAMPTZ,

    -- Provenance fields (R10) — embedded rather than separate table for query efficiency
    credibility_score       DOUBLE PRECISION
                            CHECK (credibility_score BETWEEN 0.0 AND 1.0),
    ai_generated_likelihood DOUBLE PRECISION
                            CHECK (ai_generated_likelihood BETWEEN 0.0 AND 1.0),
    scored_at               TIMESTAMPTZ,

    PRIMARY KEY (document_id, version)
);

-- Index on content_hash for idempotent re-indexing lookups (R2.4)
CREATE INDEX IF NOT EXISTS idx_document_versions_content_hash
    ON document_versions (content_hash);

-- Index on last_seen_at for freshness queries
CREATE INDEX IF NOT EXISTS idx_document_versions_last_seen_at
    ON document_versions (last_seen_at);

-- Index on credibility_score for threshold filtering (R10.3)
CREATE INDEX IF NOT EXISTS idx_document_versions_credibility_score
    ON document_versions (credibility_score)
    WHERE credibility_score IS NOT NULL;

-- Index on ai_generated_likelihood for threshold filtering (R10.4)
CREATE INDEX IF NOT EXISTS idx_document_versions_ai_generated_likelihood
    ON document_versions (ai_generated_likelihood)
    WHERE ai_generated_likelihood IS NOT NULL;

-- =============================================================================
-- Version monotonicity trigger
-- Enforces that each new version for a document_id = max(existing versions) + 1
-- This prevents gaps and out-of-order version inserts (R2.3)
-- =============================================================================

CREATE OR REPLACE FUNCTION enforce_version_monotonicity()
RETURNS TRIGGER AS $$
DECLARE
    max_version INTEGER;
BEGIN
    -- Lock the document row to prevent concurrent version inserts
    PERFORM 1 FROM documents WHERE document_id = NEW.document_id FOR UPDATE;

    -- Get the current maximum version for this document
    SELECT COALESCE(MAX(version), 0)
    INTO max_version
    FROM document_versions
    WHERE document_id = NEW.document_id;

    -- The new version must be exactly max + 1
    IF NEW.version != max_version + 1 THEN
        RAISE EXCEPTION 'Version monotonicity violation: expected version %, got %',
            max_version + 1, NEW.version
            USING ERRCODE = 'check_violation';
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_enforce_version_monotonicity
    BEFORE INSERT ON document_versions
    FOR EACH ROW
    EXECUTE FUNCTION enforce_version_monotonicity();

-- =============================================================================
-- Row-Level Security for documents table
-- Documents with NULL tenant_id are globally visible (crawl-derived).
-- Documents with a tenant_id are only visible to that tenant.
-- =============================================================================

ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE documents FORCE ROW LEVEL SECURITY;

-- Policy: app_user can see global docs (tenant_id IS NULL) OR docs belonging to their tenant
CREATE POLICY documents_tenant_isolation_policy
    ON documents
    FOR ALL
    TO app_user
    USING (
        tenant_id IS NULL
        OR tenant_id = current_setting('app.current_tenant_id')::UUID
    )
    WITH CHECK (
        tenant_id IS NULL
        OR tenant_id = current_setting('app.current_tenant_id')::UUID
    );

-- Grant table permissions to app_user
GRANT SELECT, INSERT, UPDATE, DELETE ON documents TO app_user;

-- =============================================================================
-- Row-Level Security for document_versions table
-- Visibility follows the parent document's tenant scope.
-- =============================================================================

ALTER TABLE document_versions ENABLE ROW LEVEL SECURITY;
ALTER TABLE document_versions FORCE ROW LEVEL SECURITY;

-- Policy: app_user can access versions of documents they can see
CREATE POLICY document_versions_tenant_isolation_policy
    ON document_versions
    FOR ALL
    TO app_user
    USING (
        EXISTS (
            SELECT 1 FROM documents d
            WHERE d.document_id = document_versions.document_id
            AND (d.tenant_id IS NULL
                 OR d.tenant_id = current_setting('app.current_tenant_id')::UUID)
        )
    )
    WITH CHECK (
        EXISTS (
            SELECT 1 FROM documents d
            WHERE d.document_id = document_versions.document_id
            AND (d.tenant_id IS NULL
                 OR d.tenant_id = current_setting('app.current_tenant_id')::UUID)
        )
    );

-- Grant table permissions to app_user
GRANT SELECT, INSERT, UPDATE, DELETE ON document_versions TO app_user;

-- DOWN
-- =============================================================================

-- Revoke permissions
REVOKE SELECT, INSERT, UPDATE, DELETE ON document_versions FROM app_user;
REVOKE SELECT, INSERT, UPDATE, DELETE ON documents FROM app_user;

-- Drop RLS policies
DROP POLICY IF EXISTS document_versions_tenant_isolation_policy ON document_versions;
DROP POLICY IF EXISTS documents_tenant_isolation_policy ON documents;

-- Disable RLS
ALTER TABLE document_versions DISABLE ROW LEVEL SECURITY;
ALTER TABLE documents DISABLE ROW LEVEL SECURITY;

-- Drop trigger and function
DROP TRIGGER IF EXISTS trg_enforce_version_monotonicity ON document_versions;
DROP FUNCTION IF EXISTS enforce_version_monotonicity();

-- Drop indexes
DROP INDEX IF EXISTS idx_document_versions_ai_generated_likelihood;
DROP INDEX IF EXISTS idx_document_versions_credibility_score;
DROP INDEX IF EXISTS idx_document_versions_last_seen_at;
DROP INDEX IF EXISTS idx_document_versions_content_hash;
DROP INDEX IF EXISTS idx_documents_tenant_id;

-- Drop tables (versions first due to FK)
DROP TABLE IF EXISTS document_versions;
DROP TABLE IF EXISTS documents;
