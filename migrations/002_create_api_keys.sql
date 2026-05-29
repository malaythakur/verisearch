-- Migration: 002_create_api_keys.sql
-- Description: Create api_keys table with prefix-indexed lookup, Argon2id hash storage,
--              and rotation grace period fields per R13.5
-- Requirements: R13 (API key authentication), R13.3 (tenant isolation), R13.5 (key rotation)
-- Depends on: 001_create_tenants.sql

-- UP
-- =============================================================================

-- Create the api_keys table
CREATE TABLE IF NOT EXISTS api_keys (
    api_key_id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID NOT NULL
                            REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    key_prefix              VARCHAR(12) NOT NULL
                            CHECK (char_length(key_prefix) BETWEEN 8 AND 12),
    key_hash                TEXT NOT NULL,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at              TIMESTAMPTZ,
    revoked_at              TIMESTAMPTZ,
    rotation_grace_seconds  INTEGER NOT NULL DEFAULT 3600
                            CHECK (rotation_grace_seconds BETWEEN 1 AND 86400)
);

-- Index on key_prefix for fast prefix-based lookup during authentication.
-- Auth flow: client sends full key → extract prefix → lookup rows by prefix → verify hash.
CREATE INDEX IF NOT EXISTS idx_api_keys_key_prefix
    ON api_keys (key_prefix);

-- Index on tenant_id for listing keys per tenant (dashboard, admin queries).
CREATE INDEX IF NOT EXISTS idx_api_keys_tenant_id
    ON api_keys (tenant_id);

-- Partial index on non-revoked keys to speed up authentication queries.
-- Only keys that have not been revoked (or are within rotation grace) are candidates.
CREATE INDEX IF NOT EXISTS idx_api_keys_active
    ON api_keys (key_prefix)
    WHERE revoked_at IS NULL;

-- Enable Row-Level Security on the api_keys table
ALTER TABLE api_keys ENABLE ROW LEVEL SECURITY;

-- Force RLS for table owner as well (defense in depth)
ALTER TABLE api_keys FORCE ROW LEVEL SECURITY;

-- RLS Policy: app_user can only see rows matching the current session tenant_id.
-- The application sets `app.current_tenant_id` on each request after authentication.
-- Example: SET LOCAL app.current_tenant_id = '<tenant-uuid>';
CREATE POLICY api_keys_tenant_isolation_policy
    ON api_keys
    FOR ALL
    TO app_user
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID)
    WITH CHECK (tenant_id = current_setting('app.current_tenant_id')::UUID);

-- Grant table permissions to app_user
GRANT SELECT, INSERT, UPDATE, DELETE ON api_keys TO app_user;

-- DOWN
-- =============================================================================

-- Revoke permissions
REVOKE SELECT, INSERT, UPDATE, DELETE ON api_keys FROM app_user;

-- Drop RLS policy
DROP POLICY IF EXISTS api_keys_tenant_isolation_policy ON api_keys;

-- Disable RLS
ALTER TABLE api_keys DISABLE ROW LEVEL SECURITY;

-- Drop indexes
DROP INDEX IF EXISTS idx_api_keys_active;
DROP INDEX IF EXISTS idx_api_keys_tenant_id;
DROP INDEX IF EXISTS idx_api_keys_key_prefix;

-- Drop table
DROP TABLE IF EXISTS api_keys;
