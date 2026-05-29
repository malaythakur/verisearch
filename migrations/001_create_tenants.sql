-- Migration: 001_create_tenants.sql
-- Description: Create tenants table with Row-Level Security policies
-- Requirements: R13.3 (tenant isolation), Design Data Model (Tenant entity)
-- Depends on: 000_init.sql

-- UP
-- =============================================================================

-- Create the tenants table
CREATE TABLE IF NOT EXISTS tenants (
    tenant_id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name                VARCHAR(255) NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    data_retention_days INTEGER NOT NULL DEFAULT 365
                        CHECK (data_retention_days BETWEEN 1 AND 2555),
    deletion_state      VARCHAR(20) NOT NULL DEFAULT 'active'
                        CHECK (deletion_state IN ('active', 'pending_deletion', 'deleted'))
);

-- Create indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_tenants_deletion_state
    ON tenants (deletion_state);

CREATE INDEX IF NOT EXISTS idx_tenants_created_at
    ON tenants (created_at);

-- Enable Row-Level Security on the tenants table
ALTER TABLE tenants ENABLE ROW LEVEL SECURITY;

-- Force RLS for table owner as well (defense in depth)
ALTER TABLE tenants FORCE ROW LEVEL SECURITY;

-- RLS Policy: app_user can only see rows matching the current session tenant_id.
-- The application sets `app.current_tenant_id` on each request after authentication.
-- Example: SET LOCAL app.current_tenant_id = '<tenant-uuid>';
CREATE POLICY tenant_isolation_policy
    ON tenants
    FOR ALL
    TO app_user
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID)
    WITH CHECK (tenant_id = current_setting('app.current_tenant_id')::UUID);

-- Grant table permissions to app_user
GRANT SELECT, INSERT, UPDATE, DELETE ON tenants TO app_user;

-- DOWN
-- =============================================================================

-- Revoke permissions
REVOKE SELECT, INSERT, UPDATE, DELETE ON tenants FROM app_user;

-- Drop RLS policy
DROP POLICY IF EXISTS tenant_isolation_policy ON tenants;

-- Disable RLS
ALTER TABLE tenants DISABLE ROW LEVEL SECURITY;

-- Drop indexes
DROP INDEX IF EXISTS idx_tenants_created_at;
DROP INDEX IF EXISTS idx_tenants_deletion_state;

-- Drop table
DROP TABLE IF EXISTS tenants;
