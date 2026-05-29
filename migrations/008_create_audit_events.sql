-- Migration: 008_create_audit_events.sql
-- Description: Create audit_events table with append-only enforcement (immutable constraint),
--              retention policy support, and tenant-scoped RLS for read access.
-- Requirements: R15.1 (append within 5s), R15.4 (no modification, retention [365,2555] days),
--              R15.6 (audit append failure blocks privileged action), R13.6 (null tenant_id for
--              unattributable auth_failure)
-- Depends on: 000_init.sql

-- UP
-- =============================================================================

-- Create the audit_events table (append-only ledger)
CREATE TABLE IF NOT EXISTS audit_events (
    audit_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id       UUID,  -- nullable: null for unattributable auth_failure (R13.6)
    actor           VARCHAR(255) NOT NULL,
    action          VARCHAR(100) NOT NULL,
    resource        TEXT NOT NULL,
    timestamp_utc   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    request_id      VARCHAR(64) NOT NULL
                    CHECK (char_length(request_id) BETWEEN 16 AND 64),
    detail          JSONB NOT NULL DEFAULT '{}'::jsonb
);

-- =============================================================================
-- Append-only enforcement trigger (R15.4)
-- Prevents UPDATE and DELETE operations at the database level.
-- Only INSERT is permitted on this table.
-- =============================================================================

CREATE OR REPLACE FUNCTION prevent_audit_events_modification()
RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION 'audit_events table is append-only: % operations are not permitted', TG_OP
        USING ERRCODE = 'restrict_violation';
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Block UPDATE operations
CREATE TRIGGER trg_audit_events_no_update
    BEFORE UPDATE ON audit_events
    FOR EACH ROW
    EXECUTE FUNCTION prevent_audit_events_modification();

-- Block DELETE operations
CREATE TRIGGER trg_audit_events_no_delete
    BEFORE DELETE ON audit_events
    FOR EACH ROW
    EXECUTE FUNCTION prevent_audit_events_modification();

-- =============================================================================
-- Indexes
-- =============================================================================

-- Index on tenant_id for tenant-scoped queries
CREATE INDEX IF NOT EXISTS idx_audit_events_tenant_id
    ON audit_events (tenant_id);

-- Index on timestamp_utc for time-range queries and retention cleanup
CREATE INDEX IF NOT EXISTS idx_audit_events_timestamp_utc
    ON audit_events (timestamp_utc);

-- Index on action for filtering by action type
CREATE INDEX IF NOT EXISTS idx_audit_events_action
    ON audit_events (action);

-- Composite index on (tenant_id, timestamp_utc) for common query pattern
-- e.g., "find all audit events for this tenant in the last 7 days"
CREATE INDEX IF NOT EXISTS idx_audit_events_tenant_id_timestamp
    ON audit_events (tenant_id, timestamp_utc);

-- Index on request_id for correlating audit entries with requests
CREATE INDEX IF NOT EXISTS idx_audit_events_request_id
    ON audit_events (request_id);

-- =============================================================================
-- Row-Level Security
-- Audit events have special RLS: app_user can INSERT with any tenant_id
-- (including null for auth failures) but can only SELECT their own tenant's events.
-- =============================================================================

ALTER TABLE audit_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit_events FORCE ROW LEVEL SECURITY;

-- RLS Policy: app_user can INSERT any audit event (including null tenant_id for auth failures)
CREATE POLICY audit_events_insert_policy
    ON audit_events
    FOR INSERT
    TO app_user
    WITH CHECK (true);

-- RLS Policy: app_user can only SELECT audit events for their own tenant
CREATE POLICY audit_events_select_policy
    ON audit_events
    FOR SELECT
    TO app_user
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

-- Grant only INSERT and SELECT to app_user (no UPDATE or DELETE — enforced by trigger)
GRANT INSERT, SELECT ON audit_events TO app_user;

-- DOWN
-- =============================================================================

-- Revoke permissions
REVOKE INSERT, SELECT ON audit_events FROM app_user;

-- Drop RLS policies
DROP POLICY IF EXISTS audit_events_select_policy ON audit_events;
DROP POLICY IF EXISTS audit_events_insert_policy ON audit_events;

-- Disable RLS
ALTER TABLE audit_events DISABLE ROW LEVEL SECURITY;

-- Drop indexes
DROP INDEX IF EXISTS idx_audit_events_request_id;
DROP INDEX IF EXISTS idx_audit_events_tenant_id_timestamp;
DROP INDEX IF EXISTS idx_audit_events_action;
DROP INDEX IF EXISTS idx_audit_events_timestamp_utc;
DROP INDEX IF EXISTS idx_audit_events_tenant_id;

-- Drop triggers and function
DROP TRIGGER IF EXISTS trg_audit_events_no_delete ON audit_events;
DROP TRIGGER IF EXISTS trg_audit_events_no_update ON audit_events;
DROP FUNCTION IF EXISTS prevent_audit_events_modification();

-- Drop table
DROP TABLE IF EXISTS audit_events;
