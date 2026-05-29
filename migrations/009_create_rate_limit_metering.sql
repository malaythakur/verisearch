-- Migration: 009_create_rate_limit_metering.sql
-- Description: Create rate_limit_buckets and metering_events tables for per-tenant
--              rate limiting and at-least-once metering event deduplication.
-- Requirements: R14.1 (per-tenant rate limits), R14.2 (metering events), R14.3 (at-least-once dedup)
-- Depends on: 001_create_tenants.sql

-- UP
-- =============================================================================

-- Create the rate_limit_buckets table
-- Tracks sliding-window rate limit state per tenant per endpoint.
CREATE TABLE IF NOT EXISTS rate_limit_buckets (
    tenant_id        UUID NOT NULL
                     REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    endpoint         VARCHAR(255) NOT NULL,
    window_start_utc TIMESTAMPTZ NOT NULL,
    limit_per_minute INTEGER NOT NULL,
    remaining        INTEGER NOT NULL,
    reset_at_utc     TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (tenant_id, endpoint)
);

-- Index on reset_at_utc for periodic cleanup of expired buckets
CREATE INDEX IF NOT EXISTS idx_rate_limit_buckets_reset_at_utc
    ON rate_limit_buckets (reset_at_utc);

-- =============================================================================

-- Create the metering_events table
-- Records every billable API request with at-least-once dedup via dedup_key (R14.3).
CREATE TABLE IF NOT EXISTS metering_events (
    metering_event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id        VARCHAR(64) NOT NULL,
    tenant_id         UUID NOT NULL
                      REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    endpoint          VARCHAR(255) NOT NULL,
    timestamp_utc     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    response_status   SMALLINT NOT NULL,
    tokens_consumed   INTEGER,
    dedup_key         VARCHAR(255) NOT NULL UNIQUE
);

-- Index on (tenant_id, timestamp_utc) for tenant-scoped time-range queries
-- (e.g., usage dashboards, billing aggregation)
CREATE INDEX IF NOT EXISTS idx_metering_events_tenant_timestamp
    ON metering_events (tenant_id, timestamp_utc);

-- =============================================================================
-- Row-Level Security
-- =============================================================================

-- rate_limit_buckets: tenant isolation
ALTER TABLE rate_limit_buckets ENABLE ROW LEVEL SECURITY;
ALTER TABLE rate_limit_buckets FORCE ROW LEVEL SECURITY;

CREATE POLICY rate_limit_buckets_tenant_isolation_policy
    ON rate_limit_buckets
    FOR ALL
    TO app_user
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID)
    WITH CHECK (tenant_id = current_setting('app.current_tenant_id')::UUID);

-- metering_events: tenant isolation
ALTER TABLE metering_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE metering_events FORCE ROW LEVEL SECURITY;

CREATE POLICY metering_events_tenant_isolation_policy
    ON metering_events
    FOR ALL
    TO app_user
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID)
    WITH CHECK (tenant_id = current_setting('app.current_tenant_id')::UUID);

-- =============================================================================
-- Permissions
-- =============================================================================

GRANT SELECT, INSERT, UPDATE, DELETE ON rate_limit_buckets TO app_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON metering_events TO app_user;

-- DOWN
-- =============================================================================

-- Revoke permissions
REVOKE SELECT, INSERT, UPDATE, DELETE ON metering_events FROM app_user;
REVOKE SELECT, INSERT, UPDATE, DELETE ON rate_limit_buckets FROM app_user;

-- Drop RLS policies
DROP POLICY IF EXISTS metering_events_tenant_isolation_policy ON metering_events;
DROP POLICY IF EXISTS rate_limit_buckets_tenant_isolation_policy ON rate_limit_buckets;

-- Disable RLS
ALTER TABLE metering_events DISABLE ROW LEVEL SECURITY;
ALTER TABLE rate_limit_buckets DISABLE ROW LEVEL SECURITY;

-- Drop indexes
DROP INDEX IF EXISTS idx_metering_events_tenant_timestamp;
DROP INDEX IF EXISTS idx_rate_limit_buckets_reset_at_utc;

-- Drop tables
DROP TABLE IF EXISTS metering_events;
DROP TABLE IF EXISTS rate_limit_buckets;
