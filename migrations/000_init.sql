-- Migration: 000_init.sql
-- Description: Initialize database extensions and application role
-- Requirements: Foundation for all subsequent migrations

-- UP
-- =============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Create the application role used by the backend service.
-- This role will be subject to RLS policies on tenant-scoped tables.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'app_user') THEN
        CREATE ROLE app_user LOGIN;
    END IF;
END
$$;

-- Grant usage on public schema so app_user can access tables
GRANT USAGE ON SCHEMA public TO app_user;

-- DOWN
-- =============================================================================
-- WARNING: Dropping extensions may break dependent objects.
-- Only run this if you are tearing down the entire schema.

-- REVOKE USAGE ON SCHEMA public FROM app_user;
-- DROP ROLE IF EXISTS app_user;
-- DROP EXTENSION IF EXISTS "pgcrypto";
-- DROP EXTENSION IF EXISTS "uuid-ossp";
