-- Migration: 005_create_pipelines.sql
-- Description: Create pipelines and pipeline_steps tables with tenant-scoped isolation,
--              step count enforcement (1–20), registry_name, and per-step timeout.
-- Requirements: R9.1 (1–20 steps, registry names), R9.2 (unknown step rejection),
--              R9.6 (timeout_ms [100..30000] default 2000), R13.3 (tenant isolation)
-- Depends on: 001_create_tenants.sql

-- UP
-- =============================================================================

-- Create the pipelines table
CREATE TABLE IF NOT EXISTS pipelines (
    pipeline_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id       UUID NOT NULL
                    REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    name            VARCHAR(255) NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index on tenant_id for tenant-scoped queries
CREATE INDEX IF NOT EXISTS idx_pipelines_tenant_id
    ON pipelines (tenant_id);

-- Index on created_at for ordering/pagination
CREATE INDEX IF NOT EXISTS idx_pipelines_created_at
    ON pipelines (created_at);

-- Create the pipeline_steps table
CREATE TABLE IF NOT EXISTS pipeline_steps (
    pipeline_id     UUID NOT NULL
                    REFERENCES pipelines(pipeline_id) ON DELETE CASCADE,
    ordinal         INTEGER NOT NULL
                    CHECK (ordinal >= 0),
    type            VARCHAR(20) NOT NULL
                    CHECK (type IN ('filter', 'reranker', 'transform')),
    registry_name   VARCHAR(255) NOT NULL,
    config_json     JSONB NOT NULL DEFAULT '{}'::jsonb,
    timeout_ms      INTEGER NOT NULL DEFAULT 2000
                    CHECK (timeout_ms BETWEEN 100 AND 30000),

    PRIMARY KEY (pipeline_id, ordinal)
);

-- Index on registry_name for registry lookups and validation queries
CREATE INDEX IF NOT EXISTS idx_pipeline_steps_registry_name
    ON pipeline_steps (registry_name);

-- =============================================================================
-- Step count enforcement trigger (R9.1: 1 ≤ count(steps) ≤ 20)
-- Enforces that a pipeline always has between 1 and 20 steps.
-- On INSERT: prevents adding more than 20 steps to a pipeline.
-- On DELETE: prevents removing the last step from a pipeline
--            (unless the entire pipeline is being deleted via CASCADE).
-- =============================================================================

CREATE OR REPLACE FUNCTION enforce_pipeline_step_count()
RETURNS TRIGGER AS $$
DECLARE
    step_count INTEGER;
BEGIN
    IF TG_OP = 'INSERT' THEN
        -- Count existing steps for this pipeline (before this insert)
        SELECT COUNT(*)
        INTO step_count
        FROM pipeline_steps
        WHERE pipeline_id = NEW.pipeline_id;

        -- After this insert, count will be step_count + 1
        IF step_count + 1 > 20 THEN
            RAISE EXCEPTION 'Pipeline step count exceeds maximum of 20 (would be %)',
                step_count + 1
                USING ERRCODE = 'check_violation';
        END IF;

        RETURN NEW;

    ELSIF TG_OP = 'DELETE' THEN
        -- Count remaining steps for this pipeline (before this delete)
        SELECT COUNT(*)
        INTO step_count
        FROM pipeline_steps
        WHERE pipeline_id = OLD.pipeline_id;

        -- After this delete, count will be step_count - 1
        -- Allow deletion to 0 only if the pipeline itself is being deleted (CASCADE)
        -- We check if the pipeline still exists; if not, CASCADE is in progress
        IF step_count - 1 < 1 THEN
            IF EXISTS (SELECT 1 FROM pipelines WHERE pipeline_id = OLD.pipeline_id) THEN
                RAISE EXCEPTION 'Pipeline must have at least 1 step (would have %)',
                    step_count - 1
                    USING ERRCODE = 'check_violation';
            END IF;
        END IF;

        RETURN OLD;
    END IF;

    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_enforce_pipeline_step_count
    BEFORE INSERT OR DELETE ON pipeline_steps
    FOR EACH ROW
    EXECUTE FUNCTION enforce_pipeline_step_count();

-- =============================================================================
-- Row-Level Security for pipelines table
-- Pipelines are strictly tenant-scoped (R9.7, R13.3)
-- =============================================================================

ALTER TABLE pipelines ENABLE ROW LEVEL SECURITY;
ALTER TABLE pipelines FORCE ROW LEVEL SECURITY;

-- RLS Policy: app_user can only access pipelines belonging to their tenant
CREATE POLICY pipelines_tenant_isolation_policy
    ON pipelines
    FOR ALL
    TO app_user
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID)
    WITH CHECK (tenant_id = current_setting('app.current_tenant_id')::UUID);

-- Grant table permissions to app_user
GRANT SELECT, INSERT, UPDATE, DELETE ON pipelines TO app_user;

-- =============================================================================
-- Row-Level Security for pipeline_steps table
-- Visibility follows the parent pipeline's tenant scope.
-- =============================================================================

ALTER TABLE pipeline_steps ENABLE ROW LEVEL SECURITY;
ALTER TABLE pipeline_steps FORCE ROW LEVEL SECURITY;

-- RLS Policy: app_user can access steps of pipelines they own
CREATE POLICY pipeline_steps_tenant_isolation_policy
    ON pipeline_steps
    FOR ALL
    TO app_user
    USING (
        EXISTS (
            SELECT 1 FROM pipelines p
            WHERE p.pipeline_id = pipeline_steps.pipeline_id
            AND p.tenant_id = current_setting('app.current_tenant_id')::UUID
        )
    )
    WITH CHECK (
        EXISTS (
            SELECT 1 FROM pipelines p
            WHERE p.pipeline_id = pipeline_steps.pipeline_id
            AND p.tenant_id = current_setting('app.current_tenant_id')::UUID
        )
    );

-- Grant table permissions to app_user
GRANT SELECT, INSERT, UPDATE, DELETE ON pipeline_steps TO app_user;

-- DOWN
-- =============================================================================

-- Revoke permissions
REVOKE SELECT, INSERT, UPDATE, DELETE ON pipeline_steps FROM app_user;
REVOKE SELECT, INSERT, UPDATE, DELETE ON pipelines FROM app_user;

-- Drop RLS policies
DROP POLICY IF EXISTS pipeline_steps_tenant_isolation_policy ON pipeline_steps;
DROP POLICY IF EXISTS pipelines_tenant_isolation_policy ON pipelines;

-- Disable RLS
ALTER TABLE pipeline_steps DISABLE ROW LEVEL SECURITY;
ALTER TABLE pipelines DISABLE ROW LEVEL SECURITY;

-- Drop trigger and function
DROP TRIGGER IF EXISTS trg_enforce_pipeline_step_count ON pipeline_steps;
DROP FUNCTION IF EXISTS enforce_pipeline_step_count();

-- Drop indexes
DROP INDEX IF EXISTS idx_pipeline_steps_registry_name;
DROP INDEX IF EXISTS idx_pipelines_created_at;
DROP INDEX IF EXISTS idx_pipelines_tenant_id;

-- Drop tables (steps first due to FK)
DROP TABLE IF EXISTS pipeline_steps;
DROP TABLE IF EXISTS pipelines;
