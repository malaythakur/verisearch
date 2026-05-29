-- Migration: 006_create_research.sql
-- Description: Create research_jobs, research_plans, plan_steps, and events tables
--              for the agentic research subsystem with tenant-scoped isolation,
--              budget enforcement, plan step constraints, and monotonic event streams.
-- Requirements: R7.1 (research_goal 1..4096), R7.2 (plan 1..32 steps),
--              R7.3 (strictly monotonic event_id per job), R7.6 (budgets),
--              R7.8 (output_schema), R13.3 (tenant isolation)
-- Depends on: 001_create_tenants.sql, 004_create_sessions.sql

-- UP
-- =============================================================================

-- Create the research_jobs table
CREATE TABLE IF NOT EXISTS research_jobs (
    job_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id       UUID NOT NULL
                    REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    session_id      UUID
                    REFERENCES sessions(session_id) ON DELETE SET NULL,
    research_goal   TEXT NOT NULL
                    CHECK (char_length(research_goal) BETWEEN 1 AND 4096),
    output_schema   JSONB,
    budgets         JSONB NOT NULL DEFAULT '{}'::jsonb,
    state           VARCHAR(20) NOT NULL DEFAULT 'queued'
                    CHECK (state IN ('queued', 'planning', 'running', 'succeeded', 'failed', 'budget_exceeded')),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- =============================================================================
-- Indexes for research_jobs
-- =============================================================================

-- Index on tenant_id for tenant-scoped queries
CREATE INDEX IF NOT EXISTS idx_research_jobs_tenant_id
    ON research_jobs (tenant_id);

-- Index on state for filtering jobs by status
CREATE INDEX IF NOT EXISTS idx_research_jobs_state
    ON research_jobs (state);

-- Composite index on (tenant_id, state) for common query pattern
CREATE INDEX IF NOT EXISTS idx_research_jobs_tenant_id_state
    ON research_jobs (tenant_id, state);

-- Index on session_id for session-scoped lookups
CREATE INDEX IF NOT EXISTS idx_research_jobs_session_id
    ON research_jobs (session_id);

-- Index on created_at for ordering/pagination
CREATE INDEX IF NOT EXISTS idx_research_jobs_created_at
    ON research_jobs (created_at);

-- =============================================================================
-- Create the research_plans table
-- =============================================================================

CREATE TABLE IF NOT EXISTS research_plans (
    job_id          UUID PRIMARY KEY
                    REFERENCES research_jobs(job_id) ON DELETE CASCADE,
    steps           JSONB NOT NULL DEFAULT '[]'::jsonb,
    emitted_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Check constraint: steps array must have 1..32 elements (R7.2)
ALTER TABLE research_plans
    ADD CONSTRAINT chk_research_plans_steps_count
    CHECK (jsonb_array_length(steps) BETWEEN 1 AND 32);

-- =============================================================================
-- Create the plan_steps table
-- =============================================================================

CREATE TABLE IF NOT EXISTS plan_steps (
    step_id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id          UUID NOT NULL
                    REFERENCES research_jobs(job_id) ON DELETE CASCADE,
    type            VARCHAR(20) NOT NULL
                    CHECK (type IN ('sub_query', 'retrieval', 'read', 'synthesis')),
    inputs          JSONB NOT NULL DEFAULT '{}'::jsonb,
    outputs         JSONB NOT NULL DEFAULT '{}'::jsonb
);

-- Index on job_id for job-scoped step lookups
CREATE INDEX IF NOT EXISTS idx_plan_steps_job_id
    ON plan_steps (job_id);

-- =============================================================================
-- Create the events table (research stream; R7.3)
-- Composite primary key (job_id, event_id) with strictly monotonic event_id per job
-- =============================================================================

CREATE TABLE IF NOT EXISTS events (
    job_id          UUID NOT NULL
                    REFERENCES research_jobs(job_id) ON DELETE CASCADE,
    event_id        BIGINT NOT NULL,
    type            VARCHAR(20) NOT NULL
                    CHECK (type IN ('plan_updated', 'step_started', 'step_completed', 'citation', 'report_chunk', 'done', 'error')),
    payload         JSONB NOT NULL DEFAULT '{}'::jsonb,
    emitted_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    PRIMARY KEY (job_id, event_id)
);

-- Note: The primary key on (job_id, event_id) already provides an efficient
-- index for replay queries (WHERE job_id = ? AND event_id > ?).

-- =============================================================================
-- Trigger to enforce strictly monotonic event_id per job_id (R7.3)
-- Each new event_id must be strictly greater than the current maximum for that job.
-- =============================================================================

CREATE OR REPLACE FUNCTION enforce_event_id_monotonicity()
RETURNS TRIGGER AS $$
DECLARE
    max_event_id BIGINT;
BEGIN
    -- Get the current maximum event_id for this job
    SELECT MAX(event_id)
    INTO max_event_id
    FROM events
    WHERE job_id = NEW.job_id;

    -- If there are existing events, the new event_id must be strictly greater
    IF max_event_id IS NOT NULL AND NEW.event_id <= max_event_id THEN
        RAISE EXCEPTION 'event_id must be strictly monotonic per job_id: new event_id (%) must be greater than current max (%)',
            NEW.event_id, max_event_id
            USING ERRCODE = 'check_violation';
    END IF;

    -- If this is the first event for the job, any positive event_id is valid
    IF max_event_id IS NULL AND NEW.event_id < 1 THEN
        RAISE EXCEPTION 'event_id must be >= 1, got %',
            NEW.event_id
            USING ERRCODE = 'check_violation';
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_enforce_event_id_monotonicity
    BEFORE INSERT ON events
    FOR EACH ROW
    EXECUTE FUNCTION enforce_event_id_monotonicity();

-- =============================================================================
-- Row-Level Security for research_jobs
-- =============================================================================

ALTER TABLE research_jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE research_jobs FORCE ROW LEVEL SECURITY;

CREATE POLICY research_jobs_tenant_isolation_policy
    ON research_jobs
    FOR ALL
    TO app_user
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID)
    WITH CHECK (tenant_id = current_setting('app.current_tenant_id')::UUID);

GRANT SELECT, INSERT, UPDATE, DELETE ON research_jobs TO app_user;

-- =============================================================================
-- Row-Level Security for research_plans
-- Visibility follows the parent research_job's tenant scope.
-- =============================================================================

ALTER TABLE research_plans ENABLE ROW LEVEL SECURITY;
ALTER TABLE research_plans FORCE ROW LEVEL SECURITY;

CREATE POLICY research_plans_tenant_isolation_policy
    ON research_plans
    FOR ALL
    TO app_user
    USING (
        EXISTS (
            SELECT 1 FROM research_jobs rj
            WHERE rj.job_id = research_plans.job_id
            AND rj.tenant_id = current_setting('app.current_tenant_id')::UUID
        )
    )
    WITH CHECK (
        EXISTS (
            SELECT 1 FROM research_jobs rj
            WHERE rj.job_id = research_plans.job_id
            AND rj.tenant_id = current_setting('app.current_tenant_id')::UUID
        )
    );

GRANT SELECT, INSERT, UPDATE, DELETE ON research_plans TO app_user;

-- =============================================================================
-- Row-Level Security for plan_steps
-- Visibility follows the parent research_job's tenant scope.
-- =============================================================================

ALTER TABLE plan_steps ENABLE ROW LEVEL SECURITY;
ALTER TABLE plan_steps FORCE ROW LEVEL SECURITY;

CREATE POLICY plan_steps_tenant_isolation_policy
    ON plan_steps
    FOR ALL
    TO app_user
    USING (
        EXISTS (
            SELECT 1 FROM research_jobs rj
            WHERE rj.job_id = plan_steps.job_id
            AND rj.tenant_id = current_setting('app.current_tenant_id')::UUID
        )
    )
    WITH CHECK (
        EXISTS (
            SELECT 1 FROM research_jobs rj
            WHERE rj.job_id = plan_steps.job_id
            AND rj.tenant_id = current_setting('app.current_tenant_id')::UUID
        )
    );

GRANT SELECT, INSERT, UPDATE, DELETE ON plan_steps TO app_user;

-- =============================================================================
-- Row-Level Security for events
-- Visibility follows the parent research_job's tenant scope.
-- =============================================================================

ALTER TABLE events ENABLE ROW LEVEL SECURITY;
ALTER TABLE events FORCE ROW LEVEL SECURITY;

CREATE POLICY events_tenant_isolation_policy
    ON events
    FOR ALL
    TO app_user
    USING (
        EXISTS (
            SELECT 1 FROM research_jobs rj
            WHERE rj.job_id = events.job_id
            AND rj.tenant_id = current_setting('app.current_tenant_id')::UUID
        )
    )
    WITH CHECK (
        EXISTS (
            SELECT 1 FROM research_jobs rj
            WHERE rj.job_id = events.job_id
            AND rj.tenant_id = current_setting('app.current_tenant_id')::UUID
        )
    );

GRANT SELECT, INSERT, UPDATE, DELETE ON events TO app_user;

-- DOWN
-- =============================================================================

-- Revoke permissions
REVOKE SELECT, INSERT, UPDATE, DELETE ON events FROM app_user;
REVOKE SELECT, INSERT, UPDATE, DELETE ON plan_steps FROM app_user;
REVOKE SELECT, INSERT, UPDATE, DELETE ON research_plans FROM app_user;
REVOKE SELECT, INSERT, UPDATE, DELETE ON research_jobs FROM app_user;

-- Drop RLS policies
DROP POLICY IF EXISTS events_tenant_isolation_policy ON events;
DROP POLICY IF EXISTS plan_steps_tenant_isolation_policy ON plan_steps;
DROP POLICY IF EXISTS research_plans_tenant_isolation_policy ON research_plans;
DROP POLICY IF EXISTS research_jobs_tenant_isolation_policy ON research_jobs;

-- Disable RLS
ALTER TABLE events DISABLE ROW LEVEL SECURITY;
ALTER TABLE plan_steps DISABLE ROW LEVEL SECURITY;
ALTER TABLE research_plans DISABLE ROW LEVEL SECURITY;
ALTER TABLE research_jobs DISABLE ROW LEVEL SECURITY;

-- Drop trigger and function
DROP TRIGGER IF EXISTS trg_enforce_event_id_monotonicity ON events;
DROP FUNCTION IF EXISTS enforce_event_id_monotonicity();

-- Drop constraints
ALTER TABLE research_plans DROP CONSTRAINT IF EXISTS chk_research_plans_steps_count;

-- Drop indexes
DROP INDEX IF EXISTS idx_plan_steps_job_id;
DROP INDEX IF EXISTS idx_research_jobs_created_at;
DROP INDEX IF EXISTS idx_research_jobs_session_id;
DROP INDEX IF EXISTS idx_research_jobs_tenant_id_state;
DROP INDEX IF EXISTS idx_research_jobs_state;
DROP INDEX IF EXISTS idx_research_jobs_tenant_id;

-- Drop tables (child tables first due to FK dependencies)
DROP TABLE IF EXISTS events;
DROP TABLE IF EXISTS plan_steps;
DROP TABLE IF EXISTS research_plans;
DROP TABLE IF EXISTS research_jobs;
