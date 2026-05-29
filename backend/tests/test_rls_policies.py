"""Tests validating Row-Level Security (RLS) policies block cross-tenant access.

These tests have two complementary parts:

1. **Static SQL parsing tests** (always run): parse the migration files in
   ``migrations/`` and assert the expected RLS patterns are present for every
   tenant-scoped table. This catches regressions in the migration SQL even when
   no live Postgres is available (CI default).

2. **Optional live integration test** marked with ``@pytest.mark.integration``:
   runs only when ``RUN_DB_INTEGRATION_TESTS=1`` is set in the environment and
   a reachable Postgres instance is configured. It applies the migrations,
   creates two tenants, switches ``app.current_tenant_id`` between them as the
   ``app_user`` role, and verifies that:

   - SELECT from another tenant returns 0 rows (USING clause blocks reads).
   - INSERT with another tenant's ``tenant_id`` is rejected (WITH CHECK clause
     blocks writes).

Requirements: R13.3 (tenant isolation), R7.7, R8.5, R9.7, R15.7 (uniform 404
on cross-tenant access).
"""

from __future__ import annotations

import os
import re
import uuid
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Migration discovery helpers
# ---------------------------------------------------------------------------

# Locate the migrations directory relative to the repo root. The tests file
# lives at backend/tests/test_rls_policies.py, so the repo root is two levels
# up from the tests directory.
MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"


def _read_migration(filename: str) -> str:
    """Read a migration SQL file and return its full text content."""
    path = MIGRATIONS_DIR / filename
    if not path.is_file():
        pytest.fail(f"Migration file not found: {path}")
    return path.read_text(encoding="utf-8")


def _up_section(sql: str) -> str:
    """Return only the UP section of a migration (everything before the DOWN marker).

    Migration files in this repo use ``-- DOWN`` as a marker between the apply
    and rollback sections. Policies in the DOWN section are DROP statements,
    not the policies we want to validate.
    """
    # Match "-- DOWN" on its own commented line (case-insensitive).
    parts = re.split(r"^\s*--\s*DOWN\b", sql, maxsplit=1, flags=re.IGNORECASE | re.MULTILINE)
    return parts[0]


# ---------------------------------------------------------------------------
# RLS pattern assertions
# ---------------------------------------------------------------------------

# Tenant-scoped tables that have a direct tenant_id column, mapped to the
# migration file that creates them. Each must have ENABLE/FORCE RLS, and a
# tenant isolation policy that gates rows by app.current_tenant_id and applies
# to the app_user role.
DIRECT_TENANT_TABLES: dict[str, str] = {
    "tenants": "001_create_tenants.sql",
    "api_keys": "002_create_api_keys.sql",
    "sessions": "004_create_sessions.sql",
    "pipelines": "005_create_pipelines.sql",
    "research_jobs": "006_create_research.sql",
    "citations": "007_create_citations.sql",
    "rate_limit_buckets": "009_create_rate_limit_metering.sql",
    "metering_events": "009_create_rate_limit_metering.sql",
}

# Child tables whose RLS policies use an EXISTS subquery against a parent table
# (tenant scope inherited via the parent), mapped to (migration_file, parent_table).
CHILD_TENANT_TABLES: dict[str, tuple[str, str]] = {
    "document_versions": ("003_create_documents.sql", "documents"),
    "pipeline_steps": ("005_create_pipelines.sql", "pipelines"),
    "research_plans": ("006_create_research.sql", "research_jobs"),
    "plan_steps": ("006_create_research.sql", "research_jobs"),
    "events": ("006_create_research.sql", "research_jobs"),
}


def _assert_rls_enabled_and_forced(sql_up: str, table: str) -> None:
    """Assert ENABLE and FORCE RLS statements exist for ``table`` in ``sql_up``."""
    enable_pattern = rf"ALTER\s+TABLE\s+{re.escape(table)}\s+ENABLE\s+ROW\s+LEVEL\s+SECURITY"
    force_pattern = rf"ALTER\s+TABLE\s+{re.escape(table)}\s+FORCE\s+ROW\s+LEVEL\s+SECURITY"
    assert re.search(enable_pattern, sql_up, flags=re.IGNORECASE), (
        f"{table}: missing 'ALTER TABLE {table} ENABLE ROW LEVEL SECURITY'"
    )
    assert re.search(force_pattern, sql_up, flags=re.IGNORECASE), (
        f"{table}: missing 'ALTER TABLE {table} FORCE ROW LEVEL SECURITY' (defense in depth)"
    )


def _extract_create_policy(sql_up: str, table: str) -> list[str]:
    """Return the bodies of all CREATE POLICY ... ON <table> statements (semicolon-terminated).

    Each returned string spans from "CREATE POLICY" up to and including the
    terminating semicolon.
    """
    pattern = re.compile(
        rf"CREATE\s+POLICY\s+\w+\s+ON\s+{re.escape(table)}\b.*?;",
        flags=re.IGNORECASE | re.DOTALL,
    )
    return pattern.findall(sql_up)


# ---------------------------------------------------------------------------
# Static parsing tests — direct tenant-scoped tables
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "table,migration_file",
    list(DIRECT_TENANT_TABLES.items()),
)
class TestDirectTenantScopedRLS:
    """Validate RLS patterns for tables with a direct tenant_id column."""

    def test_rls_is_enabled_and_forced(self, table: str, migration_file: str) -> None:
        """Each tenant-scoped table must ENABLE and FORCE RLS."""
        sql_up = _up_section(_read_migration(migration_file))
        _assert_rls_enabled_and_forced(sql_up, table)

    def test_has_create_policy(self, table: str, migration_file: str) -> None:
        """Each tenant-scoped table must have at least one CREATE POLICY statement."""
        sql_up = _up_section(_read_migration(migration_file))
        policies = _extract_create_policy(sql_up, table)
        assert policies, f"{table}: no CREATE POLICY statement found"

    def test_policy_applies_to_app_user(self, table: str, migration_file: str) -> None:
        """The tenant isolation policy must target the app_user role."""
        sql_up = _up_section(_read_migration(migration_file))
        policies = _extract_create_policy(sql_up, table)
        # At least one policy must reference TO app_user.
        assert any(
            re.search(r"\bTO\s+app_user\b", policy, flags=re.IGNORECASE) for policy in policies
        ), f"{table}: no CREATE POLICY targets the app_user role"

    def test_policy_uses_current_tenant_setting(self, table: str, migration_file: str) -> None:
        """Policy must compare tenant_id to current_setting('app.current_tenant_id')::UUID."""
        sql_up = _up_section(_read_migration(migration_file))
        policies = _extract_create_policy(sql_up, table)
        # Match the canonical expression used across all migrations, allowing whitespace
        # between operators and tolerating either 'app.current_tenant_id' quoted form.
        setting_pattern = re.compile(
            r"current_setting\(\s*'app\.current_tenant_id'\s*\)\s*::\s*UUID",
            flags=re.IGNORECASE,
        )
        assert any(setting_pattern.search(policy) for policy in policies), (
            f"{table}: no policy uses current_setting('app.current_tenant_id')::UUID"
        )

    def test_policy_has_using_and_with_check(self, table: str, migration_file: str) -> None:
        """The tenant isolation policy must constrain both reads (USING) and writes (WITH CHECK).

        ``audit_events`` is the documented exception (separate INSERT-only and
        SELECT-only policies); it is not in this fixture set.
        """
        sql_up = _up_section(_read_migration(migration_file))
        policies = _extract_create_policy(sql_up, table)
        # At least one policy combines USING and WITH CHECK clauses.
        assert any(
            re.search(r"\bUSING\b", policy, flags=re.IGNORECASE)
            and re.search(r"\bWITH\s+CHECK\b", policy, flags=re.IGNORECASE)
            for policy in policies
        ), f"{table}: tenant isolation policy lacks both USING and WITH CHECK clauses"


# ---------------------------------------------------------------------------
# Static parsing tests — child tables (EXISTS subquery scoping)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "table,migration_file,parent_table",
    [(t, m, p) for t, (m, p) in CHILD_TENANT_TABLES.items()],
)
class TestChildTableRLS:
    """Validate that child-table policies join to the parent table for tenant scope."""

    def test_rls_is_enabled_and_forced(
        self, table: str, migration_file: str, parent_table: str
    ) -> None:
        """Child tables must also ENABLE and FORCE RLS."""
        sql_up = _up_section(_read_migration(migration_file))
        _assert_rls_enabled_and_forced(sql_up, table)

    def test_policy_uses_exists_subquery_against_parent(
        self, table: str, migration_file: str, parent_table: str
    ) -> None:
        """Child policy must use an EXISTS subquery that references the parent table."""
        sql_up = _up_section(_read_migration(migration_file))
        policies = _extract_create_policy(sql_up, table)
        assert policies, f"{table}: no CREATE POLICY statement found"
        # The policy body must contain "EXISTS ( SELECT ... FROM <parent_table>".
        exists_pattern = re.compile(
            rf"EXISTS\s*\(\s*SELECT\b.*?\bFROM\s+{re.escape(parent_table)}\b",
            flags=re.IGNORECASE | re.DOTALL,
        )
        assert any(exists_pattern.search(policy) for policy in policies), (
            f"{table}: policy does not use EXISTS subquery against parent table "
            f"'{parent_table}' for tenant scoping"
        )

    def test_policy_references_current_tenant_via_parent(
        self, table: str, migration_file: str, parent_table: str
    ) -> None:
        """The EXISTS subquery must compare the parent's tenant_id to the session setting."""
        sql_up = _up_section(_read_migration(migration_file))
        policies = _extract_create_policy(sql_up, table)
        setting_pattern = re.compile(
            r"current_setting\(\s*'app\.current_tenant_id'\s*\)\s*::\s*UUID",
            flags=re.IGNORECASE,
        )
        assert any(setting_pattern.search(policy) for policy in policies), (
            f"{table}: policy does not gate by current_setting('app.current_tenant_id')::UUID"
        )

    def test_policy_applies_to_app_user(
        self, table: str, migration_file: str, parent_table: str
    ) -> None:
        """Child-table policies must also target the app_user role."""
        sql_up = _up_section(_read_migration(migration_file))
        policies = _extract_create_policy(sql_up, table)
        assert any(
            re.search(r"\bTO\s+app_user\b", policy, flags=re.IGNORECASE) for policy in policies
        ), f"{table}: child-table policy does not target the app_user role"


# ---------------------------------------------------------------------------
# Static parsing tests — audit_events special policies
# ---------------------------------------------------------------------------


class TestAuditEventsSpecialPolicies:
    """Validate audit_events RLS: separate INSERT (allows null tenant) and SELECT (own tenant)."""

    @pytest.fixture(scope="class")
    def sql_up(self) -> str:
        return _up_section(_read_migration("008_create_audit_events.sql"))

    def test_rls_is_enabled_and_forced(self, sql_up: str) -> None:
        _assert_rls_enabled_and_forced(sql_up, "audit_events")

    def test_has_separate_insert_policy(self, sql_up: str) -> None:
        """An INSERT-only policy with WITH CHECK (true) must exist (allows null tenant_id)."""
        policies = _extract_create_policy(sql_up, "audit_events")
        insert_policies = [
            p for p in policies if re.search(r"\bFOR\s+INSERT\b", p, flags=re.IGNORECASE)
        ]
        assert insert_policies, "audit_events: missing FOR INSERT policy"
        # The INSERT policy must use WITH CHECK (true) to allow null tenant_id rows
        # for unattributable auth_failure entries (R13.6).
        assert any(
            re.search(r"WITH\s+CHECK\s*\(\s*true\s*\)", p, flags=re.IGNORECASE)
            for p in insert_policies
        ), "audit_events: INSERT policy does not use WITH CHECK (true) to allow null tenant_id"

    def test_has_separate_select_policy(self, sql_up: str) -> None:
        """A SELECT-only policy gated by current_setting must exist."""
        policies = _extract_create_policy(sql_up, "audit_events")
        select_policies = [
            p for p in policies if re.search(r"\bFOR\s+SELECT\b", p, flags=re.IGNORECASE)
        ]
        assert select_policies, "audit_events: missing FOR SELECT policy"
        setting_pattern = re.compile(
            r"current_setting\(\s*'app\.current_tenant_id'\s*\)\s*::\s*UUID",
            flags=re.IGNORECASE,
        )
        assert any(setting_pattern.search(p) for p in select_policies), (
            "audit_events: SELECT policy does not gate rows by current_setting('app.current_tenant_id')::UUID"
        )

    def test_select_policy_has_no_with_check(self, sql_up: str) -> None:
        """The SELECT-only policy must not declare a WITH CHECK clause (SELECT cannot use it)."""
        policies = _extract_create_policy(sql_up, "audit_events")
        select_policies = [
            p for p in policies if re.search(r"\bFOR\s+SELECT\b", p, flags=re.IGNORECASE)
        ]
        for p in select_policies:
            assert not re.search(r"\bWITH\s+CHECK\b", p, flags=re.IGNORECASE), (
                "audit_events: SELECT policy must not contain WITH CHECK"
            )

    def test_no_combined_for_all_policy(self, sql_up: str) -> None:
        """audit_events must NOT use a single FOR ALL policy (would conflict with append-only)."""
        policies = _extract_create_policy(sql_up, "audit_events")
        for_all = [p for p in policies if re.search(r"\bFOR\s+ALL\b", p, flags=re.IGNORECASE)]
        assert not for_all, (
            "audit_events: must not use FOR ALL — INSERT and SELECT are split by design"
        )

    def test_no_update_or_delete_grant(self, sql_up: str) -> None:
        """audit_events grants must not include UPDATE or DELETE (append-only)."""
        # The GRANT line for audit_events should be exactly INSERT, SELECT.
        grant_match = re.search(
            r"GRANT\s+([A-Z, ]+?)\s+ON\s+audit_events\s+TO\s+app_user",
            sql_up,
            flags=re.IGNORECASE,
        )
        assert grant_match, "audit_events: missing GRANT statement for app_user"
        privileges = {p.strip().upper() for p in grant_match.group(1).split(",")}
        assert privileges <= {"INSERT", "SELECT"}, (
            f"audit_events: app_user has unexpected privileges {privileges}; "
            f"only INSERT and SELECT are permitted (append-only)"
        )


# ---------------------------------------------------------------------------
# Sanity check on migration coverage
# ---------------------------------------------------------------------------


class TestMigrationCoverage:
    """Sanity checks that all expected migrations are present and parsed."""

    def test_all_migrations_present(self) -> None:
        """All expected migration files exist on disk."""
        expected = {
            *DIRECT_TENANT_TABLES.values(),
            *(m for m, _ in CHILD_TENANT_TABLES.values()),
            "008_create_audit_events.sql",
        }
        for filename in expected:
            assert (MIGRATIONS_DIR / filename).is_file(), f"missing migration: {filename}"

    def test_documents_table_has_global_or_tenant_policy(self) -> None:
        """The documents table allows tenant_id IS NULL (global crawl docs) OR matching tenant.

        This is a special case (not in DIRECT_TENANT_TABLES because of the IS NULL
        clause), so we validate it explicitly.
        """
        sql_up = _up_section(_read_migration("003_create_documents.sql"))
        _assert_rls_enabled_and_forced(sql_up, "documents")
        policies = _extract_create_policy(sql_up, "documents")
        assert policies, "documents: no CREATE POLICY found"
        # Must contain "tenant_id IS NULL" combined with current_setting comparison.
        assert any(
            re.search(r"tenant_id\s+IS\s+NULL", p, flags=re.IGNORECASE)
            and re.search(
                r"current_setting\(\s*'app\.current_tenant_id'\s*\)\s*::\s*UUID",
                p,
                flags=re.IGNORECASE,
            )
            for p in policies
        ), "documents: policy must allow tenant_id IS NULL OR equal to current setting"


# ---------------------------------------------------------------------------
# Optional integration test (live Postgres)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_DB_INTEGRATION_TESTS") != "1",
    reason="Set RUN_DB_INTEGRATION_TESTS=1 and provide a Postgres DSN to run live RLS tests.",
)
class TestRLSCrossTenantIntegration:
    """Live RLS validation against a real Postgres instance.

    Requires:
      - RUN_DB_INTEGRATION_TESTS=1
      - A Postgres superuser DSN in DATABASE_URL_SUPERUSER (used to run migrations
        and create rows that bypass RLS for setup).
      - The app_user role's password in APP_USER_PASSWORD (DSN derived from the
        superuser DSN with role=app_user).

    The test:
      1. Connects as superuser, applies all migrations 000–009 to a clean DB.
      2. Inserts two tenants (A and B) and one row in each tenant-scoped table.
      3. Reconnects as app_user, sets app.current_tenant_id to tenant A.
      4. Asserts SELECT from each tenant-scoped table returns only tenant A rows
         (USING clause blocks tenant B rows).
      5. Attempts to INSERT a row with tenant B's tenant_id while session is set
         to tenant A — must fail (WITH CHECK clause blocks the write).
    """

    def test_cross_tenant_select_returns_zero_rows(self) -> None:
        asyncpg = pytest.importorskip("asyncpg")
        import asyncio

        superuser_dsn = os.environ.get("DATABASE_URL_SUPERUSER")
        app_user_dsn = os.environ.get("DATABASE_URL_APP_USER")
        if not superuser_dsn or not app_user_dsn:
            pytest.skip(
                "Set DATABASE_URL_SUPERUSER and DATABASE_URL_APP_USER to run this test."
            )

        tenant_a = uuid.uuid4()
        tenant_b = uuid.uuid4()

        async def run() -> None:
            # Insert seed data as superuser (bypasses RLS).
            su_conn = await asyncpg.connect(superuser_dsn)
            try:
                await su_conn.execute(
                    "INSERT INTO tenants (tenant_id, name) VALUES ($1, $2), ($3, $4)",
                    tenant_a,
                    "tenant-a",
                    tenant_b,
                    "tenant-b",
                )
                # Seed one session per tenant.
                await su_conn.execute(
                    "INSERT INTO sessions (tenant_id, retention_days) VALUES ($1, $2)",
                    tenant_a,
                    14,
                )
                await su_conn.execute(
                    "INSERT INTO sessions (tenant_id, retention_days) VALUES ($1, $2)",
                    tenant_b,
                    14,
                )
            finally:
                await su_conn.close()

            # Connect as app_user, set session tenant to A, verify isolation.
            app_conn = await asyncpg.connect(app_user_dsn)
            try:
                await app_conn.execute(
                    "SET LOCAL app.current_tenant_id = $1", str(tenant_a)
                )
                # SELECT must only see tenant A's session.
                rows = await app_conn.fetch(
                    "SELECT tenant_id FROM sessions"
                )
                tenant_ids = {row["tenant_id"] for row in rows}
                assert tenant_ids == {tenant_a}, (
                    f"RLS leak: as tenant A, saw tenant_ids {tenant_ids}"
                )

                # INSERT with tenant B's tenant_id while session is tenant A
                # must be rejected by the WITH CHECK clause.
                with pytest.raises(asyncpg.exceptions.PostgresError):
                    await app_conn.execute(
                        "INSERT INTO sessions (tenant_id, retention_days) "
                        "VALUES ($1, $2)",
                        tenant_b,
                        14,
                    )
            finally:
                await app_conn.close()

        asyncio.run(run())
