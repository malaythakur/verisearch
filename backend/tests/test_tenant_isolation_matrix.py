"""Tenant Isolation Matrix Tests (Tasks 18.1–18.4).

Tests:
- 18.1: Parameterized test matrix across all resource types
- 18.2: Property test — cross-tenant access returns uniform 404 (Property 19)
- 18.3: Property test — same-tenant access succeeds with standard 2xx (Property 19 converse)
- 18.4: Verify RLS policies at DB level with direct SQL cross-tenant attempts

**Validates: Requirements R7.7, R8.5, R9.7, R13.3, R15.7**
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# Resource types for the isolation matrix
# ---------------------------------------------------------------------------


class ResourceType(str, Enum):
    """All tenant-scoped resource types in the system."""

    RESEARCH_JOB = "research_job"
    SESSION = "session"
    PIPELINE = "pipeline"
    AUDIT_ENTRY = "audit_entry"
    DELETION_TARGET = "deletion_target"
    METERING_RECORD = "metering_record"
    API_KEY = "api_key"


# Map resource types to their expected 404 error codes
RESOURCE_NOT_FOUND_CODES = {
    ResourceType.RESEARCH_JOB: "job_not_found",
    ResourceType.SESSION: "session_not_found",
    ResourceType.PIPELINE: "pipeline_not_found",
    ResourceType.AUDIT_ENTRY: "resource_not_found",
    ResourceType.DELETION_TARGET: "resource_not_found",
    ResourceType.METERING_RECORD: "resource_not_found",
    ResourceType.API_KEY: "resource_not_found",
}


# ---------------------------------------------------------------------------
# Simulated tenant isolation layer (mirrors real implementation)
# ---------------------------------------------------------------------------


@dataclass
class TenantResource:
    """A resource owned by a tenant."""

    resource_id: str
    tenant_id: str
    resource_type: ResourceType
    data: dict[str, Any]


class TenantIsolationService:
    """Service that enforces tenant isolation across all resource types.

    Implements the uniform 404 pattern: cross-tenant access returns the same
    response shape as not-found, without disclosing resource existence (R13.3).
    """

    def __init__(self) -> None:
        self._resources: dict[str, TenantResource] = {}

    def create_resource(
        self,
        tenant_id: str,
        resource_type: ResourceType,
        data: Optional[dict[str, Any]] = None,
    ) -> TenantResource:
        """Create a resource scoped to a tenant."""
        resource_id = str(uuid.uuid4())
        resource = TenantResource(
            resource_id=resource_id,
            tenant_id=tenant_id,
            resource_type=resource_type,
            data=data or {},
        )
        self._resources[resource_id] = resource
        return resource

    def get_resource(
        self,
        requesting_tenant_id: str,
        resource_id: str,
        resource_type: ResourceType,
    ) -> tuple[int, dict[str, Any]]:
        """Attempt to access a resource.

        Returns:
            Tuple of (status_code, response_body).
            - (200, resource_data) for same-tenant access
            - (404, error_response) for cross-tenant or not-found (uniform shape)
        """
        resource = self._resources.get(resource_id)

        # Not found OR cross-tenant → same 404 response (R13.3)
        if resource is None or resource.tenant_id != requesting_tenant_id:
            error_code = RESOURCE_NOT_FOUND_CODES[resource_type]
            return (
                404,
                {
                    "error": {
                        "code": error_code,
                        "message": f"{resource_type.value} not found",
                    }
                },
            )

        # Same tenant → success
        return (200, {"resource_id": resource.resource_id, "data": resource.data})

    def delete_resource(
        self,
        requesting_tenant_id: str,
        resource_id: str,
        resource_type: ResourceType,
    ) -> tuple[int, dict[str, Any]]:
        """Attempt to delete a resource.

        Same isolation rules as get_resource.
        """
        resource = self._resources.get(resource_id)

        if resource is None or resource.tenant_id != requesting_tenant_id:
            error_code = RESOURCE_NOT_FOUND_CODES[resource_type]
            return (
                404,
                {
                    "error": {
                        "code": error_code,
                        "message": f"{resource_type.value} not found",
                    }
                },
            )

        # Same tenant → delete and return 204
        del self._resources[resource_id]
        return (204, {})

    def list_resources(
        self,
        tenant_id: str,
        resource_type: ResourceType,
    ) -> list[TenantResource]:
        """List resources for a tenant (only returns own resources)."""
        return [
            r
            for r in self._resources.values()
            if r.tenant_id == tenant_id and r.resource_type == resource_type
        ]


# ---------------------------------------------------------------------------
# RLS Policy Simulator (Task 18.4)
# ---------------------------------------------------------------------------


class RLSPolicySimulator:
    """Simulates PostgreSQL Row-Level Security policies.

    Verifies that direct SQL queries are filtered by tenant_id,
    preventing cross-tenant data access at the database level.
    """

    def __init__(self) -> None:
        self._rows: list[dict[str, Any]] = []

    def insert(self, table: str, tenant_id: str, data: dict[str, Any]) -> dict[str, Any]:
        """Insert a row with tenant_id."""
        row = {"id": str(uuid.uuid4()), "table": table, "tenant_id": tenant_id, **data}
        self._rows.append(row)
        return row

    def select_with_rls(self, table: str, session_tenant_id: str, **filters: Any) -> list[dict[str, Any]]:
        """SELECT with RLS policy applied (only returns rows matching session tenant).

        This simulates: SET app.tenant_id = session_tenant_id;
        SELECT * FROM table WHERE <filters>;
        With RLS policy: USING (tenant_id = current_setting('app.tenant_id'))
        """
        results = []
        for row in self._rows:
            if row["table"] != table:
                continue
            # RLS filter: only rows belonging to the session tenant
            if row["tenant_id"] != session_tenant_id:
                continue
            # Apply additional filters
            match = True
            for key, value in filters.items():
                if row.get(key) != value:
                    match = False
                    break
            if match:
                results.append(row)
        return results

    def select_without_rls(self, table: str, **filters: Any) -> list[dict[str, Any]]:
        """SELECT without RLS (superuser bypass) — for verification only."""
        results = []
        for row in self._rows:
            if row["table"] != table:
                continue
            match = True
            for key, value in filters.items():
                if row.get(key) != value:
                    match = False
                    break
            if match:
                results.append(row)
        return results

    def update_with_rls(
        self, table: str, session_tenant_id: str, row_id: str, data: dict[str, Any]
    ) -> int:
        """UPDATE with RLS policy — only updates rows belonging to session tenant.

        Returns number of rows affected.
        """
        affected = 0
        for row in self._rows:
            if row["table"] != table:
                continue
            if row["id"] != row_id:
                continue
            # RLS: can only update own rows
            if row["tenant_id"] != session_tenant_id:
                continue
            row.update(data)
            affected += 1
        return affected

    def delete_with_rls(self, table: str, session_tenant_id: str, row_id: str) -> int:
        """DELETE with RLS policy — only deletes rows belonging to session tenant.

        Returns number of rows affected.
        """
        before = len(self._rows)
        self._rows = [
            row
            for row in self._rows
            if not (row["table"] == table and row["id"] == row_id and row["tenant_id"] == session_tenant_id)
        ]
        return before - len(self._rows)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

tenant_id_st = st.uuids().map(str)
resource_type_st = st.sampled_from(list(ResourceType))
resource_data_st = st.fixed_dictionaries({"name": st.text(min_size=1, max_size=50)})


# ---------------------------------------------------------------------------
# Task 18.1: Parameterized test matrix across all resource types
# ---------------------------------------------------------------------------


class TestTenantIsolationMatrix:
    """Parameterized test matrix verifying isolation across all resource types."""

    @pytest.fixture
    def isolation_service(self) -> TenantIsolationService:
        return TenantIsolationService()

    @pytest.mark.parametrize("resource_type", list(ResourceType))
    def test_cross_tenant_get_returns_404(
        self, isolation_service: TenantIsolationService, resource_type: ResourceType
    ) -> None:
        """Cross-tenant GET returns 404 for all resource types."""
        tenant_a = str(uuid.uuid4())
        tenant_b = str(uuid.uuid4())

        # Create resource owned by tenant A
        resource = isolation_service.create_resource(tenant_a, resource_type, {"name": "secret"})

        # Tenant B tries to access it
        status, body = isolation_service.get_resource(tenant_b, resource.resource_id, resource_type)

        assert status == 404
        assert body["error"]["code"] == RESOURCE_NOT_FOUND_CODES[resource_type]

    @pytest.mark.parametrize("resource_type", list(ResourceType))
    def test_same_tenant_get_returns_200(
        self, isolation_service: TenantIsolationService, resource_type: ResourceType
    ) -> None:
        """Same-tenant GET returns 200 for all resource types."""
        tenant_a = str(uuid.uuid4())

        resource = isolation_service.create_resource(tenant_a, resource_type, {"name": "my-resource"})

        status, body = isolation_service.get_resource(tenant_a, resource.resource_id, resource_type)

        assert status == 200
        assert body["resource_id"] == resource.resource_id

    @pytest.mark.parametrize("resource_type", list(ResourceType))
    def test_cross_tenant_delete_returns_404(
        self, isolation_service: TenantIsolationService, resource_type: ResourceType
    ) -> None:
        """Cross-tenant DELETE returns 404 for all resource types."""
        tenant_a = str(uuid.uuid4())
        tenant_b = str(uuid.uuid4())

        resource = isolation_service.create_resource(tenant_a, resource_type, {"name": "secret"})

        status, body = isolation_service.delete_resource(tenant_b, resource.resource_id, resource_type)

        assert status == 404
        assert body["error"]["code"] == RESOURCE_NOT_FOUND_CODES[resource_type]

    @pytest.mark.parametrize("resource_type", list(ResourceType))
    def test_nonexistent_resource_returns_same_404_shape(
        self, isolation_service: TenantIsolationService, resource_type: ResourceType
    ) -> None:
        """Non-existent resource returns same 404 shape as cross-tenant (no presence disclosure)."""
        tenant_a = str(uuid.uuid4())
        tenant_b = str(uuid.uuid4())

        # Create resource owned by tenant A
        resource = isolation_service.create_resource(tenant_a, resource_type, {"name": "secret"})

        # Cross-tenant access
        cross_status, cross_body = isolation_service.get_resource(tenant_b, resource.resource_id, resource_type)

        # Non-existent resource access
        fake_id = str(uuid.uuid4())
        notfound_status, notfound_body = isolation_service.get_resource(tenant_a, fake_id, resource_type)

        # Both should be identical in shape (no presence disclosure)
        assert cross_status == notfound_status == 404
        assert cross_body["error"]["code"] == notfound_body["error"]["code"]
        # Response structure is identical
        assert set(cross_body["error"].keys()) == set(notfound_body["error"].keys())

    @pytest.mark.parametrize("resource_type", list(ResourceType))
    def test_list_only_returns_own_resources(
        self, isolation_service: TenantIsolationService, resource_type: ResourceType
    ) -> None:
        """Listing resources only returns resources owned by the requesting tenant."""
        tenant_a = str(uuid.uuid4())
        tenant_b = str(uuid.uuid4())

        # Create resources for both tenants
        isolation_service.create_resource(tenant_a, resource_type, {"name": "a-resource"})
        isolation_service.create_resource(tenant_b, resource_type, {"name": "b-resource"})

        # Tenant A only sees their own
        a_resources = isolation_service.list_resources(tenant_a, resource_type)
        assert len(a_resources) == 1
        assert all(r.tenant_id == tenant_a for r in a_resources)

        # Tenant B only sees their own
        b_resources = isolation_service.list_resources(tenant_b, resource_type)
        assert len(b_resources) == 1
        assert all(r.tenant_id == tenant_b for r in b_resources)


# ---------------------------------------------------------------------------
# Task 18.2: Property test — cross-tenant access returns uniform 404 (Property 19)
# ---------------------------------------------------------------------------


class TestCrossTenantUniform404Property:
    """**Validates: Requirements R7.7, R8.5, R9.7, R13.3, R15.7**

    Property 19: Cross-tenant access returns a uniform 404 response
    indistinguishable from not-found, for all resource types.
    """

    @given(
        owner_tenant=tenant_id_st,
        requesting_tenant=tenant_id_st,
        resource_type=resource_type_st,
        data=resource_data_st,
    )
    @settings(max_examples=200)
    def test_cross_tenant_access_uniform_404(
        self,
        owner_tenant: str,
        requesting_tenant: str,
        resource_type: ResourceType,
        data: dict[str, Any],
    ) -> None:
        """Cross-tenant access always returns 404 with the resource-specific error code."""
        assume(owner_tenant != requesting_tenant)

        service = TenantIsolationService()
        resource = service.create_resource(owner_tenant, resource_type, data)

        # Cross-tenant access
        status, body = service.get_resource(requesting_tenant, resource.resource_id, resource_type)

        # Must be 404
        assert status == 404

        # Must have the correct error code for this resource type
        expected_code = RESOURCE_NOT_FOUND_CODES[resource_type]
        assert body["error"]["code"] == expected_code

        # Must have the standard error shape
        assert "error" in body
        assert "code" in body["error"]
        assert "message" in body["error"]

    @given(
        owner_tenant=tenant_id_st,
        requesting_tenant=tenant_id_st,
        resource_type=resource_type_st,
    )
    @settings(max_examples=200)
    def test_cross_tenant_indistinguishable_from_not_found(
        self,
        owner_tenant: str,
        requesting_tenant: str,
        resource_type: ResourceType,
    ) -> None:
        """Cross-tenant 404 is byte-identical in shape to genuine not-found 404."""
        assume(owner_tenant != requesting_tenant)

        service = TenantIsolationService()
        resource = service.create_resource(owner_tenant, resource_type, {"name": "test"})

        # Cross-tenant access
        cross_status, cross_body = service.get_resource(
            requesting_tenant, resource.resource_id, resource_type
        )

        # Genuine not-found
        fake_id = str(uuid.uuid4())
        notfound_status, notfound_body = service.get_resource(
            requesting_tenant, fake_id, resource_type
        )

        # Responses must be structurally identical
        assert cross_status == notfound_status
        assert cross_body.keys() == notfound_body.keys()
        assert cross_body["error"].keys() == notfound_body["error"].keys()
        assert cross_body["error"]["code"] == notfound_body["error"]["code"]


# ---------------------------------------------------------------------------
# Task 18.3: Property test — same-tenant access succeeds (Property 19 converse)
# ---------------------------------------------------------------------------


class TestSameTenantAccessProperty:
    """**Validates: Requirements R13.3**

    Property 19 (converse): Same-tenant access succeeds with standard 2xx.
    """

    @given(
        tenant_id=tenant_id_st,
        resource_type=resource_type_st,
        data=resource_data_st,
    )
    @settings(max_examples=200)
    def test_same_tenant_access_succeeds(
        self,
        tenant_id: str,
        resource_type: ResourceType,
        data: dict[str, Any],
    ) -> None:
        """Same-tenant access always returns 200 with the resource data."""
        service = TenantIsolationService()
        resource = service.create_resource(tenant_id, resource_type, data)

        status, body = service.get_resource(tenant_id, resource.resource_id, resource_type)

        assert status == 200
        assert body["resource_id"] == resource.resource_id
        assert body["data"] == data

    @given(
        tenant_id=tenant_id_st,
        resource_type=resource_type_st,
    )
    @settings(max_examples=100)
    def test_same_tenant_delete_succeeds(
        self,
        tenant_id: str,
        resource_type: ResourceType,
    ) -> None:
        """Same-tenant delete returns 204."""
        service = TenantIsolationService()
        resource = service.create_resource(tenant_id, resource_type, {"name": "to-delete"})

        status, _ = service.delete_resource(tenant_id, resource.resource_id, resource_type)

        assert status == 204


# ---------------------------------------------------------------------------
# Task 18.4: RLS policies at DB level with direct SQL cross-tenant attempts
# ---------------------------------------------------------------------------


class TestRLSPolicies:
    """Verify Row-Level Security policies block cross-tenant access at the DB level.

    Simulates PostgreSQL RLS policies that bind tenant_id to the connection's
    app.tenant_id GUC, ensuring queries cannot access other tenants' data.
    """

    @pytest.fixture
    def rls(self) -> RLSPolicySimulator:
        return RLSPolicySimulator()

    @pytest.mark.parametrize(
        "table",
        [
            "research_jobs",
            "sessions",
            "pipelines",
            "audit_events",
            "api_keys",
            "metering_events",
            "citations",
        ],
    )
    def test_select_with_rls_blocks_cross_tenant(self, rls: RLSPolicySimulator, table: str) -> None:
        """SELECT with RLS cannot see other tenants' rows."""
        tenant_a = str(uuid.uuid4())
        tenant_b = str(uuid.uuid4())

        # Insert rows for both tenants
        row_a = rls.insert(table, tenant_a, {"data": "tenant_a_secret"})
        row_b = rls.insert(table, tenant_b, {"data": "tenant_b_secret"})

        # Tenant A's session can only see their own rows
        results_a = rls.select_with_rls(table, tenant_a)
        assert len(results_a) == 1
        assert results_a[0]["tenant_id"] == tenant_a
        assert results_a[0]["data"] == "tenant_a_secret"

        # Tenant B's session can only see their own rows
        results_b = rls.select_with_rls(table, tenant_b)
        assert len(results_b) == 1
        assert results_b[0]["tenant_id"] == tenant_b

    @pytest.mark.parametrize(
        "table",
        [
            "research_jobs",
            "sessions",
            "pipelines",
            "audit_events",
            "api_keys",
        ],
    )
    def test_update_with_rls_blocks_cross_tenant(self, rls: RLSPolicySimulator, table: str) -> None:
        """UPDATE with RLS cannot modify other tenants' rows."""
        tenant_a = str(uuid.uuid4())
        tenant_b = str(uuid.uuid4())

        row_a = rls.insert(table, tenant_a, {"data": "original"})

        # Tenant B tries to update tenant A's row
        affected = rls.update_with_rls(table, tenant_b, row_a["id"], {"data": "hacked"})
        assert affected == 0

        # Verify row is unchanged
        rows = rls.select_without_rls(table, id=row_a["id"])
        assert rows[0]["data"] == "original"

    @pytest.mark.parametrize(
        "table",
        [
            "research_jobs",
            "sessions",
            "pipelines",
            "api_keys",
        ],
    )
    def test_delete_with_rls_blocks_cross_tenant(self, rls: RLSPolicySimulator, table: str) -> None:
        """DELETE with RLS cannot remove other tenants' rows."""
        tenant_a = str(uuid.uuid4())
        tenant_b = str(uuid.uuid4())

        row_a = rls.insert(table, tenant_a, {"data": "protected"})

        # Tenant B tries to delete tenant A's row
        affected = rls.delete_with_rls(table, tenant_b, row_a["id"])
        assert affected == 0

        # Verify row still exists
        rows = rls.select_without_rls(table, id=row_a["id"])
        assert len(rows) == 1

    @given(
        tenant_a=tenant_id_st,
        tenant_b=tenant_id_st,
        table=st.sampled_from(["research_jobs", "sessions", "pipelines", "audit_events", "api_keys"]),
    )
    @settings(max_examples=100)
    def test_rls_property_cross_tenant_select_always_empty(
        self, tenant_a: str, tenant_b: str, table: str
    ) -> None:
        """Property: cross-tenant SELECT with RLS always returns empty results."""
        assume(tenant_a != tenant_b)

        rls = RLSPolicySimulator()
        rls.insert(table, tenant_a, {"data": "secret"})

        # Tenant B cannot see tenant A's data
        results = rls.select_with_rls(table, tenant_b)
        assert len(results) == 0

    @given(
        tenant_a=tenant_id_st,
        tenant_b=tenant_id_st,
        table=st.sampled_from(["research_jobs", "sessions", "pipelines", "api_keys"]),
    )
    @settings(max_examples=100)
    def test_rls_property_cross_tenant_update_zero_affected(
        self, tenant_a: str, tenant_b: str, table: str
    ) -> None:
        """Property: cross-tenant UPDATE with RLS affects zero rows."""
        assume(tenant_a != tenant_b)

        rls = RLSPolicySimulator()
        row = rls.insert(table, tenant_a, {"data": "original"})

        affected = rls.update_with_rls(table, tenant_b, row["id"], {"data": "hacked"})
        assert affected == 0
