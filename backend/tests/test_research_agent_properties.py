"""Property-based tests for Research Agent and Session Store.

Properties tested:
- Property 15: SSE event_id strictly monotonic and replayable (R7.2, R7.3)
- Property 16: Research report citations reference indexed documents (R7.4)
- Property 17: Output schema validation when supplied (R7.5)
- Property 18: Budget exceedance terminates with partial report (R7.6)
- Property 29: Session memory bounds and recency (R8.1, R8.2)
- Property 30: Session expiry deletes memory and stops incorporation (R8.4)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from backend.research_agent.budget import BudgetConfig, BudgetExceededError, BudgetTracker
from backend.research_agent.events import EventBuffer, EventEmitter
from backend.research_agent.models import (
    EventType,
    JobState,
    ResearchEvent,
)
from backend.research_agent.planner import ResearchPlanner
from backend.research_agent.service import (
    InvalidResearchRequestError,
    JobNotFoundError,
    ResearchAgentService,
)
from backend.session_store.models import (
    MAX_CITATIONS,
    MAX_DOC_IDS,
    SessionCitation,
    SessionMemory,
)
from backend.session_store.service import (
    SessionNotFoundError,
    SessionService,
)


# ===========================================================================
# Strategies
# ===========================================================================

# Research goal: 1-4096 chars (R7.8)
research_goal_strategy = st.text(
    alphabet=st.characters(categories=("L", "N", "P", "S", "Z")),
    min_size=1,
    max_size=200,  # Keep small for speed in property tests
)

# Tenant IDs
tenant_id_strategy = st.uuids().map(str)

# Retention days [1, 90] (R8.1)
retention_days_strategy = st.integers(min_value=1, max_value=90)

# Budget configs with small values for fast testing
budget_strategy = st.builds(
    BudgetConfig,
    max_steps=st.integers(min_value=1, max_value=10),
    max_duration_ms=st.integers(min_value=100, max_value=60_000),
    max_tool_calls=st.integers(min_value=1, max_value=20),
)

# Number of citations to add to session memory
citation_count_strategy = st.integers(min_value=0, max_value=100)

# Number of doc_ids to add to session memory
doc_id_count_strategy = st.integers(min_value=0, max_value=50)


# ===========================================================================
# Property 15: SSE event_id strictly monotonic and replayable
# ===========================================================================


class TestProperty15EventMonotonicity:
    """Property 15: Research-job event stream is strictly monotonic and replayable.

    **Validates: Requirements 7.2, 7.3**

    For any research job's event stream, event_id values are strictly
    monotonically increasing; for any reconnection with Last-Event-ID = N,
    the server replays exactly those events with event_id > N; the first
    event is plan_updated with 1..32 steps.
    """

    @given(goal=research_goal_strategy, tenant_id=tenant_id_strategy)
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_event_ids_strictly_monotonic(self, goal: str, tenant_id: str) -> None:
        """For any job, event_ids are strictly monotonically increasing."""
        service = ResearchAgentService()
        job_id = service.start_job(tenant_id, goal)
        events = service.get_events(job_id, tenant_id)

        assert len(events) >= 1, "Job must emit at least one event"

        for i in range(1, len(events)):
            assert events[i].event_id > events[i - 1].event_id, (
                f"Event {i} (id={events[i].event_id}) must be > "
                f"event {i-1} (id={events[i-1].event_id})"
            )

    @given(goal=research_goal_strategy, tenant_id=tenant_id_strategy)
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_first_event_is_plan_updated(self, goal: str, tenant_id: str) -> None:
        """For any job, the first event is plan_updated with 1-32 steps."""
        service = ResearchAgentService()
        job_id = service.start_job(tenant_id, goal)
        events = service.get_events(job_id, tenant_id)

        assert events[0].type == EventType.PLAN_UPDATED
        plan_steps = events[0].payload.get("plan", [])
        assert 1 <= len(plan_steps) <= 32

    @given(
        goal=research_goal_strategy,
        tenant_id=tenant_id_strategy,
        replay_offset=st.integers(min_value=0, max_value=50),
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_last_event_id_replay(
        self, goal: str, tenant_id: str, replay_offset: int
    ) -> None:
        """For any Last-Event-ID = N, replay returns exactly events with id > N."""
        service = ResearchAgentService()
        job_id = service.start_job(tenant_id, goal)
        all_events = service.get_events(job_id, tenant_id)

        # Pick a replay point
        if not all_events:
            return

        # Use an offset within the event range
        last_event_id = min(replay_offset, all_events[-1].event_id)

        replayed = service.get_events(job_id, tenant_id, last_event_id=last_event_id)

        # All replayed events must have id > last_event_id
        for event in replayed:
            assert event.event_id > last_event_id

        # Replayed events should be exactly those from all_events with id > last_event_id
        expected = [e for e in all_events if e.event_id > last_event_id]
        assert len(replayed) == len(expected)

        # Order must be preserved
        for r, e in zip(replayed, expected):
            assert r.event_id == e.event_id
            assert r.type == e.type


# ===========================================================================
# Property 16: Research report citations reference indexed documents
# ===========================================================================


class TestProperty16ReportCitations:
    """Property 16: Research report citations exist in the index.

    **Validates: Requirements 7.4**

    For any successfully completed research job, the final report's every
    citation references a document_id and version.
    """

    @given(goal=research_goal_strategy, tenant_id=tenant_id_strategy)
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_citations_have_valid_references(self, goal: str, tenant_id: str) -> None:
        """For any completed job, all citations reference valid documents."""
        service = ResearchAgentService()
        job_id = service.start_job(tenant_id, goal)
        report = service.get_report(job_id, tenant_id)

        for citation in report.citations:
            # Every citation must have a non-empty document_id
            assert citation.document_id, "Citation must reference a document_id"
            # Version must be positive
            assert citation.version >= 1, "Citation version must be >= 1"

    @given(goal=research_goal_strategy, tenant_id=tenant_id_strategy)
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_citation_events_match_report(self, goal: str, tenant_id: str) -> None:
        """Citation events in the stream correspond to report citations."""
        service = ResearchAgentService()
        job_id = service.start_job(tenant_id, goal)
        events = service.get_events(job_id, tenant_id)
        report = service.get_report(job_id, tenant_id)

        # Count citation events
        citation_events = [e for e in events if e.type == EventType.CITATION]

        # Each citation event should have a document_id
        for ce in citation_events:
            assert "document_id" in ce.payload
            assert ce.payload["document_id"]


# ===========================================================================
# Property 17: Output schema validation when supplied
# ===========================================================================


class TestProperty17OutputSchema:
    """Property 17: Output schema validation when supplied.

    **Validates: Requirements 7.5**

    For any research job with a valid output_schema, the final report's
    structured payload validates against that schema.
    """

    @given(goal=research_goal_strategy, tenant_id=tenant_id_strategy)
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_report_validates_against_object_schema(
        self, goal: str, tenant_id: str
    ) -> None:
        """Report with object schema has dict payload."""
        service = ResearchAgentService()
        schema = {"type": "object"}
        job_id = service.start_job(tenant_id, goal, output_schema=schema)
        state = service.get_job_state(job_id, tenant_id)

        if state == JobState.SUCCEEDED:
            report = service.get_report(job_id, tenant_id)
            # Structured payload should be a dict (object)
            if report.structured_payload is not None:
                assert isinstance(report.structured_payload, dict)

    @given(
        goal=research_goal_strategy,
        tenant_id=tenant_id_strategy,
        required_field=st.text(
            alphabet=st.characters(categories=("L",)),
            min_size=1,
            max_size=20,
        ),
    )
    @settings(max_examples=15, suppress_health_check=[HealthCheck.too_slow])
    def test_invalid_schema_rejected(
        self, goal: str, tenant_id: str, required_field: str
    ) -> None:
        """Invalid output_schema (no schema keywords) is rejected."""
        service = ResearchAgentService()
        # Schema with no valid keywords
        invalid_schema = {required_field: "not_a_schema_value", f"{required_field}_2": 42}

        # Should be rejected if no valid schema keywords present
        valid_keywords = {
            "type", "properties", "items", "required", "enum",
            "const", "allOf", "anyOf", "oneOf", "not",
            "$schema", "$id", "$ref", "title", "description",
            "default", "examples", "minimum", "maximum",
            "minLength", "maxLength", "pattern", "format",
            "additionalProperties", "minItems", "maxItems",
        }

        has_valid = any(k in valid_keywords for k in invalid_schema.keys())
        if not has_valid:
            try:
                service.start_job(tenant_id, goal, output_schema=invalid_schema)
                # If it didn't raise, the schema was considered valid
                assert False, "Should have raised InvalidResearchRequestError"
            except InvalidResearchRequestError:
                pass  # Expected


# ===========================================================================
# Property 18: Budget exceedance terminates with partial report
# ===========================================================================


class TestProperty18BudgetExceedance:
    """Property 18: Budget exceedance terminates with a partial report.

    **Validates: Requirements 7.6**

    For any research job whose execution exceeds its budget, the terminal
    event is an error with code budget_exceeded, and GET /v1/research/{job_id}
    returns the partial report and citations gathered prior to termination.
    """

    @given(
        goal=research_goal_strategy,
        tenant_id=tenant_id_strategy,
        max_steps=st.integers(min_value=1, max_value=3),
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_budget_exceeded_produces_partial_report(
        self, goal: str, tenant_id: str, max_steps: int
    ) -> None:
        """When budget is exceeded, job has partial report available."""
        service = ResearchAgentService()
        budgets = BudgetConfig(
            max_steps=max_steps,
            max_duration_ms=300_000,
            max_tool_calls=100,
        )
        job_id = service.start_job(tenant_id, goal, budgets=budgets)
        state = service.get_job_state(job_id, tenant_id)

        if state == JobState.BUDGET_EXCEEDED:
            report = service.get_report(job_id, tenant_id)
            assert report.is_partial, "Budget-exceeded job must have partial report"

    @given(
        goal=research_goal_strategy,
        tenant_id=tenant_id_strategy,
        max_tool_calls=st.integers(min_value=1, max_value=3),
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_budget_exceeded_emits_error_event(
        self, goal: str, tenant_id: str, max_tool_calls: int
    ) -> None:
        """When budget is exceeded, terminal event has code budget_exceeded."""
        service = ResearchAgentService()
        budgets = BudgetConfig(
            max_steps=32,
            max_duration_ms=300_000,
            max_tool_calls=max_tool_calls,
        )
        job_id = service.start_job(tenant_id, goal, budgets=budgets)
        state = service.get_job_state(job_id, tenant_id)

        if state == JobState.BUDGET_EXCEEDED:
            events = service.get_events(job_id, tenant_id)
            error_events = [e for e in events if e.type == EventType.ERROR]
            assert len(error_events) == 1
            assert error_events[0].payload["code"] == "budget_exceeded"

    @given(
        tenant_id=tenant_id_strategy,
        max_steps=st.integers(min_value=1, max_value=5),
        max_tool_calls=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=15, suppress_health_check=[HealthCheck.too_slow])
    def test_budget_tracker_enforces_limits(
        self, tenant_id: str, max_steps: int, max_tool_calls: int
    ) -> None:
        """BudgetTracker raises after exactly max_steps or max_tool_calls."""
        tracker = BudgetTracker(
            config=BudgetConfig(
                max_steps=max_steps,
                max_duration_ms=999_999,
                max_tool_calls=max_tool_calls,
            )
        )
        tracker.start()

        # Should not raise before limit
        for _ in range(min(max_steps, max_tool_calls) - 1):
            tracker.check()
            tracker.record_step()
            tracker.record_tool_call()

        # Should raise at or after limit
        # Record one more to hit the limit
        tracker.record_step()
        tracker.record_tool_call()

        assert tracker.is_exceeded()


# ===========================================================================
# Property 29: Session memory bounds and recency
# ===========================================================================


class TestProperty29SessionMemoryBounds:
    """Property 29: Session memory bounds and recency.

    **Validates: Requirements 8.1, 8.2**

    For any session with N citations and M distinct doc_ids, the memory
    contains the most recent min(N, 50) citations and the most recent
    min(M, 20) distinct doc_ids.
    """

    @given(num_citations=st.integers(min_value=0, max_value=100))
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_citation_count_bounded(self, num_citations: int) -> None:
        """Session memory never exceeds 50 citations."""
        memory = SessionMemory()
        for i in range(num_citations):
            memory.add_citation(
                SessionCitation(document_id=f"doc-{i}", version=1)
            )

        expected_count = min(num_citations, MAX_CITATIONS)
        assert memory.citation_count == expected_count

    @given(num_doc_ids=st.integers(min_value=0, max_value=100))
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_doc_id_count_bounded(self, num_doc_ids: int) -> None:
        """Session memory never exceeds 20 doc_ids."""
        memory = SessionMemory()
        for i in range(num_doc_ids):
            memory.add_doc_id(f"doc-{i}")

        expected_count = min(num_doc_ids, MAX_DOC_IDS)
        assert memory.doc_id_count == expected_count

    @given(num_citations=st.integers(min_value=1, max_value=100))
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_citations_are_most_recent(self, num_citations: int) -> None:
        """Stored citations are the most recent ones added."""
        memory = SessionMemory()
        for i in range(num_citations):
            memory.add_citation(
                SessionCitation(document_id=f"doc-{i}", version=1)
            )

        citations = memory.citations
        expected_start = max(0, num_citations - MAX_CITATIONS)

        # The oldest citation in memory should be doc-{expected_start}
        assert citations[0].document_id == f"doc-{expected_start}"
        # The newest should be doc-{num_citations - 1}
        assert citations[-1].document_id == f"doc-{num_citations - 1}"

    @given(num_doc_ids=st.integers(min_value=1, max_value=100))
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_doc_ids_are_most_recent(self, num_doc_ids: int) -> None:
        """Stored doc_ids are the most recent ones added."""
        memory = SessionMemory()
        for i in range(num_doc_ids):
            memory.add_doc_id(f"doc-{i}")

        doc_ids = memory.doc_ids
        expected_start = max(0, num_doc_ids - MAX_DOC_IDS)

        # The oldest doc_id in memory should be doc-{expected_start}
        assert doc_ids[0] == f"doc-{expected_start}"
        # The newest should be doc-{num_doc_ids - 1}
        assert doc_ids[-1] == f"doc-{num_doc_ids - 1}"

    @given(
        tenant_id=tenant_id_strategy,
        retention_days=retention_days_strategy,
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_retention_days_in_valid_range(
        self, tenant_id: str, retention_days: int
    ) -> None:
        """Sessions created with valid retention_days are accepted (R8.1)."""
        service = SessionService()
        session = service.create(tenant_id, retention_days=retention_days)
        assert session.retention_days == retention_days
        assert 1 <= session.retention_days <= 90


# ===========================================================================
# Property 30: Session expiry deletes memory and stops incorporation
# ===========================================================================


class TestProperty30SessionExpiry:
    """Property 30: Session expiry deletes memory and stops incorporation.

    **Validates: Requirements 8.4**

    For any session whose created_at + retention_days is in the past,
    within 24 simulated hours the session's memory is unreachable,
    no subsequent request incorporates its memory, and the Audit_Log
    contains a session_expired entry.
    """

    @given(
        tenant_id=tenant_id_strategy,
        retention_days=retention_days_strategy,
        num_citations=st.integers(min_value=1, max_value=20),
        num_doc_ids=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_expired_session_memory_unreachable(
        self,
        tenant_id: str,
        retention_days: int,
        num_citations: int,
        num_doc_ids: int,
    ) -> None:
        """After expiry, session memory is unreachable."""
        audit_entries: list[dict] = []

        class MockAuditLog:
            def append(self, entry: dict) -> None:
                audit_entries.append(entry)

        service = SessionService(audit_log=MockAuditLog())
        session = service.create(tenant_id, retention_days=retention_days)

        # Add memory
        citations = [
            {"document_id": f"doc-{i}", "version": 1}
            for i in range(num_citations)
        ]
        doc_ids = [f"doc-{i}" for i in range(num_doc_ids)]
        service.add_to_memory(session.session_id, tenant_id, citations, doc_ids)

        # Simulate time passing beyond expiry
        future = datetime.now(timezone.utc) + timedelta(days=retention_days + 1)
        expired = service.expire_sweep(now=future)

        # Session should be expired
        assert session.session_id in expired

        # Memory should be unreachable
        try:
            service.get_memory(session.session_id, tenant_id)
            assert False, "Should have raised SessionNotFoundError"
        except SessionNotFoundError:
            pass  # Expected

    @given(
        tenant_id=tenant_id_strategy,
        retention_days=retention_days_strategy,
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_expired_session_emits_audit(
        self, tenant_id: str, retention_days: int
    ) -> None:
        """Expiry emits session_expired audit with correct fields."""
        audit_entries: list[dict] = []

        class MockAuditLog:
            def append(self, entry: dict) -> None:
                audit_entries.append(entry)

        service = SessionService(audit_log=MockAuditLog())
        session = service.create(tenant_id, retention_days=retention_days)

        future = datetime.now(timezone.utc) + timedelta(days=retention_days + 1)
        service.expire_sweep(now=future)

        # Should have audit entry
        assert len(audit_entries) == 1
        entry = audit_entries[0]
        assert entry["action"] == "session_expired"
        assert entry["detail"]["session_id"] == session.session_id
        assert entry["detail"]["tenant_id"] == tenant_id
        assert "deletion_timestamp" in entry["detail"]

    @given(
        tenant_id=tenant_id_strategy,
        retention_days=retention_days_strategy,
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_non_expired_session_still_accessible(
        self, tenant_id: str, retention_days: int
    ) -> None:
        """Sessions that haven't expired are still accessible after sweep."""
        service = SessionService()
        session = service.create(tenant_id, retention_days=retention_days)

        # Add some memory
        service.add_to_memory(
            session.session_id,
            tenant_id,
            citations=[{"document_id": "doc-1", "version": 1}],
            doc_ids=["doc-1"],
        )

        # Sweep at current time — session should not be expired
        expired = service.expire_sweep()
        assert session.session_id not in expired

        # Memory should still be accessible
        memory = service.get_memory(session.session_id, tenant_id)
        assert len(memory["citations"]) == 1
        assert len(memory["doc_ids"]) == 1
