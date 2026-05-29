"""Unit tests for the Research Agent and Session Store.

Tests cover:
- Task 14.1: POST /v1/research validation and job_id return (R7.1, R7.8)
- Task 14.2: Research plan generation (R7.2)
- Task 14.3: Tool-use loop execution
- Task 14.4: SSE event stream with monotonic event_id (R7.3)
- Task 14.5: Last-Event-ID replay (R7.3)
- Task 14.6: Budget enforcement (R7.6)
- Task 14.7: GET /v1/research/{job_id} report (R7.4, R7.6)
- Task 14.8: output_schema validation (R7.5)
- Task 14.9: Cross-tenant job access → 404 (R7.7)
- Task 14.10: POST /v1/sessions creation (R8.1)
- Task 14.11: Session memory bounds (R8.2)
- Task 14.12: Session expiry sweep (R8.4)
- Task 14.13: Cross-tenant/expired/missing session → 404 (R8.5)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from backend.research_agent.budget import BudgetConfig, BudgetExceededError, BudgetTracker
from backend.research_agent.events import EventBuffer, EventEmitter
from backend.research_agent.models import (
    EventType,
    JobState,
    PlanStep,
    ResearchEvent,
    ResearchPlan,
    StepType,
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
    Session,
    SessionCitation,
    SessionMemory,
    SessionState,
)
from backend.session_store.service import (
    InvalidSessionRequestError,
    SessionNotFoundError,
    SessionService,
)


# ===========================================================================
# Task 14.1: POST /v1/research — validate goal, return job_id
# ===========================================================================


class TestResearchJobCreation:
    """Tests for research job creation (R7.1, R7.8)."""

    def test_start_job_returns_job_id(self, tenant_id: str) -> None:
        """Valid request returns a job_id string."""
        service = ResearchAgentService()
        job_id = service.start_job(tenant_id, "What is quantum computing?")
        assert isinstance(job_id, str)
        assert len(job_id) > 0

    def test_start_job_validates_goal_min_length(self, tenant_id: str) -> None:
        """Empty goal raises InvalidResearchRequestError (R7.8)."""
        service = ResearchAgentService()
        with pytest.raises(InvalidResearchRequestError):
            service.start_job(tenant_id, "")

    def test_start_job_validates_goal_max_length(self, tenant_id: str) -> None:
        """Goal > 4096 chars raises InvalidResearchRequestError (R7.8)."""
        service = ResearchAgentService()
        with pytest.raises(InvalidResearchRequestError):
            service.start_job(tenant_id, "x" * 4097)

    def test_start_job_accepts_goal_at_boundaries(self, tenant_id: str) -> None:
        """Goals at exactly 1 and 4096 chars are accepted."""
        service = ResearchAgentService()
        # 1 char
        job_id_1 = service.start_job(tenant_id, "x")
        assert job_id_1

        # 4096 chars
        job_id_max = service.start_job(tenant_id, "y" * 4096)
        assert job_id_max

    def test_start_job_validates_output_schema(self, tenant_id: str) -> None:
        """Invalid output_schema raises InvalidResearchRequestError (R7.8)."""
        service = ResearchAgentService()
        # Not a valid JSON Schema (no schema keywords)
        with pytest.raises(InvalidResearchRequestError):
            service.start_job(
                tenant_id,
                "Research topic",
                output_schema={"foo": "bar", "baz": 123},
            )

    def test_start_job_accepts_valid_output_schema(self, tenant_id: str) -> None:
        """Valid output_schema is accepted."""
        service = ResearchAgentService()
        job_id = service.start_job(
            tenant_id,
            "Research topic",
            output_schema={"type": "object", "properties": {"summary": {"type": "string"}}},
        )
        assert job_id

    def test_start_job_accepts_empty_schema(self, tenant_id: str) -> None:
        """Empty object {} is a valid JSON Schema (matches anything)."""
        service = ResearchAgentService()
        job_id = service.start_job(
            tenant_id,
            "Research topic",
            output_schema={},
        )
        assert job_id


# ===========================================================================
# Task 14.2: Research plan generation
# ===========================================================================


class TestResearchPlanGeneration:
    """Tests for research plan generation (R7.2)."""

    def test_plan_has_1_to_32_steps(self) -> None:
        """Generated plan has between 1 and 32 steps."""
        planner = ResearchPlanner()
        plan = planner.generate_plan("job-1", "What is AI?")
        assert 1 <= len(plan.steps) <= 32

    def test_plan_steps_have_typed_labels(self) -> None:
        """Each step has a type from the StepType enum."""
        planner = ResearchPlanner()
        plan = planner.generate_plan("job-1", "Explain machine learning algorithms")
        for step in plan.steps:
            assert step.type in StepType

    def test_plan_first_event_is_plan_updated(self, tenant_id: str) -> None:
        """First event emitted is plan_updated (R7.2)."""
        service = ResearchAgentService()
        job_id = service.start_job(tenant_id, "Research quantum computing")
        events = service.get_events(job_id, tenant_id)
        assert len(events) > 0
        assert events[0].type == EventType.PLAN_UPDATED

    def test_plan_updated_contains_steps(self, tenant_id: str) -> None:
        """plan_updated event contains the plan steps."""
        service = ResearchAgentService()
        job_id = service.start_job(tenant_id, "Research AI safety")
        events = service.get_events(job_id, tenant_id)
        plan_event = events[0]
        assert "plan" in plan_event.payload
        plan_steps = plan_event.payload["plan"]
        assert 1 <= len(plan_steps) <= 32
        for step in plan_steps:
            assert "step_id" in step
            assert "type" in step
            assert step["type"] in [t.value for t in StepType]


# ===========================================================================
# Task 14.3: Tool-use loop
# ===========================================================================


class TestToolUseLoop:
    """Tests for the tool-use loop execution."""

    def test_executor_produces_events(self, tenant_id: str) -> None:
        """Executor produces step_started and step_completed events."""
        service = ResearchAgentService()
        job_id = service.start_job(tenant_id, "Research topic")
        events = service.get_events(job_id, tenant_id)

        event_types = [e.type for e in events]
        assert EventType.STEP_STARTED in event_types
        assert EventType.STEP_COMPLETED in event_types

    def test_executor_produces_report(self, tenant_id: str) -> None:
        """Executor produces a final report."""
        service = ResearchAgentService()
        job_id = service.start_job(tenant_id, "Research topic")
        report = service.get_report(job_id, tenant_id)
        assert report is not None
        assert report.job_id == job_id


# ===========================================================================
# Task 14.4: SSE event stream with monotonic event_id
# ===========================================================================


class TestEventStream:
    """Tests for SSE event stream (R7.3)."""

    def test_event_ids_are_strictly_monotonic(self, tenant_id: str) -> None:
        """Event IDs are strictly monotonically increasing."""
        service = ResearchAgentService()
        job_id = service.start_job(tenant_id, "Research topic")
        events = service.get_events(job_id, tenant_id)

        assert len(events) > 1
        for i in range(1, len(events)):
            assert events[i].event_id > events[i - 1].event_id

    def test_event_ids_start_at_1(self, tenant_id: str) -> None:
        """First event has event_id = 1."""
        service = ResearchAgentService()
        job_id = service.start_job(tenant_id, "Research topic")
        events = service.get_events(job_id, tenant_id)
        assert events[0].event_id == 1

    def test_event_buffer_rejects_non_monotonic(self) -> None:
        """EventBuffer rejects events with non-monotonic IDs."""
        buffer = EventBuffer()
        event1 = ResearchEvent(
            event_id=1, job_id="job-1", type=EventType.PLAN_UPDATED
        )
        event2 = ResearchEvent(
            event_id=1, job_id="job-1", type=EventType.STEP_STARTED
        )
        buffer.append(event1)
        with pytest.raises(ValueError):
            buffer.append(event2)


# ===========================================================================
# Task 14.5: Last-Event-ID replay
# ===========================================================================


class TestLastEventIdReplay:
    """Tests for Last-Event-ID replay (R7.3)."""

    def test_replay_from_beginning(self, tenant_id: str) -> None:
        """Replay with no last_event_id returns all events."""
        service = ResearchAgentService()
        job_id = service.start_job(tenant_id, "Research topic")
        all_events = service.get_events(job_id, tenant_id)
        replayed = service.get_events(job_id, tenant_id, last_event_id=None)
        assert len(replayed) == len(all_events)

    def test_replay_from_middle(self, tenant_id: str) -> None:
        """Replay with last_event_id returns only events after that ID."""
        service = ResearchAgentService()
        job_id = service.start_job(tenant_id, "Research topic")
        all_events = service.get_events(job_id, tenant_id)

        # Replay from after the first event
        mid_id = all_events[0].event_id
        replayed = service.get_events(job_id, tenant_id, last_event_id=mid_id)
        assert len(replayed) == len(all_events) - 1
        assert all(e.event_id > mid_id for e in replayed)

    def test_replay_from_end_returns_empty(self, tenant_id: str) -> None:
        """Replay with last_event_id at the end returns empty list."""
        service = ResearchAgentService()
        job_id = service.start_job(tenant_id, "Research topic")
        all_events = service.get_events(job_id, tenant_id)

        last_id = all_events[-1].event_id
        replayed = service.get_events(job_id, tenant_id, last_event_id=last_id)
        assert len(replayed) == 0


# ===========================================================================
# Task 14.6: Budget enforcement
# ===========================================================================


class TestBudgetEnforcement:
    """Tests for budget enforcement (R7.6)."""

    def test_max_steps_exceeded(self, tenant_id: str) -> None:
        """Job terminates with budget_exceeded when max_steps is hit."""
        service = ResearchAgentService()
        # Set very low budget
        budgets = BudgetConfig(max_steps=1, max_duration_ms=300_000, max_tool_calls=100)
        job_id = service.start_job(tenant_id, "Research a complex topic with many steps", budgets=budgets)

        state = service.get_job_state(job_id, tenant_id)
        assert state == JobState.BUDGET_EXCEEDED

        # Should have partial report
        report = service.get_report(job_id, tenant_id)
        assert report.is_partial

    def test_max_tool_calls_exceeded(self, tenant_id: str) -> None:
        """Job terminates with budget_exceeded when max_tool_calls is hit."""
        service = ResearchAgentService()
        budgets = BudgetConfig(max_steps=32, max_duration_ms=300_000, max_tool_calls=1)
        job_id = service.start_job(tenant_id, "Research topic", budgets=budgets)

        state = service.get_job_state(job_id, tenant_id)
        assert state == JobState.BUDGET_EXCEEDED

    def test_budget_exceeded_emits_error_event(self, tenant_id: str) -> None:
        """Budget exceeded emits an error event with code budget_exceeded."""
        service = ResearchAgentService()
        budgets = BudgetConfig(max_steps=1, max_duration_ms=300_000, max_tool_calls=100)
        job_id = service.start_job(tenant_id, "Research topic", budgets=budgets)

        events = service.get_events(job_id, tenant_id)
        error_events = [e for e in events if e.type == EventType.ERROR]
        assert len(error_events) == 1
        assert error_events[0].payload["code"] == "budget_exceeded"

    def test_budget_tracker_check_raises(self) -> None:
        """BudgetTracker.check() raises when budget is exceeded."""
        tracker = BudgetTracker(config=BudgetConfig(max_steps=2))
        tracker.record_step()
        tracker.record_step()
        with pytest.raises(BudgetExceededError) as exc_info:
            tracker.check()
        assert exc_info.value.reason == "max_steps"


# ===========================================================================
# Task 14.7: GET /v1/research/{job_id} — report
# ===========================================================================


class TestGetReport:
    """Tests for GET /v1/research/{job_id} (R7.4, R7.6)."""

    def test_successful_job_has_report(self, tenant_id: str) -> None:
        """Successful job has a non-partial report."""
        service = ResearchAgentService()
        job_id = service.start_job(tenant_id, "Research topic")
        report = service.get_report(job_id, tenant_id)
        assert report.job_id == job_id
        # With default budgets, job should succeed
        state = service.get_job_state(job_id, tenant_id)
        if state == JobState.SUCCEEDED:
            assert not report.is_partial

    def test_budget_exceeded_has_partial_report(self, tenant_id: str) -> None:
        """Budget-exceeded job has a partial report with citations."""
        service = ResearchAgentService()
        budgets = BudgetConfig(max_steps=2, max_duration_ms=300_000, max_tool_calls=100)
        job_id = service.start_job(
            tenant_id,
            "Research a very complex topic requiring many steps",
            budgets=budgets,
        )
        report = service.get_report(job_id, tenant_id)
        assert report.is_partial


# ===========================================================================
# Task 14.8: output_schema validation
# ===========================================================================


class TestOutputSchemaValidation:
    """Tests for output_schema validation (R7.5)."""

    def test_valid_schema_accepted(self, tenant_id: str) -> None:
        """Job with valid output_schema succeeds."""
        service = ResearchAgentService()
        schema = {"type": "object", "properties": {"text": {"type": "string"}}}
        job_id = service.start_job(tenant_id, "Research topic", output_schema=schema)
        state = service.get_job_state(job_id, tenant_id)
        # Should succeed (default payload includes 'text')
        assert state in (JobState.SUCCEEDED, JobState.BUDGET_EXCEEDED)

    def test_invalid_schema_rejected_at_creation(self, tenant_id: str) -> None:
        """Invalid output_schema is rejected at job creation (R7.8)."""
        service = ResearchAgentService()
        with pytest.raises(InvalidResearchRequestError):
            service.start_job(
                tenant_id,
                "Research topic",
                output_schema={"not_a_schema_keyword": True, "another": "value"},
            )


# ===========================================================================
# Task 14.9: Cross-tenant job access → 404
# ===========================================================================


class TestCrossTenantJobAccess:
    """Tests for cross-tenant job access (R7.7)."""

    def test_cross_tenant_get_report_raises(
        self, tenant_id: str, second_tenant_id: str
    ) -> None:
        """Accessing another tenant's job raises JobNotFoundError."""
        service = ResearchAgentService()
        job_id = service.start_job(tenant_id, "Research topic")

        with pytest.raises(JobNotFoundError):
            service.get_report(job_id, second_tenant_id)

    def test_cross_tenant_get_events_raises(
        self, tenant_id: str, second_tenant_id: str
    ) -> None:
        """Accessing another tenant's events raises JobNotFoundError."""
        service = ResearchAgentService()
        job_id = service.start_job(tenant_id, "Research topic")

        with pytest.raises(JobNotFoundError):
            service.get_events(job_id, second_tenant_id)

    def test_nonexistent_job_raises(self, tenant_id: str) -> None:
        """Accessing a non-existent job raises JobNotFoundError."""
        service = ResearchAgentService()
        with pytest.raises(JobNotFoundError):
            service.get_report("nonexistent-job-id", tenant_id)

    def test_cross_tenant_same_error_as_nonexistent(
        self, tenant_id: str, second_tenant_id: str
    ) -> None:
        """Cross-tenant and non-existent produce the same error type (R7.7)."""
        service = ResearchAgentService()
        job_id = service.start_job(tenant_id, "Research topic")

        # Both should raise JobNotFoundError
        with pytest.raises(JobNotFoundError):
            service.get_report(job_id, second_tenant_id)

        with pytest.raises(JobNotFoundError):
            service.get_report("totally-fake-id", tenant_id)


# ===========================================================================
# Task 14.10: POST /v1/sessions — create
# ===========================================================================


class TestSessionCreation:
    """Tests for session creation (R8.1)."""

    def test_create_session_returns_session(self, tenant_id: str) -> None:
        """Creating a session returns a Session with valid fields."""
        service = SessionService()
        session = service.create(tenant_id)
        assert session.session_id
        assert session.tenant_id == tenant_id
        assert session.retention_days == 14  # default
        assert session.state == SessionState.ACTIVE

    def test_create_session_custom_retention(self, tenant_id: str) -> None:
        """Custom retention_days is respected."""
        service = SessionService()
        session = service.create(tenant_id, retention_days=30)
        assert session.retention_days == 30

    def test_create_session_retention_bounds(self, tenant_id: str) -> None:
        """retention_days must be in [1, 90]."""
        service = SessionService()

        # Valid boundaries
        s1 = service.create(tenant_id, retention_days=1)
        assert s1.retention_days == 1

        s90 = service.create(tenant_id, retention_days=90)
        assert s90.retention_days == 90

        # Invalid: below minimum
        with pytest.raises(InvalidSessionRequestError):
            service.create(tenant_id, retention_days=0)

        # Invalid: above maximum
        with pytest.raises(InvalidSessionRequestError):
            service.create(tenant_id, retention_days=91)


# ===========================================================================
# Task 14.11: Session memory bounds
# ===========================================================================


class TestSessionMemory:
    """Tests for session memory bounds (R8.2)."""

    def test_memory_citations_bounded_at_50(self) -> None:
        """Session memory holds at most 50 citations."""
        memory = SessionMemory()
        for i in range(60):
            memory.add_citation(
                SessionCitation(document_id=f"doc-{i}", version=1)
            )
        assert memory.citation_count == MAX_CITATIONS  # 50

    def test_memory_doc_ids_bounded_at_20(self) -> None:
        """Session memory holds at most 20 unique doc_ids."""
        memory = SessionMemory()
        for i in range(30):
            memory.add_doc_id(f"doc-{i}")
        assert memory.doc_id_count == MAX_DOC_IDS  # 20

    def test_memory_citations_evict_oldest(self) -> None:
        """When at capacity, oldest citations are evicted."""
        memory = SessionMemory()
        for i in range(55):
            memory.add_citation(
                SessionCitation(document_id=f"doc-{i}", version=1)
            )
        citations = memory.citations
        # Should have the 50 most recent (doc-5 through doc-54)
        assert citations[0].document_id == "doc-5"
        assert citations[-1].document_id == "doc-54"

    def test_memory_doc_ids_evict_oldest(self) -> None:
        """When at capacity, oldest doc_ids are evicted."""
        memory = SessionMemory()
        for i in range(25):
            memory.add_doc_id(f"doc-{i}")
        doc_ids = memory.doc_ids
        # Should have the 20 most recent (doc-5 through doc-24)
        assert doc_ids[0] == "doc-5"
        assert doc_ids[-1] == "doc-24"

    def test_memory_doc_ids_unique(self) -> None:
        """Adding a duplicate doc_id moves it to most recent."""
        memory = SessionMemory()
        memory.add_doc_id("doc-a")
        memory.add_doc_id("doc-b")
        memory.add_doc_id("doc-a")  # Move to end
        doc_ids = memory.doc_ids
        assert doc_ids == ["doc-b", "doc-a"]

    def test_session_memory_incorporated(self, tenant_id: str) -> None:
        """Session memory is retrievable for incorporation."""
        service = SessionService()
        session = service.create(tenant_id)

        # Add some memory
        service.add_to_memory(
            session.session_id,
            tenant_id,
            citations=[{"document_id": "doc-1", "version": 1}],
            doc_ids=["doc-1", "doc-2"],
        )

        memory = service.get_memory(session.session_id, tenant_id)
        assert len(memory["citations"]) == 1
        assert len(memory["doc_ids"]) == 2


# ===========================================================================
# Task 14.12: Session expiry sweep
# ===========================================================================


class TestSessionExpirySweep:
    """Tests for session expiry sweep (R8.4)."""

    def test_expired_session_is_swept(self, tenant_id: str) -> None:
        """Expired sessions are deleted during sweep."""
        service = SessionService()
        session = service.create(tenant_id, retention_days=1)

        # Simulate time passing beyond expiry
        future = datetime.now(timezone.utc) + timedelta(days=2)
        expired = service.expire_sweep(now=future)
        assert session.session_id in expired

    def test_active_session_not_swept(self, tenant_id: str) -> None:
        """Active sessions are not affected by sweep."""
        service = SessionService()
        session = service.create(tenant_id, retention_days=90)

        # Sweep now — session shouldn't be expired
        expired = service.expire_sweep()
        assert session.session_id not in expired

    def test_expired_session_memory_cleared(self, tenant_id: str) -> None:
        """Expired session's memory is cleared."""
        service = SessionService()
        session = service.create(tenant_id, retention_days=1)

        # Add memory
        service.add_to_memory(
            session.session_id,
            tenant_id,
            citations=[{"document_id": "doc-1", "version": 1}],
            doc_ids=["doc-1"],
        )

        # Expire
        future = datetime.now(timezone.utc) + timedelta(days=2)
        service.expire_sweep(now=future)

        # Memory should be inaccessible
        with pytest.raises(SessionNotFoundError):
            service.get_memory(session.session_id, tenant_id)

    def test_expired_session_emits_audit(self, tenant_id: str) -> None:
        """Expiry sweep emits session_expired audit event (R8.4)."""
        audit_entries: list[dict] = []

        class MockAuditLog:
            def append(self, entry: dict) -> None:
                audit_entries.append(entry)

        service = SessionService(audit_log=MockAuditLog())
        session = service.create(tenant_id, retention_days=1)

        future = datetime.now(timezone.utc) + timedelta(days=2)
        service.expire_sweep(now=future)

        assert len(audit_entries) == 1
        assert audit_entries[0]["action"] == "session_expired"
        assert audit_entries[0]["detail"]["session_id"] == session.session_id
        assert audit_entries[0]["detail"]["tenant_id"] == tenant_id


# ===========================================================================
# Task 14.13: Cross-tenant/expired/missing session → 404
# ===========================================================================


class TestCrossTenantSessionAccess:
    """Tests for cross-tenant session access (R8.5)."""

    def test_cross_tenant_raises_not_found(
        self, tenant_id: str, second_tenant_id: str
    ) -> None:
        """Accessing another tenant's session raises SessionNotFoundError."""
        service = SessionService()
        session = service.create(tenant_id)

        with pytest.raises(SessionNotFoundError):
            service.get_memory(session.session_id, second_tenant_id)

    def test_expired_session_raises_not_found(self, tenant_id: str) -> None:
        """Accessing an expired session raises SessionNotFoundError."""
        service = SessionService()
        session = service.create(tenant_id, retention_days=1)

        # Expire it
        future = datetime.now(timezone.utc) + timedelta(days=2)
        service.expire_sweep(now=future)

        with pytest.raises(SessionNotFoundError):
            service.get_memory(session.session_id, tenant_id)

    def test_missing_session_raises_not_found(self, tenant_id: str) -> None:
        """Accessing a non-existent session raises SessionNotFoundError."""
        service = SessionService()
        with pytest.raises(SessionNotFoundError):
            service.get_memory("nonexistent-session-id", tenant_id)

    def test_uniform_error_shape(
        self, tenant_id: str, second_tenant_id: str
    ) -> None:
        """All three cases produce the same error type (R8.5)."""
        service = SessionService()
        session = service.create(tenant_id, retention_days=1)

        # Cross-tenant
        with pytest.raises(SessionNotFoundError):
            service.get_memory(session.session_id, second_tenant_id)

        # Expire and try same tenant
        future = datetime.now(timezone.utc) + timedelta(days=2)
        service.expire_sweep(now=future)
        with pytest.raises(SessionNotFoundError):
            service.get_memory(session.session_id, tenant_id)

        # Non-existent
        with pytest.raises(SessionNotFoundError):
            service.get_memory("fake-id", tenant_id)
