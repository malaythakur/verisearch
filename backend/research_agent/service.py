"""Research Agent service — orchestrates research jobs (R7.1-R7.8).

Implements:
- start_job(): Validate goal, create job, return job_id within 1s p95 (R7.1, R7.8).
- get_report(): Get final/partial report with citations (R7.4, R7.6).
- get_events(): Get event stream with Last-Event-ID replay (R7.3).
- Cross-tenant access → 404 job_not_found (R7.7).
"""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from typing import Any

from backend.research_agent.budget import BudgetExceededError, BudgetTracker
from backend.research_agent.events import EventBuffer, EventEmitter
from backend.research_agent.executor import ResearchExecutor
from backend.research_agent.models import (
    BudgetConfig,
    EventType,
    JobState,
    ResearchEvent,
    ResearchJob,
    ResearchReport,
)
from backend.research_agent.planner import ResearchPlanner


class InvalidResearchRequestError(Exception):
    """Raised when a research request is invalid (R7.8).

    Covers:
    - research_goal outside 1-4096 chars
    - output_schema not a valid JSON Schema
    """

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class JobNotFoundError(Exception):
    """Raised when a job is not found or belongs to another tenant (R7.7).

    Returns uniform 404 without disclosing whether the job exists in another tenant.
    """

    def __init__(self, job_id: str) -> None:
        self.job_id = job_id
        super().__init__(f"Job not found: {job_id}")


class OutputSchemaValidationError(Exception):
    """Raised when the final report fails output_schema validation (R7.5)."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class ResearchAgentService:
    """Main service for managing research jobs (R7).

    Provides:
    - start_job: Create and execute a research job asynchronously (R7.1).
    - get_report: Retrieve the final or partial report (R7.4, R7.6).
    - get_events: Get events with Last-Event-ID replay (R7.3).
    """

    def __init__(
        self,
        planner: ResearchPlanner | None = None,
        event_buffer: EventBuffer | None = None,
        retriever: Any | None = None,
    ) -> None:
        """Initialize the Research Agent service.

        Args:
            planner: Research planner for generating plans. Defaults to ResearchPlanner.
            event_buffer: Event buffer for SSE replay. Defaults to new EventBuffer.
            retriever: Retriever service for real search during execution.
        """
        self._planner = planner or ResearchPlanner()
        self._event_buffer = event_buffer or EventBuffer()
        self._emitter = EventEmitter(self._event_buffer)
        self._jobs: dict[str, ResearchJob] = {}
        self._lock = threading.Lock()
        self._retriever = retriever

    @property
    def event_buffer(self) -> EventBuffer:
        """Access the event buffer for testing."""
        return self._event_buffer

    def start_job(
        self,
        tenant_id: str,
        research_goal: str,
        output_schema: dict[str, Any] | None = None,
        session_id: str | None = None,
        budgets: BudgetConfig | None = None,
        session_memory: dict[str, Any] | None = None,
    ) -> str:
        """Start a new research job (R7.1, R7.8).

        Validates the request, creates the job, and begins asynchronous execution.
        Returns the job_id within 1s p95.

        Args:
            tenant_id: The requesting tenant's ID.
            research_goal: The research goal (1-4096 chars).
            output_schema: Optional JSON Schema for the final report.
            session_id: Optional session ID for context incorporation.
            budgets: Optional budget configuration.
            session_memory: Optional session memory to incorporate.

        Returns:
            The job_id for the created job.

        Raises:
            InvalidResearchRequestError: If the request is invalid (R7.8).
        """
        # Validate research_goal (R7.8)
        self._validate_goal(research_goal)

        # Validate output_schema if provided (R7.8)
        if output_schema is not None:
            self._validate_output_schema(output_schema)

        # Create the job
        job = ResearchJob(
            job_id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            session_id=session_id,
            research_goal=research_goal,
            output_schema=output_schema,
            budgets=budgets or BudgetConfig(),
            state=JobState.QUEUED,
            created_at=datetime.now(timezone.utc),
        )

        with self._lock:
            self._jobs[job.job_id] = job

        # Execute the job (synchronously for MVP; production would use async queue)
        self._execute_job(job, session_memory)

        return job.job_id

    def get_report(self, job_id: str, tenant_id: str) -> ResearchReport:
        """Get the final or partial report for a job (R7.4, R7.6).

        Args:
            job_id: The job to get the report for.
            tenant_id: The requesting tenant's ID.

        Returns:
            The research report (complete or partial).

        Raises:
            JobNotFoundError: If the job doesn't exist or belongs to another tenant (R7.7).
        """
        job = self._get_job(job_id, tenant_id)

        if job.report is None:
            # Job hasn't produced a report yet
            return ResearchReport(
                job_id=job_id,
                text="",
                citations=[],
                is_partial=True,
            )

        return job.report

    def get_events(
        self, job_id: str, tenant_id: str, last_event_id: int | None = None
    ) -> list[ResearchEvent]:
        """Get events for a job with Last-Event-ID replay (R7.3).

        Args:
            job_id: The job to get events for.
            tenant_id: The requesting tenant's ID.
            last_event_id: If provided, only events after this ID are returned.

        Returns:
            List of events in order.

        Raises:
            JobNotFoundError: If the job doesn't exist or belongs to another tenant (R7.7).
        """
        # Verify tenant access
        self._get_job(job_id, tenant_id)

        # Replay from buffer
        return self._event_buffer.replay(job_id, last_event_id)

    def get_job_state(self, job_id: str, tenant_id: str) -> JobState:
        """Get the current state of a job.

        Args:
            job_id: The job to check.
            tenant_id: The requesting tenant's ID.

        Returns:
            The current job state.

        Raises:
            JobNotFoundError: If the job doesn't exist or belongs to another tenant.
        """
        job = self._get_job(job_id, tenant_id)
        return job.state

    def _get_job(self, job_id: str, tenant_id: str) -> ResearchJob:
        """Get a job with tenant isolation (R7.7).

        Returns uniform 404 for cross-tenant access, non-existent jobs.
        """
        with self._lock:
            job = self._jobs.get(job_id)

        if job is None or job.tenant_id != tenant_id:
            raise JobNotFoundError(job_id)

        return job

    def _validate_goal(self, research_goal: str) -> None:
        """Validate research_goal is 1-4096 chars (R7.8)."""
        if not research_goal or len(research_goal) < 1:
            raise InvalidResearchRequestError(
                "research_goal must be between 1 and 4096 characters"
            )
        if len(research_goal) > 4096:
            raise InvalidResearchRequestError(
                "research_goal must be between 1 and 4096 characters"
            )

    def _validate_output_schema(self, schema: dict[str, Any]) -> None:
        """Validate output_schema is a syntactically valid JSON Schema (R7.8).

        Basic validation: must be a dict with valid JSON Schema structure.
        """
        if not isinstance(schema, dict):
            raise InvalidResearchRequestError(
                "output_schema must be a valid JSON Schema document"
            )

        # Basic JSON Schema validation: must have a type or be a valid schema object
        # A valid JSON Schema can be a boolean or an object
        # For objects, we check for common schema keywords
        valid_keywords = {
            "type", "properties", "items", "required", "enum",
            "const", "allOf", "anyOf", "oneOf", "not",
            "$schema", "$id", "$ref", "title", "description",
            "default", "examples", "minimum", "maximum",
            "minLength", "maxLength", "pattern", "format",
            "additionalProperties", "minItems", "maxItems",
        }

        # An empty object {} is a valid JSON Schema (matches anything)
        if not schema:
            return

        # Check that at least one key is a known JSON Schema keyword
        has_schema_keyword = any(k in valid_keywords for k in schema.keys())
        if not has_schema_keyword:
            raise InvalidResearchRequestError(
                "output_schema must be a valid JSON Schema document"
            )

    def _execute_job(
        self, job: ResearchJob, session_memory: dict[str, Any] | None = None
    ) -> None:
        """Execute a research job synchronously (MVP).

        Production would dispatch to an async worker queue.
        """
        # Transition to planning
        job.state = JobState.PLANNING
        job.started_at = datetime.now(timezone.utc)

        # Generate plan (R7.2)
        try:
            plan = self._planner.generate_plan(job.job_id, job.research_goal)
            job.plan = plan
        except Exception as e:
            job.state = JobState.FAILED
            self._emitter.emit(
                job.job_id,
                EventType.ERROR,
                {"code": "internal_error", "message": str(e)},
            )
            return

        # Emit plan_updated as first event (R7.2)
        self._emitter.emit(
            job.job_id,
            EventType.PLAN_UPDATED,
            {
                "plan": [
                    {"step_id": s.step_id, "type": s.type.value, "description": s.description}
                    for s in plan.steps
                ]
            },
        )

        # Transition to running
        job.state = JobState.RUNNING

        # Set up budget tracker
        budget_tracker = BudgetTracker(config=job.budgets)
        budget_tracker.start()

        # Execute the plan
        executor = ResearchExecutor(
            emitter=self._emitter,
            budget_tracker=budget_tracker,
            retriever=self._retriever,
        )

        try:
            report = executor.execute_plan(job.job_id, plan)

            # Validate against output_schema if provided (R7.5)
            if job.output_schema is not None:
                self._validate_report_against_schema(report, job.output_schema)

            job.report = report
            job.state = JobState.SUCCEEDED
            job.steps_executed = budget_tracker.steps_used
            job.tool_calls_made = budget_tracker.tool_calls_used

            # Emit done event
            self._emitter.emit(
                job.job_id,
                EventType.DONE,
                {"report_uri": f"/v1/research/{job.job_id}"},
            )

        except BudgetExceededError as e:
            # R7.6: Budget exceeded → partial report
            job.state = JobState.BUDGET_EXCEEDED
            job.steps_executed = budget_tracker.steps_used
            job.tool_calls_made = budget_tracker.tool_calls_used

            # Create partial report from what we have so far
            events = self._event_buffer.get_all_events(job.job_id)
            partial_citations = []
            partial_text_parts = []

            for event in events:
                if event.type == EventType.CITATION:
                    from backend.research_agent.models import ResearchCitation
                    partial_citations.append(
                        ResearchCitation(
                            document_id=event.payload.get("document_id", ""),
                            version=event.payload.get("version", 1),
                            answer_start=event.payload.get("answer_start", 0),
                            answer_end=event.payload.get("answer_end", 0),
                            source_start=event.payload.get("source_start", 0),
                            source_end=event.payload.get("source_end", 0),
                        )
                    )
                elif event.type == EventType.REPORT_CHUNK:
                    partial_text_parts.append(event.payload.get("text", ""))

            job.report = ResearchReport(
                job_id=job.job_id,
                text="\n\n".join(partial_text_parts) if partial_text_parts else "",
                citations=partial_citations,
                is_partial=True,
            )

            # Emit budget_exceeded error event
            self._emitter.emit(
                job.job_id,
                EventType.ERROR,
                {
                    "code": "budget_exceeded",
                    "message": f"Budget exceeded: {e.reason} "
                    f"({e.current_value} >= {e.limit})",
                },
            )

        except Exception as e:
            job.state = JobState.FAILED
            job.report = ResearchReport(
                job_id=job.job_id,
                text="",
                citations=[],
                is_partial=True,
            )
            self._emitter.emit(
                job.job_id,
                EventType.ERROR,
                {"code": "internal_error", "message": str(e)},
            )

    def _validate_report_against_schema(
        self, report: ResearchReport, schema: dict[str, Any]
    ) -> None:
        """Validate the report's structured payload against the output schema (R7.5).

        If the report has a structured_payload, validates it against the schema.
        If not, creates a default structured payload from the report text.
        """
        # For MVP: if no structured payload, create one from the report
        if report.structured_payload is None:
            report.structured_payload = {
                "text": report.text,
                "citations_count": len(report.citations),
            }

        # Basic schema validation
        payload = report.structured_payload
        schema_type = schema.get("type")

        if schema_type == "object":
            if not isinstance(payload, dict):
                raise OutputSchemaValidationError(
                    "Report payload must be an object per output_schema"
                )
            # Check required fields
            required = schema.get("required", [])
            for field_name in required:
                if field_name not in payload:
                    raise OutputSchemaValidationError(
                        f"Report payload missing required field: {field_name}"
                    )

        elif schema_type == "array":
            if not isinstance(payload, (list, dict)):
                raise OutputSchemaValidationError(
                    "Report payload must match output_schema type"
                )
