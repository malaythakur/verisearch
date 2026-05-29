"""Research Agent data models — jobs, plans, events, reports.

Core types:
- ResearchJob: A multi-hop research job with goal, budgets, and state.
- ResearchPlan: A plan with 1-32 typed steps.
- PlanStep: A single step in a research plan.
- ResearchEvent: An event in the job's SSE stream with monotonic event_id.
- ResearchReport: Final or partial report with citations.
- BudgetConfig: Budget limits for a research job.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class JobState(str, Enum):
    """Research job lifecycle states."""

    QUEUED = "queued"
    PLANNING = "planning"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BUDGET_EXCEEDED = "budget_exceeded"


class StepType(str, Enum):
    """Typed labels for research plan steps (R7.2)."""

    SUB_QUERY = "sub_query"
    RETRIEVAL = "retrieval"
    READ = "read"
    SYNTHESIS = "synthesis"


class EventType(str, Enum):
    """Event types for the research job SSE stream (R7.3)."""

    PLAN_UPDATED = "plan_updated"
    STEP_STARTED = "step_started"
    STEP_COMPLETED = "step_completed"
    CITATION = "citation"
    REPORT_CHUNK = "report_chunk"
    DONE = "done"
    ERROR = "error"


@dataclass
class BudgetConfig:
    """Budget limits for a research job (R7.6).

    Defaults are applied from tenant-level configuration when omitted.
    """

    max_steps: int = 32
    max_duration_ms: int = 300_000  # 5 minutes default
    max_tool_calls: int = 100


@dataclass
class PlanStep:
    """A single step in a research plan (R7.2)."""

    step_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: StepType = StepType.RETRIEVAL
    description: str = ""
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchPlan:
    """A research plan with 1-32 steps (R7.2).

    The plan is emitted as the first event (plan_updated) before any retrieval.
    """

    job_id: str = ""
    steps: list[PlanStep] = field(default_factory=list)
    emitted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def validate(self) -> bool:
        """Validate plan has 1-32 steps."""
        return 1 <= len(self.steps) <= 32


@dataclass
class ResearchEvent:
    """An event in the research job's SSE stream (R7.3).

    event_id is strictly monotonically increasing per job.
    """

    event_id: int
    job_id: str
    type: EventType
    payload: dict[str, Any] = field(default_factory=dict)
    emitted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ResearchCitation:
    """A citation in a research report linking a claim to a source document."""

    citation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str = ""
    version: int = 1
    answer_start: int = 0
    answer_end: int = 0
    source_start: int = 0
    source_end: int = 0


@dataclass
class ResearchReport:
    """Final or partial research report (R7.4, R7.6).

    Contains the report text, structured payload (if output_schema was provided),
    and all citations gathered during the job.
    """

    job_id: str = ""
    text: str = ""
    structured_payload: dict[str, Any] | None = None
    citations: list[ResearchCitation] = field(default_factory=list)
    is_partial: bool = False


@dataclass
class ResearchJob:
    """A multi-hop research job (R7.1).

    Tracks the full lifecycle from creation through planning, execution,
    and completion (or budget exceedance).
    """

    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str = ""
    session_id: str | None = None
    research_goal: str = ""
    output_schema: dict[str, Any] | None = None
    budgets: BudgetConfig = field(default_factory=BudgetConfig)
    state: JobState = JobState.QUEUED
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    plan: ResearchPlan | None = None
    report: ResearchReport | None = None

    # Execution tracking
    steps_executed: int = 0
    tool_calls_made: int = 0
    started_at: datetime | None = None
