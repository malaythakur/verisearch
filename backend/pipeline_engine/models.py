"""Pipeline Engine data models (Task 12, R9).

Defines types for pipeline definitions, steps, execution results, and warnings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class StepType(str, Enum):
    """Pipeline step types (R9.4)."""

    FILTER = "filter"
    RERANKER = "reranker"
    TRANSFORM = "transform"


@dataclass(frozen=True, slots=True)
class PipelineStep:
    """A single step in a pipeline definition (R9.1).

    Attributes:
        name: Registry name of the step (must exist in registry).
        type: Step type (filter, reranker, transform).
        config: Optional configuration for the step.
        timeout_ms: Per-step timeout in [100, 30000]ms, default 2000 (R9.6).
    """

    name: str
    type: StepType
    config: dict[str, Any] = field(default_factory=dict)
    timeout_ms: int = 2000

    def __post_init__(self) -> None:
        """Validate timeout_ms is in [100, 30000]."""
        if not (100 <= self.timeout_ms <= 30000):
            object.__setattr__(self, "timeout_ms", max(100, min(30000, self.timeout_ms)))


@dataclass
class PipelineDefinition:
    """A complete pipeline definition (R9.1).

    Attributes:
        pipeline_id: Unique identifier for the pipeline.
        tenant_id: Owning tenant's ID.
        name: Human-readable pipeline name.
        steps: Ordered list of pipeline steps (1-20).
        created_at: Creation timestamp.
    """

    pipeline_id: str
    tenant_id: str
    name: str
    steps: list[PipelineStep]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True, slots=True)
class StepWarning:
    """A warning produced during pipeline execution (R9.6).

    Attributes:
        code: Warning code (e.g., 'step_timeout').
        step: Name of the step that produced the warning.
        message: Human-readable description.
    """

    code: str
    step: str
    message: str = ""


@dataclass
class ExecutionResult:
    """Result of executing a pipeline (R9.3).

    Attributes:
        results: The final ranked result list after all steps.
        warnings: Any warnings produced during execution.
    """

    results: list[Any]
    warnings: list[StepWarning] = field(default_factory=list)
