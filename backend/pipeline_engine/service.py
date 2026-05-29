"""Pipeline service (Tasks 12.2, 12.3, 12.7, R9.1, R9.2, R9.7).

Provides CRUD operations for pipelines with tenant scoping:
- Create pipeline with validation (R9.1, R9.2)
- Get pipeline by ID with tenant isolation (R9.7)
- Execute pipeline (R9.3, R9.4, R9.5, R9.6)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from backend.pipeline_engine.executor import execute_pipeline
from backend.pipeline_engine.models import (
    ExecutionResult,
    PipelineDefinition,
    PipelineStep,
    StepType,
)
from backend.pipeline_engine.registry import PipelineRegistry, create_default_registry


class PipelineNotFoundError(Exception):
    """Raised when a pipeline is not found or belongs to another tenant (R9.7)."""

    pass


class UnknownPipelineStepError(Exception):
    """Raised when a pipeline definition references unknown step names (R9.2).

    Attributes:
        unknown_steps: List of step names not found in the registry.
    """

    def __init__(self, unknown_steps: list[str]) -> None:
        self.unknown_steps = unknown_steps
        super().__init__(
            f"Unknown pipeline steps: {', '.join(unknown_steps)}"
        )


class InvalidPipelineError(Exception):
    """Raised when a pipeline definition is invalid (e.g., wrong step count)."""

    pass


class PipelineService:
    """Service for managing and executing pipelines (R9).

    Provides:
    - create_pipeline: Validate and persist a pipeline (R9.1, R9.2)
    - get_pipeline: Retrieve a pipeline with tenant isolation (R9.7)
    - execute: Run a pipeline against candidates (R9.3, R9.4, R9.6)
    """

    def __init__(self, registry: PipelineRegistry | None = None) -> None:
        """Initialize the pipeline service.

        Args:
            registry: Step registry to use. Defaults to the built-in registry.
        """
        self._registry = registry or create_default_registry()
        self._pipelines: dict[str, PipelineDefinition] = {}

    @property
    def registry(self) -> PipelineRegistry:
        """Get the step registry."""
        return self._registry

    def create_pipeline(
        self,
        tenant_id: str,
        name: str,
        steps: list[dict[str, Any]],
    ) -> PipelineDefinition:
        """Create and persist a new pipeline (R9.1, R9.2).

        Validates:
        - Step count is in [1, 20] (R9.1)
        - All step names exist in the registry (R9.1, R9.2)

        Args:
            tenant_id: The owning tenant's ID.
            name: Human-readable pipeline name.
            steps: List of step definitions, each with 'name' and optional
                   'config' and 'timeout_ms'.

        Returns:
            The created PipelineDefinition with a generated pipeline_id.

        Raises:
            InvalidPipelineError: If step count is outside [1, 20].
            UnknownPipelineStepError: If any step name is not in the registry.
        """
        # Validate step count (R9.1)
        if not steps or len(steps) < 1:
            raise InvalidPipelineError("Pipeline must have at least 1 step.")
        if len(steps) > 20:
            raise InvalidPipelineError("Pipeline must have at most 20 steps.")

        # Extract step names and validate against registry (R9.2)
        step_names = [s.get("name", "") for s in steps]
        unknown = self._registry.get_unknown_steps(step_names)
        if unknown:
            raise UnknownPipelineStepError(unknown)

        # Build PipelineStep objects
        pipeline_steps: list[PipelineStep] = []
        for step_def in steps:
            step_name = step_def["name"]
            entry = self._registry.get(step_name)
            assert entry is not None  # Already validated

            timeout_ms = step_def.get("timeout_ms", 2000)
            # Clamp timeout to valid range
            timeout_ms = max(100, min(30000, timeout_ms))

            config = step_def.get("config", {})

            pipeline_steps.append(PipelineStep(
                name=step_name,
                type=entry.step_type,
                config=config,
                timeout_ms=timeout_ms,
            ))

        # Generate pipeline_id and persist
        pipeline_id = str(uuid.uuid4())
        pipeline = PipelineDefinition(
            pipeline_id=pipeline_id,
            tenant_id=tenant_id,
            name=name,
            steps=pipeline_steps,
        )

        self._pipelines[pipeline_id] = pipeline
        return pipeline

    def get_pipeline(self, pipeline_id: str, tenant_id: str) -> PipelineDefinition:
        """Retrieve a pipeline by ID with tenant isolation (R9.7).

        Args:
            pipeline_id: The pipeline's unique identifier.
            tenant_id: The requesting tenant's ID.

        Returns:
            The pipeline definition.

        Raises:
            PipelineNotFoundError: If the pipeline doesn't exist or belongs
                to a different tenant (uniform 404, R9.7).
        """
        pipeline = self._pipelines.get(pipeline_id)
        if pipeline is None or pipeline.tenant_id != tenant_id:
            raise PipelineNotFoundError(
                f"Pipeline not found: {pipeline_id}"
            )
        return pipeline

    def execute(
        self,
        pipeline_id: str,
        tenant_id: str,
        candidates: list[Any],
    ) -> ExecutionResult:
        """Execute a pipeline against a candidate set (R9.3, R9.4, R9.6).

        Args:
            pipeline_id: The pipeline to execute.
            tenant_id: The requesting tenant's ID.
            candidates: The initial candidate set (e.g., from Retriever).

        Returns:
            ExecutionResult with final results and warnings.

        Raises:
            PipelineNotFoundError: If the pipeline doesn't exist or belongs
                to a different tenant.
        """
        pipeline = self.get_pipeline(pipeline_id, tenant_id)
        return execute_pipeline(pipeline, candidates, self._registry)

    def delete_pipeline(self, pipeline_id: str, tenant_id: str) -> bool:
        """Delete a pipeline (tenant-scoped).

        Args:
            pipeline_id: The pipeline to delete.
            tenant_id: The requesting tenant's ID.

        Returns:
            True if deleted, False if not found.

        Raises:
            PipelineNotFoundError: If the pipeline doesn't exist or belongs
                to a different tenant.
        """
        pipeline = self._pipelines.get(pipeline_id)
        if pipeline is None or pipeline.tenant_id != tenant_id:
            raise PipelineNotFoundError(
                f"Pipeline not found: {pipeline_id}"
            )
        del self._pipelines[pipeline_id]
        return True

    def list_pipelines(self, tenant_id: str) -> list[PipelineDefinition]:
        """List all pipelines for a tenant.

        Args:
            tenant_id: The tenant's ID.

        Returns:
            List of pipeline definitions owned by the tenant.
        """
        return [
            p for p in self._pipelines.values()
            if p.tenant_id == tenant_id
        ]
