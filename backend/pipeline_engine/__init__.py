"""Pipeline Engine - Programmable retrieval pipeline execution with step ordering and timeouts.

Exports:
- PipelineService: Main service for CRUD and execution of pipelines.
- PipelineDefinition, PipelineStep, ExecutionResult, StepWarning: Data models.
- StepType: Enum for step types (filter, reranker, transform).
- PipelineRegistry: Step registry for built-in and custom steps.
- PipelineNotFoundError: Error for missing/cross-tenant pipelines (R9.7).
- UnknownPipelineStepError: Error for unknown step names (R9.2).
- InvalidPipelineError: Error for invalid pipeline definitions.
- create_default_registry: Factory for the built-in step registry.
- execute_pipeline: Low-level pipeline execution function.
- apply_type_ordering: Implicit type ordering (R9.4).
- has_explicit_type_ordering: Check if steps already follow type order.
"""

from backend.pipeline_engine.executor import (
    StepTimeoutError,
    apply_type_ordering,
    execute_pipeline,
    has_explicit_type_ordering,
)
from backend.pipeline_engine.models import (
    ExecutionResult,
    PipelineDefinition,
    PipelineStep,
    StepType,
    StepWarning,
)
from backend.pipeline_engine.registry import (
    PipelineRegistry,
    create_default_registry,
)
from backend.pipeline_engine.service import (
    InvalidPipelineError,
    PipelineNotFoundError,
    PipelineService,
    UnknownPipelineStepError,
)

__all__ = [
    "PipelineService",
    "PipelineDefinition",
    "PipelineStep",
    "ExecutionResult",
    "StepWarning",
    "StepType",
    "PipelineRegistry",
    "PipelineNotFoundError",
    "UnknownPipelineStepError",
    "InvalidPipelineError",
    "StepTimeoutError",
    "create_default_registry",
    "execute_pipeline",
    "apply_type_ordering",
    "has_explicit_type_ordering",
]
