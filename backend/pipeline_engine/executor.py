"""Pipeline executor (Task 12.4–12.6, R9.3, R9.4, R9.6).

Executes pipeline steps in declared order with:
- Output→input chaining (R9.3)
- Implicit type ordering when no explicit cross-type ordering is given (R9.4)
- Per-step timeout with pass-through fallback (R9.6)
"""

from __future__ import annotations

import concurrent.futures
import threading
from typing import Any

from backend.pipeline_engine.models import (
    ExecutionResult,
    PipelineDefinition,
    PipelineStep,
    StepType,
    StepWarning,
)
from backend.pipeline_engine.registry import PipelineRegistry


# Type ordering priority: filters first, then rerankers, then transforms (R9.4)
_TYPE_ORDER = {
    StepType.FILTER: 0,
    StepType.RERANKER: 1,
    StepType.TRANSFORM: 2,
}


def apply_type_ordering(steps: list[PipelineStep]) -> list[PipelineStep]:
    """Apply implicit type ordering to pipeline steps (R9.4).

    When a pipeline definition lacks explicit cross-type ordering,
    steps are sorted so that:
    - All filter steps execute first
    - Then all reranker steps
    - Then all transform steps

    Within the same type, the original declared order is preserved.

    Args:
        steps: The pipeline steps in declared order.

    Returns:
        Steps reordered by type precedence.
    """
    # Use a stable sort so that within-type order is preserved
    return sorted(steps, key=lambda s: _TYPE_ORDER.get(s.type, 99))


def has_explicit_type_ordering(steps: list[PipelineStep]) -> bool:
    """Check if steps already follow type ordering (R9.4).

    Returns True if the steps are already in filter→reranker→transform order,
    meaning the user explicitly declared them in the correct type order.

    Args:
        steps: The pipeline steps in declared order.

    Returns:
        True if steps already respect type precedence.
    """
    last_type_order = -1
    for step in steps:
        current_order = _TYPE_ORDER.get(step.type, 99)
        if current_order < last_type_order:
            return False
        last_type_order = current_order
    return True


def execute_pipeline(
    pipeline: PipelineDefinition,
    candidates: list[Any],
    registry: PipelineRegistry,
) -> ExecutionResult:
    """Execute a pipeline against a candidate set (R9.3, R9.4, R9.6).

    Steps are executed in order with output→input chaining.
    If the pipeline lacks explicit type ordering, implicit ordering is applied (R9.4).
    Each step has a timeout; on timeout, the step is skipped and a warning is appended (R9.6).

    Args:
        pipeline: The pipeline definition to execute.
        candidates: Initial candidate set (e.g., from Retriever).
        registry: The step registry for looking up step implementations.

    Returns:
        ExecutionResult with final results and any warnings.
    """
    warnings: list[StepWarning] = []

    # Determine step execution order
    steps = pipeline.steps
    if not has_explicit_type_ordering(steps):
        # Apply implicit type ordering (R9.4)
        steps = apply_type_ordering(steps)

    # Execute steps in order with output→input chaining (R9.3)
    current_results = list(candidates)

    for step in steps:
        entry = registry.get(step.name)
        if entry is None:
            # This shouldn't happen if validation was done at creation time,
            # but handle gracefully
            warnings.append(StepWarning(
                code="step_not_found",
                step=step.name,
                message=f"Step '{step.name}' not found in registry, skipping.",
            ))
            continue

        # Execute with timeout (R9.6)
        timeout_ms = step.timeout_ms
        try:
            result = _execute_step_with_timeout(
                entry.fn,
                current_results,
                step.config,
                timeout_ms,
            )
            current_results = result
        except StepTimeoutError:
            # On timeout: skip step, pass-through input, append warning (R9.6)
            warnings.append(StepWarning(
                code="step_timeout",
                step=step.name,
                message=f"Step '{step.name}' exceeded timeout of {timeout_ms}ms.",
            ))
            # current_results remains unchanged (pass-through)

    return ExecutionResult(results=current_results, warnings=warnings)


class StepTimeoutError(Exception):
    """Raised when a pipeline step exceeds its timeout."""

    pass


def _execute_step_with_timeout(
    fn: Any,
    results: list[Any],
    config: dict[str, Any],
    timeout_ms: int,
) -> list[Any]:
    """Execute a step function with a timeout (R9.6).

    Args:
        fn: The step function to execute.
        results: Input results.
        config: Step configuration.
        timeout_ms: Timeout in milliseconds [100, 30000].

    Returns:
        The step's output results.

    Raises:
        StepTimeoutError: If the step exceeds its timeout.
    """
    timeout_seconds = timeout_ms / 1000.0

    # Use a thread pool for timeout enforcement
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn, results, config)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            # Cancel the future (best effort)
            future.cancel()
            raise StepTimeoutError(
                f"Step exceeded timeout of {timeout_ms}ms"
            )
