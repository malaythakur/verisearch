"""Property-based tests for the Pipeline Engine (Tasks 12.8–12.10).

Properties tested:
- Property 26: Pipeline composition is order-preserving and respects type precedence.
- Property 27: Step timeout falls back to pass-through with warning.
- Property 28: Unknown step names rejected atomically.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pytest
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from backend.pipeline_engine import (
    ExecutionResult,
    InvalidPipelineError,
    PipelineDefinition,
    PipelineNotFoundError,
    PipelineService,
    PipelineStep,
    StepType,
    StepWarning,
    UnknownPipelineStepError,
    apply_type_ordering,
    create_default_registry,
    execute_pipeline,
    has_explicit_type_ordering,
)
from backend.pipeline_engine.registry import PipelineRegistry


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


@dataclass
class SimpleResult:
    """A simple result type for property testing."""

    document_id: str
    url: str
    title: str
    score: float


# Strategy for generating simple results
result_strategy = st.builds(
    SimpleResult,
    document_id=st.uuids().map(str),
    url=st.from_regex(r"https://[a-z]{3,10}\.[a-z]{2,4}/[a-z]{1,10}", fullmatch=True),
    title=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("L", "N", "Z"))),
    score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)

results_list_strategy = st.lists(result_strategy, min_size=0, max_size=20)

# Strategy for step types
step_type_strategy = st.sampled_from([StepType.FILTER, StepType.RERANKER, StepType.TRANSFORM])

# Strategy for valid step names from the default registry
valid_step_names = st.sampled_from([
    "domain_filter",
    "freshness_filter",
    "language_filter",
    "score_reranker",
    "credibility_reranker",
    "reciprocal_rank_reranker",
    "title_transform",
    "snippet_transform",
    "dedup_transform",
])

# Strategy for invalid step names (guaranteed not in registry)
invalid_step_names = st.text(
    min_size=1,
    max_size=30,
    alphabet=st.characters(whitelist_categories=("L",)),
).filter(lambda s: s not in {
    "domain_filter", "freshness_filter", "language_filter",
    "score_reranker", "credibility_reranker", "reciprocal_rank_reranker",
    "title_transform", "snippet_transform", "dedup_transform",
})

# Strategy for timeout values in valid range
timeout_strategy = st.integers(min_value=100, max_value=30000)

# Strategy for pipeline step counts in valid range [1, 20]
step_count_strategy = st.integers(min_value=1, max_value=20)

# Strategy for tenant IDs
tenant_id_strategy = st.text(
    min_size=1,
    max_size=36,
    alphabet=st.characters(whitelist_categories=("L", "N")),
).map(lambda s: f"tenant-{s}")


# ---------------------------------------------------------------------------
# Property 26: Pipeline composition is order-preserving and respects type precedence
# ---------------------------------------------------------------------------


class TestProperty26PipelineComposition:
    """Property 26: Pipeline composition is order-preserving and respects type precedence.

    **Validates: Requirements 9.3, 9.4**

    For any tenant-scoped pipeline definition with steps s_1..s_n and any candidate set C:
    - Executing the pipeline produces the same observed input/output chain as
      sequential composition applied to C.
    - When no explicit cross-type ordering is given, all filter steps execute before
      all reranker steps, which execute before all transform steps.
    """

    @given(
        step_types=st.lists(step_type_strategy, min_size=1, max_size=10),
        results=results_list_strategy,
    )
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=50)
    def test_type_precedence_respected(self, step_types: list[StepType], results: list[SimpleResult]):
        """When types are mixed, implicit ordering ensures filters→rerankers→transforms."""
        registry = PipelineRegistry()
        execution_order: list[StepType] = []

        # Register steps that track execution order
        for i, stype in enumerate(step_types):
            name = f"step_{i}_{stype.value}"

            def make_fn(st=stype):
                def fn(r, config=None):
                    execution_order.append(st)
                    return r
                return fn

            registry.register(name, stype, make_fn())

        service = PipelineService(registry=registry)
        step_defs = [{"name": f"step_{i}_{stype.value}"} for i, stype in enumerate(step_types)]

        pipeline = service.create_pipeline(
            tenant_id="test-tenant",
            name="Property 26 Test",
            steps=step_defs,
        )

        execution_order.clear()
        service.execute(pipeline.pipeline_id, "test-tenant", list(results))

        # Verify type precedence: all filters before all rerankers before all transforms
        filter_indices = [i for i, t in enumerate(execution_order) if t == StepType.FILTER]
        reranker_indices = [i for i, t in enumerate(execution_order) if t == StepType.RERANKER]
        transform_indices = [i for i, t in enumerate(execution_order) if t == StepType.TRANSFORM]

        if filter_indices and reranker_indices:
            assert max(filter_indices) < min(reranker_indices), (
                "Filters must execute before rerankers"
            )
        if reranker_indices and transform_indices:
            assert max(reranker_indices) < min(transform_indices), (
                "Rerankers must execute before transforms"
            )
        if filter_indices and transform_indices:
            assert max(filter_indices) < min(transform_indices), (
                "Filters must execute before transforms"
            )

    @given(
        num_steps=st.integers(min_value=2, max_value=8),
        results=results_list_strategy,
    )
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=50)
    def test_within_type_order_preserved(self, num_steps: int, results: list[SimpleResult]):
        """Within the same type, declared order is preserved."""
        registry = PipelineRegistry()
        execution_order: list[int] = []

        # Register multiple filter steps
        for i in range(num_steps):
            name = f"filter_{i}"

            def make_fn(idx=i):
                def fn(r, config=None):
                    execution_order.append(idx)
                    return r
                return fn

            registry.register(name, StepType.FILTER, make_fn())

        service = PipelineService(registry=registry)
        step_defs = [{"name": f"filter_{i}"} for i in range(num_steps)]

        pipeline = service.create_pipeline(
            tenant_id="test-tenant",
            name="Order Preservation",
            steps=step_defs,
        )

        execution_order.clear()
        service.execute(pipeline.pipeline_id, "test-tenant", list(results))

        # Within-type order should match declared order
        assert execution_order == list(range(num_steps)), (
            f"Expected order {list(range(num_steps))}, got {execution_order}"
        )

    @given(results=results_list_strategy)
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=50)
    def test_output_input_chaining(self, results: list[SimpleResult]):
        """Each step receives the output of the previous step (R9.3)."""
        registry = PipelineRegistry()
        observed_inputs: list[int] = []

        def step_remove_first(r, config=None):
            """Remove the first result."""
            observed_inputs.append(len(r))
            return r[1:] if r else r

        def step_count(r, config=None):
            """Just observe the input length."""
            observed_inputs.append(len(r))
            return r

        registry.register("remove_first", StepType.FILTER, step_remove_first)
        registry.register("count", StepType.FILTER, step_count)

        service = PipelineService(registry=registry)
        pipeline = service.create_pipeline(
            tenant_id="test-tenant",
            name="Chaining Test",
            steps=[
                {"name": "remove_first"},
                {"name": "count"},
            ],
        )

        observed_inputs.clear()
        service.execute(pipeline.pipeline_id, "test-tenant", list(results))

        if len(results) > 0:
            # First step sees original length
            assert observed_inputs[0] == len(results)
            # Second step sees length - 1 (one removed)
            assert observed_inputs[1] == len(results) - 1
        else:
            # Both see 0
            assert observed_inputs == [0, 0]


# ---------------------------------------------------------------------------
# Property 27: Step timeout falls back to pass-through with warning
# ---------------------------------------------------------------------------


class TestProperty27StepTimeout:
    """Property 27: Step timeout falls back to pass-through with warning.

    **Validates: Requirements 9.6**

    For any pipeline step whose execution exceeds its configured timeout_ms ∈ [100, 30000]:
    - The step's effective output equals its input (pass-through).
    - The response's warnings array contains a step_timeout entry naming the step.
    """

    @given(
        results=st.lists(result_strategy, min_size=1, max_size=10),
        timeout_ms=st.integers(min_value=100, max_value=200),
    )
    @settings(
        suppress_health_check=[HealthCheck.too_slow],
        max_examples=10,
        deadline=None,
    )
    def test_timeout_produces_passthrough(self, results: list[SimpleResult], timeout_ms: int):
        """On timeout, output equals input (pass-through)."""
        registry = PipelineRegistry()

        def slow_step(r, config=None):
            time.sleep(1)  # Always exceeds the short timeout
            return []  # Would return empty, but timeout prevents this

        registry.register("slow", StepType.FILTER, slow_step)

        service = PipelineService(registry=registry)
        pipeline = service.create_pipeline(
            tenant_id="test-tenant",
            name="Timeout Test",
            steps=[{"name": "slow", "timeout_ms": timeout_ms}],
        )

        execution = service.execute(pipeline.pipeline_id, "test-tenant", list(results))

        # Output should equal input (pass-through)
        assert len(execution.results) == len(results)
        for orig, out in zip(results, execution.results):
            assert orig.document_id == out.document_id

    @given(
        results=st.lists(result_strategy, min_size=1, max_size=10),
        timeout_ms=st.integers(min_value=100, max_value=200),
    )
    @settings(
        suppress_health_check=[HealthCheck.too_slow],
        max_examples=10,
        deadline=None,
    )
    def test_timeout_produces_warning(self, results: list[SimpleResult], timeout_ms: int):
        """On timeout, warnings array contains step_timeout entry."""
        registry = PipelineRegistry()

        def slow_step(r, config=None):
            time.sleep(1)
            return []

        registry.register("slow_step", StepType.FILTER, slow_step)

        service = PipelineService(registry=registry)
        pipeline = service.create_pipeline(
            tenant_id="test-tenant",
            name="Warning Test",
            steps=[{"name": "slow_step", "timeout_ms": timeout_ms}],
        )

        execution = service.execute(pipeline.pipeline_id, "test-tenant", list(results))

        # Should have exactly one step_timeout warning
        timeout_warnings = [w for w in execution.warnings if w.code == "step_timeout"]
        assert len(timeout_warnings) == 1
        assert timeout_warnings[0].step == "slow_step"

    @given(results=st.lists(result_strategy, min_size=1, max_size=10))
    @settings(
        suppress_health_check=[HealthCheck.too_slow],
        max_examples=10,
        deadline=None,
    )
    def test_timeout_does_not_affect_subsequent_steps(self, results: list[SimpleResult]):
        """After a timeout, subsequent steps still execute normally."""
        registry = PipelineRegistry()
        executed: list[str] = []

        def slow_step(r, config=None):
            time.sleep(1)
            return []

        def fast_step(r, config=None):
            executed.append("fast")
            return r

        registry.register("slow", StepType.FILTER, slow_step)
        registry.register("fast", StepType.FILTER, fast_step)

        service = PipelineService(registry=registry)
        pipeline = service.create_pipeline(
            tenant_id="test-tenant",
            name="Continue Test",
            steps=[
                {"name": "slow", "timeout_ms": 100},
                {"name": "fast", "timeout_ms": 5000},
            ],
        )

        executed.clear()
        execution = service.execute(pipeline.pipeline_id, "test-tenant", list(results))

        # Fast step should have executed
        assert "fast" in executed
        # Results should be the original (passed through slow, then through fast)
        assert len(execution.results) == len(results)


# ---------------------------------------------------------------------------
# Property 28: Unknown step names rejected atomically
# ---------------------------------------------------------------------------


class TestProperty28UnknownStepRejection:
    """Property 28: Pipeline persistence rejects unknown step names atomically.

    **Validates: Requirements 9.1, 9.2**

    For any POST /v1/pipelines request containing one or more step names not present
    in the registry:
    - The response is HTTP 400 with code unknown_pipeline_step.
    - The error lists every offending name.
    - No pipeline is persisted.

    For any request whose every step name resolves and step count is in [1, 20]:
    - The response is 201 with a generated pipeline_id.
    """

    @given(
        valid_names=st.lists(valid_step_names, min_size=0, max_size=5),
        invalid_names=st.lists(invalid_step_names, min_size=1, max_size=5),
    )
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=50)
    def test_unknown_steps_rejected_with_all_names(
        self, valid_names: list[str], invalid_names: list[str]
    ):
        """All unknown step names are listed in the rejection error."""
        service = PipelineService()

        # Mix valid and invalid names
        all_steps = [{"name": n} for n in valid_names + invalid_names]
        assume(1 <= len(all_steps) <= 20)

        with pytest.raises(UnknownPipelineStepError) as exc_info:
            service.create_pipeline(
                tenant_id="test-tenant",
                name="Invalid Pipeline",
                steps=all_steps,
            )

        # Every invalid name should be in the error
        for name in invalid_names:
            assert name in exc_info.value.unknown_steps

        # No valid name should be in the error
        for name in valid_names:
            assert name not in exc_info.value.unknown_steps

    @given(
        invalid_names=st.lists(invalid_step_names, min_size=1, max_size=5),
    )
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=50)
    def test_unknown_steps_not_persisted(self, invalid_names: list[str]):
        """No pipeline is persisted when unknown steps are present."""
        service = PipelineService()

        steps = [{"name": n} for n in invalid_names]
        assume(1 <= len(steps) <= 20)

        try:
            service.create_pipeline(
                tenant_id="test-tenant",
                name="Should Not Persist",
                steps=steps,
            )
        except UnknownPipelineStepError:
            pass

        # No pipelines should exist
        assert service.list_pipelines("test-tenant") == []

    @given(
        step_names=st.lists(valid_step_names, min_size=1, max_size=20),
        tenant_id=tenant_id_strategy,
    )
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=50)
    def test_valid_steps_accepted_with_pipeline_id(
        self, step_names: list[str], tenant_id: str
    ):
        """Valid step names with count in [1, 20] produce a pipeline_id."""
        service = PipelineService()

        steps = [{"name": n} for n in step_names]
        pipeline = service.create_pipeline(
            tenant_id=tenant_id,
            name="Valid Pipeline",
            steps=steps,
        )

        assert pipeline.pipeline_id is not None
        assert len(pipeline.pipeline_id) > 0
        assert pipeline.tenant_id == tenant_id
        assert len(pipeline.steps) == len(step_names)

    @given(
        step_names=st.lists(valid_step_names, min_size=1, max_size=20),
    )
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=50)
    def test_valid_pipeline_retrievable(self, step_names: list[str]):
        """A successfully created pipeline can be retrieved by its owner."""
        service = PipelineService()

        steps = [{"name": n} for n in step_names]
        pipeline = service.create_pipeline(
            tenant_id="test-tenant",
            name="Retrievable",
            steps=steps,
        )

        retrieved = service.get_pipeline(pipeline.pipeline_id, "test-tenant")
        assert retrieved.pipeline_id == pipeline.pipeline_id
        assert len(retrieved.steps) == len(step_names)

    @given(
        valid_names=st.lists(valid_step_names, min_size=1, max_size=5),
        invalid_names=st.lists(invalid_step_names, min_size=1, max_size=5),
    )
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=50)
    def test_rejection_is_atomic(self, valid_names: list[str], invalid_names: list[str]):
        """Rejection is atomic: even if some steps are valid, nothing is persisted."""
        service = PipelineService()

        # Create one valid pipeline first
        service.create_pipeline(
            tenant_id="test-tenant",
            name="Pre-existing",
            steps=[{"name": valid_names[0]}],
        )
        initial_count = len(service.list_pipelines("test-tenant"))

        # Try to create an invalid pipeline
        all_steps = [{"name": n} for n in valid_names + invalid_names]
        assume(1 <= len(all_steps) <= 20)

        try:
            service.create_pipeline(
                tenant_id="test-tenant",
                name="Atomic Reject",
                steps=all_steps,
            )
        except UnknownPipelineStepError:
            pass

        # Count should not have changed
        assert len(service.list_pipelines("test-tenant")) == initial_count
