"""Unit tests for the Pipeline Engine (Tasks 12.1–12.7).

Tests cover:
- Registry: built-in step registration and lookup (12.1)
- Pipeline creation with validation (12.2, 12.3)
- Pipeline execution with step chaining (12.4)
- Implicit type ordering (12.5)
- Per-step timeout with pass-through fallback (12.6)
- Pipeline not found / cross-tenant isolation (12.7)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pytest

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
# Test fixtures / helpers
# ---------------------------------------------------------------------------


@dataclass
class MockResult:
    """A simple mock search result for testing."""

    document_id: str
    url: str
    title: str
    score: float
    published_at: datetime | None = None


def make_results(n: int = 5) -> list[MockResult]:
    """Create a list of mock results."""
    return [
        MockResult(
            document_id=f"doc-{i}",
            url=f"https://example{i}.com/page",
            title=f"Title {i}",
            score=1.0 - (i * 0.1),
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Task 12.1: Pipeline Registry Tests
# ---------------------------------------------------------------------------


class TestPipelineRegistry:
    """Tests for the pipeline step registry (Task 12.1)."""

    def test_default_registry_has_filters(self):
        """Default registry contains built-in filter steps."""
        registry = create_default_registry()
        assert registry.exists("domain_filter")
        assert registry.exists("freshness_filter")
        assert registry.exists("language_filter")

    def test_default_registry_has_rerankers(self):
        """Default registry contains built-in reranker steps."""
        registry = create_default_registry()
        assert registry.exists("score_reranker")
        assert registry.exists("credibility_reranker")
        assert registry.exists("reciprocal_rank_reranker")

    def test_default_registry_has_transforms(self):
        """Default registry contains built-in transform steps."""
        registry = create_default_registry()
        assert registry.exists("title_transform")
        assert registry.exists("snippet_transform")
        assert registry.exists("dedup_transform")

    def test_registry_get_returns_entry(self):
        """Registry.get returns the entry for a known step."""
        registry = create_default_registry()
        entry = registry.get("domain_filter")
        assert entry is not None
        assert entry.name == "domain_filter"
        assert entry.step_type == StepType.FILTER

    def test_registry_get_returns_none_for_unknown(self):
        """Registry.get returns None for unknown step names."""
        registry = create_default_registry()
        assert registry.get("nonexistent_step") is None

    def test_registry_get_unknown_steps(self):
        """get_unknown_steps returns names not in registry."""
        registry = create_default_registry()
        unknown = registry.get_unknown_steps(["domain_filter", "fake_step", "another_fake"])
        assert "fake_step" in unknown
        assert "another_fake" in unknown
        assert "domain_filter" not in unknown

    def test_registry_get_unknown_steps_all_valid(self):
        """get_unknown_steps returns empty list when all names are valid."""
        registry = create_default_registry()
        unknown = registry.get_unknown_steps(["domain_filter", "score_reranker"])
        assert unknown == []

    def test_registry_step_types_correct(self):
        """Each built-in step has the correct type."""
        registry = create_default_registry()
        assert registry.get("domain_filter").step_type == StepType.FILTER
        assert registry.get("score_reranker").step_type == StepType.RERANKER
        assert registry.get("title_transform").step_type == StepType.TRANSFORM

    def test_registry_custom_step_registration(self):
        """Custom steps can be registered."""
        registry = PipelineRegistry()
        registry.register("my_filter", StepType.FILTER, lambda r, c: r, "Custom filter")
        assert registry.exists("my_filter")
        entry = registry.get("my_filter")
        assert entry.step_type == StepType.FILTER

    def test_registry_list_steps(self):
        """list_steps returns all registered entries."""
        registry = create_default_registry()
        steps = registry.list_steps()
        assert len(steps) == 9  # 3 filters + 3 rerankers + 3 transforms


# ---------------------------------------------------------------------------
# Task 12.2: Pipeline Creation Tests
# ---------------------------------------------------------------------------


class TestPipelineCreation:
    """Tests for POST /v1/pipelines — validate and persist (Task 12.2, R9.1)."""

    def test_create_valid_pipeline(self):
        """Creating a pipeline with valid steps returns 201 + pipeline_id."""
        service = PipelineService()
        pipeline = service.create_pipeline(
            tenant_id="tenant-1",
            name="My Pipeline",
            steps=[
                {"name": "domain_filter", "config": {"domains": ["example.com"]}},
                {"name": "score_reranker"},
            ],
        )
        assert pipeline.pipeline_id is not None
        assert pipeline.tenant_id == "tenant-1"
        assert pipeline.name == "My Pipeline"
        assert len(pipeline.steps) == 2

    def test_create_pipeline_with_single_step(self):
        """Pipeline with exactly 1 step is valid (R9.1 lower bound)."""
        service = PipelineService()
        pipeline = service.create_pipeline(
            tenant_id="tenant-1",
            name="Single Step",
            steps=[{"name": "domain_filter"}],
        )
        assert len(pipeline.steps) == 1

    def test_create_pipeline_with_20_steps(self):
        """Pipeline with exactly 20 steps is valid (R9.1 upper bound)."""
        service = PipelineService()
        steps = [{"name": "domain_filter"} for _ in range(20)]
        pipeline = service.create_pipeline(
            tenant_id="tenant-1",
            name="Max Steps",
            steps=steps,
        )
        assert len(pipeline.steps) == 20

    def test_create_pipeline_zero_steps_rejected(self):
        """Pipeline with 0 steps is rejected."""
        service = PipelineService()
        with pytest.raises(InvalidPipelineError):
            service.create_pipeline(
                tenant_id="tenant-1",
                name="Empty",
                steps=[],
            )

    def test_create_pipeline_21_steps_rejected(self):
        """Pipeline with >20 steps is rejected (R9.1)."""
        service = PipelineService()
        steps = [{"name": "domain_filter"} for _ in range(21)]
        with pytest.raises(InvalidPipelineError):
            service.create_pipeline(
                tenant_id="tenant-1",
                name="Too Many",
                steps=steps,
            )

    def test_create_pipeline_persists(self):
        """Created pipeline can be retrieved."""
        service = PipelineService()
        pipeline = service.create_pipeline(
            tenant_id="tenant-1",
            name="Persisted",
            steps=[{"name": "domain_filter"}],
        )
        retrieved = service.get_pipeline(pipeline.pipeline_id, "tenant-1")
        assert retrieved.pipeline_id == pipeline.pipeline_id
        assert retrieved.name == "Persisted"

    def test_create_pipeline_with_custom_timeout(self):
        """Pipeline steps can have custom timeout_ms."""
        service = PipelineService()
        pipeline = service.create_pipeline(
            tenant_id="tenant-1",
            name="Custom Timeout",
            steps=[{"name": "domain_filter", "timeout_ms": 5000}],
        )
        assert pipeline.steps[0].timeout_ms == 5000

    def test_create_pipeline_timeout_clamped_to_range(self):
        """Timeout values are clamped to [100, 30000]."""
        service = PipelineService()
        pipeline = service.create_pipeline(
            tenant_id="tenant-1",
            name="Clamped",
            steps=[
                {"name": "domain_filter", "timeout_ms": 50},  # Below min
                {"name": "score_reranker", "timeout_ms": 50000},  # Above max
            ],
        )
        assert pipeline.steps[0].timeout_ms == 100
        assert pipeline.steps[1].timeout_ms == 30000

    def test_create_pipeline_default_timeout(self):
        """Default timeout is 2000ms."""
        service = PipelineService()
        pipeline = service.create_pipeline(
            tenant_id="tenant-1",
            name="Default Timeout",
            steps=[{"name": "domain_filter"}],
        )
        assert pipeline.steps[0].timeout_ms == 2000


# ---------------------------------------------------------------------------
# Task 12.3: Unknown Step Rejection Tests
# ---------------------------------------------------------------------------


class TestUnknownStepRejection:
    """Tests for unknown step name rejection (Task 12.3, R9.2)."""

    def test_single_unknown_step_rejected(self):
        """A single unknown step name causes HTTP 400."""
        service = PipelineService()
        with pytest.raises(UnknownPipelineStepError) as exc_info:
            service.create_pipeline(
                tenant_id="tenant-1",
                name="Bad Pipeline",
                steps=[{"name": "nonexistent_step"}],
            )
        assert "nonexistent_step" in exc_info.value.unknown_steps

    def test_multiple_unknown_steps_all_listed(self):
        """All unknown step names are listed in the error (R9.2)."""
        service = PipelineService()
        with pytest.raises(UnknownPipelineStepError) as exc_info:
            service.create_pipeline(
                tenant_id="tenant-1",
                name="Bad Pipeline",
                steps=[
                    {"name": "fake_filter"},
                    {"name": "domain_filter"},  # valid
                    {"name": "bogus_reranker"},
                    {"name": "score_reranker"},  # valid
                    {"name": "invalid_transform"},
                ],
            )
        unknown = exc_info.value.unknown_steps
        assert "fake_filter" in unknown
        assert "bogus_reranker" in unknown
        assert "invalid_transform" in unknown
        assert "domain_filter" not in unknown
        assert "score_reranker" not in unknown

    def test_unknown_step_does_not_persist(self):
        """Pipeline with unknown steps is not persisted."""
        service = PipelineService()
        try:
            service.create_pipeline(
                tenant_id="tenant-1",
                name="Not Persisted",
                steps=[{"name": "fake_step"}],
            )
        except UnknownPipelineStepError:
            pass

        # No pipelines should exist
        assert service.list_pipelines("tenant-1") == []


# ---------------------------------------------------------------------------
# Task 12.4: Pipeline Execution Tests
# ---------------------------------------------------------------------------


class TestPipelineExecution:
    """Tests for pipeline execution with step chaining (Task 12.4, R9.3)."""

    def test_single_step_execution(self):
        """A single-step pipeline executes correctly."""
        service = PipelineService()
        pipeline = service.create_pipeline(
            tenant_id="tenant-1",
            name="Single Step",
            steps=[{"name": "domain_filter", "config": {"domains": ["example1.com"]}}],
        )
        results = make_results(5)
        execution = service.execute(pipeline.pipeline_id, "tenant-1", results)
        # Only results with example1.com in URL should remain
        assert all("example1.com" in r.url for r in execution.results)

    def test_multi_step_chaining(self):
        """Multi-step pipeline chains output→input correctly (R9.3)."""
        service = PipelineService()
        pipeline = service.create_pipeline(
            tenant_id="tenant-1",
            name="Multi Step",
            steps=[
                {"name": "domain_filter", "config": {"domains": ["example"]}},
                {"name": "title_transform", "config": {"prefix": "[Filtered] "}},
            ],
        )
        results = make_results(3)
        execution = service.execute(pipeline.pipeline_id, "tenant-1", results)
        # All results should have the prefix (since all URLs contain "example")
        for r in execution.results:
            assert r.title.startswith("[Filtered] ")

    def test_empty_candidate_set(self):
        """Pipeline handles empty candidate set gracefully."""
        service = PipelineService()
        pipeline = service.create_pipeline(
            tenant_id="tenant-1",
            name="Empty Input",
            steps=[{"name": "domain_filter"}],
        )
        execution = service.execute(pipeline.pipeline_id, "tenant-1", [])
        assert execution.results == []
        assert execution.warnings == []

    def test_execution_preserves_order_within_type(self):
        """Steps of the same type execute in declared order."""
        # Create a registry with custom steps that track execution order
        registry = PipelineRegistry()
        execution_order: list[str] = []

        def step_a(results, config=None):
            execution_order.append("a")
            return results

        def step_b(results, config=None):
            execution_order.append("b")
            return results

        def step_c(results, config=None):
            execution_order.append("c")
            return results

        registry.register("step_a", StepType.FILTER, step_a)
        registry.register("step_b", StepType.FILTER, step_b)
        registry.register("step_c", StepType.FILTER, step_c)

        service = PipelineService(registry=registry)
        pipeline = service.create_pipeline(
            tenant_id="tenant-1",
            name="Order Test",
            steps=[
                {"name": "step_a"},
                {"name": "step_b"},
                {"name": "step_c"},
            ],
        )
        service.execute(pipeline.pipeline_id, "tenant-1", make_results(1))
        assert execution_order == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Task 12.5: Implicit Type Ordering Tests
# ---------------------------------------------------------------------------


class TestImplicitTypeOrdering:
    """Tests for implicit type ordering: filters → rerankers → transforms (Task 12.5, R9.4)."""

    def test_type_ordering_applied_when_mixed(self):
        """When types are mixed, implicit ordering is applied (R9.4)."""
        registry = PipelineRegistry()
        execution_order: list[str] = []

        def filter_step(results, config=None):
            execution_order.append("filter")
            return results

        def reranker_step(results, config=None):
            execution_order.append("reranker")
            return results

        def transform_step(results, config=None):
            execution_order.append("transform")
            return results

        registry.register("my_filter", StepType.FILTER, filter_step)
        registry.register("my_reranker", StepType.RERANKER, reranker_step)
        registry.register("my_transform", StepType.TRANSFORM, transform_step)

        service = PipelineService(registry=registry)
        # Declare in wrong order: transform, filter, reranker
        pipeline = service.create_pipeline(
            tenant_id="tenant-1",
            name="Mixed Order",
            steps=[
                {"name": "my_transform"},
                {"name": "my_filter"},
                {"name": "my_reranker"},
            ],
        )
        service.execute(pipeline.pipeline_id, "tenant-1", make_results(1))
        # Should execute in type order: filter → reranker → transform
        assert execution_order == ["filter", "reranker", "transform"]

    def test_explicit_type_ordering_preserved(self):
        """When steps are already in type order, declared order is preserved."""
        registry = PipelineRegistry()
        execution_order: list[str] = []

        def filter_step(results, config=None):
            execution_order.append("filter")
            return results

        def reranker_step(results, config=None):
            execution_order.append("reranker")
            return results

        def transform_step(results, config=None):
            execution_order.append("transform")
            return results

        registry.register("my_filter", StepType.FILTER, filter_step)
        registry.register("my_reranker", StepType.RERANKER, reranker_step)
        registry.register("my_transform", StepType.TRANSFORM, transform_step)

        service = PipelineService(registry=registry)
        # Declare in correct type order
        pipeline = service.create_pipeline(
            tenant_id="tenant-1",
            name="Correct Order",
            steps=[
                {"name": "my_filter"},
                {"name": "my_reranker"},
                {"name": "my_transform"},
            ],
        )
        service.execute(pipeline.pipeline_id, "tenant-1", make_results(1))
        assert execution_order == ["filter", "reranker", "transform"]

    def test_has_explicit_type_ordering_true(self):
        """has_explicit_type_ordering returns True for correctly ordered steps."""
        steps = [
            PipelineStep(name="f", type=StepType.FILTER),
            PipelineStep(name="r", type=StepType.RERANKER),
            PipelineStep(name="t", type=StepType.TRANSFORM),
        ]
        assert has_explicit_type_ordering(steps) is True

    def test_has_explicit_type_ordering_false(self):
        """has_explicit_type_ordering returns False for incorrectly ordered steps."""
        steps = [
            PipelineStep(name="t", type=StepType.TRANSFORM),
            PipelineStep(name="f", type=StepType.FILTER),
        ]
        assert has_explicit_type_ordering(steps) is False

    def test_apply_type_ordering_stable_within_type(self):
        """apply_type_ordering preserves relative order within same type."""
        steps = [
            PipelineStep(name="t1", type=StepType.TRANSFORM),
            PipelineStep(name="f1", type=StepType.FILTER),
            PipelineStep(name="f2", type=StepType.FILTER),
            PipelineStep(name="t2", type=StepType.TRANSFORM),
        ]
        ordered = apply_type_ordering(steps)
        # Filters first, then transforms, preserving within-type order
        assert [s.name for s in ordered] == ["f1", "f2", "t1", "t2"]

    def test_multiple_filters_before_rerankers(self):
        """Multiple filters all execute before any reranker."""
        registry = PipelineRegistry()
        execution_order: list[str] = []

        def f1(results, config=None):
            execution_order.append("f1")
            return results

        def f2(results, config=None):
            execution_order.append("f2")
            return results

        def r1(results, config=None):
            execution_order.append("r1")
            return results

        registry.register("f1", StepType.FILTER, f1)
        registry.register("f2", StepType.FILTER, f2)
        registry.register("r1", StepType.RERANKER, r1)

        service = PipelineService(registry=registry)
        # Declare reranker between filters
        pipeline = service.create_pipeline(
            tenant_id="tenant-1",
            name="Interleaved",
            steps=[
                {"name": "f1"},
                {"name": "r1"},
                {"name": "f2"},
            ],
        )
        service.execute(pipeline.pipeline_id, "tenant-1", make_results(1))
        # Both filters should execute before the reranker
        assert execution_order == ["f1", "f2", "r1"]


# ---------------------------------------------------------------------------
# Task 12.6: Per-Step Timeout Tests
# ---------------------------------------------------------------------------


class TestPerStepTimeout:
    """Tests for per-step timeout with pass-through fallback (Task 12.6, R9.6)."""

    def test_timeout_produces_warning(self):
        """A step that exceeds timeout produces a step_timeout warning."""
        registry = PipelineRegistry()

        def slow_step(results, config=None):
            time.sleep(2)  # Sleep longer than timeout
            return []

        registry.register("slow_step", StepType.FILTER, slow_step)

        service = PipelineService(registry=registry)
        pipeline = service.create_pipeline(
            tenant_id="tenant-1",
            name="Slow Pipeline",
            steps=[{"name": "slow_step", "timeout_ms": 100}],  # 100ms timeout
        )
        results = make_results(3)
        execution = service.execute(pipeline.pipeline_id, "tenant-1", results)

        # Should have a step_timeout warning
        assert len(execution.warnings) == 1
        assert execution.warnings[0].code == "step_timeout"
        assert execution.warnings[0].step == "slow_step"

    def test_timeout_passes_through_input(self):
        """On timeout, the step's input is passed through unchanged (R9.6)."""
        registry = PipelineRegistry()

        def slow_step(results, config=None):
            time.sleep(2)
            return []  # Would return empty, but timeout should prevent this

        registry.register("slow_step", StepType.FILTER, slow_step)

        service = PipelineService(registry=registry)
        pipeline = service.create_pipeline(
            tenant_id="tenant-1",
            name="Pass-through",
            steps=[{"name": "slow_step", "timeout_ms": 100}],
        )
        results = make_results(3)
        execution = service.execute(pipeline.pipeline_id, "tenant-1", results)

        # Results should be unchanged (pass-through)
        assert len(execution.results) == 3
        assert execution.results == results

    def test_fast_step_no_warning(self):
        """A step that completes within timeout produces no warning."""
        registry = PipelineRegistry()

        def fast_step(results, config=None):
            return results

        registry.register("fast_step", StepType.FILTER, fast_step)

        service = PipelineService(registry=registry)
        pipeline = service.create_pipeline(
            tenant_id="tenant-1",
            name="Fast Pipeline",
            steps=[{"name": "fast_step", "timeout_ms": 5000}],
        )
        results = make_results(3)
        execution = service.execute(pipeline.pipeline_id, "tenant-1", results)

        assert execution.warnings == []

    def test_timeout_continues_pipeline(self):
        """After a timeout, subsequent steps still execute."""
        registry = PipelineRegistry()
        execution_order: list[str] = []

        def slow_step(results, config=None):
            time.sleep(2)
            execution_order.append("slow")
            return []

        def fast_step(results, config=None):
            execution_order.append("fast")
            return results

        registry.register("slow_step", StepType.FILTER, slow_step)
        registry.register("fast_step", StepType.FILTER, fast_step)

        service = PipelineService(registry=registry)
        pipeline = service.create_pipeline(
            tenant_id="tenant-1",
            name="Continue After Timeout",
            steps=[
                {"name": "slow_step", "timeout_ms": 100},
                {"name": "fast_step", "timeout_ms": 5000},
            ],
        )
        results = make_results(3)
        execution = service.execute(pipeline.pipeline_id, "tenant-1", results)

        # Fast step should have executed
        assert "fast" in execution_order
        # Should have warning for slow step
        assert any(w.code == "step_timeout" for w in execution.warnings)


# ---------------------------------------------------------------------------
# Task 12.7: Pipeline Not Found Tests
# ---------------------------------------------------------------------------


class TestPipelineNotFound:
    """Tests for pipeline_not_found 404 (Task 12.7, R9.7)."""

    def test_nonexistent_pipeline_raises(self):
        """Requesting a non-existent pipeline_id raises PipelineNotFoundError."""
        service = PipelineService()
        with pytest.raises(PipelineNotFoundError):
            service.get_pipeline("nonexistent-id", "tenant-1")

    def test_cross_tenant_access_raises(self):
        """Accessing another tenant's pipeline raises PipelineNotFoundError (R9.7)."""
        service = PipelineService()
        pipeline = service.create_pipeline(
            tenant_id="tenant-1",
            name="Tenant 1 Pipeline",
            steps=[{"name": "domain_filter"}],
        )
        # Tenant-2 tries to access tenant-1's pipeline
        with pytest.raises(PipelineNotFoundError):
            service.get_pipeline(pipeline.pipeline_id, "tenant-2")

    def test_cross_tenant_execute_raises(self):
        """Executing another tenant's pipeline raises PipelineNotFoundError."""
        service = PipelineService()
        pipeline = service.create_pipeline(
            tenant_id="tenant-1",
            name="Tenant 1 Pipeline",
            steps=[{"name": "domain_filter"}],
        )
        with pytest.raises(PipelineNotFoundError):
            service.execute(pipeline.pipeline_id, "tenant-2", make_results(3))

    def test_cross_tenant_delete_raises(self):
        """Deleting another tenant's pipeline raises PipelineNotFoundError."""
        service = PipelineService()
        pipeline = service.create_pipeline(
            tenant_id="tenant-1",
            name="Tenant 1 Pipeline",
            steps=[{"name": "domain_filter"}],
        )
        with pytest.raises(PipelineNotFoundError):
            service.delete_pipeline(pipeline.pipeline_id, "tenant-2")

    def test_same_tenant_access_succeeds(self):
        """Same tenant can access their own pipeline."""
        service = PipelineService()
        pipeline = service.create_pipeline(
            tenant_id="tenant-1",
            name="My Pipeline",
            steps=[{"name": "domain_filter"}],
        )
        retrieved = service.get_pipeline(pipeline.pipeline_id, "tenant-1")
        assert retrieved.pipeline_id == pipeline.pipeline_id

    def test_uniform_error_shape(self):
        """Cross-tenant and non-existent both raise the same error type."""
        service = PipelineService()
        pipeline = service.create_pipeline(
            tenant_id="tenant-1",
            name="Pipeline",
            steps=[{"name": "domain_filter"}],
        )

        # Both should raise PipelineNotFoundError (uniform 404)
        with pytest.raises(PipelineNotFoundError):
            service.get_pipeline("totally-fake-id", "tenant-1")

        with pytest.raises(PipelineNotFoundError):
            service.get_pipeline(pipeline.pipeline_id, "tenant-2")


# ---------------------------------------------------------------------------
# Built-in step function tests
# ---------------------------------------------------------------------------


class TestBuiltInSteps:
    """Tests for built-in step implementations."""

    def test_domain_filter_includes(self):
        """domain_filter includes results matching specified domains."""
        from backend.pipeline_engine.registry import domain_filter

        results = make_results(5)
        filtered = domain_filter(results, {"domains": ["example0.com"]})
        assert len(filtered) == 1
        assert "example0.com" in filtered[0].url

    def test_domain_filter_excludes(self):
        """domain_filter with exclude=True removes matching domains."""
        from backend.pipeline_engine.registry import domain_filter

        results = make_results(5)
        filtered = domain_filter(results, {"domains": ["example0.com"], "exclude": True})
        assert len(filtered) == 4
        assert all("example0.com" not in r.url for r in filtered)

    def test_domain_filter_no_config_passthrough(self):
        """domain_filter with no config passes all results through."""
        from backend.pipeline_engine.registry import domain_filter

        results = make_results(5)
        filtered = domain_filter(results, None)
        assert len(filtered) == 5

    def test_dedup_transform(self):
        """dedup_transform removes duplicates by URL."""
        from backend.pipeline_engine.registry import dedup_transform

        results = [
            MockResult("doc-1", "https://example.com/a", "A", 0.9),
            MockResult("doc-2", "https://example.com/a", "A dup", 0.8),
            MockResult("doc-3", "https://example.com/b", "B", 0.7),
        ]
        deduped = dedup_transform(results, {"field": "url"})
        assert len(deduped) == 2
        assert deduped[0].document_id == "doc-1"
        assert deduped[1].document_id == "doc-3"

    def test_snippet_transform_passthrough(self):
        """snippet_transform passes results through unchanged."""
        from backend.pipeline_engine.registry import snippet_transform

        results = make_results(3)
        transformed = snippet_transform(results)
        assert len(transformed) == 3
