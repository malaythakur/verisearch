"""Pipeline step registry (Task 12.1, R9.1, R9.2).

Maintains a registry of available pipeline steps (filters, rerankers, transforms).
Each step is a callable that takes a list of search results and optional config,
and returns a modified list of search results.

Built-in steps are registered at module load time.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol

from backend.pipeline_engine.models import StepType


class StepFunction(Protocol):
    """Protocol for pipeline step functions.

    A step function takes a list of results and optional config,
    and returns a modified list of results.
    """

    def __call__(self, results: list[Any], config: dict[str, Any] | None = None) -> list[Any]: ...


class StepRegistryEntry:
    """An entry in the step registry.

    Attributes:
        name: Unique name of the step.
        step_type: Type of step (filter, reranker, transform).
        fn: The step function to execute.
        description: Human-readable description.
    """

    def __init__(
        self,
        name: str,
        step_type: StepType,
        fn: StepFunction,
        description: str = "",
    ) -> None:
        self.name = name
        self.step_type = step_type
        self.fn = fn
        self.description = description


class PipelineRegistry:
    """Registry of available pipeline steps (R9.1, R9.2).

    Maintains a mapping of step names to their implementations.
    Used to validate pipeline definitions and execute steps.
    """

    def __init__(self) -> None:
        self._steps: dict[str, StepRegistryEntry] = {}

    def register(
        self,
        name: str,
        step_type: StepType,
        fn: StepFunction,
        description: str = "",
    ) -> None:
        """Register a new step in the registry.

        Args:
            name: Unique name for the step.
            step_type: Type of step (filter, reranker, transform).
            fn: The step function.
            description: Human-readable description.
        """
        self._steps[name] = StepRegistryEntry(
            name=name,
            step_type=step_type,
            fn=fn,
            description=description,
        )

    def get(self, name: str) -> StepRegistryEntry | None:
        """Look up a step by name.

        Args:
            name: The step name to look up.

        Returns:
            The registry entry, or None if not found.
        """
        return self._steps.get(name)

    def exists(self, name: str) -> bool:
        """Check if a step name exists in the registry.

        Args:
            name: The step name to check.

        Returns:
            True if the step exists.
        """
        return name in self._steps

    def get_unknown_steps(self, names: list[str]) -> list[str]:
        """Find step names that don't exist in the registry (R9.2).

        Args:
            names: List of step names to validate.

        Returns:
            List of names not found in the registry.
        """
        return [name for name in names if name not in self._steps]

    def list_steps(self) -> list[StepRegistryEntry]:
        """List all registered steps.

        Returns:
            All registry entries.
        """
        return list(self._steps.values())

    @property
    def step_names(self) -> set[str]:
        """Get all registered step names."""
        return set(self._steps.keys())


# ---------------------------------------------------------------------------
# Built-in step implementations
# ---------------------------------------------------------------------------


def domain_filter(results: list[Any], config: dict[str, Any] | None = None) -> list[Any]:
    """Filter results by domain.

    Config:
        domains: List of allowed domains. If empty, passes all through.
        exclude: If True, exclude matching domains instead of including.
    """
    if not config:
        return results

    domains = config.get("domains", [])
    exclude = config.get("exclude", False)

    if not domains:
        return results

    filtered = []
    for result in results:
        url = getattr(result, "url", "") or ""
        domain_match = any(d in url for d in domains)
        if exclude and not domain_match:
            filtered.append(result)
        elif not exclude and domain_match:
            filtered.append(result)

    return filtered


def freshness_filter(results: list[Any], config: dict[str, Any] | None = None) -> list[Any]:
    """Filter results by publication freshness.

    Config:
        max_age_days: Maximum age in days. Results older than this are excluded.
    """
    if not config:
        return results

    from datetime import timedelta

    max_age_days = config.get("max_age_days")
    if max_age_days is None:
        return results

    from datetime import datetime, timezone

    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)

    filtered = []
    for result in results:
        published_at = getattr(result, "published_at", None)
        if published_at is None:
            # Include results without a publication date
            filtered.append(result)
        elif published_at >= cutoff:
            filtered.append(result)

    return filtered


def language_filter(results: list[Any], config: dict[str, Any] | None = None) -> list[Any]:
    """Filter results by language.

    Config:
        languages: List of allowed language codes (e.g., ['en', 'fr']).
    """
    if not config:
        return results

    languages = config.get("languages", [])
    if not languages:
        return results

    filtered = []
    for result in results:
        lang = getattr(result, "language", None)
        if lang is None or lang in languages:
            filtered.append(result)

    return filtered


def score_reranker(results: list[Any], config: dict[str, Any] | None = None) -> list[Any]:
    """Rerank results by adjusting scores with a boost factor.

    Config:
        boost_factor: Multiplier for scores (default 1.0).
        boost_domains: List of domains to boost.
    """
    if not config:
        return results

    boost_factor = config.get("boost_factor", 1.0)
    boost_domains = config.get("boost_domains", [])

    reranked = []
    for result in results:
        url = getattr(result, "url", "") or ""
        score = getattr(result, "score", 0.0)

        if boost_domains and any(d in url for d in boost_domains):
            new_score = min(1.0, score * boost_factor)
        else:
            new_score = score

        # Create a copy with the new score if it changed
        if new_score != score and hasattr(result, "__class__"):
            from dataclasses import fields, asdict

            try:
                field_names = {f.name for f in fields(result)}
                kwargs = {f.name: getattr(result, f.name) for f in fields(result)}
                kwargs["score"] = new_score
                reranked.append(result.__class__(**kwargs))
            except (TypeError, AttributeError):
                reranked.append(result)
        else:
            reranked.append(result)

    # Sort by score descending
    reranked.sort(key=lambda r: getattr(r, "score", 0.0), reverse=True)
    return reranked


def credibility_reranker(results: list[Any], config: dict[str, Any] | None = None) -> list[Any]:
    """Rerank results by blending relevance score with credibility.

    Config:
        credibility_weight: Weight for credibility score (0.0-1.0, default 0.3).
    """
    if not config:
        config = {}

    weight = config.get("credibility_weight", 0.3)

    reranked = []
    for result in results:
        score = getattr(result, "score", 0.0)
        provenance = getattr(result, "provenance", None)
        credibility = getattr(provenance, "credibility_score", 0.5) if provenance else 0.5

        blended_score = min(1.0, (1 - weight) * score + weight * credibility)

        if hasattr(result, "__class__"):
            from dataclasses import fields

            try:
                kwargs = {f.name: getattr(result, f.name) for f in fields(result)}
                kwargs["score"] = blended_score
                reranked.append(result.__class__(**kwargs))
            except (TypeError, AttributeError):
                reranked.append(result)
        else:
            reranked.append(result)

    reranked.sort(key=lambda r: getattr(r, "score", 0.0), reverse=True)
    return reranked


def reciprocal_rank_reranker(results: list[Any], config: dict[str, Any] | None = None) -> list[Any]:
    """Rerank using reciprocal rank fusion with original positions.

    Config:
        k: RRF constant (default 60).
    """
    if not config:
        config = {}

    k = config.get("k", 60)

    # Apply RRF scoring based on current position
    reranked = []
    for i, result in enumerate(results):
        rrf_score = 1.0 / (k + i + 1)

        if hasattr(result, "__class__"):
            from dataclasses import fields

            try:
                kwargs = {f.name: getattr(result, f.name) for f in fields(result)}
                kwargs["score"] = min(1.0, rrf_score)
                reranked.append(result.__class__(**kwargs))
            except (TypeError, AttributeError):
                reranked.append(result)
        else:
            reranked.append(result)

    reranked.sort(key=lambda r: getattr(r, "score", 0.0), reverse=True)
    return reranked


def title_transform(results: list[Any], config: dict[str, Any] | None = None) -> list[Any]:
    """Transform result titles (e.g., truncate, prefix).

    Config:
        max_length: Maximum title length (default 100).
        prefix: Optional prefix to add.
        suffix: Optional suffix to add.
    """
    if not config:
        return results

    max_length = config.get("max_length", 100)
    prefix = config.get("prefix", "")
    suffix = config.get("suffix", "")

    transformed = []
    for result in results:
        title = getattr(result, "title", "") or ""
        new_title = f"{prefix}{title}{suffix}"
        if len(new_title) > max_length:
            new_title = new_title[:max_length]

        if new_title != title and hasattr(result, "__class__"):
            from dataclasses import fields

            try:
                kwargs = {f.name: getattr(result, f.name) for f in fields(result)}
                kwargs["title"] = new_title
                transformed.append(result.__class__(**kwargs))
            except (TypeError, AttributeError):
                transformed.append(result)
        else:
            transformed.append(result)

    return transformed


def snippet_transform(results: list[Any], config: dict[str, Any] | None = None) -> list[Any]:
    """Transform results by adding/modifying snippet fields.

    This is a pass-through transform that preserves results unchanged.
    In production, it would extract and attach text snippets.
    """
    # Pass-through: results are unchanged
    return list(results)


def dedup_transform(results: list[Any], config: dict[str, Any] | None = None) -> list[Any]:
    """Remove duplicate results based on URL.

    Config:
        field: Field to deduplicate on (default 'url').
    """
    if not config:
        config = {}

    dedup_field = config.get("field", "url")

    seen: set[str] = set()
    deduped = []
    for result in results:
        value = str(getattr(result, dedup_field, ""))
        if value not in seen:
            seen.add(value)
            deduped.append(result)

    return deduped


# ---------------------------------------------------------------------------
# Default registry with built-in steps
# ---------------------------------------------------------------------------


def create_default_registry() -> PipelineRegistry:
    """Create a registry with all built-in steps.

    Returns:
        A PipelineRegistry populated with built-in filters, rerankers, and transforms.
    """
    registry = PipelineRegistry()

    # Filters
    registry.register("domain_filter", StepType.FILTER, domain_filter, "Filter results by domain")
    registry.register("freshness_filter", StepType.FILTER, freshness_filter, "Filter by publication freshness")
    registry.register("language_filter", StepType.FILTER, language_filter, "Filter by language")

    # Rerankers
    registry.register("score_reranker", StepType.RERANKER, score_reranker, "Rerank with score boost")
    registry.register("credibility_reranker", StepType.RERANKER, credibility_reranker, "Rerank by credibility blend")
    registry.register(
        "reciprocal_rank_reranker", StepType.RERANKER, reciprocal_rank_reranker, "Rerank using RRF"
    )

    # Transforms
    registry.register("title_transform", StepType.TRANSFORM, title_transform, "Transform result titles")
    registry.register("snippet_transform", StepType.TRANSFORM, snippet_transform, "Add/modify snippets")
    registry.register("dedup_transform", StepType.TRANSFORM, dedup_transform, "Remove duplicate results")

    return registry
