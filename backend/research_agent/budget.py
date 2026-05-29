"""Research Agent budget enforcement (R7.6).

Enforces:
- max_steps: Maximum number of plan steps executed.
- max_duration_ms: Maximum wall-clock time for the job.
- max_tool_calls: Maximum number of tool invocations (Retriever, Pipeline, Answer).

When any budget is exceeded, the job terminates with `budget_exceeded` error
and the partial report + citations are made available via GET /v1/research/{job_id}.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from backend.research_agent.models import BudgetConfig


class BudgetExceededError(Exception):
    """Raised when a research job exceeds its budget (R7.6).

    Attributes:
        reason: Which budget was exceeded (max_steps, max_duration_ms, max_tool_calls).
        current_value: The current value that exceeded the budget.
        limit: The configured limit.
    """

    def __init__(self, reason: str, current_value: int, limit: int) -> None:
        self.reason = reason
        self.current_value = current_value
        self.limit = limit
        super().__init__(
            f"Budget exceeded: {reason} ({current_value} > {limit})"
        )


@dataclass
class BudgetTracker:
    """Tracks resource consumption against configured budgets (R7.6).

    Call check() before each operation to verify the budget hasn't been exceeded.
    Call record_step() / record_tool_call() after each operation.
    """

    config: BudgetConfig
    steps_used: int = 0
    tool_calls_used: int = 0
    start_time_ms: float = 0.0

    def start(self) -> None:
        """Mark the start of job execution for duration tracking."""
        self.start_time_ms = time.time() * 1000

    def elapsed_ms(self) -> float:
        """Get elapsed time since job start in milliseconds."""
        if self.start_time_ms == 0:
            return 0.0
        return (time.time() * 1000) - self.start_time_ms

    def check(self) -> None:
        """Check all budgets and raise BudgetExceededError if any is exceeded.

        Raises:
            BudgetExceededError: If any budget limit has been reached or exceeded.
        """
        # Check steps
        if self.steps_used >= self.config.max_steps:
            raise BudgetExceededError(
                reason="max_steps",
                current_value=self.steps_used,
                limit=self.config.max_steps,
            )

        # Check duration
        elapsed = self.elapsed_ms()
        if self.start_time_ms > 0 and elapsed >= self.config.max_duration_ms:
            raise BudgetExceededError(
                reason="max_duration_ms",
                current_value=int(elapsed),
                limit=self.config.max_duration_ms,
            )

        # Check tool calls
        if self.tool_calls_used >= self.config.max_tool_calls:
            raise BudgetExceededError(
                reason="max_tool_calls",
                current_value=self.tool_calls_used,
                limit=self.config.max_tool_calls,
            )

    def record_step(self) -> None:
        """Record that a plan step has been executed."""
        self.steps_used += 1

    def record_tool_call(self) -> None:
        """Record that a tool call has been made."""
        self.tool_calls_used += 1

    def is_exceeded(self) -> bool:
        """Check if any budget is exceeded without raising.

        Returns:
            True if any budget limit has been reached or exceeded.
        """
        try:
            self.check()
            return False
        except BudgetExceededError:
            return True
