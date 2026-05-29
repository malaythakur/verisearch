"""Research Agent plan generation (R7.2).

Generates a Research_Plan with 1-32 steps as the first event before any retrieval.
Each step is labeled with a type from: sub_query, retrieval, read, synthesis.

The planner analyzes the research goal and produces a structured plan that
the executor will follow during the tool-use loop.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from backend.research_agent.models import PlanStep, ResearchPlan, StepType


# Maximum and minimum plan steps (R7.2)
MIN_PLAN_STEPS = 1
MAX_PLAN_STEPS = 32


class PlanGenerationError(Exception):
    """Raised when plan generation fails."""

    pass


class ResearchPlanner:
    """Generates research plans from a research goal (R7.2).

    Analyzes the goal and produces a structured plan with typed steps.
    For MVP, uses a heuristic approach; production would use an LLM.
    """

    def __init__(self, max_steps: int = MAX_PLAN_STEPS) -> None:
        """Initialize the planner.

        Args:
            max_steps: Maximum number of steps to generate (capped at 32).
        """
        self._max_steps = min(max_steps, MAX_PLAN_STEPS)

    def generate_plan(self, job_id: str, research_goal: str) -> ResearchPlan:
        """Generate a research plan for the given goal (R7.2).

        The plan always has between 1 and 32 steps, each with a typed label.
        The plan is emitted as the first event before any retrieval.

        Args:
            job_id: The research job ID this plan belongs to.
            research_goal: The user's research goal (1-4096 chars).

        Returns:
            A ResearchPlan with 1-32 typed steps.

        Raises:
            PlanGenerationError: If plan generation fails.
        """
        steps = self._decompose_goal(research_goal)

        # Ensure we have at least 1 step and at most max_steps
        if not steps:
            steps = [
                PlanStep(
                    step_id=str(uuid.uuid4()),
                    type=StepType.RETRIEVAL,
                    description=f"Search for: {research_goal[:200]}",
                )
            ]

        if len(steps) > self._max_steps:
            steps = steps[: self._max_steps]

        plan = ResearchPlan(
            job_id=job_id,
            steps=steps,
            emitted_at=datetime.now(timezone.utc),
        )

        if not plan.validate():
            raise PlanGenerationError(
                f"Generated plan has {len(plan.steps)} steps, "
                f"expected 1-{MAX_PLAN_STEPS}"
            )

        return plan

    def _decompose_goal(self, research_goal: str) -> list[PlanStep]:
        """Decompose a research goal into typed plan steps.

        Uses LLM-based planning when OPENAI_API_KEY is set, otherwise
        falls back to heuristic decomposition.
        """
        import os

        if os.environ.get("OPENAI_API_KEY"):
            try:
                return self._llm_decompose(research_goal)
            except Exception:
                pass

        return self._heuristic_decompose(research_goal)

    def _llm_decompose(self, research_goal: str) -> list[PlanStep]:
        """Use LLM to generate a research plan."""
        import json
        import openai
        import os

        client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        response = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": (
                    "You are a research planning assistant. Given a research goal, "
                    "decompose it into a structured plan of 3-10 steps. "
                    "Each step must have a 'type' (one of: sub_query, retrieval, read, synthesis) "
                    "and a 'description' explaining what to do. "
                    "Return ONLY a JSON array of steps like: "
                    '[{"type": "sub_query", "description": "..."}, {"type": "retrieval", "description": "..."}]'
                    "\n\nRules:\n"
                    "- Start with sub_query to break down the goal\n"
                    "- Use retrieval to search for information\n"
                    "- Use read to analyze retrieved documents\n"
                    "- End with synthesis to compile findings\n"
                    "- Keep descriptions concise (under 100 chars)"
                )},
                {"role": "user", "content": research_goal[:2000]},
            ],
            temperature=0.2,
            max_tokens=1000,
        )

        content = response.choices[0].message.content.strip()
        # Handle markdown code blocks
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        steps_data = json.loads(content)

        steps = []
        type_map = {
            "sub_query": StepType.SUB_QUERY,
            "retrieval": StepType.RETRIEVAL,
            "read": StepType.READ,
            "synthesis": StepType.SYNTHESIS,
        }

        for step_data in steps_data:
            step_type = type_map.get(step_data.get("type", "retrieval"), StepType.RETRIEVAL)
            steps.append(PlanStep(
                step_id=str(uuid.uuid4()),
                type=step_type,
                description=step_data.get("description", ""),
            ))

        return steps if steps else self._heuristic_decompose(research_goal)

    def _heuristic_decompose(self, research_goal: str) -> list[PlanStep]:
        """Heuristic decomposition of a research goal into typed plan steps.

        Uses goal length and structure to determine plan complexity.
        """
        goal_length = len(research_goal)

        if goal_length < 50:
            num_retrieval_rounds = 1
        elif goal_length < 200:
            num_retrieval_rounds = 2
        elif goal_length < 500:
            num_retrieval_rounds = 3
        else:
            num_retrieval_rounds = min(4, self._max_steps // 4)

        steps: list[PlanStep] = []

        # Step 1: Decompose the goal into sub-queries
        steps.append(
            PlanStep(
                step_id=str(uuid.uuid4()),
                type=StepType.SUB_QUERY,
                description="Decompose research goal into sub-queries",
                inputs={"goal": research_goal[:500]},
            )
        )

        # Steps 2-N: Retrieval and read rounds
        for i in range(num_retrieval_rounds):
            steps.append(
                PlanStep(
                    step_id=str(uuid.uuid4()),
                    type=StepType.RETRIEVAL,
                    description=f"Retrieve documents for sub-query {i + 1}",
                )
            )
            steps.append(
                PlanStep(
                    step_id=str(uuid.uuid4()),
                    type=StepType.READ,
                    description=f"Analyze retrieved documents from round {i + 1}",
                )
            )

        # Final step: Synthesis
        steps.append(
            PlanStep(
                step_id=str(uuid.uuid4()),
                type=StepType.SYNTHESIS,
                description="Synthesize findings into final report",
            )
        )

        return steps
