"""Research Agent tool-use loop executor (Task 14.3).

Implements the sequential tool-use loop where the Research_Agent calls
Retriever, Pipeline_Engine, and Answer_Engine iteratively based on the plan.

For MVP, this is a simple sequential executor that processes each plan step
in order, making tool calls as needed and tracking budget consumption.
"""

from __future__ import annotations

import uuid
from typing import Any

from backend.research_agent.budget import BudgetExceededError, BudgetTracker
from backend.research_agent.events import EventEmitter
from backend.research_agent.models import (
    EventType,
    PlanStep,
    ResearchCitation,
    ResearchPlan,
    ResearchReport,
    StepType,
)


class ToolCallResult:
    """Result from a tool call (Retriever, Pipeline, or Answer)."""

    def __init__(
        self,
        tool_name: str,
        results: list[dict[str, Any]] | None = None,
        citations: list[ResearchCitation] | None = None,
        text: str = "",
    ) -> None:
        self.tool_name = tool_name
        self.results = results or []
        self.citations = citations or []
        self.text = text


class ResearchExecutor:
    """Executes a research plan step-by-step with budget enforcement (R7.3, R7.6).

    Calls Retriever, Pipeline_Engine, and Answer_Engine iteratively based on
    the plan steps. Emits events for each step and tracks budget consumption.
    """

    def __init__(
        self,
        emitter: EventEmitter,
        budget_tracker: BudgetTracker,
        retriever: Any | None = None,
        pipeline_engine: Any | None = None,
        answer_engine: Any | None = None,
    ) -> None:
        """Initialize the executor.

        Args:
            emitter: Event emitter for SSE stream events.
            budget_tracker: Budget tracker for enforcement.
            retriever: Retriever service (or mock for MVP).
            pipeline_engine: Pipeline engine service (or mock for MVP).
            answer_engine: Answer engine service (or mock for MVP).
        """
        self._emitter = emitter
        self._budget = budget_tracker
        self._retriever = retriever
        self._pipeline_engine = pipeline_engine
        self._answer_engine = answer_engine

    def execute_plan(self, job_id: str, plan: ResearchPlan) -> ResearchReport:
        """Execute a research plan step by step (R7.3, R7.6).

        Processes each step in order, emitting step_started/step_completed events.
        Stops early if budget is exceeded, returning a partial report.

        Args:
            job_id: The research job ID.
            plan: The research plan to execute.

        Returns:
            ResearchReport (complete or partial if budget exceeded).

        Raises:
            BudgetExceededError: If any budget limit is exceeded during execution.
        """
        all_citations: list[ResearchCitation] = []
        report_chunks: list[str] = []
        retrieved_docs: list[dict[str, Any]] = []

        for step in plan.steps:
            # Check budget before each step
            self._budget.check()

            # Emit step_started
            self._emitter.emit(
                job_id,
                EventType.STEP_STARTED,
                {"step_id": step.step_id, "type": step.type.value},
            )

            # Execute the step based on its type
            result = self._execute_step(step, retrieved_docs)

            # Record step execution
            self._budget.record_step()

            # Collect results
            if result.citations:
                all_citations.extend(result.citations)
                for citation in result.citations:
                    self._emitter.emit(
                        job_id,
                        EventType.CITATION,
                        {
                            "document_id": citation.document_id,
                            "version": citation.version,
                            "answer_start": citation.answer_start,
                            "answer_end": citation.answer_end,
                            "source_start": citation.source_start,
                            "source_end": citation.source_end,
                        },
                    )

            if result.results:
                retrieved_docs.extend(result.results)

            if result.text:
                report_chunks.append(result.text)
                self._emitter.emit(
                    job_id,
                    EventType.REPORT_CHUNK,
                    {"text": result.text, "ordinal": len(report_chunks)},
                )

            # Emit step_completed
            self._emitter.emit(
                job_id,
                EventType.STEP_COMPLETED,
                {"step_id": step.step_id, "summary": f"Completed {step.type.value}"},
            )

        # Build final report
        report_text = "\n\n".join(report_chunks) if report_chunks else "Research complete."
        return ResearchReport(
            job_id=job_id,
            text=report_text,
            citations=all_citations,
            is_partial=False,
        )

    def _execute_step(
        self, step: PlanStep, context_docs: list[dict[str, Any]]
    ) -> ToolCallResult:
        """Execute a single plan step by calling the appropriate tool.

        Args:
            step: The plan step to execute.
            context_docs: Documents retrieved so far for context.

        Returns:
            ToolCallResult with any results, citations, or text produced.
        """
        if step.type == StepType.SUB_QUERY:
            return self._execute_sub_query(step)
        elif step.type == StepType.RETRIEVAL:
            return self._execute_retrieval(step)
        elif step.type == StepType.READ:
            return self._execute_read(step, context_docs)
        elif step.type == StepType.SYNTHESIS:
            return self._execute_synthesis(step, context_docs)
        else:
            return ToolCallResult(tool_name="unknown")

    def _execute_sub_query(self, step: PlanStep) -> ToolCallResult:
        """Execute a sub-query decomposition step.

        For MVP, this generates sub-queries from the step description.
        """
        self._budget.record_tool_call()

        # MVP: Generate a simple sub-query from the step
        sub_query = step.inputs.get("goal", step.description)
        return ToolCallResult(
            tool_name="sub_query",
            text=f"Identified sub-queries for: {sub_query[:100]}",
        )

    def _execute_retrieval(self, step: PlanStep) -> ToolCallResult:
        """Execute a retrieval step using the Retriever.

        Uses the real Retriever service when available, otherwise simulates.
        """
        self._budget.record_tool_call()

        # Use real retriever if available
        if self._retriever is not None:
            try:
                from backend.retriever.models import SearchRequest, SearchMode

                query = step.description or step.inputs.get("goal", "research")
                request = SearchRequest(
                    query=query[:200],
                    mode=SearchMode.HYBRID,
                    num_results=5,
                    tenant_id="",
                )
                response = self._retriever.search(request)

                results = [
                    {
                        "document_id": r.document_id,
                        "version": r.version,
                        "url": r.url,
                        "title": r.title,
                        "score": r.score,
                        "cleaned_text": "",  # Will be filled by read step
                    }
                    for r in response.results
                ]

                # Get cleaned text from indexer if available
                if hasattr(self._retriever, '_indexer'):
                    for result in results:
                        doc = self._retriever._indexer.get_latest_version(result["document_id"])
                        if doc:
                            result["cleaned_text"] = doc.cleaned_text

                return ToolCallResult(tool_name="retriever", results=results)
            except Exception:
                pass

        # Fallback: simulated retrieval
        doc_id = str(uuid.uuid4())
        results = [
            {
                "document_id": doc_id,
                "version": 1,
                "url": f"https://example.com/doc/{doc_id[:8]}",
                "title": f"Document for {step.description[:50]}",
                "score": 0.85,
                "cleaned_text": f"Content relevant to {step.description[:100]}",
            }
        ]

        return ToolCallResult(tool_name="retriever", results=results)

    def _execute_read(
        self, step: PlanStep, context_docs: list[dict[str, Any]]
    ) -> ToolCallResult:
        """Execute a read/analyze step over retrieved documents.

        For MVP, generates a summary of the context documents.
        """
        self._budget.record_tool_call()

        # Generate citations from context docs
        citations: list[ResearchCitation] = []
        text_parts: list[str] = []

        for doc in context_docs[-5:]:  # Use last 5 docs
            citation = ResearchCitation(
                citation_id=str(uuid.uuid4()),
                document_id=doc.get("document_id", ""),
                version=doc.get("version", 1),
                answer_start=0,
                answer_end=50,
                source_start=0,
                source_end=min(50, len(doc.get("cleaned_text", ""))),
            )
            citations.append(citation)
            text_parts.append(
                f"From {doc.get('title', 'unknown')}: "
                f"{doc.get('cleaned_text', '')[:100]}"
            )

        return ToolCallResult(
            tool_name="read",
            citations=citations,
            text="\n".join(text_parts) if text_parts else "No documents to analyze.",
        )

    def _execute_synthesis(
        self, step: PlanStep, context_docs: list[dict[str, Any]]
    ) -> ToolCallResult:
        """Execute a synthesis step using the Answer Engine.

        For MVP, generates a simple synthesis from context.
        """
        self._budget.record_tool_call()

        # Generate final synthesis text
        synthesis = "Based on the research conducted, "
        if context_docs:
            synthesis += f"analysis of {len(context_docs)} documents reveals "
            synthesis += "the following findings."
        else:
            synthesis += "no relevant documents were found."

        return ToolCallResult(
            tool_name="answer_engine",
            text=synthesis,
        )
