"""Multi-step financial research workflow using LangGraph-style execution."""
from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Optional, Any

from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class WorkflowState:
    """Mutable workflow state passed between steps."""
    query: str
    ticker: Optional[str] = None
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Retrieved documents
    sec_documents: list[dict] = field(default_factory=list)
    earnings_documents: list[dict] = field(default_factory=list)
    research_documents: list[dict] = field(default_factory=list)

    # Generated content
    sec_analysis: Optional[str] = None
    earnings_analysis: Optional[str] = None
    research_synthesis: Optional[str] = None
    executive_summary: Optional[str] = None

    # Evaluation
    grounding_score: Optional[float] = None
    hallucination_risk: Optional[str] = None

    # Metadata
    total_tokens: int = 0
    steps_completed: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class FinancialResearchWorkflow:
    """
    Multi-step financial research workflow.

    Steps:
    1. SEC filing analysis
    2. Earnings transcript analysis
    3. Research synthesis
    4. Executive summary generation
    5. Grounding evaluation
    """

    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator

    async def run(
        self,
        query: str,
        ticker: Optional[str] = None,
        steps: Optional[list[str]] = None,
    ) -> WorkflowState:
        """Execute the full research workflow."""
        state = WorkflowState(query=query, ticker=ticker)
        default_steps = ["sec_analysis", "earnings_analysis", "synthesis", "summary", "evaluation"]
        workflow_steps = steps or default_steps

        logger.info(
            "Research workflow started",
            extra={
                "query": query[:100],
                "ticker": ticker,
                "steps": workflow_steps,
                "session_id": state.session_id,
            },
        )

        for step in workflow_steps:
            try:
                state = await self._execute_step(step, state)
                state.steps_completed.append(step)
            except Exception as e:
                error_msg = f"Step '{step}' failed: {str(e)}"
                logger.error(error_msg, exc_info=True)
                state.errors.append(error_msg)

        logger.info(
            "Research workflow completed",
            extra={
                "session_id": state.session_id,
                "steps_completed": len(state.steps_completed),
                "errors": len(state.errors),
                "total_tokens": state.total_tokens,
            },
        )
        return state

    async def _execute_step(self, step: str, state: WorkflowState) -> WorkflowState:
        """Execute a single workflow step."""
        if not self.orchestrator:
            logger.warning("Orchestrator not configured, skipping step")
            return state

        if step == "sec_analysis":
            response = await self.orchestrator.route(
                query=state.query,
                agent_name="sec_filing_agent",
                ticker=state.ticker,
            )
            state.sec_analysis = response.content
            state.total_tokens += response.tokens_used

        elif step == "earnings_analysis":
            response = await self.orchestrator.route(
                query=state.query,
                agent_name="earnings_agent",
                ticker=state.ticker,
            )
            state.earnings_analysis = response.content
            state.total_tokens += response.tokens_used

        elif step == "synthesis":
            combined_query = (
                f"Synthesize the following analyses for {state.ticker or 'the company'}: {state.query}"
            )
            response = await self.orchestrator.route(
                query=combined_query,
                agent_name="research_agent",
                ticker=state.ticker,
            )
            state.research_synthesis = response.content
            state.total_tokens += response.tokens_used

        elif step == "summary":
            response = await self.orchestrator.route(
                query=state.query,
                agent_name="executive_summary_agent",
                ticker=state.ticker,
            )
            state.executive_summary = response.content
            state.total_tokens += response.tokens_used

        elif step == "evaluation":
            final_content = state.research_synthesis or state.sec_analysis or ""
            if final_content:
                response = await self.orchestrator.route(
                    query=state.query,
                    agent_name="evaluation_agent",
                    response=final_content,
                    context_documents=state.sec_documents,
                )
                state.grounding_score = response.grounding_score
                state.hallucination_risk = response.hallucination_risk
                state.total_tokens += response.tokens_used

        return state
