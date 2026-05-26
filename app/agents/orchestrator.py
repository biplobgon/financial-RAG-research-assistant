"""Multi-agent orchestration router."""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from app.agents.base_agent import BaseFinancialAgent, AgentResponse
from app.config.constants import (
    AGENT_SEC_FILING,
    AGENT_EARNINGS,
    AGENT_PORTFOLIO,
    AGENT_EXECUTIVE_SUMMARY,
    AGENT_RESEARCH,
    AGENT_EVALUATION,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Intent-to-agent routing keyword map.
# Earlier entries take precedence (checked in order).
_ROUTING_RULES: list[tuple[str, list[str]]] = [
    (
        AGENT_SEC_FILING,
        [
            "10-k", "10-q", "10k", "10q", "sec filing", "annual report",
            "quarterly report", "risk factor", "item 1", "item 7", "item 8",
            "md&a", "management discussion", "management's discussion",
            "form 10", "edgar", "reg s-k", "financial statement",
        ],
    ),
    (
        AGENT_EARNINGS,
        [
            "earnings call", "transcript", "quarterly results", "guidance",
            "earnings per share", "eps beat", "eps miss", "revenue beat",
            "revenue miss", "q&a session", "management commentary",
            "analyst question", "conference call", "fiscal quarter",
        ],
    ),
    (
        AGENT_PORTFOLIO,
        [
            "portfolio", "holdings", "position", "allocation", "diversification",
            "benchmark", "sector exposure", "concentration", "weighting",
            "rebalance", "factor exposure", "asset allocation",
        ],
    ),
    (
        AGENT_EXECUTIVE_SUMMARY,
        [
            "summarize", "summary", "brief", "executive summary", "overview",
            "key points", "highlights", "tldr", "tl;dr", "give me a summary",
            "quick summary",
        ],
    ),
    (
        AGENT_EVALUATION,
        [
            "evaluate", "check", "verify", "hallucination", "accuracy check",
            "grounding", "quality score", "fact check", "is this correct",
            "validate response",
        ],
    ),
]


class AgentOrchestrator:
    """
    Routes financial analysis requests to the appropriate specialized agent.

    Features:
    - Keyword-based intent detection with ordered precedence rules
    - Direct named routing (bypass auto-detection)
    - Parallel multi-agent fan-out execution
    - Graceful fallback to ResearchAgent for unrecognized intent
    - Per-agent routing statistics for observability
    - Agent registration / hot-swap support
    """

    def __init__(
        self,
        agents: Optional[dict[str, BaseFinancialAgent]] = None,
        llm=None,
        retriever=None,
        evaluator=None,
        rag_pipeline=None,
    ):
        self.agents: dict[str, BaseFinancialAgent] = agents or {}
        self.llm = llm
        self.retriever = retriever
        self.evaluator = evaluator
        self.rag_pipeline = rag_pipeline
        self._routing_count: dict[str, int] = {}
        self._fallback_count: int = 0

    # ------------------------------------------------------------------
    # Agent registration
    # ------------------------------------------------------------------

    def register_agent(self, agent: BaseFinancialAgent) -> None:
        """Register or replace a named agent."""
        self.agents[agent.AGENT_NAME] = agent
        logger.info("Agent registered", extra={"agent": agent.AGENT_NAME})

    def unregister_agent(self, agent_name: str) -> None:
        """Remove a registered agent."""
        if agent_name in self.agents:
            del self.agents[agent_name]
            logger.info("Agent unregistered", extra={"agent": agent_name})

    def list_agents(self) -> list[str]:
        """Return list of currently registered agent names."""
        return list(self.agents.keys())

    # ------------------------------------------------------------------
    # Core routing interface
    # ------------------------------------------------------------------

    async def route(
        self,
        query: str,
        agent_name: Optional[str] = None,
        **kwargs,
    ) -> AgentResponse:
        """
        Route a query to the appropriate agent.

        Args:
            query: User's financial analysis question.
            agent_name: If provided, route directly to this agent (no intent detection).
            **kwargs: Passed through to the target agent's _execute() method.

        Returns:
            AgentResponse from the selected agent.
        """
        if not query or not query.strip():
            return AgentResponse(
                content="Query cannot be empty.",
                agent_name="orchestrator",
                trace_id="",
                latency_ms=0.0,
                status="error",
                error="empty query",
            )

        # Direct routing when agent is explicitly specified
        if agent_name:
            if agent_name in self.agents:
                return await self._run_agent(agent_name, query, **kwargs)
            logger.warning(
                "Requested agent not registered, falling back to auto-routing",
                extra={"requested_agent": agent_name},
            )

        # Auto-routing by intent detection
        detected_agent = self._detect_intent(query)
        logger.info(
            "Auto-routing decision",
            extra={
                "detected_agent": detected_agent,
                "query_preview": query[:120],
            },
        )
        return await self._run_agent(detected_agent, query, **kwargs)

    async def run_parallel(
        self,
        query: str,
        agent_names: list[str],
        **kwargs,
    ) -> dict[str, AgentResponse]:
        """
        Execute multiple agents in parallel against the same query.

        Args:
            query: The financial analysis question.
            agent_names: List of agent names to run concurrently.
            **kwargs: Passed through to each agent.

        Returns:
            Dict mapping agent_name → AgentResponse (includes error responses
            for any agents that failed or were not registered).
        """
        valid_agents = [a for a in agent_names if a in self.agents]
        missing = [a for a in agent_names if a not in self.agents]

        if missing:
            logger.warning(
                "Some agents not registered for parallel run",
                extra={"missing": missing, "running": valid_agents},
            )

        if not valid_agents:
            return {
                name: AgentResponse(
                    content=f"Agent '{name}' is not registered.",
                    agent_name=name,
                    trace_id="",
                    latency_ms=0.0,
                    status="error",
                    error="agent not registered",
                )
                for name in agent_names
            }

        # Fan out concurrently
        results = await asyncio.gather(
            *[self._run_agent(name, query, **kwargs) for name in valid_agents],
            return_exceptions=True,
        )

        output: dict[str, AgentResponse] = {}
        for name, result in zip(valid_agents, results):
            if isinstance(result, Exception):
                logger.error(
                    "Parallel agent execution failed",
                    extra={"agent": name, "error": str(result)},
                    exc_info=result,
                )
                output[name] = AgentResponse(
                    content=f"Agent {name} raised an exception: {result}",
                    agent_name=name,
                    trace_id="",
                    latency_ms=0.0,
                    status="error",
                    error=str(result),
                )
            else:
                output[name] = result  # type: ignore[assignment]

        # Add error stubs for missing agents
        for name in missing:
            output[name] = AgentResponse(
                content=f"Agent '{name}' is not registered.",
                agent_name=name,
                trace_id="",
                latency_ms=0.0,
                status="error",
                error="agent not registered",
            )

        return output

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _run_agent(
        self, agent_name: str, query: str, **kwargs
    ) -> AgentResponse:
        """
        Execute a named agent, falling back to ResearchAgent if not found.
        """
        agent = self.agents.get(agent_name)
        if not agent:
            fallback = self.agents.get(AGENT_RESEARCH)
            if fallback:
                self._fallback_count += 1
                logger.warning(
                    "Agent not found — falling back to research_agent",
                    extra={"requested": agent_name},
                )
                self._routing_count[AGENT_RESEARCH] = (
                    self._routing_count.get(AGENT_RESEARCH, 0) + 1
                )
                return await fallback.run(query, **kwargs)

            return AgentResponse(
                content=f"Agent '{agent_name}' is not available and no fallback is registered.",
                agent_name=agent_name,
                trace_id="",
                latency_ms=0.0,
                status="error",
                error=f"agent not registered: {agent_name}",
            )

        self._routing_count[agent_name] = self._routing_count.get(agent_name, 0) + 1
        return await agent.run(query, **kwargs)

    def _detect_intent(self, query: str) -> str:
        """
        Determine the most appropriate agent for a query using keyword matching.

        Rules are evaluated in priority order (_ROUTING_RULES).
        Falls back to AGENT_RESEARCH if no rule matches.
        """
        query_lower = query.lower()
        for agent_name, keywords in _ROUTING_RULES:
            if any(kw in query_lower for kw in keywords):
                return agent_name
        return AGENT_RESEARCH

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    @property
    def routing_stats(self) -> dict:
        """Return orchestrator-level routing statistics."""
        total = sum(self._routing_count.values())
        return {
            "registered_agents": list(self.agents.keys()),
            "routing_counts": dict(self._routing_count),
            "fallback_count": self._fallback_count,
            "total_queries_routed": total,
            "routing_distribution": {
                k: round(v / total, 4) if total > 0 else 0.0
                for k, v in self._routing_count.items()
            },
        }

    @property
    def agent_metrics(self) -> dict[str, dict]:
        """Aggregate per-agent metrics from all registered agents."""
        return {name: agent.metrics for name, agent in self.agents.items()}

    def __repr__(self) -> str:
        return (
            f"AgentOrchestrator("
            f"agents={list(self.agents.keys())}, "
            f"total_routed={sum(self._routing_count.values())})"
        )
