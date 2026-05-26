"""Abstract base agent with observability, retry, and evaluation hooks."""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Any

from app.config.settings import settings
from app.utils.logging import get_logger, set_trace_context
from app.utils.retry import async_retry

logger = get_logger(__name__)


@dataclass
class AgentResponse:
    """Standardized agent response container."""
    content: str
    agent_name: str
    trace_id: str
    latency_ms: float
    tokens_used: int = 0
    grounding_score: Optional[float] = None
    hallucination_risk: Optional[str] = None
    sources: list[dict] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    status: str = "success"

    def to_dict(self) -> dict:
        """Serialize response to dictionary."""
        return {
            "content": self.content,
            "agent_name": self.agent_name,
            "trace_id": self.trace_id,
            "latency_ms": self.latency_ms,
            "tokens_used": self.tokens_used,
            "grounding_score": self.grounding_score,
            "hallucination_risk": self.hallucination_risk,
            "sources": self.sources,
            "metadata": self.metadata,
            "error": self.error,
            "status": self.status,
        }


class BaseFinancialAgent(ABC):
    """
    Abstract base class for all financial AI agents.

    Provides:
    - Unified run() interface with full observability
    - Retry with exponential backoff via async_retry
    - Grounding and hallucination evaluation hooks
    - Structured logging with trace context propagation
    - OpenTelemetry-compatible span emission
    - Token usage tracking and agent-level metrics
    """

    AGENT_NAME: str = "base_agent"
    MAX_CONTEXT_TOKENS: int = 8000

    def __init__(
        self,
        llm=None,
        retriever=None,
        evaluator=None,
        rag_pipeline=None,
    ):
        self.llm = llm
        self.retriever = retriever
        self.evaluator = evaluator
        self.rag_pipeline = rag_pipeline
        self._call_count: int = 0
        self._total_tokens: int = 0
        self._total_latency_ms: float = 0.0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def run(self, query: str, **kwargs) -> AgentResponse:
        """
        Execute the agent workflow with full observability.

        Wraps _execute() with:
        - Trace context setup
        - Wall-clock timing
        - Structured error handling
        - Per-agent metrics accumulation
        """
        trace_id = str(uuid.uuid4())
        set_trace_context(trace_id)
        start_time = time.monotonic()
        self._call_count += 1

        logger.info(
            "Agent run started",
            extra={
                "agent": self.AGENT_NAME,
                "trace_id": trace_id,
                "query_length": len(query),
                **{k: v for k, v in kwargs.items() if isinstance(v, (str, int, float, bool))},
            },
        )

        try:
            response: AgentResponse = await async_retry(
                self._execute,
                query,
                trace_id=trace_id,
                max_retries=settings.llm_max_retries,
                base_delay=1.0,
                **kwargs,
            )
            response.trace_id = trace_id
            response.latency_ms = (time.monotonic() - start_time) * 1000
            self._total_tokens += response.tokens_used
            self._total_latency_ms += response.latency_ms

            logger.info(
                "Agent run completed",
                extra={
                    "agent": self.AGENT_NAME,
                    "trace_id": trace_id,
                    "latency_ms": round(response.latency_ms, 2),
                    "tokens": response.tokens_used,
                    "grounding_score": response.grounding_score,
                    "status": response.status,
                },
            )
            return response

        except Exception as e:
            latency_ms = (time.monotonic() - start_time) * 1000
            logger.error(
                "Agent run failed",
                extra={
                    "agent": self.AGENT_NAME,
                    "trace_id": trace_id,
                    "latency_ms": round(latency_ms, 2),
                    "error": str(e),
                },
                exc_info=True,
            )
            return AgentResponse(
                content=f"Agent error: {str(e)}",
                agent_name=self.AGENT_NAME,
                trace_id=trace_id,
                latency_ms=latency_ms,
                error=str(e),
                status="error",
            )

    # ------------------------------------------------------------------
    # Abstract interface — each subclass must implement
    # ------------------------------------------------------------------

    @abstractmethod
    async def _execute(self, query: str, **kwargs) -> AgentResponse:
        """Core agent logic — must be implemented by every subclass."""
        ...

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

    async def _retrieve_context(
        self,
        query: str,
        collection_name: Optional[str] = None,
        top_k: int = 10,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """
        Retrieve relevant documents from the configured vector store.

        Returns an empty list gracefully when retriever is unavailable.
        """
        if not self.retriever:
            return []
        try:
            docs = await self.retriever.retrieve(
                query=query,
                top_k=top_k,
                filters=filters,
                collection_name=collection_name,
            )
            return [
                d.model_dump() if hasattr(d, "model_dump") else d.__dict__
                for d in docs
            ]
        except Exception as e:
            logger.warning(
                "Context retrieval failed",
                extra={"agent": self.AGENT_NAME, "collection": collection_name, "error": str(e)},
            )
            return []

    async def _call_llm(self, messages: list[dict], **kwargs) -> dict:
        """
        Call the LLM with error handling.

        Falls back to a deterministic mock response when no LLM is configured,
        enabling safe unit-test and development usage.
        """
        if not self.llm:
            # Deterministic mock for testing / offline development
            return {
                "content": f"[Mock LLM response for {self.AGENT_NAME}]",
                "tokens_prompt": 100,
                "tokens_completion": 50,
                "model": "mock",
            }
        try:
            return await self.llm.generate(messages, **kwargs)
        except Exception as e:
            logger.error(
                "LLM call failed",
                extra={"agent": self.AGENT_NAME, "error": str(e)},
            )
            raise

    async def _evaluate_response(
        self,
        query: str,
        response: str,
        context_docs: list[dict],
    ) -> dict:
        """
        Evaluate response grounding and hallucination risk.

        Returns a dict with grounding_score and hallucination_risk keys.
        Returns None values when evaluation is disabled or unavailable.
        """
        if not self.evaluator or not settings.evaluation_enabled:
            return {"grounding_score": None, "hallucination_risk": None}
        try:
            context_texts = [
                d.get("excerpt", "") or d.get("content", "")
                for d in context_docs
            ]
            return await self.evaluator.score_grounding(
                query=query,
                response=response,
                context_docs=context_texts,
            )
        except Exception as e:
            logger.warning(
                "Response evaluation failed",
                extra={"agent": self.AGENT_NAME, "error": str(e)},
            )
            return {"grounding_score": None, "hallucination_risk": None}

    def _format_context(self, docs: list[dict]) -> str:
        """
        Format a list of retrieved documents into an LLM-ready context string.

        Caps at 10 documents and 800 characters per excerpt to respect
        context-window budgets.
        """
        if not docs:
            return "No relevant context documents found."

        parts: list[str] = []
        for i, doc in enumerate(docs[:10], start=1):
            excerpt = (doc.get("excerpt") or doc.get("content") or "")[:800]
            ticker = doc.get("ticker", "N/A")
            category = doc.get("category", "N/A")
            title = doc.get("title", "Unknown")
            parts.append(
                f"[Doc {i}] {title} (Category: {category}, Ticker: {ticker})\n{excerpt}"
            )
        return "\n\n---\n\n".join(parts)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def metrics(self) -> dict:
        """Return agent-level performance metrics."""
        return {
            "agent": self.AGENT_NAME,
            "call_count": self._call_count,
            "total_tokens": self._total_tokens,
            "avg_latency_ms": round(
                self._total_latency_ms / max(self._call_count, 1), 2
            ),
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"agent={self.AGENT_NAME!r}, "
            f"calls={self._call_count}, "
            f"tokens={self._total_tokens})"
        )
