"""FastAPI dependency injection providers."""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

_rag_pipeline = None
_orchestrator = None
_governance_service = None


async def get_rag_pipeline():
    """Provide the RAG pipeline instance."""
    global _rag_pipeline
    if _rag_pipeline is None:
        from app.rag.pipeline import RAGPipeline
        from app.rag.retriever import ChromaRetriever
        from app.embeddings.embedder import FinancialEmbedder

        embedder = FinancialEmbedder()
        retriever = ChromaRetriever(embedder=embedder)
        _rag_pipeline = RAGPipeline(retriever=retriever, embedder=embedder)
        logger.info("RAG pipeline initialized")
    return _rag_pipeline


async def get_orchestrator():
    """Provide the agent orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        from app.agents.orchestrator import AgentOrchestrator
        from app.agents.sec_filing_agent import SECFilingAgent
        from app.agents.earnings_agent import EarningsAgent
        from app.agents.portfolio_agent import PortfolioAgent
        from app.agents.executive_summary_agent import ExecutiveSummaryAgent
        from app.agents.research_agent import ResearchAgent
        from app.agents.evaluation_agent import EvaluationAgent
        from app.rag.retriever import ChromaRetriever
        from app.embeddings.embedder import FinancialEmbedder

        embedder = FinancialEmbedder()
        retriever = ChromaRetriever(embedder=embedder)

        _orchestrator = AgentOrchestrator()
        for AgentClass in [SECFilingAgent, EarningsAgent, PortfolioAgent,
                            ExecutiveSummaryAgent, ResearchAgent, EvaluationAgent]:
            agent = AgentClass(retriever=retriever)
            _orchestrator.register_agent(agent)

        logger.info(f"Orchestrator initialized with {len(_orchestrator.agents)} agents")
    return _orchestrator


async def get_governance_service():
    """Provide the governance service instance."""
    global _governance_service
    if _governance_service is None:
        from app.governance.guardrails import GovernanceService
        _governance_service = GovernanceService()
        logger.info("Governance service initialized")
    return _governance_service
