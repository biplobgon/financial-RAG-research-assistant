"""Financial data service — orchestrates RAG + agents for API endpoints."""
from __future__ import annotations

import logging
from typing import Optional, Any

from app.utils.logging import get_logger

logger = get_logger(__name__)


class FinancialService:
    """
    Business logic layer coordinating RAG pipeline and agents.

    Provides high-level methods consumed by API route handlers.
    Handles caching, error recovery, and response assembly.
    """

    def __init__(self, rag_pipeline=None, orchestrator=None):
        self.rag_pipeline = rag_pipeline
        self.orchestrator = orchestrator

    async def search_filings(
        self,
        query: str,
        ticker: Optional[str] = None,
        filing_types: Optional[list[str]] = None,
        top_k: int = 10,
    ) -> dict:
        """Search SEC filings with semantic retrieval."""
        filters = {}
        if ticker:
            filters["ticker"] = ticker.upper()
        if filing_types and len(filing_types) == 1:
            filters["filing_type"] = filing_types[0]

        if self.rag_pipeline:
            result = await self.rag_pipeline.retrieve(
                query=query,
                top_k=top_k,
                filters=filters if filters else None,
                collection_name="sec_filings",
            )
            return result.model_dump()
        return {"documents": [], "total_found": 0}

    async def analyze_ticker(
        self,
        ticker: str,
        analysis_type: str = "comprehensive",
        time_period: Optional[str] = None,
    ) -> dict:
        """Full financial analysis for a ticker."""
        if self.orchestrator:
            response = await self.orchestrator.route(
                query=f"Comprehensive financial analysis for {ticker}",
                agent_name="sec_filing_agent",
                ticker=ticker,
                analysis_type=analysis_type,
                time_period=time_period,
            )
            return response.to_dict()
        return {"error": "Orchestrator not initialized"}

    async def get_earnings_insights(
        self,
        ticker: str,
        fiscal_quarter: Optional[str] = None,
        fiscal_year: Optional[str] = None,
    ) -> dict:
        """Extract earnings insights for a ticker."""
        if self.orchestrator:
            query = f"Earnings call insights and guidance for {ticker}"
            if fiscal_quarter:
                query += f" {fiscal_quarter}"

            response = await self.orchestrator.route(
                query=query,
                agent_name="earnings_agent",
                ticker=ticker,
                fiscal_quarter=fiscal_quarter,
                fiscal_year=fiscal_year,
            )
            return response.to_dict()
        return {"error": "Orchestrator not initialized"}
