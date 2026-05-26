"""Multi-Document Financial Research Synthesis Agent."""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from app.agents.base_agent import BaseFinancialAgent, AgentResponse
from app.config.constants import (
    COLLECTION_SEC_FILINGS,
    COLLECTION_EARNINGS,
    COLLECTION_RESEARCH,
)

logger = logging.getLogger(__name__)

# Supported analysis types with their retrieval weight tuning
_ANALYSIS_TYPES: dict[str, str] = {
    "comprehensive": "Produce a full investment research report covering all dimensions.",
    "investment_memo": "Produce a concise investment committee memo with a clear Buy/Hold/Sell recommendation.",
    "competitive": "Focus on competitive positioning, market share dynamics, and peer comparison.",
    "industry": "Focus on industry structure, secular trends, and macro tailwinds/headwinds.",
    "valuation": "Focus on valuation methodology, intrinsic value estimation, and upside/downside scenarios.",
    "esg": "Focus on ESG factors: environmental exposure, governance quality, and social risk.",
}


class ResearchAgent(BaseFinancialAgent):
    """
    Multi-document financial research synthesis agent.

    Capabilities:
    - Cross-collection research synthesis (SEC + earnings + research reports)
    - Investment memo / full research report generation
    - Competitive analysis and peer benchmarking
    - Industry trend and secular driver analysis
    - Multi-source evidence aggregation with citation tracking
    - Configurable analysis types (comprehensive, investment_memo, competitive, etc.)
    - Evidence conflict detection across sources

    Routing keywords: research, analysis, investment thesis, competitive,
                      industry, valuation, peer comparison, deep-dive
    """

    AGENT_NAME = "research_agent"

    SYSTEM_PROMPT = """You are a senior financial research analyst at a top-tier investment firm.
You synthesize information from multiple financial document sources to produce comprehensive,
evidence-based research reports with institutional-grade rigor.

Research Methodology:
1. Gather and cross-reference evidence from SEC filings, earnings calls, and analyst reports
2. Verify consistency of claims across sources; flag discrepancies explicitly
3. Identify key investment thesis drivers and quantify their impact
4. Size upside and downside scenarios with probability weighting
5. Benchmark against sector peers using standardized valuation multiples
6. Formulate a clear, evidence-based investment perspective with a rating

Standard Research Report Structure:
**Executive Summary** — thesis in 3 sentences, rating, and 12-month target
**Investment Thesis** — bull case drivers with evidence citations
**Business Analysis** — competitive moat, TAM, market position
**Financial Analysis** — revenue model, margins, FCF, balance sheet (3-year view)
**Risk Factors** — top 5 risks with probability and impact matrix
**Catalysts** — near-term (0–3 months) and medium-term (3–12 months) events
**Valuation Commentary** — methodology, comparable multiples, DCF assumptions
**Conclusion** — final recommendation with conviction level (high/medium/low)

Always cite the source document type and period for every factual claim.
Maintain analytical objectivity — if evidence is insufficient, say so explicitly."""

    async def _execute(self, query: str, **kwargs) -> AgentResponse:
        """
        Execute multi-document research synthesis workflow.

        Steps:
        1. Parse analysis parameters
        2. Parallel retrieval from SEC filings, earnings, and research collections
        3. Combine and rank documents by relevance
        4. Build research synthesis prompt
        5. Generate comprehensive report
        6. Evaluate grounding quality
        7. Return structured response with full source attribution
        """
        ticker: Optional[str] = kwargs.get("ticker")
        analysis_type: str = kwargs.get("analysis_type", "comprehensive")
        trace_id: str = kwargs.get("trace_id", "")

        # Validate analysis type
        if analysis_type not in _ANALYSIS_TYPES:
            logger.warning(
                "Unknown analysis_type, defaulting to comprehensive",
                extra={"requested": analysis_type, "valid": list(_ANALYSIS_TYPES.keys())},
            )
            analysis_type = "comprehensive"

        analysis_instruction = _ANALYSIS_TYPES[analysis_type]

        # Build per-collection retrieval filters
        retrieval_filters = {"ticker": ticker} if ticker else None

        # Retrieve from all three collections in parallel to maximize evidence coverage
        sec_docs, earnings_docs, research_docs = await asyncio.gather(
            self._retrieve_context(
                query=query,
                collection_name=COLLECTION_SEC_FILINGS,
                top_k=8,
                filters=retrieval_filters,
            ),
            self._retrieve_context(
                query=query,
                collection_name=COLLECTION_EARNINGS,
                top_k=5,
                filters=retrieval_filters,
            ),
            self._retrieve_context(
                query=query,
                collection_name=COLLECTION_RESEARCH,
                top_k=5,
                filters=retrieval_filters,
            ),
        )

        all_docs = sec_docs + earnings_docs + research_docs
        context = self._format_context(all_docs)

        # Source provenance summary for prompt transparency
        source_summary = (
            f"{len(sec_docs)} SEC filing segments, "
            f"{len(earnings_docs)} earnings transcript segments, "
            f"{len(research_docs)} analyst research report segments"
        )

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Financial Research Request\n"
                    f"Primary Focus  : {ticker or 'General Market / Sector'}\n"
                    f"Analysis Type  : {analysis_type.replace('_', ' ').title()}\n"
                    f"Analysis Focus : {analysis_instruction}\n"
                    f"Source Coverage: {source_summary}\n\n"
                    f"Multi-Source Evidence:\n{context}\n\n"
                    f"Research Question: {query}\n\n"
                    "Produce a comprehensive financial research report following your standard structure. "
                    "Cite every factual claim with its source type (SEC filing / earnings / research report) "
                    "and fiscal period. Flag any conflicting evidence across sources. "
                    "If data is insufficient for a section, explicitly note the limitation."
                ),
            },
        ]

        # Near-zero temperature for maximum analytical consistency
        llm_result = await self._call_llm(messages, temperature=0.05)
        answer: str = llm_result.get("content", "")
        eval_result = await self._evaluate_response(query, answer, all_docs)

        return AgentResponse(
            content=answer,
            agent_name=self.AGENT_NAME,
            trace_id=trace_id,
            latency_ms=0.0,
            tokens_used=(
                llm_result.get("tokens_prompt", 0)
                + llm_result.get("tokens_completion", 0)
            ),
            grounding_score=eval_result.get("grounding_score"),
            hallucination_risk=eval_result.get("hallucination_risk"),
            sources=[
                {
                    "title": d.get("title", ""),
                    "category": d.get("category"),
                    "ticker": d.get("ticker"),
                    "fiscal_period": d.get("fiscal_period"),
                    "relevance_score": d.get("relevance_score", 0.0),
                }
                for d in all_docs[:8]
            ],
            metadata={
                "ticker": ticker,
                "analysis_type": analysis_type,
                "sec_docs": len(sec_docs),
                "earnings_docs": len(earnings_docs),
                "research_docs": len(research_docs),
                "total_docs": len(all_docs),
                "collections": [
                    COLLECTION_SEC_FILINGS,
                    COLLECTION_EARNINGS,
                    COLLECTION_RESEARCH,
                ],
            },
        )
