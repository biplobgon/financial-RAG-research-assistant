"""Earnings Call Transcript Analysis Agent."""
from __future__ import annotations

import logging
from typing import Optional

from app.agents.base_agent import BaseFinancialAgent, AgentResponse
from app.config.constants import COLLECTION_EARNINGS

logger = logging.getLogger(__name__)


class EarningsAgent(BaseFinancialAgent):
    """
    Agent for earnings call transcript analysis and financial signal detection.

    Capabilities:
    - Management commentary and tone extraction
    - Forward guidance identification and quantification
    - Analyst Q&A sentiment and concern analysis
    - Earnings surprise detection vs. consensus estimates
    - Period-over-period language drift analysis
    - Key financial metric commentary extraction
    - Strategic initiative identification
    - Risk acknowledgment and mitigation assessment

    Routing keywords: earnings call, transcript, quarterly results, guidance,
                      EPS beat, revenue miss, Q&A session, management commentary
    """

    AGENT_NAME = "earnings_agent"

    SYSTEM_PROMPT = """You are an expert financial analyst specializing in earnings call transcript analysis.
You extract actionable intelligence from earnings call transcripts with deep expertise in:
- Management tone, confidence level, and language consistency vs. prior quarters
- Forward guidance interpretation and quantification (EPS, revenue, margin targets)
- Analyst question sentiment — identifying investor concerns embedded in questions
- Key financial metric commentary (gross margin, operating leverage, free cash flow)
- Strategic initiative identification and execution credibility assessment
- Risk acknowledgment language and specificity of mitigation plans
- Beat/miss attribution — what drove performance relative to expectations

Analysis Framework:
1. Identify the most impactful management statements on near-term and long-term performance
2. Extract all quantitative guidance figures with explicit fiscal periods
3. Note analyst themes from Q&A — what concerns recur across multiple questions?
4. Assess language changes vs. prior quarter — more/less optimistic, new caveats added
5. Flag any hedged, evasive, or unusually cautious management language
6. Summarize actionable investment signals: buy catalysts, risk flags, thesis changers

Output Format:
**Management Highlights** — top 5 most impactful statements with direct quotes
**Forward Guidance** — all numerical guidance with period labels
**Analyst Q&A Signals** — recurring themes and management responsiveness
**Financial Metric Commentary** — margins, cash flow, growth commentary
**Risk Factors Discussed** — acknowledged headwinds and mitigation plans
**Investment Implications** — bullish signals, bearish flags, thesis-changing disclosures"""

    async def _execute(self, query: str, **kwargs) -> AgentResponse:
        """
        Execute earnings transcript analysis workflow.

        Steps:
        1. Build ticker/period filters
        2. Retrieve earnings transcript chunks
        3. Assemble structured prompt
        4. Generate analysis
        5. Evaluate grounding
        6. Return structured response
        """
        ticker: Optional[str] = kwargs.get("ticker")
        fiscal_quarter: Optional[str] = kwargs.get("fiscal_quarter")
        fiscal_year: Optional[str] = kwargs.get("fiscal_year")
        trace_id: str = kwargs.get("trace_id", "")

        # Build retrieval filters
        filters: dict = {}
        if ticker:
            filters["ticker"] = ticker.upper()
        if fiscal_quarter:
            filters["fiscal_quarter"] = fiscal_quarter
        if fiscal_year:
            filters["fiscal_year"] = str(fiscal_year)

        # Retrieve earnings transcript context
        context_docs = await self._retrieve_context(
            query=query,
            collection_name=COLLECTION_EARNINGS,
            top_k=10,
            filters=filters if filters else None,
        )

        context = self._format_context(context_docs)

        # Build period description for prompt
        period_parts: list[str] = []
        if fiscal_quarter:
            period_parts.append(fiscal_quarter)
        if fiscal_year:
            period_parts.append(f"FY{fiscal_year}")
        period_info = f"Focus Period: {' '.join(period_parts)}" if period_parts else ""

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Earnings Analysis Request\n"
                    f"Ticker        : {ticker or 'Not specified'}\n"
                    f"{period_info + chr(10) if period_info else ''}"
                    f"Retrieved Docs: {len(context_docs)} transcript segments\n\n"
                    f"Transcript Context:\n{context}\n\n"
                    f"Analysis Question: {query}\n\n"
                    "Structure your response with all six sections from your output format. "
                    "Include direct quotes from management where available. "
                    "Explicitly flag any language that differs materially from prior quarters."
                ),
            },
        ]

        llm_result = await self._call_llm(messages, temperature=0.1)
        answer: str = llm_result.get("content", "")
        eval_result = await self._evaluate_response(query, answer, context_docs)

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
                    "doc_id": d.get("doc_id", ""),
                    "title": d.get("title", ""),
                    "ticker": d.get("ticker"),
                    "fiscal_quarter": d.get("fiscal_quarter"),
                    "fiscal_year": d.get("fiscal_year"),
                    "relevance_score": d.get("relevance_score", 0.0),
                }
                for d in context_docs[:5]
            ],
            metadata={
                "ticker": ticker,
                "fiscal_quarter": fiscal_quarter,
                "fiscal_year": fiscal_year,
                "retrieved_docs": len(context_docs),
                "collection": COLLECTION_EARNINGS,
            },
        )
