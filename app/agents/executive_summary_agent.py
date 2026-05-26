"""Executive Summary Generation Agent."""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from app.agents.base_agent import BaseFinancialAgent, AgentResponse
from app.config.constants import COLLECTION_SEC_FILINGS, COLLECTION_EARNINGS

logger = logging.getLogger(__name__)


# Target word counts per summary length tier
SUMMARY_LENGTH_WORDS: dict[str, int] = {
    "short": 200,
    "medium": 500,
    "long": 1000,
    "executive": 750,
}

# Audience-specific style instructions injected into the prompt
AUDIENCE_INSTRUCTIONS: dict[str, str] = {
    "executive": (
        "Write for C-suite executives. Lead with business impact and strategic implications. "
        "Use concise, declarative language. Avoid accounting jargon. "
        "Prioritize: revenue trajectory, margin trends, key risks, and strategic moves."
    ),
    "analyst": (
        "Write for buy-side/sell-side financial analysts. Include specific KPIs, "
        "period-over-period comparisons, and ratio analysis. "
        "Use standard financial terminology (EBITDA, FCF, DSO, etc.). "
        "Quantify all material changes."
    ),
    "investor": (
        "Write for institutional investors. Focus on investment thesis drivers, "
        "risk/reward trade-offs, valuation context, and return potential. "
        "Address both bull and bear cases. Highlight any thesis-changing developments."
    ),
    "retail": (
        "Write for retail investors. Use plain, accessible language. "
        "Briefly explain any technical financial terms. "
        "Focus on: what the company does, how it performed, and what it means for shareholders."
    ),
}


class ExecutiveSummaryAgent(BaseFinancialAgent):
    """
    Agent for generating professional, audience-tailored executive summaries.

    Capabilities:
    - Audience-adapted content (executive, analyst, investor, retail)
    - Configurable length tiers: short / medium / long / executive
    - Cross-source synthesis (SEC filings + earnings transcripts)
    - Key insights and action items extraction
    - Risk highlights and strategic outlook
    - Parallel multi-collection retrieval for comprehensive coverage

    Routing keywords: summarize, summary, brief, executive, overview,
                      key points, highlights, TL;DR
    """

    AGENT_NAME = "executive_summary_agent"

    SYSTEM_PROMPT = """You are an expert financial communications specialist with 20+ years of experience
producing executive-level financial summaries for Fortune 500 boards and institutional investors.

Every summary you produce follows this structure:
**Executive Overview** — 2–3 sentences capturing the single most important takeaway
**Key Financial Highlights** — bullet points with specific metrics and period labels
**Strategic Developments** — material business changes, acquisitions, initiatives
**Critical Risks** — top 3 risks with estimated financial exposure where available
**Outlook & Forward Guidance** — management's stated expectations for coming periods
**Recommended Actions** — concrete, prioritized action items for the target audience

Quality Standards:
- Every financial figure must reference a specific period (Q3 FY2024, FY2023, etc.)
- No unattributed superlatives or marketing language
- Flag any discrepancies between management guidance and analyst consensus
- Distinguish between confirmed results and forward-looking statements
- Adapt depth and vocabulary to the target audience"""

    async def _execute(self, query: str, **kwargs) -> AgentResponse:
        """
        Execute executive summarization workflow.

        Steps:
        1. Parse audience and length parameters
        2. Parallel retrieval from SEC filings + earnings transcripts
        3. Build audience/length-adapted prompt
        4. Generate summary
        5. Evaluate grounding
        6. Return structured response
        """
        ticker: Optional[str] = kwargs.get("ticker")
        summary_length: str = kwargs.get("summary_length", "medium")
        audience: str = kwargs.get("audience", "executive")
        trace_id: str = kwargs.get("trace_id", "")

        # Resolve length/audience params with safe defaults
        target_words = SUMMARY_LENGTH_WORDS.get(summary_length, SUMMARY_LENGTH_WORDS["medium"])
        audience_instruction = AUDIENCE_INSTRUCTIONS.get(
            audience.lower(), AUDIENCE_INSTRUCTIONS["executive"]
        )

        # Build retrieval filters
        retrieval_filters = {"ticker": ticker} if ticker else None

        # Retrieve from both SEC filings and earnings in parallel
        sec_docs, earnings_docs = await asyncio.gather(
            self._retrieve_context(
                query=query,
                collection_name=COLLECTION_SEC_FILINGS,
                top_k=6,
                filters=retrieval_filters,
            ),
            self._retrieve_context(
                query=query,
                collection_name=COLLECTION_EARNINGS,
                top_k=4,
                filters=retrieval_filters,
            ),
        )

        all_docs = sec_docs + earnings_docs
        context = self._format_context(all_docs)

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Executive Summary Request\n"
                    f"Company Ticker : {ticker or 'Multiple / General'}\n"
                    f"Target Length  : ~{target_words} words\n"
                    f"Target Audience: {audience.title()}\n"
                    f"Audience Style : {audience_instruction}\n\n"
                    f"Source Documents ({len(all_docs)} total — "
                    f"{len(sec_docs)} SEC filings, {len(earnings_docs)} earnings transcripts):\n"
                    f"{context}\n\n"
                    f"Summarization Focus: {query}\n\n"
                    "Generate a structured executive summary following your six-section format. "
                    f"Keep the total length to approximately {target_words} words. "
                    "Every financial figure must include a period reference."
                ),
            },
        ]

        llm_result = await self._call_llm(messages, temperature=0.2)
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
                    "relevance_score": d.get("relevance_score", 0.0),
                }
                for d in all_docs[:5]
            ],
            metadata={
                "ticker": ticker,
                "summary_length": summary_length,
                "target_words": target_words,
                "audience": audience,
                "sec_docs": len(sec_docs),
                "earnings_docs": len(earnings_docs),
            },
        )
