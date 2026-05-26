"""SEC Filing Retrieval and Analysis Agent."""
from __future__ import annotations

import logging
from typing import Optional

from app.agents.base_agent import BaseFinancialAgent, AgentResponse
from app.config.settings import settings
from app.config.constants import COLLECTION_SEC_FILINGS, SEC_FILING_TYPES

logger = logging.getLogger(__name__)


class SECFilingAgent(BaseFinancialAgent):
    """
    Agent specializing in SEC filing retrieval and deep financial analysis.

    Capabilities:
    - 10-K annual report analysis (revenue, margins, cash flow)
    - 10-Q quarterly report comparison and trend analysis
    - Risk factor identification and severity assessment
    - Financial statement ratio analysis
    - MD&A interpretation and management tone scoring
    - SEC EDGAR document metadata extraction
    - Earnings quality and accrual analysis
    - Segment reporting breakdown

    Routing keywords: 10-K, 10-Q, SEC, risk factors, MD&A, annual report,
                      quarterly report, Item 1A, Item 7, management discussion
    """

    AGENT_NAME = "sec_filing_agent"
    MAX_CONTEXT_TOKENS = 12000

    SYSTEM_PROMPT = """You are a financial analyst expert specializing in SEC filing analysis.
You have deep expertise in:
- 10-K and 10-Q financial statement analysis (income statement, balance sheet, cash flows)
- Risk factor identification and materiality assessment
- Revenue recognition and accounting policy review under ASC 606
- MD&A (Management's Discussion and Analysis) interpretation
- Earnings quality assessment using accrual ratios
- Balance sheet strength and liquidity analysis
- Free cash flow conversion and capex intensity analysis
- Segment reporting and geographic revenue breakdown

When analyzing SEC filings:
1. Always cite specific sections (Item 1A Risk Factors, Item 7 MD&A, Item 8 Financial Statements, etc.)
2. Highlight key financial figures with explicit fiscal periods (e.g., FY2023, Q3 FY2024)
3. Identify material risk factors and their potential financial impact
4. Assess management's forward guidance accuracy vs. prior disclosures
5. Flag any unusual accounting changes, restatements, or material weaknesses
6. Provide quantitative analysis: YoY growth rates, margin trends, leverage ratios
7. Note any related-party transactions or off-balance-sheet arrangements

Output Format:
**Financial Highlights** — key metrics with period-over-period comparisons
**Risk Factors** — top material risks with impact assessment
**MD&A Commentary** — management narrative interpretation
**Balance Sheet & Liquidity** — capital structure and liquidity metrics
**Cash Flow Analysis** — operating, investing, financing activity summary
**Key Concerns / Flags** — accounting irregularities or disclosure issues

Respond with structured, professional analysis suitable for institutional investors."""

    async def _execute(self, query: str, **kwargs) -> AgentResponse:
        """
        Execute SEC filing retrieval and analysis workflow.

        Steps:
        1. Build metadata filters from ticker / filing_type parameters
        2. Retrieve top-K relevant chunks from the SEC filings collection
        3. Assemble LLM prompt with system instructions and document context
        4. Generate analysis via LLM
        5. Run grounding evaluation
        6. Return structured AgentResponse
        """
        ticker: Optional[str] = kwargs.get("ticker")
        filing_types: list[str] = kwargs.get("filing_types", ["10-K", "10-Q"])
        time_period: Optional[str] = kwargs.get("time_period")
        trace_id: str = kwargs.get("trace_id", "")

        # Validate requested filing types
        valid_types = [ft for ft in filing_types if ft in SEC_FILING_TYPES]
        if not valid_types:
            valid_types = ["10-K", "10-Q"]

        # Build metadata filters for targeted retrieval
        filters: dict = {"category": "sec_filing"}
        if ticker:
            filters["ticker"] = ticker.upper()
        if len(valid_types) == 1:
            filters["filing_type"] = valid_types[0]

        # Retrieve relevant SEC document chunks
        context_docs = await self._retrieve_context(
            query=query,
            collection_name=COLLECTION_SEC_FILINGS,
            top_k=12,
            filters=filters if len(filters) > 1 else None,
        )

        # Format retrieved documents as LLM context
        context = self._format_context(context_docs)
        sec_context_header = self._build_sec_context(ticker, valid_types, time_period)

        # Assemble messages
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"{sec_context_header}\n\n"
                    f"Retrieved SEC Filing Context ({len(context_docs)} documents):\n"
                    f"{context}\n\n"
                    f"Analysis Request: {query}\n\n"
                    "Provide a detailed, structured analysis with specific references to "
                    "filing sections (Item numbers), financial figures, and fiscal periods. "
                    "Flag any material concerns or unusual disclosures."
                ),
            },
        ]

        # Generate LLM response (low temperature for factual precision)
        llm_result = await self._call_llm(messages, temperature=0.05)
        answer: str = llm_result.get("content", "")

        # Evaluate grounding quality
        eval_result = await self._evaluate_response(query, answer, context_docs)

        return AgentResponse(
            content=answer,
            agent_name=self.AGENT_NAME,
            trace_id=trace_id,
            latency_ms=0.0,  # Overwritten by BaseFinancialAgent.run()
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
                    "filing_type": d.get("filing_type"),
                    "ticker": d.get("ticker"),
                    "fiscal_period": d.get("fiscal_period"),
                    "relevance_score": d.get("relevance_score", 0.0),
                }
                for d in context_docs[:5]
            ],
            metadata={
                "ticker": ticker,
                "filing_types": valid_types,
                "time_period": time_period,
                "retrieved_docs": len(context_docs),
                "collection": COLLECTION_SEC_FILINGS,
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_sec_context(
        self,
        ticker: Optional[str],
        filing_types: list[str],
        time_period: Optional[str],
    ) -> str:
        """Build a descriptive SEC context header injected into the prompt."""
        lines = ["SEC Filing Analysis Context:"]
        if ticker:
            lines.append(f"  Company Ticker : {ticker.upper()}")
        if filing_types:
            lines.append(f"  Filing Types   : {', '.join(filing_types)}")
        if time_period:
            lines.append(f"  Time Period    : {time_period}")
        lines.append("  Analysis Standards : US GAAP, SEC Regulation S-K, Regulation S-X")
        lines.append("  Data Source    : SEC EDGAR")
        return "\n".join(lines)
