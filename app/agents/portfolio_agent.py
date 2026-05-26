"""Portfolio Intelligence Agent."""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from app.agents.base_agent import BaseFinancialAgent, AgentResponse
from app.config.constants import COLLECTION_SEC_FILINGS, COLLECTION_RESEARCH

logger = logging.getLogger(__name__)

# Maximum tickers processed in a single portfolio request to prevent
# runaway parallel retrieval and context-window overflow
_MAX_TICKERS = 10


class PortfolioAgent(BaseFinancialAgent):
    """
    Portfolio intelligence agent for multi-ticker cross-sectional analysis.

    Capabilities:
    - Portfolio-wide Q&A aggregating evidence across multiple positions
    - Sector and sub-sector exposure analysis
    - Risk factor aggregation and correlation assessment
    - Relative valuation commentary across holdings
    - Investment thesis synthesis per holding
    - Portfolio-level concentration and diversification analysis
    - Factor exposure (value, growth, momentum, quality, size)
    - ESG factor integration

    Routing keywords: portfolio, holdings, position, allocation, diversification,
                      benchmark, sector exposure, concentration risk
    """

    AGENT_NAME = "portfolio_agent"

    SYSTEM_PROMPT = """You are a senior portfolio analyst AI with expertise in:
- Multi-company comparative and cross-sectional analysis
- Portfolio risk assessment: concentration, correlation, and tail risk
- Sector rotation, thematic investing, and macro factor sensitivity
- Investment thesis evaluation and stress-testing
- ESG factor integration and sustainability risk assessment
- Factor exposure analysis (value, growth, momentum, quality, low-volatility)
- Relative valuation: EV/EBITDA, P/E, P/FCF, EV/Sales across peer groups

Portfolio Analysis Framework:
1. Analyze each holding individually: business quality, growth profile, valuation
2. Assess portfolio-level correlation and hidden concentration risks
3. Map sector/sub-sector exposures against benchmark weights
4. Evaluate macro sensitivities: rates, FX, commodity, credit
5. Identify diversification gaps and over-concentration in single themes
6. Provide actionable rebalancing insights grounded in fundamental data

Output Format:
**Individual Position Analysis** — per-ticker fundamental snapshot
**Portfolio-Level Insights** — aggregate themes and cross-holding observations
**Risk Assessment** — concentration, correlation, factor tilt risks
**Sector & Factor Breakdown** — exposure heatmap and benchmark delta
**Recommendations** — trim/add/monitor suggestions grounded in document evidence

Always ground analysis in factual financial data from filings and research."""

    async def _execute(self, query: str, **kwargs) -> AgentResponse:
        """
        Execute portfolio intelligence workflow.

        Steps:
        1. Validate tickers list (cap at _MAX_TICKERS)
        2. Fan out parallel retrieval across all tickers
        3. Aggregate context, de-duplicate, rank by relevance
        4. Assemble multi-ticker prompt
        5. Generate portfolio analysis
        6. Evaluate grounding and return structured response
        """
        tickers: list[str] = [t.upper() for t in kwargs.get("tickers", [])]
        include_risk: bool = kwargs.get("include_risk_analysis", True)
        benchmark: str = kwargs.get("benchmark", "SPY")
        trace_id: str = kwargs.get("trace_id", "")

        if not tickers:
            return AgentResponse(
                content=(
                    "No tickers provided. Please supply a `tickers` list to run "
                    "portfolio analysis (e.g., tickers=['AAPL', 'MSFT', 'GOOGL'])."
                ),
                agent_name=self.AGENT_NAME,
                trace_id=trace_id,
                latency_ms=0.0,
                status="error",
                error="tickers list is required for portfolio analysis",
            )

        capped_tickers = tickers[:_MAX_TICKERS]
        if len(tickers) > _MAX_TICKERS:
            logger.warning(
                "Portfolio ticker list truncated",
                extra={
                    "requested": len(tickers),
                    "capped": _MAX_TICKERS,
                    "dropped": tickers[_MAX_TICKERS:],
                },
            )

        # Fan out retrieval in parallel across all tickers
        retrieval_tasks = [
            self._retrieve_context(
                query=f"{query} {ticker}",
                collection_name=COLLECTION_SEC_FILINGS,
                top_k=5,
                filters={"ticker": ticker},
            )
            for ticker in capped_tickers
        ]
        ticker_results = await asyncio.gather(*retrieval_tasks, return_exceptions=True)

        # Aggregate documents, skipping failed retrievals
        all_docs: list[dict] = []
        failed_tickers: list[str] = []
        for ticker, result in zip(capped_tickers, ticker_results):
            if isinstance(result, Exception):
                logger.warning(
                    "Retrieval failed for ticker",
                    extra={"ticker": ticker, "error": str(result)},
                )
                failed_tickers.append(ticker)
            else:
                all_docs.extend(result)

        context = self._format_context(all_docs)
        ticker_list = ", ".join(capped_tickers)

        # Build risk analysis instruction
        risk_instruction = (
            "Include a dedicated Risk Assessment section covering: "
            "position concentration, sector correlation, factor tilt risks, "
            "and macro sensitivities."
            if include_risk
            else "Omit detailed risk analysis; focus on fundamental position overview."
        )

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Portfolio Analysis Request\n"
                    f"Portfolio Tickers : {ticker_list}\n"
                    f"Benchmark        : {benchmark}\n"
                    f"Risk Analysis    : {'Enabled' if include_risk else 'Disabled'}\n"
                    f"Document Context : {len(all_docs)} chunks across "
                    f"{len(capped_tickers) - len(failed_tickers)} tickers\n"
                    + (f"Data Gaps        : No documents found for {failed_tickers}\n" if failed_tickers else "")
                    + f"\nDocument Context:\n{context}\n\n"
                    f"Portfolio Question: {query}\n\n"
                    f"{risk_instruction}\n"
                    "Ensure each holding from the portfolio is addressed in your analysis."
                ),
            },
        ]

        llm_result = await self._call_llm(messages)
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
                    "ticker": d.get("ticker"),
                    "title": d.get("title", ""),
                    "category": d.get("category"),
                    "relevance_score": d.get("relevance_score", 0.0),
                }
                for d in all_docs[:8]
            ],
            metadata={
                "tickers": capped_tickers,
                "failed_tickers": failed_tickers,
                "benchmark": benchmark,
                "include_risk": include_risk,
                "doc_count": len(all_docs),
                "collection": COLLECTION_SEC_FILINGS,
            },
        )
