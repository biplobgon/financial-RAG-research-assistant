"""Portfolio intelligence endpoint."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from app.models.requests import PortfolioRequest
from app.models.responses import PortfolioResponse
from app.api.v1.dependencies import get_orchestrator

router = APIRouter()


@router.post("", response_model=PortfolioResponse, summary="Portfolio intelligence")
async def portfolio_intelligence(
    request: PortfolioRequest,
    orchestrator=Depends(get_orchestrator),
) -> PortfolioResponse:
    """Multi-ticker portfolio analysis and insight generation."""
    try:
        agent_response = await orchestrator.route(
            query=request.query,
            agent_name="portfolio_agent",
            tickers=request.tickers,
            include_risk_analysis=request.include_risk_analysis,
            benchmark=request.benchmark,
        )

        return PortfolioResponse(
            status=agent_response.status,
            tickers=request.tickers,
            insights=agent_response.content,
            sector_breakdown={},
            recommendations=[],
            individual_analyses={},
            metadata=agent_response.metadata,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
