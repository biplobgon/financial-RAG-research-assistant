"""Financial analysis endpoint."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from app.models.requests import AnalysisRequest
from app.models.responses import AnalysisResponse
from app.api.v1.dependencies import get_orchestrator

router = APIRouter()


@router.post("", response_model=AnalysisResponse, summary="Deep financial analysis")
async def analyze_company(
    request: AnalysisRequest,
    orchestrator=Depends(get_orchestrator),
) -> AnalysisResponse:
    """Deep financial analysis using SEC filing agent."""
    try:
        query = f"Provide comprehensive financial analysis for {request.ticker or request.company_name}"
        if request.focus_areas:
            query += f" focusing on: {', '.join(request.focus_areas)}"

        agent_response = await orchestrator.route(
            query=query,
            agent_name="sec_filing_agent",
            ticker=request.ticker,
            filing_types=request.filing_types,
            time_period=request.time_period,
        )

        return AnalysisResponse(
            status=agent_response.status,
            ticker=request.ticker,
            company_name=request.company_name,
            analysis=agent_response.content,
            key_findings=[],
            risk_factors=[],
            financial_metrics={},
            sources=[],
            metadata=agent_response.metadata,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
