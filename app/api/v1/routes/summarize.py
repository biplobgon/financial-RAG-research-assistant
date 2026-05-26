"""Executive summarization endpoint."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from app.models.requests import SummarizationRequest
from app.models.responses import SummarizationResponse
from app.api.v1.dependencies import get_orchestrator

router = APIRouter()


@router.post("", response_model=SummarizationResponse, summary="Executive summarization")
async def summarize_documents(
    request: SummarizationRequest,
    orchestrator=Depends(get_orchestrator),
) -> SummarizationResponse:
    """Generate executive-grade summaries from financial documents."""
    try:
        query = request.query or f"Summarize financial highlights for {request.ticker or 'the company'}"

        agent_response = await orchestrator.route(
            query=query,
            agent_name="executive_summary_agent",
            ticker=request.ticker,
            summary_length=request.summary_length,
            audience=request.audience,
        )

        return SummarizationResponse(
            status=agent_response.status,
            summary=agent_response.content,
            key_points=[],
            action_items=[],
            risk_highlights=[],
            sources=[],
            metadata=agent_response.metadata,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
