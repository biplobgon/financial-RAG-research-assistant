"""Response evaluation endpoint."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from app.models.requests import EvaluationRequest
from app.models.responses import EvaluationResponse, EvaluationScore
from app.api.v1.dependencies import get_orchestrator

router = APIRouter()


@router.post("", response_model=EvaluationResponse, summary="Response evaluation and scoring")
async def evaluate_response(
    request: EvaluationRequest,
    orchestrator=Depends(get_orchestrator),
) -> EvaluationResponse:
    """Evaluate AI response for grounding, hallucination, and quality."""
    try:
        agent_response = await orchestrator.route(
            query=request.query,
            agent_name="evaluation_agent",
            response=request.response,
            context_documents=request.context_documents,
            evaluation_types=request.evaluation_types,
        )

        grounding = agent_response.metadata.get("grounding_score") or 0.0
        relevance = agent_response.metadata.get("relevance_score") or 0.0
        quality = agent_response.metadata.get("quality_score") or 0.0
        hallucination_risk = agent_response.hallucination_risk or "unknown"
        overall = (grounding + relevance + quality) / 3.0
        passed = overall >= 0.70

        return EvaluationResponse(
            status=agent_response.status,
            overall_score=overall,
            passed=passed,
            scores=[
                EvaluationScore(
                    metric="grounding",
                    score=grounding,
                    explanation="Factual grounding in source documents",
                    passed=grounding >= 0.85,
                ),
                EvaluationScore(
                    metric="relevance",
                    score=relevance,
                    explanation="Relevance to original query",
                    passed=relevance >= 0.70,
                ),
                EvaluationScore(
                    metric="quality",
                    score=quality,
                    explanation="Professional quality and completeness",
                    passed=quality >= 0.70,
                ),
            ],
            hallucination_detected=hallucination_risk in ("high", "critical"),
            grounding_score=grounding,
            relevance_score=relevance,
            quality_score=quality,
            recommendations=[],
            metadata=agent_response.metadata,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
