"""Primary financial Q&A query endpoint."""
from __future__ import annotations

import logging
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from app.models.requests import QueryRequest
from app.models.responses import QueryResponse, ErrorResponse
from app.api.v1.dependencies import get_rag_pipeline, get_orchestrator

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("", response_model=QueryResponse, summary="Financial Q&A query")
async def financial_query(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    rag_pipeline=Depends(get_rag_pipeline),
    orchestrator=Depends(get_orchestrator),
) -> QueryResponse:
    """
    Primary endpoint for financial questions answered with RAG.

    Routes to appropriate agent based on query intent or explicit agent selection.
    Supports semantic retrieval, LLM generation, and grounding evaluation.
    """
    try:
        if request.agent and request.agent != "research_agent":
            # Route to specific agent
            agent_response = await orchestrator.route(
                query=request.query,
                agent_name=request.agent,
                filters=request.filters,
                top_k=request.top_k,
                session_id=request.session_id,
            )
            from app.models.responses import ResponseMetadata
            return QueryResponse(
                status=agent_response.status,
                query=request.query,
                answer=agent_response.content,
                sources=[],
                metadata=ResponseMetadata(
                    agent=agent_response.agent_name,
                    latency_ms=agent_response.latency_ms,
                    tokens_used=agent_response.tokens_used,
                    grounding_score=agent_response.grounding_score,
                    hallucination_risk=agent_response.hallucination_risk,
                    trace_id=agent_response.trace_id,
                    retrieval_count=len(agent_response.sources),
                ),
            )
        else:
            # Use RAG pipeline
            return await rag_pipeline.query(
                query=request.query,
                filters=request.filters if request.filters else None,
                top_k=request.top_k,
                session_id=request.session_id,
            )
    except Exception as e:
        logger.error(f"Query endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
