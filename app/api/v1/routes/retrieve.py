"""Document retrieval endpoint."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from app.models.requests import RetrievalRequest
from app.models.responses import RetrievalResponse
from app.api.v1.dependencies import get_rag_pipeline

router = APIRouter()


@router.post("", response_model=RetrievalResponse, summary="Semantic document retrieval")
async def retrieve_documents(
    request: RetrievalRequest,
    rag_pipeline=Depends(get_rag_pipeline),
) -> RetrievalResponse:
    """Retrieve semantically relevant financial documents."""
    try:
        return await rag_pipeline.retrieve(
            query=request.query,
            top_k=request.top_k,
            filters=request.filters if request.filters else None,
            collection_name=request.collection,
            retrieval_mode=request.retrieval_mode,
            rerank=request.rerank,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
