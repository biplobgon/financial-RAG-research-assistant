"""AI governance endpoint."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from app.models.requests import GovernanceRequest
from app.models.responses import GovernanceResponse, GovernanceCheckResult
from app.api.v1.dependencies import get_governance_service

router = APIRouter()


@router.post("", response_model=GovernanceResponse, summary="AI governance checks")
async def governance_check(
    request: GovernanceRequest,
    governance_service=Depends(get_governance_service),
) -> GovernanceResponse:
    """Run AI governance and compliance checks on content."""
    try:
        results = await governance_service.run_checks(
            content=request.content,
            check_types=request.check_types,
            context=request.context,
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
