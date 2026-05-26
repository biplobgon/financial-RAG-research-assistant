"""Prometheus metrics endpoint."""
from __future__ import annotations

from fastapi import APIRouter, Response

router = APIRouter()


@router.get("", summary="Prometheus metrics")
async def get_metrics() -> Response:
    """Expose Prometheus metrics for scraping."""
    try:
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
    except ImportError:
        return Response(content="# Prometheus client not installed\n", media_type="text/plain")
