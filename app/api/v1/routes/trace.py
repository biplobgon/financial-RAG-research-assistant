"""Distributed trace endpoint."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

router = APIRouter()


class TraceRequest(BaseModel):
    trace_id: str
    include_spans: bool = True


@router.get("/{trace_id}", summary="Get distributed trace")
async def get_trace(trace_id: str, include_spans: bool = True) -> dict:
    """Retrieve distributed trace by ID."""
    # In production, this would query Jaeger/Tempo
    return {
        "trace_id": trace_id,
        "service": "financial-rag-api",
        "status": "trace_lookup_requires_jaeger",
        "message": "Configure OTEL_EXPORTER_OTLP_ENDPOINT to enable full tracing",
        "spans": [] if include_spans else None,
    }


@router.post("", summary="Get trace for request")
async def query_trace(request: TraceRequest) -> dict:
    """Query trace information for a given trace ID."""
    return {
        "trace_id": request.trace_id,
        "spans": [],
        "message": "Connect to Jaeger/Grafana Tempo for full trace visualization",
    }
