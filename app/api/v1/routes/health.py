"""Health check endpoints."""
from __future__ import annotations

import time
from fastapi import APIRouter
from app.config.settings import settings
from app.models.responses import HealthResponse, HealthStatus

router = APIRouter()

_APP_START_TIME = time.time()


@router.get("/health", response_model=HealthResponse, summary="Application health check")
async def health_check() -> HealthResponse:
    """Liveness and readiness health check with component status."""
    components = []
    overall_status = "healthy"

    # Check ChromaDB
    try:
        import chromadb
        client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
        t0 = time.monotonic()
        client.heartbeat()
        latency = (time.monotonic() - t0) * 1000
        components.append(HealthStatus(name="chromadb", status="healthy", latency_ms=round(latency, 2)))
    except Exception as e:
        components.append(HealthStatus(name="chromadb", status="degraded", message=str(e)[:100]))
        overall_status = "degraded"

    # Check Redis
    try:
        import redis
        r = redis.from_url(settings.redis_url, socket_timeout=2.0)
        t0 = time.monotonic()
        r.ping()
        latency = (time.monotonic() - t0) * 1000
        components.append(HealthStatus(name="redis", status="healthy", latency_ms=round(latency, 2)))
    except Exception as e:
        components.append(HealthStatus(name="redis", status="degraded", message=str(e)[:100]))

    # API status
    components.append(HealthStatus(name="api", status="healthy"))

    return HealthResponse(
        status=overall_status,
        version=settings.app_version,
        environment=settings.environment,
        uptime_seconds=round(time.time() - _APP_START_TIME, 2),
        components=components,
    )
