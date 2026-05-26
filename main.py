"""
Financial RAG Research Assistant — FastAPI Application Entry Point.
"""
from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi

from app.config.settings import settings
from app.utils.logging import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)

APP_START_TIME = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler — startup and shutdown events."""
    logger.info(
        "Financial RAG Research Assistant starting up",
        extra={
            "version": settings.app_version,
            "environment": settings.environment,
            "llm_provider": settings.llm_provider,
            "chroma_host": settings.chroma_host,
        },
    )

    # Initialize observability
    try:
        from app.observability.telemetry import initialize_telemetry
        initialize_telemetry()
        logger.info("OpenTelemetry initialized")
    except Exception as e:
        logger.warning(f"Telemetry initialization skipped: {e}")

    # Warm up embedder
    try:
        from app.embeddings.embedder import FinancialEmbedder
        embedder = FinancialEmbedder()
        await embedder._get_client()
        logger.info("Embedding client warmed up")
    except Exception as e:
        logger.warning(f"Embedder warmup skipped: {e}")

    logger.info("Application startup complete")
    yield

    # Shutdown
    logger.info("Financial RAG Research Assistant shutting down")
    try:
        from app.utils.cache import _redis_client
        if _redis_client:
            await _redis_client.aclose()
    except Exception:
        pass


def create_app() -> FastAPI:
    """Application factory."""
    app = FastAPI(
        title=settings.app_name,
        description=(
            "Enterprise-grade Financial Intelligence Platform powered by RAG, "
            "Multi-Agent AI, and LLM workflows. Supports SEC filing analysis, "
            "earnings intelligence, portfolio Q&A, and investment research synthesis."
        ),
        version=settings.app_version,
        lifespan=lifespan,
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        openapi_url="/openapi.json",
    )

    # Middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins.split(","),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request timing middleware
    @app.middleware("http")
    async def add_timing_header(request: Request, call_next):
        start = time.monotonic()
        response = await call_next(request)
        duration_ms = (time.monotonic() - start) * 1000
        response.headers["X-Response-Time-Ms"] = f"{duration_ms:.2f}"
        response.headers["X-API-Version"] = settings.app_version
        return response

    # Exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error_code": "INTERNAL_ERROR",
                "message": "An unexpected error occurred",
                "detail": str(exc) if settings.is_debug else None,
            },
        )

    # Include routers
    from app.api.v1.routes import (
        health, query, retrieve, analyze, summarize,
        portfolio, evaluate, metrics, trace, governance,
    )

    app.include_router(health.router, tags=["Health"])
    app.include_router(query.router, prefix="/query", tags=["Query"])
    app.include_router(retrieve.router, prefix="/retrieve", tags=["Retrieval"])
    app.include_router(analyze.router, prefix="/analyze", tags=["Analysis"])
    app.include_router(summarize.router, prefix="/summarize", tags=["Summarization"])
    app.include_router(portfolio.router, prefix="/portfolio", tags=["Portfolio"])
    app.include_router(evaluate.router, prefix="/evaluate", tags=["Evaluation"])
    app.include_router(metrics.router, prefix="/metrics", tags=["Observability"])
    app.include_router(trace.router, prefix="/trace", tags=["Observability"])
    app.include_router(governance.router, prefix="/governance", tags=["Governance"])

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers if settings.is_production else 1,
        reload=not settings.is_production,
        log_config=None,  # Use our structured logger
    )
