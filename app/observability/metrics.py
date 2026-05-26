"""Prometheus metrics instrumentation."""
from __future__ import annotations

import logging
import time
from functools import wraps
from typing import Callable

from app.config.constants import (
    METRIC_REQUEST_DURATION, METRIC_REQUEST_COUNT, METRIC_LLM_TOKENS,
    METRIC_RETRIEVAL_COUNT, METRIC_GROUNDING_SCORE, METRIC_CACHE_HITS,
    METRIC_CACHE_MISSES,
)

logger = logging.getLogger(__name__)

# Prometheus metric objects (initialized lazily)
_metrics = {}


def _get_metrics() -> dict:
    """Lazily initialize Prometheus metrics."""
    global _metrics
    if _metrics:
        return _metrics

    try:
        from prometheus_client import Counter, Histogram, Gauge, Summary

        _metrics = {
            "request_duration": Histogram(
                METRIC_REQUEST_DURATION,
                "API request duration in seconds",
                ["method", "endpoint", "status"],
                buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
            ),
            "request_count": Counter(
                METRIC_REQUEST_COUNT,
                "Total API requests",
                ["method", "endpoint", "status"],
            ),
            "llm_tokens": Counter(
                METRIC_LLM_TOKENS,
                "Total LLM tokens consumed",
                ["model", "token_type"],
            ),
            "retrieval_count": Histogram(
                METRIC_RETRIEVAL_COUNT,
                "Documents retrieved per query",
                ["collection", "mode"],
                buckets=[1, 3, 5, 10, 20, 50],
            ),
            "grounding_score": Histogram(
                METRIC_GROUNDING_SCORE,
                "Response grounding scores",
                ["agent"],
                buckets=[0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0],
            ),
            "cache_hits": Counter(METRIC_CACHE_HITS, "Cache hit count"),
            "cache_misses": Counter(METRIC_CACHE_MISSES, "Cache miss count"),
        }
        logger.info("Prometheus metrics initialized")
    except ImportError:
        logger.warning("prometheus_client not installed, metrics disabled")
        _metrics = {}

    return _metrics


def record_request(method: str, endpoint: str, status: str, duration: float) -> None:
    """Record API request metrics."""
    m = _get_metrics()
    if m:
        m["request_count"].labels(method=method, endpoint=endpoint, status=status).inc()
        m["request_duration"].labels(method=method, endpoint=endpoint, status=status).observe(duration)


def record_llm_tokens(model: str, prompt_tokens: int, completion_tokens: int) -> None:
    """Record LLM token usage."""
    m = _get_metrics()
    if m:
        m["llm_tokens"].labels(model=model, token_type="prompt").inc(prompt_tokens)
        m["llm_tokens"].labels(model=model, token_type="completion").inc(completion_tokens)


def record_retrieval(collection: str, mode: str, count: int) -> None:
    """Record retrieval document count."""
    m = _get_metrics()
    if m:
        m["retrieval_count"].labels(collection=collection, mode=mode).observe(count)


def record_grounding_score(agent: str, score: float) -> None:
    """Record grounding evaluation score."""
    m = _get_metrics()
    if m:
        m["grounding_score"].labels(agent=agent).observe(score)


def record_cache_hit() -> None:
    """Increment cache hit counter."""
    m = _get_metrics()
    if m:
        m["cache_hits"].inc()


def record_cache_miss() -> None:
    """Increment cache miss counter."""
    m = _get_metrics()
    if m:
        m["cache_misses"].inc()
