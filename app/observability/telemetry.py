"""OpenTelemetry initialization and instrumentation."""
from __future__ import annotations

import logging
from typing import Optional

from app.config.settings import settings

logger = logging.getLogger(__name__)

_tracer = None
_meter = None


def initialize_telemetry() -> None:
    """Initialize OpenTelemetry tracing and metrics."""
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.resources import Resource

        resource = Resource.create({
            "service.name": settings.otel_service_name,
            "service.version": settings.app_version,
            "deployment.environment": settings.environment,
        })

        provider = TracerProvider(resource=resource)

        # OTLP exporter
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            otlp_exporter = OTLPSpanExporter(endpoint=settings.otel_endpoint)
            provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            logger.info(f"OTLP trace exporter configured: {settings.otel_endpoint}")
        except Exception as e:
            logger.warning(f"OTLP exporter not available: {e}. Using console exporter.")
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

        trace.set_tracer_provider(provider)

        global _tracer
        _tracer = trace.get_tracer(settings.otel_service_name)
        logger.info("OpenTelemetry tracing initialized")

    except ImportError as e:
        logger.warning(f"OpenTelemetry not installed: {e}")


def get_tracer():
    """Get the global tracer instance."""
    global _tracer
    if _tracer is None:
        try:
            from opentelemetry import trace
            _tracer = trace.get_tracer(settings.otel_service_name)
        except ImportError:
            pass
    return _tracer


def create_span(name: str, attributes: dict = None):
    """Create an OpenTelemetry span context manager."""
    tracer = get_tracer()
    if tracer:
        span = tracer.start_span(name)
        if attributes:
            for k, v in attributes.items():
                span.set_attribute(k, str(v))
        return span
    return None
