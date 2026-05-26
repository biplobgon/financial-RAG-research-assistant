# API Documentation Notes

## OpenAPI Specification

Interactive API documentation is available at:
- Development: http://localhost:8000/docs (Swagger UI)
- Development: http://localhost:8000/redoc (ReDoc)
- Spec: http://localhost:8000/openapi.json

## Authentication

For production deployments, API key authentication is supported via the `X-API-Key` header.

Configure valid keys in the `API_KEYS` environment variable (comma-separated).

## Rate Limiting

Rate limiting is not implemented in the default configuration. For production:
- Use NGINX rate limiting at the gateway level
- Implement Redis-based rate limiting middleware

## Response Format

All responses follow the standard format:

```json
{
  "status": "success | error | partial",
  "data": {},
  "metadata": {
    "agent": "agent_name",
    "latency_ms": 1843,
    "tokens_used": 2847,
    "grounding_score": 0.93,
    "hallucination_risk": "low",
    "trace_id": "uuid",
    "model": "gpt-4o",
    "cache_hit": false
  }
}
```

## Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| VALIDATION_ERROR | 422 | Request schema validation failed |
| AGENT_NOT_FOUND | 404 | Requested agent not registered |
| LLM_ERROR | 500 | LLM provider call failed |
| RETRIEVAL_ERROR | 500 | ChromaDB retrieval failed |
| INTERNAL_ERROR | 500 | Unexpected server error |
