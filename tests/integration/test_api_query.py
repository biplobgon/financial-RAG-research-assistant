"""Integration tests for query endpoint."""
import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_query_endpoint_returns_200(async_app_client):
    """Query endpoint returns 200 for valid request."""
    payload = {
        "query": "What is Apple's revenue for FY2023?",
        "agent": "research_agent",
        "top_k": 5,
    }
    with patch("app.api.v1.dependencies._rag_pipeline") as mock_pipeline:
        from app.models.responses import QueryResponse, ResponseMetadata
        mock_pipeline.query = AsyncMock(return_value=QueryResponse(
            status="success",
            query=payload["query"],
            answer="Apple reported $383.3 billion in revenue for FY2023.",
            sources=[],
            metadata=ResponseMetadata(latency_ms=100.0, tokens_used=300),
        ))
        response = await async_app_client.post("/query", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "answer" in data
        assert "metadata" in data


@pytest.mark.asyncio
async def test_query_validates_agent_name(async_app_client):
    """Query endpoint rejects invalid agent names."""
    payload = {
        "query": "Test query",
        "agent": "nonexistent_agent",
    }
    response = await async_app_client.post("/query", json=payload)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_query_requires_minimum_length(async_app_client):
    """Query endpoint rejects very short queries."""
    payload = {"query": "ab"}
    response = await async_app_client.post("/query", json=payload)
    assert response.status_code == 422
