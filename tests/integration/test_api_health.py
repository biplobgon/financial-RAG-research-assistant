"""Integration tests for API health endpoint."""
import pytest
from unittest.mock import patch, AsyncMock


@pytest.mark.asyncio
async def test_health_endpoint_returns_200(async_app_client):
    """Health check returns 200 OK."""
    response = await async_app_client.get("/health")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_health_response_structure(async_app_client):
    """Health check response has required fields."""
    response = await async_app_client.get("/health")
    data = response.json()
    assert "status" in data
    assert "version" in data
    assert "environment" in data
    assert "uptime_seconds" in data
    assert "components" in data


@pytest.mark.asyncio
async def test_health_includes_component_list(async_app_client):
    """Health check includes component statuses."""
    response = await async_app_client.get("/health")
    data = response.json()
    components = {c["name"]: c for c in data["components"]}
    assert "api" in components
    assert components["api"]["status"] == "healthy"
