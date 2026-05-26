"""Unit tests for grounding evaluator."""
import pytest
from unittest.mock import AsyncMock

from app.evaluation.grounding import GroundingEvaluator


@pytest.fixture
def evaluator():
    return GroundingEvaluator(llm=None, threshold=0.85)


@pytest.mark.asyncio
async def test_heuristic_grounding_high_overlap(evaluator):
    """High token overlap yields high grounding score."""
    context = ["Apple reported revenue of 383 billion and net income of 97 billion in fiscal 2023."]
    response = "Apple reported revenue of 383 billion in fiscal 2023 and net income reached 97 billion."
    result = await evaluator.score_grounding("Query", response, context)
    assert result["grounding_score"] > 0.0
    assert "hallucination_risk" in result


@pytest.mark.asyncio
async def test_empty_response_fails(evaluator):
    """Empty response should return low grounding score."""
    result = await evaluator.score_grounding("Query", "", ["Some context"])
    assert result["grounding_score"] == 0.0
    assert result["passed"] is False


@pytest.mark.asyncio
async def test_empty_context_fails(evaluator):
    """No context should return low grounding score."""
    result = await evaluator.score_grounding("Query", "Some response", [])
    assert result["grounding_score"] == 0.0
    assert result["passed"] is False


def test_score_to_risk_level(evaluator):
    """Risk levels map correctly to score ranges."""
    assert evaluator._score_to_risk(0.96) == "low"
    assert evaluator._score_to_risk(0.86) == "low"
    assert evaluator._score_to_risk(0.75) == "medium"
    assert evaluator._score_to_risk(0.55) == "high"
    assert evaluator._score_to_risk(0.3) == "critical"
