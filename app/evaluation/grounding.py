"""Factual grounding evaluation for RAG responses."""
from __future__ import annotations

import logging
import re
from typing import Optional

from app.config.settings import settings
from app.config.constants import (
    HALLUCINATION_RISK_LOW, HALLUCINATION_RISK_MEDIUM,
    HALLUCINATION_RISK_HIGH, HALLUCINATION_RISK_CRITICAL,
    GROUNDING_SCORE_EXCELLENT, GROUNDING_SCORE_GOOD,
)

logger = logging.getLogger(__name__)


class GroundingEvaluator:
    """
    Evaluates factual grounding of LLM responses against retrieved context.

    Methods:
    - LLM-based grounding (using judge model)
    - Token overlap scoring
    - Named entity overlap
    - Financial figure verification
    """

    def __init__(self, llm=None, threshold: float = None):
        self.llm = llm
        self.threshold = threshold or settings.hallucination_threshold

    async def score_grounding(
        self,
        query: str,
        response: str,
        context_docs: list[str],
    ) -> dict:
        """
        Score how well the response is grounded in context documents.

        Returns dict with:
        - grounding_score: float 0.0-1.0
        - hallucination_risk: str (low/medium/high/critical)
        - passed: bool
        """
        if not response or not context_docs:
            return {
                "grounding_score": 0.0,
                "hallucination_risk": HALLUCINATION_RISK_HIGH,
                "passed": False,
            }

        # Try LLM-based evaluation first
        if self.llm:
            try:
                return await self._llm_grounding_score(query, response, context_docs)
            except Exception as e:
                logger.warning(f"LLM grounding evaluation failed, using heuristic: {e}")

        # Fallback to heuristic scoring
        return self._heuristic_grounding_score(response, context_docs)

    async def _llm_grounding_score(
        self,
        query: str,
        response: str,
        context_docs: list[str],
    ) -> dict:
        """LLM-as-judge grounding evaluation."""
        context_text = "\n---\n".join(context_docs[:5])

        judge_prompt = f"""You are evaluating factual grounding of a financial AI response.

Query: {query}

Source Documents:
{context_text[:4000]}

AI Response:
{response[:2000]}

Score the factual grounding on a scale of 0.0 to 1.0:
- 1.0: Every claim directly supported by source documents
- 0.8-0.9: Most claims supported, minor inferences acceptable
- 0.6-0.7: Some claims supported, notable gaps
- 0.4-0.5: Many claims unsupported or questionable
- 0.0-0.3: Response is largely not grounded in sources

Respond with only: SCORE: X.XX"""

        messages = [
            {"role": "system", "content": "You are a rigorous financial fact-checker. Be conservative in your scoring."},
            {"role": "user", "content": judge_prompt},
        ]

        result = await self.llm.generate(messages, temperature=0.0)
        score_text = result.get("content", "")

        # Extract score
        match = re.search(r"SCORE:\s*(\d+\.?\d*)", score_text)
        if match:
            score = min(max(float(match.group(1)), 0.0), 1.0)
        else:
            score = 0.7  # Default if parsing fails

        return {
            "grounding_score": score,
            "hallucination_risk": self._score_to_risk(score),
            "passed": score >= self.threshold,
        }

    def _heuristic_grounding_score(self, response: str, context_docs: list[str]) -> dict:
        """Token overlap heuristic for grounding scoring."""
        response_tokens = set(response.lower().split())
        context_combined = " ".join(context_docs).lower()
        context_tokens = set(context_combined.split())

        if not response_tokens:
            return {"grounding_score": 0.0, "hallucination_risk": HALLUCINATION_RISK_HIGH, "passed": False}

        # Compute Jaccard-style overlap
        intersection = len(response_tokens & context_tokens)
        coverage = intersection / max(len(response_tokens), 1)

        # Scale to reasonable range (pure overlap tends to be low)
        score = min(coverage * 3.0, 1.0)

        return {
            "grounding_score": round(score, 3),
            "hallucination_risk": self._score_to_risk(score),
            "passed": score >= self.threshold,
        }

    def _score_to_risk(self, score: float) -> str:
        """Convert grounding score to risk level."""
        if score >= GROUNDING_SCORE_EXCELLENT:
            return HALLUCINATION_RISK_LOW
        elif score >= GROUNDING_SCORE_GOOD:
            return HALLUCINATION_RISK_LOW
        elif score >= 0.70:
            return HALLUCINATION_RISK_MEDIUM
        elif score >= 0.50:
            return HALLUCINATION_RISK_HIGH
        return HALLUCINATION_RISK_CRITICAL
