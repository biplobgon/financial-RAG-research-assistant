"""Response quality scoring pipeline."""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class QualityScorer:
    """
    Multi-dimensional quality scoring for financial AI responses.

    Dimensions:
    - Completeness (does it answer the full question?)
    - Coherence (is the reasoning logical and clear?)
    - Precision (are claims specific and precise?)
    - Professional quality (suitable for institutional use?)
    """

    async def score(
        self,
        query: str,
        response: str,
        context_docs: list[str] = None,
    ) -> dict:
        """Score response quality across multiple dimensions."""
        scores = {}

        scores["completeness"] = self._score_completeness(query, response)
        scores["coherence"] = self._score_coherence(response)
        scores["precision"] = self._score_precision(response)
        scores["professional_quality"] = self._score_professional_quality(response)

        overall = sum(scores.values()) / len(scores)

        return {
            "overall_quality": round(overall, 3),
            "dimension_scores": scores,
            "passed": overall >= 0.65,
        }

    def _score_completeness(self, query: str, response: str) -> float:
        """Score how completely the response addresses the query."""
        if len(response) < 100:
            return 0.3

        query_terms = set(query.lower().split())
        response_lower = response.lower()

        covered = sum(1 for t in query_terms if t in response_lower)
        coverage = covered / max(len(query_terms), 1)

        # Length bonus (longer responses generally more complete)
        length_score = min(len(response) / 2000, 1.0)

        return round((coverage * 0.7 + length_score * 0.3), 3)

    def _score_coherence(self, response: str) -> float:
        """Score logical coherence and structure of response."""
        score = 0.6  # Base score

        # Check for structured output markers
        structure_markers = ["1.", "2.", "3.", "•", "-", "**", "##", ":", "\n\n"]
        if any(m in response for m in structure_markers):
            score += 0.1

        # Check for financial reasoning patterns
        reasoning_patterns = [
            "because", "therefore", "as a result", "indicates", "suggests",
            "compared to", "relative to", "year-over-year", "quarter-over-quarter",
        ]
        found_reasoning = sum(1 for p in reasoning_patterns if p in response.lower())
        score += min(found_reasoning * 0.02, 0.15)

        return min(round(score, 3), 1.0)

    def _score_precision(self, response: str) -> float:
        """Score specificity and precision of claims."""
        from app.utils.text_processing import extract_financial_figures
        score = 0.5

        figures = extract_financial_figures(response)
        # More specific figures = higher precision score
        score += min(len(figures) * 0.05, 0.3)

        # Check for specific dates/periods
        import re
        periods = re.findall(r"(?:Q[1-4]\s+\d{4}|FY\s*\d{4}|\d{4})", response)
        score += min(len(periods) * 0.03, 0.2)

        return min(round(score, 3), 1.0)

    def _score_professional_quality(self, response: str) -> float:
        """Score professional suitability for institutional use."""
        score = 0.7

        # Penalize overly casual language
        casual_indicators = ["just", "basically", "kind of", "sort of", "like", "stuff", "thing"]
        casual_count = sum(1 for c in casual_indicators if f" {c} " in response.lower())
        score -= casual_count * 0.05

        # Reward financial terminology
        financial_terms = [
            "revenue", "EBITDA", "EPS", "P/E", "ROE", "ROA", "FCF", "CAGR",
            "basis points", "yield", "spread", "volatility", "liquidity",
        ]
        term_count = sum(1 for t in financial_terms if t.lower() in response.lower())
        score += min(term_count * 0.02, 0.2)

        return max(min(round(score, 3), 1.0), 0.0)
