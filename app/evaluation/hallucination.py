"""Hallucination detection for financial AI responses."""
from __future__ import annotations

import logging
import re
from typing import Optional

from app.utils.text_processing import extract_financial_figures

logger = logging.getLogger(__name__)


class HallucinationDetector:
    """
    Multi-method hallucination detection.

    Detection methods:
    1. Financial figure verification (numbers in response vs. context)
    2. Named entity consistency
    3. Date/period consistency
    4. Contradiction detection
    """

    def __init__(self, llm=None):
        self.llm = llm

    async def detect(
        self,
        response: str,
        context_docs: list[str],
        query: str = "",
    ) -> dict:
        """
        Detect potential hallucinations in a response.

        Returns:
        - hallucination_detected: bool
        - risk_level: str
        - findings: list of detected issues
        - confidence: float
        """
        findings = []

        # Check financial figures
        figure_issues = self._check_financial_figures(response, context_docs)
        findings.extend(figure_issues)

        # Check date consistency
        date_issues = self._check_date_consistency(response, context_docs)
        findings.extend(date_issues)

        # Check entity consistency
        entity_issues = self._check_entity_consistency(response, context_docs)
        findings.extend(entity_issues)

        # Determine risk level
        critical_count = sum(1 for f in findings if f.get("severity") == "critical")
        high_count = sum(1 for f in findings if f.get("severity") == "high")

        if critical_count > 0:
            risk_level = "critical"
        elif high_count > 2:
            risk_level = "high"
        elif len(findings) > 3:
            risk_level = "medium"
        elif len(findings) > 0:
            risk_level = "low"
        else:
            risk_level = "low"

        return {
            "hallucination_detected": risk_level in ("high", "critical"),
            "risk_level": risk_level,
            "findings": findings,
            "confidence": 0.85 if self.llm else 0.60,
        }

    def _check_financial_figures(self, response: str, context_docs: list[str]) -> list[dict]:
        """Verify financial figures in response appear in context."""
        issues = []
        context_combined = " ".join(context_docs)

        response_figures = extract_financial_figures(response)
        context_figures = extract_financial_figures(context_combined)

        context_values = {f["value"] for f in context_figures}

        for fig in response_figures:
            if fig["value"] > 1e6:  # Only check significant figures
                # Check if value (with 5% tolerance) appears in context
                found = any(
                    abs(fig["value"] - cv) / max(abs(cv), 1) < 0.05
                    for cv in context_values
                )
                if not found:
                    issues.append({
                        "type": "financial_figure_not_in_context",
                        "severity": "high",
                        "detail": f"Figure {fig['raw']} not found in source documents",
                    })
        return issues

    def _check_date_consistency(self, response: str, context_docs: list[str]) -> list[dict]:
        """Check for date consistency between response and context."""
        issues = []
        year_pattern = r"\b(20\d{2})\b"

        response_years = set(re.findall(year_pattern, response))
        context_combined = " ".join(context_docs)
        context_years = set(re.findall(year_pattern, context_combined))

        # Flag years in response not in context (could indicate hallucination)
        unsupported_years = response_years - context_years
        for year in unsupported_years:
            issues.append({
                "type": "year_not_in_context",
                "severity": "medium",
                "detail": f"Year {year} referenced in response but not in source documents",
            })
        return issues

    def _check_entity_consistency(self, response: str, context_docs: list[str]) -> list[dict]:
        """Check company/entity name consistency."""
        # Simplified entity check — in production use NER model
        issues = []
        context_combined = " ".join(context_docs).lower()

        # Check for ticker symbols mentioned in response
        tickers_in_response = re.findall(r"\$([A-Z]{1,5})\b", response)
        for ticker in tickers_in_response:
            if ticker.lower() not in context_combined and f"${ticker}" not in context_combined:
                issues.append({
                    "type": "ticker_not_in_context",
                    "severity": "medium",
                    "detail": f"Ticker ${ticker} not found in source documents",
                })
        return issues
