"""AI governance guardrails: PII detection, prompt injection, compliance."""
from __future__ import annotations

import logging
import re
from typing import Any

from app.models.responses import GovernanceResponse, GovernanceCheckResult, ResponseMetadata

logger = logging.getLogger(__name__)


PII_PATTERNS = {
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b(?:\d{4}[- ]){3}\d{4}\b",
    "email": r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b",
    "phone": r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "account_number": r"\b\d{8,17}\b",
}

INJECTION_PATTERNS = [
    r"ignore\s+(?:all\s+)?(?:previous|prior)\s+instructions",
    r"you\s+are\s+now\s+(?:a\s+)?(?:different|new|another)",
    r"disregard\s+(?:all\s+)?(?:system|above)\s+(?:prompt|instructions)",
    r"act\s+as\s+(?:if\s+you\s+are\s+)?(?:DAN|jailbreak|evil)",
    r"system\s*:\s*you\s+must",
    r"<\s*system\s*>",
    r"override\s+safety",
]

FINANCIAL_COMPLIANCE_PATTERNS = {
    "guarantee_of_returns": r"(?:guaranteed?|certain|definite|sure)\s+(?:return|profit|gain|yield)",
    "specific_investment_advice": r"you\s+should\s+(?:definitely|certainly|absolutely)\s+(?:buy|sell|short)",
    "price_prediction": r"(?:stock|price|share)\s+will\s+(?:definitely|certainly|absolutely)\s+(?:go|rise|fall)",
}


class GovernanceService:
    """
    AI governance service for financial platform compliance.

    Checks:
    - PII detection and redaction
    - Prompt injection detection
    - Financial regulatory compliance
    - Content appropriateness
    """

    async def run_checks(
        self,
        content: str,
        check_types: list[str],
        context: dict[str, Any] = None,
    ) -> GovernanceResponse:
        """Run all configured governance checks."""
        results = []
        pii_detected = False
        injection_detected = False
        compliance_issues = []

        if "pii" in check_types:
            pii_result = self._check_pii(content)
            results.append(pii_result)
            if not pii_result.passed:
                pii_detected = True

        if "injection" in check_types:
            injection_result = self._check_injection(content)
            results.append(injection_result)
            if not injection_result.passed:
                injection_detected = True

        if "compliance" in check_types:
            compliance_result = self._check_financial_compliance(content)
            results.append(compliance_result)
            if not compliance_result.passed:
                compliance_issues = compliance_result.findings

        overall_passed = all(r.passed for r in results)
        critical_checks = [r for r in results if r.severity == "critical" and not r.passed]
        if critical_checks:
            overall_passed = False

        return GovernanceResponse(
            status="success",
            overall_passed=overall_passed,
            checks=results,
            pii_detected=pii_detected,
            injection_detected=injection_detected,
            compliance_issues=compliance_issues,
            metadata=ResponseMetadata(agent="governance_service"),
        )

    def _check_pii(self, content: str) -> GovernanceCheckResult:
        """Detect PII in content."""
        findings = []
        for pii_type, pattern in PII_PATTERNS.items():
            if re.search(pattern, content):
                findings.append(f"Potential {pii_type.upper()} detected in content")

        return GovernanceCheckResult(
            check_type="pii",
            passed=len(findings) == 0,
            severity="critical" if findings else "low",
            findings=findings,
        )

    def _check_injection(self, content: str) -> GovernanceCheckResult:
        """Detect prompt injection attempts."""
        findings = []
        content_lower = content.lower()
        for pattern in INJECTION_PATTERNS:
            if re.search(pattern, content_lower):
                findings.append(f"Potential prompt injection detected: pattern '{pattern[:50]}'")

        return GovernanceCheckResult(
            check_type="injection",
            passed=len(findings) == 0,
            severity="critical" if findings else "low",
            findings=findings,
        )

    def _check_financial_compliance(self, content: str) -> GovernanceCheckResult:
        """Check for financial regulatory compliance issues."""
        findings = []
        content_lower = content.lower()
        for issue_type, pattern in FINANCIAL_COMPLIANCE_PATTERNS.items():
            if re.search(pattern, content_lower):
                findings.append(f"Potential compliance issue ({issue_type.replace('_', ' ')})")

        return GovernanceCheckResult(
            check_type="compliance",
            passed=len(findings) == 0,
            severity="high" if findings else "low",
            findings=findings,
        )

    def redact_pii(self, content: str) -> str:
        """Redact detected PII from content."""
        redacted = content
        replacements = {
            "ssn": "[SSN REDACTED]",
            "credit_card": "[CARD REDACTED]",
            "email": "[EMAIL REDACTED]",
            "phone": "[PHONE REDACTED]",
            "account_number": "[ACCOUNT REDACTED]",
        }
        for pii_type, pattern in PII_PATTERNS.items():
            redacted = re.sub(pattern, replacements[pii_type], redacted)
        return redacted
