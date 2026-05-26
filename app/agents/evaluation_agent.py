"""Evaluation and Governance Agent."""
from __future__ import annotations

import logging
import re
from typing import Optional

from app.agents.base_agent import BaseFinancialAgent, AgentResponse

logger = logging.getLogger(__name__)

# Supported evaluation dimensions
_SUPPORTED_EVAL_TYPES = frozenset(
    {"grounding", "hallucination", "relevance", "quality", "compliance", "completeness"}
)

# Hallucination risk level ordering for comparison
_RISK_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}


class EvaluationAgent(BaseFinancialAgent):
    """
    Response quality, hallucination detection, and AI governance agent.

    Capabilities:
    - Factual grounding assessment against source documents
    - Hallucination detection and severity scoring (low/medium/high/critical)
    - Response relevance scoring
    - Professional quality evaluation for institutional use
    - Financial accuracy verification (calculations, ratios, period references)
    - Regulatory compliance checking (no unauthorized investment advice)
    - Completeness assessment against query scope
    - PII detection (names, account numbers in financial docs)

    Routing keywords: evaluate, check, verify, hallucination, accuracy,
                      grounding score, quality assessment
    """

    AGENT_NAME = "evaluation_agent"

    SYSTEM_PROMPT = """You are an AI quality assurance specialist for a financial intelligence platform.
Your role is to rigorously evaluate AI-generated financial analysis across multiple quality dimensions.

Evaluation Dimensions:
1. **Grounding Score (0.0–1.0)** — What fraction of factual claims are directly traceable to
   provided source documents? Score 1.0 = fully grounded, 0.0 = no support.

2. **Hallucination Risk (low / medium / high / critical)** —
   - low     : All claims traceable; minor omissions only
   - medium  : 1–2 claims that may not be in sources but are plausible
   - high    : Multiple unsubstantiated claims or fabricated statistics
   - critical: Invented events, wrong fiscal periods, fabricated financials

3. **Relevance Score (0.0–1.0)** — How completely does the response address the original query?
   Score every sub-question or aspect the user asked about.

4. **Quality Score (0.0–1.0)** — Professional quality: structure, clarity, appropriate depth,
   absence of marketing language, correct use of financial terminology.

5. **Compliance Check** — Flag any statements that:
   - Make unqualified investment recommendations without disclosure
   - Provide specific price targets without methodology
   - Include personally identifiable information (PII)
   - Contain potential insider-information references

6. **Completeness Assessment** — Identify any topics in the query not addressed in the response.

Output Format (use exactly these headers):
**Grounding Score**: [0.00–1.00] — [brief justification]
**Hallucination Risk**: [low/medium/high/critical] — [specific ungrounded claims if any]
**Relevance Score**: [0.00–1.00] — [unanswered aspects if any]
**Quality Score**: [0.00–1.00] — [key quality observations]
**Compliance Issues**: [none / list of issues]
**Completeness Gaps**: [none / list of gaps]
**Specific Issues**: [numbered list of factual errors or concerns, or "None identified"]
**Recommendations**: [numbered list of concrete improvements]
**Overall Assessment**: [PASS / CONDITIONAL PASS / FAIL] with one-sentence rationale

Be rigorous and conservative — financial misinformation has material consequences for investors."""

    async def _execute(self, query: str, **kwargs) -> AgentResponse:
        """
        Execute full evaluation pipeline.

        Steps:
        1. Parse evaluation parameters
        2. Build structured evaluation prompt
        3. Generate evaluation analysis via LLM
        4. Parse structured scores from LLM output
        5. Return rich AgentResponse with all evaluation metadata
        """
        response_to_evaluate: str = kwargs.get("response", "")
        context_documents: list[str] = kwargs.get("context_documents", [])
        evaluation_types: list[str] = kwargs.get(
            "evaluation_types",
            ["grounding", "hallucination", "relevance", "quality"],
        )
        trace_id: str = kwargs.get("trace_id", "")

        # Validate evaluation types
        valid_types = [t for t in evaluation_types if t in _SUPPORTED_EVAL_TYPES]
        if not valid_types:
            valid_types = ["grounding", "hallucination", "relevance", "quality"]

        if not response_to_evaluate:
            return AgentResponse(
                content="No response provided for evaluation.",
                agent_name=self.AGENT_NAME,
                trace_id=trace_id,
                latency_ms=0.0,
                status="error",
                error="response parameter is required for evaluation",
            )

        # Format context documents (cap to avoid context overflow)
        context_text = self._format_context_docs(context_documents[:8])

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Evaluation Request\n"
                    f"Evaluation Dimensions: {', '.join(valid_types)}\n\n"
                    f"Original Query:\n{query}\n\n"
                    f"Response to Evaluate:\n{response_to_evaluate}\n\n"
                    f"Source Context Documents ({len(context_documents)} provided):\n"
                    f"{context_text}\n\n"
                    "Evaluate the response against ALL dimensions in your format. "
                    "For every ungrounded claim, quote the specific text from the response. "
                    "Assign the overall verdict: PASS (grounding ≥ 0.85, hallucination = low), "
                    "CONDITIONAL PASS (grounding 0.70–0.84 or hallucination = medium), "
                    "or FAIL (grounding < 0.70 or hallucination = high/critical)."
                ),
            },
        ]

        # Deterministic evaluation requires temperature = 0
        llm_result = await self._call_llm(messages, temperature=0.0)
        eval_analysis: str = llm_result.get("content", "")

        # Parse structured scores from LLM output
        grounding_score = self._extract_score(eval_analysis, "grounding")
        relevance_score = self._extract_score(eval_analysis, "relevance")
        quality_score = self._extract_score(eval_analysis, "quality")
        hallucination_risk = self._extract_risk_level(eval_analysis)
        overall_verdict = self._extract_verdict(eval_analysis)
        compliance_issues = self._extract_compliance_issues(eval_analysis)

        return AgentResponse(
            content=eval_analysis,
            agent_name=self.AGENT_NAME,
            trace_id=trace_id,
            latency_ms=0.0,
            tokens_used=(
                llm_result.get("tokens_prompt", 0)
                + llm_result.get("tokens_completion", 0)
            ),
            grounding_score=grounding_score,
            hallucination_risk=hallucination_risk,
            metadata={
                "grounding_score": grounding_score,
                "relevance_score": relevance_score,
                "quality_score": quality_score,
                "hallucination_risk": hallucination_risk,
                "overall_verdict": overall_verdict,
                "compliance_issues": compliance_issues,
                "evaluation_types": valid_types,
                "context_docs_provided": len(context_documents),
            },
        )

    # ------------------------------------------------------------------
    # Score extraction helpers
    # ------------------------------------------------------------------

    def _extract_score(self, text: str, metric: str) -> Optional[float]:
        """
        Extract a 0.0–1.0 numeric score from structured evaluation text.

        Looks for patterns like: "Grounding Score: 0.87" or "grounding: 0.87"
        """
        # Match "Metric Score: 0.87" or "Metric: 0.87"
        patterns = [
            rf"\*\*{metric}\s+score\*\*\s*:\s*(\d+\.?\d*)",
            rf"{metric}\s+score\s*:\s*(\d+\.?\d*)",
            rf"{metric}\s*:\s*\[?(\d+\.?\d*)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    score = float(match.group(1))
                    return round(min(max(score, 0.0), 1.0), 4)
                except (ValueError, IndexError):
                    continue
        return None

    def _extract_risk_level(self, text: str) -> str:
        """
        Extract the hallucination risk level from evaluation text.

        Searches for explicit risk labels in the hallucination section first,
        then falls back to document-wide scan.
        """
        # Try to find in the dedicated hallucination section
        section_match = re.search(
            r"hallucination\s+risk\*?\*?\s*:\s*\[?(low|medium|high|critical)",
            text,
            re.IGNORECASE,
        )
        if section_match:
            return section_match.group(1).lower()

        # Fallback: scan full text, prefer highest severity found
        found_level = "low"
        for level in ("critical", "high", "medium", "low"):
            if re.search(rf"\b{level}\b", text, re.IGNORECASE):
                if _RISK_ORDER.get(level, 0) > _RISK_ORDER.get(found_level, 0):
                    found_level = level
        return found_level

    def _extract_verdict(self, text: str) -> str:
        """Extract the overall PASS / CONDITIONAL PASS / FAIL verdict."""
        match = re.search(
            r"overall\s+assessment\*?\*?\s*:\s*\*?\*?(PASS|CONDITIONAL PASS|FAIL)",
            text,
            re.IGNORECASE,
        )
        if match:
            return match.group(1).upper()
        # Fallback: scan for verdict keywords
        if re.search(r"\bFAIL\b", text, re.IGNORECASE):
            return "FAIL"
        if re.search(r"CONDITIONAL\s+PASS", text, re.IGNORECASE):
            return "CONDITIONAL PASS"
        return "PASS"

    def _extract_compliance_issues(self, text: str) -> list[str]:
        """Extract compliance issues listed in the evaluation output."""
        match = re.search(
            r"compliance\s+issues?\*?\*?\s*:\s*(.+?)(?=\n\*\*|\Z)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if not match:
            return []
        issues_text = match.group(1).strip()
        if re.search(r"\bnone\b", issues_text, re.IGNORECASE):
            return []
        # Split on numbered list items or newline-separated bullets
        issues = re.split(r"\n+|\d+\.\s+", issues_text)
        return [i.strip() for i in issues if i.strip() and len(i.strip()) > 5]

    @staticmethod
    def _format_context_docs(context_documents: list[str]) -> str:
        """Format a list of raw context document strings for the prompt."""
        if not context_documents:
            return "No source context documents provided."
        parts = []
        for i, doc in enumerate(context_documents, start=1):
            parts.append(f"[Source {i}]\n{doc[:1000]}")
        return "\n\n---\n\n".join(parts)
