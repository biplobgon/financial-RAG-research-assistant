"""Text processing utilities for financial documents."""
from __future__ import annotations

import re
import unicodedata
from typing import Optional


def clean_financial_text(text: str) -> str:
    """Clean and normalize financial document text."""
    if not text:
        return ""
    # Normalize unicode
    text = unicodedata.normalize("NFKD", text)
    # Remove null bytes and control characters (except newlines and tabs)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Collapse multiple whitespace (preserve single newlines)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove form feeds and page markers
    text = re.sub(r"\f", "\n", text)
    # Clean up SEC-specific artifacts
    text = re.sub(r"-{5,}", "---", text)
    text = re.sub(r"={5,}", "===", text)
    return text.strip()


def extract_ticker_symbols(text: str) -> list[str]:
    """Extract stock ticker symbols from text."""
    # Match uppercase 1-5 letter tickers, optionally preceded by $
    pattern = r"\$?([A-Z]{1,5})(?:\b)"
    matches = re.findall(pattern, text)
    # Filter common false positives
    false_positives = {"I", "A", "AN", "THE", "OR", "AND", "FOR", "AT", "IN",
                       "ON", "TO", "BE", "AS", "IS", "IT", "OF", "BY", "US",
                       "CEO", "CFO", "COO", "CTO", "MD", "VP", "SEC", "FY",
                       "Q1", "Q2", "Q3", "Q4", "YTD", "YOY", "QOQ"}
    return list(set(t for t in matches if t not in false_positives))


def extract_financial_figures(text: str) -> list[dict]:
    """Extract monetary figures and percentages from text."""
    figures = []

    # Monetary amounts: $X.Xm/b/t, $X,XXX
    money_pattern = r"\$\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*(million|billion|trillion|m|b|t|M|B|T)?"
    for match in re.finditer(money_pattern, text):
        value_str = match.group(1).replace(",", "")
        multiplier_str = match.group(2) or ""
        multipliers = {"million": 1e6, "m": 1e6, "M": 1e6,
                       "billion": 1e9, "b": 1e9, "B": 1e9,
                       "trillion": 1e12, "t": 1e12, "T": 1e12}
        multiplier = multipliers.get(multiplier_str.lower(), 1.0)
        try:
            value = float(value_str) * multiplier
            figures.append({
                "type": "monetary",
                "raw": match.group(0),
                "value": value,
                "start": match.start(),
                "end": match.end(),
            })
        except ValueError:
            pass

    # Percentages
    pct_pattern = r"(\d+(?:\.\d+)?)\s*%"
    for match in re.finditer(pct_pattern, text):
        try:
            figures.append({
                "type": "percentage",
                "raw": match.group(0),
                "value": float(match.group(1)),
                "start": match.start(),
                "end": match.end(),
            })
        except ValueError:
            pass

    return figures


def truncate_text(text: str, max_chars: int = 8000, suffix: str = "...") -> str:
    """Truncate text to max characters at sentence boundary."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    # Try to break at sentence boundary
    last_period = truncated.rfind(".")
    if last_period > max_chars * 0.8:
        return truncated[:last_period + 1] + suffix
    return truncated + suffix


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences for financial documents."""
    # Simple sentence splitter that handles common financial abbreviations
    abbreviations = r"(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|Inc|Corp|Ltd|LLC|Co|vs|etc|i\.e|e\.g|U\.S|U\.K|Fig|No|Vol)"
    pattern = rf"(?<!\b{abbreviations})(?<=[.!?])\s+"
    sentences = re.split(pattern, text)
    return [s.strip() for s in sentences if s.strip()]


def count_tokens_approx(text: str) -> int:
    """Approximate token count (4 chars ≈ 1 token for English text)."""
    return len(text) // 4


def extract_section(text: str, section_name: str) -> Optional[str]:
    """Extract a specific section from SEC filing text."""
    pattern = rf"(?i)(item\s+\d+[a-z]?\.\s+{re.escape(section_name)})(.*?)(?=item\s+\d+[a-z]?\.|$)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(2).strip()
    return None
