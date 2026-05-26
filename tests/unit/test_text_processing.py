"""Unit tests for text processing utilities."""
import pytest
from app.utils.text_processing import (
    clean_financial_text,
    extract_ticker_symbols,
    extract_financial_figures,
    truncate_text,
    count_tokens_approx,
)


def test_clean_financial_text_basic():
    text = "Apple  Inc.   reported  revenue."
    cleaned = clean_financial_text(text)
    assert "  " not in cleaned


def test_clean_financial_text_removes_control_chars():
    text = "Revenue\x00\x01\x07 growth was strong."
    cleaned = clean_financial_text(text)
    assert "\x00" not in cleaned


def test_clean_financial_text_preserves_newlines():
    text = "Line 1\n\nLine 2\n\nLine 3"
    cleaned = clean_financial_text(text)
    assert "\n" in cleaned


def test_extract_ticker_symbols():
    text = "AAPL reported strong earnings. MSFT and GOOGL also performed well. The $NVDA stock rose 10%."
    tickers = extract_ticker_symbols(text)
    assert "AAPL" in tickers
    assert "MSFT" in tickers
    assert "NVDA" in tickers


def test_extract_ticker_filters_false_positives():
    text = "THE company AND its CEO said OR the CFO"
    tickers = extract_ticker_symbols(text)
    assert "THE" not in tickers
    assert "AND" not in tickers
    assert "CEO" not in tickers


def test_extract_financial_figures_monetary():
    text = "Revenue was $383.3 billion and net income was $97 million."
    figures = extract_financial_figures(text)
    monetary = [f for f in figures if f["type"] == "monetary"]
    assert len(monetary) >= 2
    values = [f["value"] for f in monetary]
    assert any(abs(v - 383.3e9) < 1e9 for v in values)
    assert any(abs(v - 97e6) < 1e6 for v in values)


def test_extract_financial_figures_percentage():
    text = "Gross margin improved to 45.2% from 42.3%."
    figures = extract_financial_figures(text)
    percentages = [f for f in figures if f["type"] == "percentage"]
    assert len(percentages) == 2
    values = [f["value"] for f in percentages]
    assert 45.2 in values
    assert 42.3 in values


def test_truncate_text_short():
    text = "Short text."
    result = truncate_text(text, max_chars=100)
    assert result == text


def test_truncate_text_long():
    text = "A" * 1000 + "."
    result = truncate_text(text, max_chars=100)
    assert len(result) <= 103  # Allow for suffix


def test_count_tokens_approx():
    text = "a" * 400  # 400 chars / 4 = 100 tokens
    assert count_tokens_approx(text) == 100


def test_count_tokens_empty():
    assert count_tokens_approx("") == 0
