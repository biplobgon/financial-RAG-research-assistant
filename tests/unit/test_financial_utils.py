"""Unit tests for financial utility functions."""
import pytest
from app.utils.financial_utils import (
    normalize_ticker,
    validate_ticker,
    format_currency,
    compute_returns,
    compute_cagr,
    compute_max_drawdown,
    fiscal_quarter_to_dates,
    parse_market_cap_tier,
)


def test_normalize_ticker():
    assert normalize_ticker("aapl") == "AAPL"
    assert normalize_ticker(" MSFT ") == "MSFT"
    assert normalize_ticker("googl") == "GOOGL"


def test_validate_ticker_valid():
    assert validate_ticker("AAPL") is True
    assert validate_ticker("MSFT") is True
    assert validate_ticker("A") is True
    assert validate_ticker("GOOGL") is True


def test_validate_ticker_invalid():
    assert validate_ticker("TOOLONG") is False
    assert validate_ticker("123") is False
    assert validate_ticker("") is False


def test_format_currency_billions():
    assert format_currency(383.3e9) == "$383.30B"


def test_format_currency_millions():
    assert format_currency(97e6) == "$97.00M"


def test_format_currency_thousands():
    assert format_currency(500_000) == "$500.00K"


def test_compute_returns():
    prices = [100.0, 110.0, 105.0, 115.0]
    returns = compute_returns(prices)
    assert len(returns) == 3
    assert abs(returns[0] - 0.10) < 0.001
    assert abs(returns[1] - (-0.04545)) < 0.001


def test_compute_returns_single_price():
    assert compute_returns([100.0]) == []


def test_compute_cagr():
    cagr = compute_cagr(100.0, 200.0, 10.0)
    assert abs(cagr - 0.07177) < 0.001


def test_compute_cagr_invalid():
    assert compute_cagr(0.0, 200.0, 10.0) is None
    assert compute_cagr(100.0, 200.0, 0.0) is None


def test_compute_max_drawdown():
    prices = [100.0, 120.0, 80.0, 90.0, 150.0]
    dd = compute_max_drawdown(prices)
    assert abs(dd - (40.0 / 120.0)) < 0.001


def test_fiscal_quarter_to_dates():
    result = fiscal_quarter_to_dates("Q3 2024")
    assert result == ("2024-07-01", "2024-09-30")


def test_fiscal_quarter_to_dates_q1():
    result = fiscal_quarter_to_dates("Q1 2023")
    assert result == ("2023-01-01", "2023-03-31")


def test_fiscal_quarter_to_dates_invalid():
    assert fiscal_quarter_to_dates("Q5 2024") is None
    assert fiscal_quarter_to_dates("invalid") is None


def test_parse_market_cap_tier():
    assert parse_market_cap_tier(3e12) == "Mega-Cap"
    assert parse_market_cap_tier(50e9) == "Large-Cap"
    assert parse_market_cap_tier(5e9) == "Mid-Cap"
    assert parse_market_cap_tier(1e9) == "Small-Cap"
    assert parse_market_cap_tier(100e6) == "Micro-Cap"
