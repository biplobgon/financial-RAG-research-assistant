"""Financial computation and data utilities."""
from __future__ import annotations

import re
from typing import Optional


def normalize_ticker(ticker: str) -> str:
    """Normalize stock ticker to uppercase, strip whitespace."""
    return ticker.strip().upper()


def validate_ticker(ticker: str) -> bool:
    """Validate stock ticker format (1-5 uppercase letters)."""
    return bool(re.match(r"^[A-Z]{1,5}$", ticker.strip().upper()))


def format_currency(value: float, currency: str = "USD") -> str:
    """Format a float value as currency string."""
    if abs(value) >= 1e12:
        return f"${value/1e12:.2f}T"
    elif abs(value) >= 1e9:
        return f"${value/1e9:.2f}B"
    elif abs(value) >= 1e6:
        return f"${value/1e6:.2f}M"
    elif abs(value) >= 1e3:
        return f"${value/1e3:.2f}K"
    else:
        return f"${value:.2f}"


def compute_returns(prices: list[float]) -> list[float]:
    """Compute period returns from price series."""
    if len(prices) < 2:
        return []
    return [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]


def compute_cagr(start_value: float, end_value: float, years: float) -> Optional[float]:
    """Compute Compound Annual Growth Rate."""
    if start_value <= 0 or years <= 0:
        return None
    return (end_value / start_value) ** (1 / years) - 1


def compute_sharpe_ratio(
    returns: list[float],
    risk_free_rate: float = 0.05,
    periods_per_year: int = 252,
) -> Optional[float]:
    """Compute Sharpe ratio from return series."""
    if len(returns) < 2:
        return None
    import statistics
    mean_return = statistics.mean(returns)
    std_return = statistics.stdev(returns)
    if std_return == 0:
        return None
    excess_return = mean_return - risk_free_rate / periods_per_year
    return (excess_return / std_return) * (periods_per_year ** 0.5)


def compute_max_drawdown(prices: list[float]) -> Optional[float]:
    """Compute maximum drawdown from price series."""
    if len(prices) < 2:
        return None
    peak = prices[0]
    max_dd = 0.0
    for price in prices[1:]:
        if price > peak:
            peak = price
        drawdown = (peak - price) / peak if peak != 0 else 0
        max_dd = max(max_dd, drawdown)
    return max_dd


def fiscal_quarter_to_dates(quarter_str: str) -> Optional[tuple[str, str]]:
    """Convert 'Q3 2024' to (start_date, end_date) strings."""
    match = re.match(r"Q(\d)\s+(\d{4})", quarter_str.strip())
    if not match:
        return None
    q, year = int(match.group(1)), int(match.group(2))
    quarter_months = {1: ("01-01", "03-31"), 2: ("04-01", "06-30"),
                      3: ("07-01", "09-30"), 4: ("10-01", "12-31")}
    if q not in quarter_months:
        return None
    start_m, end_m = quarter_months[q]
    return f"{year}-{start_m}", f"{year}-{end_m}"


def parse_market_cap_tier(market_cap: float) -> str:
    """Classify company by market cap tier."""
    if market_cap >= 200e9:
        return "Mega-Cap"
    elif market_cap >= 10e9:
        return "Large-Cap"
    elif market_cap >= 2e9:
        return "Mid-Cap"
    elif market_cap >= 300e6:
        return "Small-Cap"
    elif market_cap >= 50e6:
        return "Micro-Cap"
    else:
        return "Nano-Cap"
