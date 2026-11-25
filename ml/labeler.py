"""Label generation module for market data.

Provides multiple labeling strategies:
1. Fixed threshold (baseline)
2. Quantile-based (top/bottom N% per quarter)
3. Sector-neutral excess returns
4. Risk-adjusted (Sharpe-like)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import numpy as np
import pandas as pd


# Sector ETF mapping for sector-neutral returns
SECTOR_ETFS = {
    "XLK": "Technology",
    "XLF": "Financials", 
    "XLV": "Healthcare",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLE": "Energy",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "XLC": "Communication Services",
}

# Default sector mapping for major tickers (extendable)
TICKER_SECTOR = {
    # Technology
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "ADBE": "XLK", "CRM": "XLK",
    "AVGO": "XLK", "QCOM": "XLK", "TXN": "XLK", "AMD": "XLK", "INTC": "XLK",
    "ORCL": "XLK", "CSCO": "XLK", "NOW": "XLK", "INTU": "XLK", "AMAT": "XLK",
    "MU": "XLK", "LRCX": "XLK", "KLAC": "XLK", "SNPS": "XLK", "CDNS": "XLK",
    
    # Financials
    "JPM": "XLF", "BAC": "XLF", "WFC": "XLF", "GS": "XLF", "MS": "XLF",
    "C": "XLF", "AXP": "XLF", "BLK": "XLF", "SCHW": "XLF", "USB": "XLF",
    "PNC": "XLF", "TFC": "XLF", "COF": "XLF", "AXP": "XLF",
    
    # Healthcare
    "JNJ": "XLV", "UNH": "XLV", "PFE": "XLV", "ABBV": "XLV", "MRK": "XLV",
    "TMO": "XLV", "DHR": "XLV", "ABT": "XLV", "AMGN": "XLV", "BMY": "XLV",
    "GILD": "XLV", "CVS": "XLV", "CI": "XLV", "HUM": "XLV", "ELV": "XLV",
    
    # Consumer Discretionary
    "AMZN": "XLY", "TSLA": "XLY", "HD": "XLY", "MCD": "XLY", "NKE": "XLY",
    "SBUX": "XLY", "TJX": "XLY", "LOW": "XLY", "BKNG": "XLY", "GM": "XLY",
    "F": "XLY", "CMG": "XLY", "YUM": "XLY", "MAR": "XLY",
    
    # Consumer Staples
    "PG": "XLP", "KO": "XLP", "PEP": "XLP", "WMT": "XLP", "COST": "XLP",
    "PM": "XLP", "MO": "XLP", "CL": "XLP", "KMB": "XLP", "GIS": "XLP",
    
    # Energy
    "XOM": "XLE", "CVX": "XLE", "COP": "XLE", "EOG": "XLE", "SLB": "XLE",
    "MPC": "XLE", "PSX": "XLE", "VLO": "XLE", "OXY": "XLE", "HAL": "XLE",
    
    # Industrials
    "HON": "XLI", "UNP": "XLI", "UPS": "XLI", "BA": "XLI", "CAT": "XLI",
    "GE": "XLI", "MMM": "XLI", "LMT": "XLI", "RTX": "XLI", "DE": "XLI",
    
    # Materials
    "LIN": "XLB", "APD": "XLB", "SHW": "XLB", "FCX": "XLB", "NEM": "XLB",
    "DD": "XLB", "DOW": "XLB", "PPG": "XLB", "ECL": "XLB",
    
    # Utilities
    "NEE": "XLU", "DUK": "XLU", "SO": "XLU", "D": "XLU", "AEP": "XLU",
    
    # Real Estate
    "PLD": "XLRE", "AMT": "XLRE", "EQIX": "XLRE", "CCI": "XLRE", "SPG": "XLRE",
    
    # Communication Services
    "GOOGL": "XLC", "META": "XLC", "NFLX": "XLC", "DIS": "XLC", "CMCSA": "XLC",
    "VZ": "XLC", "T": "XLC", "TMUS": "XLC", "CHTR": "XLC",
}


@dataclass
class LabelConfig:
    """Configuration for label generation."""
    # Fixed threshold (baseline)
    buy_threshold: float = 0.05
    avoid_threshold: float = -0.05
    
    # Quantile-based
    quantile_low: float = 0.30
    quantile_high: float = 0.70
    
    # Sector-neutral
    use_sector_neutral: bool = False
    sector_etf_returns: Optional[Dict[str, float]] = None
    
    # Risk-adjusted
    use_risk_adjusted: bool = False
    vol_window_months: int = 6


def compute_sector_neutral_excess(
    stock_return: float,
    ticker: str,
    sector_etf_returns: Dict[str, float]
) -> Optional[float]:
    """Compute sector-neutral excess return."""
    sector_etf = TICKER_SECTOR.get(ticker)
    if not sector_etf or sector_etf not in sector_etf_returns:
        return None
    sector_return = sector_etf_returns[sector_etf]
    return stock_return - sector_return


def compute_risk_adjusted_return(
    excess_return: float,
    ticker: str,
    market_data: Dict,
    vol_window_months: int = 6
) -> Optional[float]:
    """Compute risk-adjusted excess return (Sharpe-like)."""
    # Would need historical volatility data
    # Placeholder for now
    return excess_return


def get_quarter(dt: datetime) -> str:
    """Get quarter string (e.g., '2023Q1') from datetime."""
    quarter = (dt.month - 1) // 3 + 1
    return f"{dt.year}Q{quarter}"


def compute_quantile_thresholds(
    excess_returns: pd.Series,
    low_quantile: float = 0.30,
    high_quantile: float = 0.70
) -> Tuple[float, float]:
    """Compute quantile thresholds for labeling."""
    low = excess_returns.quantile(low_quantile)
    high = excess_returns.quantile(high_quantile)
    return float(low), float(high)


def compute_quantile_labels(
    excess_returns: pd.Series,
    filing_dates: pd.Series,
    low_quantile: float = 0.30,
    high_quantile: float = 0.70
) -> pd.Series:
    """
    Compute quantile-based labels per quarter.
    
    Returns:
        Series with 1 (buy), 0 (avoid), -1 (hold/neutral)
    """
    # Convert filing_dates to datetime if they're strings
    filing_dates_dt = pd.to_datetime(filing_dates, errors='coerce')
    df = pd.DataFrame({
        'excess': excess_returns,
        'quarter': filing_dates_dt.apply(get_quarter)
    })
    
    labels = pd.Series(-1, index=excess_returns.index)
    
    for quarter, group in df.groupby('quarter'):
        if len(group) < 10:  # Too few samples
            continue
        low, high = compute_quantile_thresholds(
            group['excess'], 0.30, 0.70
        )
        mask = df['quarter'] == quarter
        labels.loc[mask & (df['excess'] >= high)] = 1
        labels.loc[mask & (df['excess'] <= low)] = 0
        # -1 remains for hold
    
    return labels


def compute_sector_neutral_labels(
    excess_returns: pd.Series,
    tickers: pd.Series,
    sector_etf_returns: Dict[str, float],
    buy_threshold: float = 0.05,
    avoid_threshold: float = -0.05
) -> pd.Series:
    """Compute labels using sector-neutral excess returns."""
    # Align tickers to excess_returns index
    tickers_aligned = tickers.reindex(excess_returns.index)
    labels = pd.Series(index=excess_returns.index, dtype=int)
    
    for idx in excess_returns.index:
        ticker = tickers_aligned.loc[idx]
        excess = excess_returns.loc[idx]
        
        if pd.isna(ticker) or pd.isna(excess):
            labels.loc[idx] = -1
            continue
        
        sector_etf = TICKER_SECTOR.get(ticker)
        if not sector_etf or sector_etf not in sector_etf_returns:
            # Fall back to absolute
            if excess > 0.05:
                labels.loc[idx] = 1
            elif excess < -0.05:
                labels.loc[idx] = 0
            else:
                labels.loc[idx] = -1
            continue
        
        sector_return = sector_etf_returns[sector_etf]
        sector_neutral = excess - sector_return
        
        if sector_neutral > 0.05:
            labels.loc[idx] = 1
        elif sector_neutral < -0.05:
            labels.loc[idx] = 0
        else:
            labels.loc[idx] = -1
    
    return labels


def compute_risk_adjusted_labels(
    excess_returns: pd.Series,
    volatilities: pd.Series,
    buy_threshold: float = 0.05,
    avoid_threshold: float = -0.05
) -> pd.Series:
    """Compute risk-adjusted labels (excess return / volatility)."""
    risk_adjusted = excess_returns / (volatilities + 1e-6)
    
    labels = pd.Series(index=excess_returns.index, dtype=int)
    labels[risk_adjusted > 0.5] = 1
    labels[risk_adjusted < -0.5] = 0
    labels[(risk_adjusted >= -0.5) & (risk_adjusted <= 0.5)] = -1
    
    return labels


@dataclass
class LabelResult:
    """Result of label generation."""
    labels: pd.Series  # 1=buy, 0=avoid, -1=hold
    method: str
    metadata: dict


def generate_labels(
    excess_returns: pd.Series,
    filing_dates: pd.Series,
    tickers: pd.Series,
    method: str = "fixed",
    **kwargs
) -> LabelResult:
    """
    Generate labels using specified method.
    
    Methods:
    - 'fixed': Fixed threshold (±5%)
    - 'quantile': Quantile-based per quarter (top/bottom 30%)
    - 'sector_neutral': Sector-neutral excess returns
    - 'risk_adjusted': Risk-adjusted (excess/volatility)
    - 'ensemble': Weighted combination of above
    """
    if method == "fixed":
        buy_thresh = kwargs.get("buy_threshold", 0.05)
        avoid_thresh = kwargs.get("avoid_threshold", -0.05)
        labels = pd.Series(-1, index=excess_returns.index, dtype=int)
        labels[excess_returns > kwargs.get("buy_threshold", 0.05)] = 1
        labels[excess_returns < kwargs.get("avoid_threshold", -0.05)] = 0
        return LabelResult(labels, "fixed", {"buy_threshold": buy_threshold, "avoid_threshold": avoid_thresh})
    
    elif method == "quantile":
        low_q = kwargs.get("low_quantile", 0.30)
        high_q = kwargs.get("high_quantile", 0.70)
        labels = compute_quantile_labels(
            excess_returns, 
            pd.Series([get_quarter(pd.Timestamp(d)) for d in filing_dates]),
            low_q, high_q
        )
        return LabelResult(labels, "quantile", {"low_quantile": low_q, "high_quantile": high_q})
    
    elif method == "sector_neutral":
        sector_etf_returns = kwargs.get("sector_etf_returns", {})
        labels = compute_sector_neutral_labels(
            excess_returns, 
            kwargs.get("tickers", pd.Series()),
            kwargs.get("sector_etf_returns", {}),
            kwargs.get("buy_threshold", 0.05),
            kwargs.get("avoid_threshold", -0.05)
        )
        return LabelResult(labels, "sector_neutral", {"sector_etf_returns": list(sector_etf_returns.keys())})
    
    elif method == "risk_adjusted":
        volatilities = kwargs.get("volatilities")
        if volatilities is None:
            raise ValueError("volatilities required for risk_adjusted method")
        labels = compute_risk_adjusted_labels(
            excess_returns,
            volatilities,
            kwargs.get("buy_threshold", 0.5),
            kwargs.get("avoid_threshold", -0.5)
        )
        return LabelResult(labels, "risk_adjusted", {})
    
    elif method == "ensemble":
        # Weighted combination
        weights = kwargs.get("weights", {"fixed": 0.3, "quantile": 0.3, "sector": 0.2, "risk": 0.2})
        # Simplified: just use quantile as primary
        labels = compute_quantile_labels(excess_returns, filing_dates)
        return LabelResult(labels, "ensemble", {"weights": weights})
    
    else:
        raise ValueError(f"Unknown method: {method}")