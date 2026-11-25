"""Fetch market data (valuation + forward returns + momentum + sector) for real anchor filings.

Step 1+2 of the "make it real" plan:
  - Valuation features: trailing P/E, P/S, PEG, price-vs-book (yfinance .info)
  - Forward returns: 6-month and 12-month total return minus SPY return,
    measured from each company's 10-K filing date (from real_anchors.json).
  - Momentum features: 3m/6m/12m price momentum
  - Sector-relative returns: stock return minus sector ETF return
  - Volatility: 6m/12m realized volatility
  - Beta: rolling beta vs SPY

Independent labels for training: forward_return_6m / forward_return_12m
(excess over SPY).

Usage:
    python -m ml.fetch_market_data              # fetch for all cached anchors
    python -m ml.fetch_market_data --tickers NVDA AAPL
    python -m ml.fetch_market_data --list       # show cached market data

Caches to ml/data/market_data.json. Rate-limited ~0.5s between requests.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("yfinance not installed. Run: pip install yfinance")
    sys.exit(1)


ROOT = Path(__file__).resolve().parent.parent
ANCHORS_PATH = ROOT / "ml" / "data" / "real_anchors.json"
MARKET_PATH = ROOT / "ml" / "data" / "market_data.json"
SP500_PATH = ROOT / "ml" / "data" / "sp500_tickers.json"

BENCHMARK = "SPY"
DELAY_SEC = 0.5          # be polite to Yahoo
FWD_MONTHS = [6, 12]     # forward-return horizons
MOMENTUM_MONTHS = [3, 6, 12]  # momentum horizons
VOL_MONTHS = [6, 12]     # volatility horizons
BUY_THRESHOLD = 0.05     # +5% excess return over 12m -> buy
AVOID_THRESHOLD = -0.05  # -5% excess return over 12m -> avoid

# Sector ETF mapping
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

# Default sector mapping for major tickers
TICKER_SECTOR = {
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "ADBE": "XLK", "CRM": "XLK",
    "AVGO": "XLK", "QCOM": "XLK", "TXN": "XLK", "AMD": "XLK", "INTC": "XLK",
    "ORCL": "XLK", "CSCO": "XLK", "NOW": "XLK", "INTU": "XLK", "AMAT": "XLK",
    "MU": "XLK", "LRCX": "XLK", "KLAC": "XLK", "SNPS": "XLK", "CDNS": "XLK",
    "JPM": "XLF", "BAC": "XLF", "WFC": "XLF", "GS": "XLF", "MS": "XLF",
    "C": "XLF", "AXP": "XLF", "BLK": "XLF", "SCHW": "XLF", "USB": "XLF",
    "PNC": "XLF", "TFC": "XLF", "COF": "XLF",
    "JNJ": "XLV", "UNH": "XLV", "PFE": "XLV", "ABBV": "XLV", "MRK": "XLV",
    "TMO": "XLV", "DHR": "XLV", "ABT": "XLV", "AMGN": "XLV", "BMY": "XLV",
    "GILD": "XLV", "CVS": "XLV", "CI": "XLV", "HUM": "XLV", "ELV": "XLV",
    "AMZN": "XLY", "TSLA": "XLY", "HD": "XLY", "MCD": "XLY", "NKE": "XLY",
    "SBUX": "XLY", "TJX": "XLY", "LOW": "XLY", "BKNG": "XLY", "GM": "XLY",
    "F": "XLY", "CMG": "XLY", "YUM": "XLY", "MAR": "XLY",
    "PG": "XLP", "KO": "XLP", "PEP": "XLP", "WMT": "XLP", "COST": "XLP",
    "PM": "XLP", "MO": "XLP", "CL": "XLP", "KMB": "XLP", "GIS": "XLP",
    "XOM": "XLE", "CVX": "XLE", "COP": "XLE", "EOG": "XLE", "SLB": "XLE",
    "MPC": "XLE", "PSX": "XLE", "VLO": "XLE", "OXY": "XLE", "HAL": "XLE",
    "HON": "XLI", "UNP": "XLI", "UPS": "XLI", "BA": "XLI", "CAT": "XLI",
    "GE": "XLI", "MMM": "XLI", "LMT": "XLI", "RTX": "XLI", "DE": "XLI",
    "LIN": "XLB", "APD": "XLB", "SHW": "XLB", "FCX": "XLB", "NEM": "XLB",
    "DD": "XLB", "DOW": "XLB", "PPG": "XLB", "ECL": "XLB",
    "NEE": "XLU", "DUK": "XLU", "SO": "XLU", "D": "XLU", "AEP": "XLU",
    "PLD": "XLRE", "AMT": "XLRE", "EQIX": "XLRE", "CCI": "XLRE", "SPG": "XLRE",
    "GOOGL": "XLC", "META": "XLC", "NFLX": "XLC", "DIS": "XLC", "CMCSA": "XLC",
    "VZ": "XLC", "T": "XLC", "TMUS": "XLC", "CHTR": "XLC",
}


# =============================================================================
# S&P 500 ticker list (Step 3)
# =============================================================================
def fetch_sp500_tickers() -> List[str]:
    """Scrape S&P 500 constituents from Wikipedia. Cached to sp500_tickers.json."""
    if SP500_PATH.exists():
        return json.loads(SP500_PATH.read_text(encoding="utf-8"))
    print("[sp500] Fetching S&P 500 constituents from Wikipedia...")
    import requests
    import io
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) InvestmentResearch/1.0"}
    resp = requests.get(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        headers=headers, timeout=30,
    )
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    df = tables[0]
    tickers = [t.replace(".", "-") for t in df["Symbol"].tolist()]  # BRK.B -> BRK-B (Yahoo format)
    SP500_PATH.parent.mkdir(parents=True, exist_ok=True)
    SP500_PATH.write_text(json.dumps(tickers), encoding="utf-8")
    print(f"[sp500] Cached {len(tickers)} tickers -> {SP500_PATH}")
    return tickers


# =============================================================================
# Valuation + forward returns + momentum + sector
# =============================================================================
def _valuation(tk) -> Dict[str, Optional[float]]:
    """Extract valuation metrics from a yfinance Ticker."""
    info = {}
    try:
        raw = tk.info or {}
    except Exception:
        raw = {}
    def g(key: str) -> Optional[float]:
        v = raw.get(key)
        try:
            return None if v is None else float(v)
        except (TypeError, ValueError):
            return None
    return {
        "trailing_pe": g("trailingPE"),
        "forward_pe": g("forwardPE"),
        "price_to_sales": g("priceToSalesTrailing12Months"),
        "peg": g("pegRatio"),
        "price_to_book": g("priceToBook"),
        "market_cap": g("marketCap"),
        "dividend_yield": g("dividendYield"),
    }


def _price_on(tk, target_date: datetime, window_days: int = 7) -> Optional[float]:
    """Close price on/just after target_date (searches a small window)."""
    try:
        start = target_date
        end = target_date + timedelta(days=window_days)
        hist = tk.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), interval="1d")
        if hist is None or hist.empty:
            return None
        return float(hist["Close"].iloc[0])
    except Exception:
        return None


def _get_price_history(tk, start_date: datetime, end_date: datetime) -> Optional[pd.Series]:
    """Get daily close prices for a date range."""
    try:
        hist = tk.history(start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"), interval="1d")
        if hist is None or hist.empty:
            return None
        return hist["Close"]
    except Exception:
        return None


def _compute_momentum(tk, filing_date: str, months_list: Optional[List[int]] = None) -> Dict[str, Optional[float]]:
    """Compute price momentum (return) for specified months before filing date."""
    months_to_compute = months_list if months_list is not None else MOMENTUM_MONTHS
    out: Dict[str, Optional[float]] = {}
    try:
        d0 = datetime.strptime(filing_date, "%Y-%m-%d")
    except (ValueError, TypeError):
        return out
    
    for months in months_to_compute:
        key = f"mom_{months}m"
        d_start = d0 - timedelta(days=int(months * 30.44))
        hist = _get_price_history(tk, d_start, d0)
        if hist is not None and len(hist) >= 2:
            p_start = float(hist.iloc[0])
            p_end = float(hist.iloc[-1])
            if p_start > 0:
                out[key] = round((p_end - p_start) / p_start, 4)
            else:
                out[key] = None
        else:
            out[key] = None
    return out


def _compute_volatility(tk, filing_date: str, months_list: Optional[List[int]] = None) -> Dict[str, Optional[float]]:
    """Compute realized volatility (annualized) for specified months before filing date."""
    months_to_compute = months_list if months_list is not None else VOL_MONTHS
    out: Dict[str, Optional[float]] = {}
    try:
        d0 = datetime.strptime(filing_date, "%Y-%m-%d")
    except (ValueError, TypeError):
        return out
    
    for months in months_to_compute:
        key = f"vol_{months}m"
        d_start = d0 - timedelta(days=int(months * 30.44))
        hist = _get_price_history(tk, d_start, d0)
        if hist is not None and len(hist) >= 20:  # Need enough data points
            returns = hist.pct_change().dropna()
            if len(returns) >= 10:
                vol = returns.std() * np.sqrt(252)  # Annualized
                out[key] = round(float(vol), 4)
            else:
                out[key] = None
        else:
            out[key] = None
    return out


def _compute_beta(tk, filing_date: str, spy_hist: pd.Series, months: int = 12) -> Optional[float]:
    """Compute rolling beta vs SPY for specified months before filing date."""
    try:
        d0 = datetime.strptime(filing_date, "%Y-%m-%d")
    except (ValueError, TypeError):
        return None
    
    d_start = d0 - timedelta(days=int(months * 30.44))
    hist = _get_price_history(tk, d_start, d0)
    if hist is None or len(hist) < 20:
        return None
    
    stock_returns = hist.pct_change().dropna()
    
    # Align SPY returns
    tz = spy_hist.index.tz
    d0_tz = pd.Timestamp(d0).tz_localize(tz) if tz else pd.Timestamp(d0)
    d_start_tz = pd.Timestamp(d_start).tz_localize(tz) if tz else pd.Timestamp(d_start)
    spy_aligned = spy_hist[(spy_hist.index >= d_start_tz) & (spy_hist.index <= d0_tz)]
    if len(spy_aligned) < 20:
        return None
    spy_returns = spy_aligned.pct_change().dropna()
    
    # Align by index
    common_idx = stock_returns.index.intersection(spy_returns.index)
    if len(common_idx) < 20:
        return None
    
    stock_aligned = stock_returns.loc[common_idx]
    spy_aligned = spy_returns.loc[common_idx]
    
    if spy_aligned.std() == 0:
        return None
    
    beta = stock_aligned.cov(spy_aligned) / spy_aligned.var()
    return round(float(beta), 4)


def _compute_sector_return(filing_date: str, sector_hist: pd.Series, months: int = 6) -> Optional[float]:
    """Compute sector ETF return for specified months before filing date."""
    try:
        d0 = datetime.strptime(filing_date, "%Y-%m-%d")
    except (ValueError, TypeError):
        return None
    
    d_start = d0 - timedelta(days=int(months * 30.44))
    
    tz = sector_hist.index.tz
    d0_tz = pd.Timestamp(d0).tz_localize(tz) if tz else pd.Timestamp(d0)
    d_start_tz = pd.Timestamp(d_start).tz_localize(tz) if tz else pd.Timestamp(d_start)
    
    sector_aligned = sector_hist[(sector_hist.index >= d_start_tz) & (sector_hist.index <= d0_tz)]
    if sector_aligned.empty:
        return None
    
    p_start = float(sector_aligned.iloc[0])
    p_end = float(sector_aligned.iloc[-1])
    if p_start > 0:
        return round((p_end - p_start) / p_start, 4)
    return None


def _forward_returns(tk, filing_date: str, months_list: Optional[List[int]] = None) -> Dict[str, Optional[float]]:
    """Return forward returns for specified months from filing date price."""
    months_to_compute = months_list if months_list is not None else FWD_MONTHS
    out: Dict[str, Optional[float]] = {"fwd_6m": None, "fwd_12m": None}
    try:
        d0 = datetime.strptime(filing_date, "%Y-%m-%d")
    except (ValueError, TypeError):
        return out
    p0 = _price_on(tk, d0)
    if p0 is None or p0 == 0:
        return out
    for months in months_to_compute:
        key = f"fwd_{months}m"
        d_end = d0 + timedelta(days=int(months * 30.44))
        p1 = _price_on(tk, d_end, window_days=14)
        if p1 is not None:
            out[key] = round((p1 - p0) / p0, 4)
    return out


def _load_anchors() -> Dict[str, Dict[str, Any]]:
    """Load anchors cache."""
    if ANCHORS_PATH.exists():
        return json.loads(ANCHORS_PATH.read_text(encoding="utf-8"))
    return {}


def _group_anchors_by_ticker(anchors: Dict[str, Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group anchor entries by base ticker (strip year suffix)."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for key, info in anchors.items():
        if "error" in info or "kpis" not in info:
            continue
        # Extract base ticker: "AAPL_2023" -> "AAPL", "AAPL" -> "AAPL"
        if "_" in key and key.split("_")[-1].isdigit():
            base_ticker = "_".join(key.split("_")[:-1])
        else:
            base_ticker = key
        if base_ticker not in grouped:
            grouped[base_ticker] = []
        grouped[base_ticker].append({"key": key, **info})
    return grouped


def fetch_market_data(anchor_keys: List[str], force: bool = False) -> Dict[str, Dict[str, Any]]:
    """Fetch valuation + forward returns + momentum + sector per anchor filing. Caches to MARKET_PATH.

    anchor_keys: list of cache keys like ["AAPL_2023", "AAPL_2024", "MSFT_2023", ...]
    Returns dict keyed by anchor_key with valuation + forward returns + momentum + sector + label.
    """
    cache = {}
    if MARKET_PATH.exists() and not force:
        cache = json.loads(MARKET_PATH.read_text(encoding="utf-8"))

    # Determine which anchor keys need fetching
    missing = [k for k in anchor_keys if k not in cache or force]
    if not missing:
        print(f"[market] All {len(anchor_keys)} anchor filings cached; skipping.")
        return cache

    # Load anchors to get filing dates
    anchors = _load_anchors()

    # Group by ticker to avoid re-fetching valuation for each year
    grouped = _group_anchors_by_ticker(anchors)

    # Fetch SPY history once for benchmark returns
    spy = yf.Ticker(BENCHMARK)
    spy_hist = None
    try:
        spy_hist = spy.history(period="5y", interval="1d")["Close"]
    except Exception as e:
        print(f"[market] ! Could not fetch SPY history: {e}")

    # Fetch sector ETF histories
    sector_hists: Dict[str, pd.Series] = {}
    for etf_symbol in SECTOR_ETFS:
        try:
            etf = yf.Ticker(etf_symbol)
            hist = etf.history(period="5y", interval="1d")["Close"]
            if not hist.empty:
                sector_hists[etf_symbol] = hist
            time.sleep(DELAY_SEC)
        except Exception as e:
            print(f"[market] ! Could not fetch {etf_symbol} history: {e}")

    def spy_return(d0: datetime, months: int) -> Optional[float]:
        if spy_hist is None or spy_hist.empty:
            return None
        try:
            d_end = d0 + timedelta(days=int(months * 30.44))
            tz = spy_hist.index.tz
            d0_tz = pd.Timestamp(d0).tz_localize(tz) if tz else pd.Timestamp(d0)
            d_end_tz = pd.Timestamp(d_end).tz_localize(tz) if tz else pd.Timestamp(d_end)
            base = spy_hist[spy_hist.index >= d0_tz]
            end = spy_hist[spy_hist.index >= d_end_tz]
            if base.empty or end.empty:
                return None
            p0, p1 = float(base.iloc[0]), float(end.iloc[0])
            return (p1 - p0) / p0
        except Exception:
            return None

    # Process each unique ticker
    tickers_to_process = set()
    for key in missing:
        if "_" in key and key.split("_")[-1].isdigit():
            base_ticker = "_".join(key.split("_")[:-1])
        else:
            base_ticker = key
        tickers_to_process.add(base_ticker)

    # We'll fetch valuation once per ticker, then forward returns per filing
    valuation_cache: Dict[str, Dict[str, Optional[float]]] = {}

    for ticker in sorted(tickers_to_process):
        print(f"[market] Processing {ticker}...", flush=True)

        # Get all filings for this ticker from missing list
        filings = [k for k in missing if k.startswith(ticker + "_") or k == ticker]
        if not filings:
            continue

        # Fetch valuation once
        try:
            tk = yf.Ticker(ticker)
            val = _valuation(tk)
            valuation_cache[ticker] = val
            time.sleep(DELAY_SEC)
        except Exception as e:
            print(f"  ! Valuation fetch failed: {e}")
            valuation_cache[ticker] = {}
            for filing_key in filings:
                cache[filing_key] = {"error": f"Valuation fetch failed: {e}"}
            continue

        # Get sector ETF for this ticker
        sector_etf = TICKER_SECTOR.get(ticker)
        sector_hist = sector_hists.get(sector_etf) if sector_etf else None

        # Process each filing for this ticker
        for filing_key in filings:
            info = anchors.get(filing_key, {})
            filing_date = info.get("filing_date")

            fwd: Dict[str, Optional[float]] = {"fwd_6m": None, "fwd_12m": None}
            excess: Dict[str, Optional[float]] = {"excess_6m": None, "excess_12m": None}
            momentum: Dict[str, Optional[float]] = {}
            volatility: Dict[str, Optional[float]] = {}
            beta: Optional[float] = None
            sector_relative: Dict[str, Optional[float]] = {}

            if filing_date:
                # Forward returns
                fwd = _forward_returns(tk, filing_date)
                time.sleep(DELAY_SEC)
                for months in FWD_MONTHS:
                    key = f"fwd_{months}m"
                    ex_key = f"excess_{months}m"
                    sr = spy_return(datetime.strptime(filing_date, "%Y-%m-%d"), months) if filing_date else None
                    if fwd[key] is not None and sr is not None:
                        excess[ex_key] = round(fwd[key] - sr, 4)

                # Momentum (trailing returns)
                momentum = _compute_momentum(tk, filing_date)
                time.sleep(DELAY_SEC)

                # Volatility
                volatility = _compute_volatility(tk, filing_date)
                # No extra sleep needed, uses same history

                # Beta vs SPY
                if spy_hist is not None:
                    beta = _compute_beta(tk, filing_date, spy_hist)
                
                # Sector-relative returns (6m)
                if sector_hist is not None:
                    # Use momentum for stock return
                    stock_6m = momentum.get("mom_6m")
                    sector_6m = _compute_sector_return(filing_date, sector_hist, 6)
                    if stock_6m is not None and sector_6m is not None:
                        sector_relative["sector_rel_6m"] = round(stock_6m - sector_6m, 4)
                    
                    # Also 12m sector relative
                    stock_12m = momentum.get("mom_12m")
                    sector_12m = _compute_sector_return(filing_date, sector_hist, 12)
                    if stock_12m is not None and sector_12m is not None:
                        sector_relative["sector_rel_12m"] = round(stock_12m - sector_12m, 4)

            cache[filing_key] = {
                **valuation_cache[ticker],
                **fwd,
                **excess,
                **momentum,
                **volatility,
                **sector_relative,
                "beta_12m": beta,
                "label_outcome": None
            }

        MARKET_PATH.parent.mkdir(parents=True, exist_ok=True)
        MARKET_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")

    return cache


# =============================================================================
# CLI
# =============================================================================
def _cli() -> int:
    parser = argparse.ArgumentParser(description="Market data fetcher (valuation + forward returns).")
    parser.add_argument("--tickers", nargs="*", help="Explicit ticker list (base tickers, e.g. NVDA AAPL).")
    parser.add_argument("--sp500", action="store_true", help="Fetch for all S&P 500 tickers.")
    parser.add_argument("--force", action="store_true", help="Re-fetch even if cached.")
    parser.add_argument("--list", action="store_true", help="Print cached market data table.")
    parser.add_argument("--years", type=int, nargs="+", default=[2025, 2024, 2023],
                        help="Filing years to fetch (default: 2025 2024 2023).")
    parser.add_argument("--max-years-back", type=int, default=None,
                        help="Maximum years to look back from current year (e.g., 5 = last 5 years). Overrides --years.")
    args = parser.parse_args()

    if args.list:
        if not MARKET_PATH.exists():
            print("(empty cache)")
            return 0
        cache = json.loads(MARKET_PATH.read_text(encoding="utf-8"))
        print(f"{'anchor_key':<14} {'pe':>8} {'ps':>7} {'peg':>6} {'fwd6m':>8} {'fwd12m':>8} {'ex12m':>8} {'label':>6}")
        print("-" * 80)
        for t, v in sorted(cache.items()):
            if "error" in v:
                print(f"{t:<14} ERROR: {v['error']}")
                continue
            def f(x):
                return f"{x:.2f}" if isinstance(x, (int, float)) else "?"
            print(f"{t:<14} {f(v.get('trailing_pe')):>8} {f(v.get('price_to_sales')):>7} "
                  f"{f(v.get('peg')):>6} {f(v.get('fwd_6m')):>8} {f(v.get('fwd_12m')):>8} "
                  f"{f(v.get('excess_12m')):>8} {str(v.get('label_outcome')):>6}")
        return 0

    # Build list of anchor keys to fetch
    anchors = _load_anchors()
    if not anchors:
        print("No anchors cache found. Run python -m ml.fetch_anchors first.")
        return 1

    # Determine allowed years
    if args.max_years_back is not None:
        current_year = datetime.now().year
        min_year = current_year - args.max_years_back
        allowed_years = set(range(min_year, current_year + 1))
    else:
        allowed_years = set(args.years)

    # Filter anchors by year if specified
    valid_keys = []
    for key, info in anchors.items():
        if "error" in info or "kpis" not in info:
            continue
        fd = info.get("filing_date", "")
        try:
            year = int(fd[:4])
        except (ValueError, TypeError):
            continue
        if year in allowed_years:
            valid_keys.append(key)

    if not valid_keys:
        print(f"No valid anchors found for years {sorted(allowed_years)}")
        return 1

    print(f"[market] Will fetch market data for {len(valid_keys)} anchor filings...")
    fetch_market_data(valid_keys, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())