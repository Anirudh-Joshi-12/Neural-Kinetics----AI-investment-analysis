"""Shared types and utilities for the ML package — no back-references.

This module contains only constants and pure functions. It must NOT import
from ml.pipeline, ml.ml, ml.ood, or any other ml submodule that might
import from this module. This breaks circular import chains.
"""
from __future__ import annotations

from typing import Final, List

import numpy as np
import pandas as pd

# Fixed KPI schema: must match generate_data() columns AND build_feature_vector().
KPI_SCHEMA: Final[List[str]] = [
    "revenue_growth",   # YoY revenue growth (decimal, e.g., 0.12 = 12%)
    "net_margin",       # Net profit margin (decimal, e.g., 0.08 = 8%)
    "debt_equity",      # Debt-to-equity ratio (e.g., 1.5)
    "roe",              # Return on equity (decimal, e.g., 0.18 = 18%)
    "current_ratio",    # Current assets / current liabilities (e.g., 2.1)
]

# Valuation features (from yfinance .info at scoring time)
VALUATION_SCHEMA: Final[List[str]] = [
    "trailing_pe_log",      # log(1 + trailing P/E)
    "forward_pe_log",       # log(1 + forward P/E)
    "price_to_sales_log",   # log(1 + P/S)
    "peg_clipped",          # PEG ratio clipped to [-5, 5]
    "price_to_book_log",    # log(1 + P/B)
    "market_cap_log",       # log10(market cap)
    "dividend_yield",       # dividend yield (decimal, 0 if None)
]

# Momentum features (trailing returns)
MOMENTUM_SCHEMA: Final[List[str]] = [
    "mom_3m",               # 3-month trailing return
    "mom_6m",               # 6-month trailing return
    "mom_12m",              # 12-month trailing return
]

# Volatility features (realized vol)
VOLATILITY_SCHEMA: Final[List[str]] = [
    "vol_6m",               # 6-month realized volatility (annualized)
    "vol_12m",              # 12-month realized volatility (annualized)
]

# Beta feature
BETA_SCHEMA: Final[List[str]] = [
    "beta_12m",             # 12-month rolling beta vs SPY
]

# Sector-relative features
SECTOR_SCHEMA: Final[List[str]] = [
    "sector_rel_6m",        # 6-month stock return minus sector ETF return
    "sector_rel_12m",       # 12-month stock return minus sector ETF return
]

# Combined feature schema for training + scoring (20 features total)
ALL_FEATURES: Final[List[str]] = (
    KPI_SCHEMA + VALUATION_SCHEMA + MOMENTUM_SCHEMA + VOLATILITY_SCHEMA + BETA_SCHEMA + SECTOR_SCHEMA
)

# Versioned artifacts (for reproducibility / future DB version tracking)
MODEL_VERSION: Final[str] = "rf_v1"
KPI_PROMPT_VERSION: Final[str] = "kpi_extract_v1"
BRIEF_PROMPT_VERSION: Final[str] = "brief_v1"


def generate_default_medians(n: int = 200, seed: int = 42) -> dict:
    """Generate synthetic data and return column medians.

    This is a pure function with no side effects, safe to import anywhere.
    Uses the same data generation logic as ml.ml.generate_data but without
    importing from ml.ml (which would create a cycle).
    """
    rng = np.random.default_rng(seed=seed)

    revenue_growth = rng.uniform(-0.2, 0.4, n)
    net_margin = rng.normal(0.05, 0.05, n)
    debt_equity = rng.uniform(0.1, 3.0, n)
    roe = rng.normal(0.1, 0.06, n)
    current_ratio = rng.uniform(0.5, 3.5, n)

    X = pd.DataFrame({
        "revenue_growth": revenue_growth,
        "net_margin": net_margin,
        "debt_equity": debt_equity,
        "roe": roe,
        "current_ratio": current_ratio,
    })

    return X.median().to_dict()


def generate_valuation_medians(n: int = 200, seed: int = 42) -> dict:
    """Generate realistic valuation feature medians for imputation."""
    rng = np.random.default_rng(seed=seed)

    # Simulate log(1+PE) etc. ranges
    trailing_pe_log = rng.uniform(0.5, 4.0, n)
    forward_pe_log = rng.uniform(0.5, 4.0, n)
    price_to_sales_log = rng.uniform(0.0, 3.0, n)
    peg_clipped = rng.uniform(-1.0, 3.0, n)
    price_to_book_log = rng.uniform(0.0, 3.0, n)
    market_cap_log = rng.uniform(9.0, 12.5, n)
    dividend_yield = rng.uniform(0.0, 0.05, n)

    X = pd.DataFrame({
        "trailing_pe_log": trailing_pe_log,
        "forward_pe_log": forward_pe_log,
        "price_to_sales_log": price_to_sales_log,
        "peg_clipped": peg_clipped,
        "price_to_book_log": price_to_book_log,
        "market_cap_log": market_cap_log,
        "dividend_yield": dividend_yield,
    })

    return X.median().to_dict()


def generate_momentum_medians(n: int = 200, seed: int = 42) -> dict:
    """Generate realistic momentum feature medians for imputation."""
    rng = np.random.default_rng(seed=seed)

    mom_3m = rng.uniform(-0.3, 0.3, n)
    mom_6m = rng.uniform(-0.4, 0.4, n)
    mom_12m = rng.uniform(-0.5, 0.5, n)

    X = pd.DataFrame({
        "mom_3m": mom_3m,
        "mom_6m": mom_6m,
        "mom_12m": mom_12m,
    })

    return X.median().to_dict()


def generate_volatility_medians(n: int = 200, seed: int = 42) -> dict:
    """Generate realistic volatility feature medians for imputation."""
    rng = np.random.default_rng(seed=seed)

    vol_6m = rng.uniform(0.15, 0.6, n)
    vol_12m = rng.uniform(0.15, 0.6, n)

    X = pd.DataFrame({
        "vol_6m": vol_6m,
        "vol_12m": vol_12m,
    })

    return X.median().to_dict()


def generate_beta_medians(n: int = 200, seed: int = 42) -> dict:
    """Generate realistic beta feature medians for imputation."""
    rng = np.random.default_rng(seed=seed)

    beta_12m = rng.uniform(0.5, 2.0, n)

    X = pd.DataFrame({
        "beta_12m": beta_12m,
    })

    return X.median().to_dict()


def generate_sector_medians(n: int = 200, seed: int = 42) -> dict:
    """Generate realistic sector-relative feature medians for imputation."""
    rng = np.random.default_rng(seed=seed)

    sector_rel_6m = rng.uniform(-0.3, 0.3, n)
    sector_rel_12m = rng.uniform(-0.3, 0.3, n)

    X = pd.DataFrame({
        "sector_rel_6m": sector_rel_6m,
        "sector_rel_12m": sector_rel_12m,
    })

    return X.median().to_dict()


# Module-level cached medians (computed once at import).
# This replaces the import-time call to ml.ml.generate_data in ml.pipeline.
_DEFAULT_MEDIANS: Final[dict] = generate_default_medians()
_DEFAULT_VALUATION_MEDIANS: Final[dict] = generate_valuation_medians()
_DEFAULT_MOMENTUM_MEDIANS: Final[dict] = generate_momentum_medians()
_DEFAULT_VOLATILITY_MEDIANS: Final[dict] = generate_volatility_medians()
_DEFAULT_BETA_MEDIANS: Final[dict] = generate_beta_medians()
_DEFAULT_SECTOR_MEDIANS: Final[dict] = generate_sector_medians()


def get_default_medians() -> dict:
    """Return the cached default medians for KPI imputation."""
    return _DEFAULT_MEDIANS


def get_valuation_medians() -> dict:
    """Return the cached default medians for valuation feature imputation."""
    return _DEFAULT_VALUATION_MEDIANS


def get_momentum_medians() -> dict:
    """Return the cached default medians for momentum feature imputation."""
    return _DEFAULT_MOMENTUM_MEDIANS


def get_volatility_medians() -> dict:
    """Return the cached default medians for volatility feature imputation."""
    return _DEFAULT_VOLATILITY_MEDIANS


def get_beta_medians() -> dict:
    """Return the cached default medians for beta feature imputation."""
    return _DEFAULT_BETA_MEDIANS


def get_sector_medians() -> dict:
    """Return the cached default medians for sector-relative feature imputation."""
    return _DEFAULT_SECTOR_MEDIANS