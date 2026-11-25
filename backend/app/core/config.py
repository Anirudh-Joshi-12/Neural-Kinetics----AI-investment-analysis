"""Backend Settings — loads from the shared .env at project root."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Final, List

from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root is two levels up from this file.
ROOT: Final[Path] = Path(__file__).resolve().parents[3]

PLACEHOLDER_KEY = "your_nvidia_api_key_here"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Database
    database_url: str
    db_echo: bool = False

    # NVIDIA NIM
    nvidia_api_key: str
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nemotron_model: str = "nvidia/nemotron-3.5-bolt-e4b"

    # ML / paths
    sample_pdf_path: str = "./sample_annual_report.pdf"
    model_save_path: str = "./ml/scorer.joblib"
    shap_output_dir: str = "./ml/shap_output"

    # LLM
    kpi_chunk_max_chars: int = 4000
    llm_temperature: float = 0.1
    llm_max_tokens: int = 800

    @property
    def model_path_obj(self) -> Path:
        # Resolve model path relative to project root.
        p = Path(self.model_save_path)
        if not p.is_absolute():
            p = ROOT / p
        return p

    def validate_nvidia_api_key(self) -> None:
        """Fail fast if the NVIDIA API key is still the placeholder."""
        if self.nvidia_api_key == PLACEHOLDER_KEY:
            raise RuntimeError(
                "NVIDIA_API_KEY is still the placeholder value. "
                "Set a real API key in .env before starting the application."
            )


settings: Final[Settings] = Settings()
# NOTE: Validation is now lazy — call settings.validate_nvidia_api_key()
# before making Nemotron API calls. The health check endpoint will
# validate at startup to fail fast in production.

# KPI schema must match the ML package.
KPI_SCHEMA: Final[List[str]] = [
    "revenue_growth",
    "net_margin",
    "debt_equity",
    "roe",
    "current_ratio",
]

# Valuation features (from yfinance .info at scoring time)
VALUATION_SCHEMA: Final[List[str]] = [
    "trailing_pe_log",
    "forward_pe_log",
    "price_to_sales_log",
    "peg_clipped",
    "price_to_book_log",
    "market_cap_log",
    "dividend_yield",
]

# Momentum features (trailing returns)
MOMENTUM_SCHEMA: Final[List[str]] = [
    "mom_3m",
    "mom_6m",
    "mom_12m",
]

# Volatility features (realized vol)
VOLATILITY_SCHEMA: Final[List[str]] = [
    "vol_6m",
    "vol_12m",
]

# Beta feature
BETA_SCHEMA: Final[List[str]] = [
    "beta_12m",
]

# Sector-relative features
SECTOR_SCHEMA: Final[List[str]] = [
    "sector_rel_6m",
    "sector_rel_12m",
]

# Combined feature schema for training + scoring (20 features total)
ALL_FEATURES: Final[List[str]] = (
    KPI_SCHEMA + VALUATION_SCHEMA + MOMENTUM_SCHEMA + VOLATILITY_SCHEMA + BETA_SCHEMA + SECTOR_SCHEMA
)

MODEL_VERSION: Final[str] = "rf_v1.1_20feat"
