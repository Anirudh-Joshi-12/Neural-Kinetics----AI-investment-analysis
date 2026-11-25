"""Central configuration for the Phase 1 ML + GenAI pipeline.

All constants and API keys live here. Every other module imports from here,
so updating a value updates it everywhere. Values are loaded from the .env
file via pydantic-settings.
"""
from __future__ import annotations

from pathlib import Path
from typing import Final, List

from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve .env at project root (one level up from this file's parent dir).
_ROOT = Path(__file__).resolve().parent.parent

PLACEHOLDER_KEY = "your_nvidia_api_key_here"


class Settings(BaseSettings):
    """Typed settings loaded from the .env file."""

    model_config = SettingsConfigDict(
        env_file=str(_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # NVIDIA NIM / Nemotron
    nvidia_api_key: str
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nemotron_model: str = "nvidia/nemotron-3-ultra-550b-a55b"

    # File paths
    sample_pdf_path: str = "./nvda-20260125.pdf"
    model_save_path: str = "./ml/scorer.joblib"
    calibration_path: str = "./ml/calibration.json"
    shap_output_dir: str = "./ml/shap_output"

    # ML hyperparameters
    n_samples: int = 200
    test_size: float = 0.2
    min_accuracy: float = 0.75
    min_confidence: float = 0.65

    # LLM
    kpi_chunk_max_chars: int = 4000
    llm_temperature: float = 0.1
    llm_max_tokens: int = 800

    @property
    def model_path_obj(self) -> Path:
        return Path(self.model_save_path)

    @property
    def calibration_path_obj(self) -> Path:
        return Path(self.calibration_path)

    @property
    def shap_dir_obj(self) -> Path:
        return Path(self.shap_output_dir)

    @property
    def sample_pdf_obj(self) -> Path:
        return Path(self.sample_pdf_path)

    def validate_nvidia_api_key(self) -> None:
        """Fail fast if the NVIDIA API key is still the placeholder.

        Call this before making any API calls to Nemotron.
        """
        if self.nvidia_api_key == PLACEHOLDER_KEY:
            raise RuntimeError(
                "NVIDIA_API_KEY is still the placeholder value. "
                "Set a real API key in .env before starting the application."
            )


# Single shared instance — every module imports `settings`.
settings: Final[Settings] = Settings()
# NOTE: Validation is now lazy — call settings.validate_nvidia_api_key()
# before making Nemotron API calls. This allows training scripts to run
# without a real API key.

# Fixed KPI schema: must match generate_data() columns AND build_feature_vector().
KPI_SCHEMA: Final[List[str]] = [
    "revenue_growth",   # YoY revenue growth (decimal, e.g., 0.12 = 12%)
    "net_margin",       # Net profit margin (decimal, e.g., 0.08 = 8%)
    "debt_equity",      # Debt-to-equity ratio (e.g., 1.5)
    "roe",              # Return on equity (decimal, e.g., 0.18 = 18%)
    "current_ratio",    # Current assets / current liabilities (e.g., 2.1)
]

# Valuation features (from yfinance .info at filing date)
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

# Combined feature schema for training + scoring
ALL_FEATURES: Final[List[str]] = KPI_SCHEMA + VALUATION_SCHEMA + MOMENTUM_SCHEMA + VOLATILITY_SCHEMA + BETA_SCHEMA + SECTOR_SCHEMA

# Versioned artifacts (for reproducibility / future DB version tracking)
MODEL_VERSION: Final[str] = "rf_v1"
KPI_PROMPT_VERSION: Final[str] = "kpi_extract_v1"
BRIEF_PROMPT_VERSION: Final[str] = "brief_v1"

# Nemotron number extraction prompt (extract raw financial numbers).
NUMBER_EXTRACTION_PROMPT: Final[str] = """You are a financial data extractor. Find these exact line items in the financial statements and return their dollar amounts (in millions). Look at the Consolidated Statements of Operations and Balance Sheets.

Extract these 8 numbers (return as JSON, use null if not found):
- revenue_current
- revenue_prior
- net_income_current
- total_current_assets
- total_current_liabilities
- short_term_debt
- long_term_debt
- total_shareholders_equity

SEARCH INSTRUCTIONS:
- Scan the text for tables with column headers like "Jan 26, 2025", "Jan 28, 2024", "Jan 29, 2023" or "Year Ended"
- Look for these EXACT row labels (case-insensitive):
  * "Revenue" or "Total revenue" - get current year and prior year values
  - "Net income" or "Net earnings" - get current year value
  * "Total current assets" - get current year value
  - "Total current liabilities" - get current year value
  - "Short-term debt" or "Current portion of long-term debt" - get current year value
  - "Long-term debt" or "Long-term debt, net of current portion" - get current year value
  - "Total shareholders' equity" or "Total shareholders equity" or "Total equity" - get current year value
- Numbers are in MILLIONS (e.g., "130,497" means 130497)
- If a value has parentheses like "(1,250)" it's negative
- If a value has "$" or commas, ignore them and return just the number

Return ONLY a valid JSON object with these 8 keys. Use null if not found.
No markdown, no prose, no code fences.

Example:
{{"revenue_current": 130497, "revenue_prior": 60922, "net_income_current": 72880, "total_current_assets": 80126, "total_current_liabilities": 18047, "short_term_debt": 1250, "long_term_debt": 8463, "total_shareholders_equity": 79328}}

Excerpt:
\"\"\"{chunk}\"\"\"
"""

# Nemotron KPI extraction prompt (strict JSON output) - legacy, kept for compatibility
KPI_EXTRACTION_PROMPT: Final[str] = NUMBER_EXTRACTION_PROMPT

# Nemotron investment brief generation prompt.
BRIEF_GENERATION_PROMPT: Final[str] = """You are an investment analyst. Write a concise (~200 words) investment brief for a company based on its extracted KPIs, ML-derived score (0-100), label, and SHAP feature attributions.

KPIs:
{kpis}

ML Score: {score}/100
Label: {label}
Confidence: {confidence}%

SHAP feature attributions (higher = more positive contribution to the score):
{shap}

The brief should:
- Open with a one-sentence verdict referencing score/label.
- Highlight the top 2 positive and top 1 negative KPI drivers (using SHAP).
- Note confidence quality (e.g., high confidence reads cleanly, low confidence: "interpret with caution").
- End with a single forward-looking caveat sentence.

Return only the brief prose — no markdown, no headers, no bullets. ~200 words.
"""