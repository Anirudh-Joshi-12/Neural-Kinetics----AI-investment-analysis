"""GenAI backend pipeline: PDF -> text -> Nemotron KPIs -> score -> brief.

Functional style: small pure helpers, Nemotron interaction isolated behind a
single function, strict JSON validation with repair + retry.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import fitz  # PyMuPDF
import numpy as np
import pandas as pd
from openai import OpenAI

from ml.common import (
    KPI_SCHEMA,
    VALUATION_SCHEMA,
    ALL_FEATURES,
    MOMENTUM_SCHEMA,
    VOLATILITY_SCHEMA,
    BETA_SCHEMA,
    SECTOR_SCHEMA,
    MODEL_VERSION,
    KPI_PROMPT_VERSION,
    BRIEF_PROMPT_VERSION,
    get_default_medians,
    get_valuation_medians,
    get_momentum_medians,
    get_volatility_medians,
    get_beta_medians,
    get_sector_medians,
)
from ml.config import (
    BRIEF_GENERATION_PROMPT,
    KPI_EXTRACTION_PROMPT,
    NUMBER_EXTRACTION_PROMPT,
    settings,
)


# ---------------------------------------------------------------------------
# Cache for KPI extraction (keyed by PDF hash)
# ---------------------------------------------------------------------------
CACHE_DIR = Path("ml/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "kpi_cache.json"


def _pdf_hash(pdf_bytes: bytes) -> str:
    return hashlib.sha256(pdf_bytes).hexdigest()[:16]


def _load_cache() -> Dict[str, Any]:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_cache(cache: Dict[str, Any]) -> None:
    try:
        CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _get_cached_kpis(pdf_bytes: bytes) -> Optional[Dict[str, Any]]:
    cache = _load_cache()
    key = _pdf_hash(pdf_bytes)
    return cache.get(key)


def _set_cached_kpis(pdf_bytes: bytes, kpis: Dict[str, Any]) -> None:
    cache = _load_cache()
    key = _pdf_hash(pdf_bytes)
    cache[key] = kpis
    _save_cache(cache)


def extract_text(pdf_path: str) -> str:
    """Read a PDF file and return the concatenated raw text string."""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    doc = fitz.open(path)
    try:
        text = "\n".join(page.get_text() for page in doc)
    finally:
        doc.close()

    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        raise ValueError(f"No extractable text in PDF (scanned PDF?): {path}")
    return text


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def chunk_text(text: str, max_chars: int = None) -> List[str]:
    """Split text into chunks <= max_chars.

    Strategy:
      1. Split on likely section headers (uppercase title lines).
      2. Merge tiny sections into a single buffer until max_chars is reached,
         flushing when the buffer is large enough.
      3. Hard-split long sections on paragraph boundaries.
    """
    max_chars = max_chars or settings.kpi_chunk_max_chars
    chunks: List[str] = []

    # Initial split on likely section headers (uppercase title lines, 5-80 chars).
    raw_sections = re.split(r"\n(?=[A-Z0-9][A-Z0-9 \-]{4,79}\s*$)", text, flags=re.M)

    # Merge tiny sections into a buffer to avoid 95+ tiny chunks → 95 API calls.
    buffer = ""
    for section in raw_sections:
        section = section.strip()
        if not section:
            continue
        candidate = f"{buffer}\n{section}" if buffer else section
        if len(candidate) <= max_chars:
            buffer = candidate
        else:
            if buffer:
                chunks.append(buffer.strip())
                buffer = ""
            if len(section) <= max_chars:
                buffer = section
            else:
                # Hard-split oversized sections.
                chunks.extend(_hard_chunk(section, max_chars))
                buffer = ""
    if buffer:
        chunks.append(buffer.strip())

    return chunks or [text[:max_chars]]


def _hard_chunk(text: str, max_chars: int) -> List[str]:
    """Fallback chunker: splits long sections on paragraphs, then hard cuts."""
    out: List[str] = []
    buffer = ""
    for para in text.split("\n"):
        if len(buffer) + len(para) + 1 > max_chars:
            if buffer:
                out.append(buffer.strip())
            if len(para) > max_chars:
                # Hard split a giant paragraph
                for i in range(0, len(para), max_chars):
                    out.append(para[i:i + max_chars])
                buffer = ""
            else:
                buffer = para
        else:
            buffer = f"{buffer}\n{para}" if buffer else para
    if buffer:
        out.append(buffer.strip())
    return out


# ---------------------------------------------------------------------------
# Nemotron client
# ---------------------------------------------------------------------------
def make_client() -> OpenAI:
    """Build an OpenAI-compatible client pointed at NVIDIA NIM with retry on 429."""
    return OpenAI(
        api_key=settings.nvidia_api_key,
        base_url=settings.nvidia_base_url,
        timeout=60.0,
        max_retries=6,  # Exponential backoff on 429 (1s, 2s, 4s, 8s, 16s, 32s)
    )


from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

# Retry on rate limit (429), timeout, connection errors
_nemotron_retry_config = {
    "wait": wait_exponential_jitter(initial=1, max=30),
    "stop": stop_after_attempt(3),
    "retry": retry_if_exception_type((Exception,)),
    "reraise": True,
}

def _is_rate_limit_error(e: Exception) -> bool:
    msg = str(e).lower()
    return "429" in msg or "rate limit" in msg or "too many requests" in msg


@retry(**_nemotron_retry_config)
def call_nemotron(client: OpenAI, prompt: str) -> str:
    """Send a single prompt to the Nemotron model and return the raw text.

    Nemotron 3 Ultra defaults to reasoning mode; disable thinking for direct
    instruction-following output.
    """
    try:
        response = client.chat.completions.create(
            model=settings.nemotron_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": False,
                    "force_nonempty_content": True,
                }
            },
        )
    except Exception as e:
        if _is_rate_limit_error(e):
            raise  # will trigger retry
        raise
    return (response.choices[0].message.content or "").strip()


# ---------------------------------------------------------------------------
# KPI extraction (Call #1)
# ---------------------------------------------------------------------------
def extract_kpis(chunk: str) -> Dict[str, Any]:
    """Extract raw financial numbers from chunk, then compute KPIs in Python."""
    # First, extract raw numbers using LLM
    client = make_client()
    prompt = NUMBER_EXTRACTION_PROMPT.format(chunk=chunk[:settings.kpi_chunk_max_chars])
    raw = call_nemotron(client, prompt)
    
    print(f"[Number Extraction] Raw response (first 500 chars): {raw[:500]}")
    
    parsed = _parse_json_lenient(raw)
    if parsed is None:
        print(f"[Number Extraction] FAILED to parse JSON. Raw: {raw[:1000]}")
        return {k: None for k in KPI_SCHEMA}
    
    # Compute KPIs from extracted numbers
    return compute_kpis_from_numbers(parsed)


def compute_kpis_from_numbers(nums: Dict[str, Any]) -> Dict[str, Any]:
    """Compute KPIs from extracted raw numbers."""
    def to_float(v):
        if v is None:
            return None
        try:
            return float(v)
        except (ValueError, TypeError):
            return None
    
    rev_cur = to_float(nums.get("revenue_current"))
    rev_prior = to_float(nums.get("revenue_prior"))
    ni_cur = to_float(nums.get("net_income_current"))
    ca_cur = to_float(nums.get("total_current_assets"))
    cl_cur = to_float(nums.get("total_current_liabilities"))
    std_cur = to_float(nums.get("short_term_debt"))
    ltd_cur = to_float(nums.get("long_term_debt"))
    eq_cur = to_float(nums.get("total_shareholders_equity"))
    
    out = {k: None for k in KPI_SCHEMA}
    
    if rev_cur is not None and rev_prior is not None and rev_prior != 0:
        out["revenue_growth"] = (rev_cur - rev_prior) / rev_prior
    if ni_cur is not None and rev_cur is not None and rev_cur != 0:
        out["net_margin"] = ni_cur / rev_cur
    if std_cur is not None and ltd_cur is not None and eq_cur is not None and eq_cur != 0:
        out["debt_equity"] = (std_cur + ltd_cur) / eq_cur
    if ni_cur is not None and eq_cur is not None and eq_cur != 0:
        out["roe"] = ni_cur / eq_cur
    if ca_cur is not None and cl_cur is not None and cl_cur != 0:
        out["current_ratio"] = ca_cur / cl_cur
    
    return out


def _parse_json_lenient(raw: str) -> Dict[str, Any] | None:
    """Parse JSON from LLM output, tolerating code fences and surrounding prose."""
    if not raw:
        return None

    # Strip markdown code fences if present
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.M)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try to locate the first {...} block
    match = re.search(r"\{.*\}", cleaned, flags=re.S)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


# ---------------------------------------------------------------------------
# Merge KPIs from multiple chunks
# ---------------------------------------------------------------------------
def merge_kpi_results(kpi_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge per-chunk KPI dicts: first non-null numeric value wins per key."""
    merged: Dict[str, Any] = {k: None for k in KPI_SCHEMA}
    for kpi_dict in kpi_list:
        for key in KPI_SCHEMA:
            if merged[key] is None:
                val = kpi_dict.get(key)
                if isinstance(val, (int, float)) and not (isinstance(val, bool)):
                    merged[key] = float(val)
                elif isinstance(val, str):
                    parsed = _maybe_num(val)
                    if parsed is not None:
                        merged[key] = parsed
    return merged


def _maybe_num(value: str) -> float | None:
    """Best-effort numeric coercion for strings like '12.5%' or '1.5x'."""
    if value is None:
        return None
    cleaned = re.sub(r"[%x,\s]", "", str(value))
    try:
        return float(cleaned)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Feature vectorization
# ---------------------------------------------------------------------------
def _fetch_market_features(ticker: str | None) -> dict:
    """Fetch all market features from yfinance for a given ticker.
    
    Returns dict with VALUATION_SCHEMA + MOMENTUM_SCHEMA + VOLATILITY_SCHEMA + 
    BETA_SCHEMA + SECTOR_SCHEMA features.
    Returns NaN for all features if ticker is None or fetch fails.
    """
    if not ticker:
        return {k: np.nan for k in VALUATION_SCHEMA + MOMENTUM_SCHEMA + VOLATILITY_SCHEMA + BETA_SCHEMA + SECTOR_SCHEMA}
    
    try:
        import yfinance as yf
        from datetime import datetime, timedelta
        tk = yf.Ticker(ticker)
        info = tk.info or {}
        
        def _log1p(x):
            return np.log1p(x) if x is not None and x > 0 else np.nan
        
        def _log10(x):
            return np.log10(x) if x is not None and x > 0 else np.nan
        
        # Valuation features
        pe = info.get("trailingPE")
        fpe = info.get("forwardPE")
        ps = info.get("priceToSalesTrailing12Months")
        peg = info.get("pegRatio")
        pb = info.get("priceToBook")
        mcap = info.get("marketCap")
        div = info.get("dividendYield")
        
        val_features = {
            "trailing_pe_log": _log1p(pe),
            "forward_pe_log": _log1p(fpe),
            "price_to_sales_log": _log1p(ps),
            "peg_clipped": np.clip(peg, -5, 5) if peg is not None else np.nan,
            "price_to_book_log": _log1p(pb),
            "market_cap_log": _log10(mcap),
            "dividend_yield": div if div is not None else 0.0,
        }
        
        # Momentum, Volatility, Beta, Sector features
        # Fetch price history for calculations
        try:
            hist = tk.history(period="2y", interval="1d")["Close"]
            if hist is None or hist.empty:
                hist = pd.Series(dtype=float)
        except Exception:
            hist = pd.Series(dtype=float)
        
        mom_features = {}
        vol_features = {}
        beta_features = {}
        sector_features = {}
        
        if len(hist) >= 20:
            returns = hist.pct_change().dropna()
            current_price = float(hist.iloc[-1])
            
            # Momentum: trailing returns
            for months, key in [(3, "mom_3m"), (6, "mom_6m"), (12, "mom_12m")]:
                days = int(months * 21)  # ~21 trading days per month
                if len(hist) > days:
                    past_price = float(hist.iloc[-days-1])
                    if past_price > 0:
                        mom_features[key] = (current_price - past_price) / past_price
                    else:
                        mom_features[key] = np.nan
                else:
                    mom_features[key] = np.nan
            
            # Volatility: annualized realized vol
            for months, key in [(6, "vol_6m"), (12, "vol_12m")]:
                days = int(months * 21)
                if len(returns) >= days:
                    vol = returns.tail(days).std() * np.sqrt(252)
                    vol_features[key] = float(vol)
                else:
                    vol_features[key] = np.nan
            
            # Beta vs SPY
            try:
                spy = yf.Ticker("SPY")
                spy_hist = spy.history(period="1y", interval="1d")["Close"]
                spy_returns = spy_hist.pct_change().dropna()
                
                # Align dates
                common_idx = returns.index.intersection(spy_returns.index)
                if len(common_idx) >= 60:
                    stock_aligned = returns.loc[common_idx]
                    spy_aligned = spy_returns.loc[common_idx]
                    if spy_aligned.std() > 0:
                        beta = stock_aligned.cov(spy_aligned) / spy_aligned.var()
                        beta_features["beta_12m"] = float(beta)
                    else:
                        beta_features["beta_12m"] = np.nan
                else:
                    beta_features["beta_12m"] = np.nan
            except Exception:
                beta_features["beta_12m"] = np.nan
            
            # Sector-relative returns (need sector ETF mapping)
            sector_etf_map = {
                "XLK": ["AAPL", "MSFT", "NVDA", "ADBE", "CRM", "AVGO", "QCOM", "TXN", "AMD", "INTC", "ORCL", "CSCO", "NOW", "INTU", "AMAT", "MU", "LRCX", "KLAC", "SNPS", "CDNS"],
                "XLF": ["JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "BLK", "SCHW", "USB", "PNC", "TFC", "COF"],
                "XLV": ["JNJ", "UNH", "PFE", "ABBV", "MRK", "TMO", "DHR", "ABT", "AMGN", "BMY", "GILD", "CVS", "CI", "HUM", "ELV"],
                "XLY": ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "TJX", "LOW", "BKNG", "GM", "F", "CMG", "YUM", "MAR"],
                "XLP": ["PG", "KO", "PEP", "WMT", "COST", "PM", "MO", "CL", "KMB", "GIS"],
                "XLE": ["XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO", "OXY", "HAL"],
                "XLI": ["HON", "UNP", "UPS", "BA", "CAT", "GE", "MMM", "LMT", "RTX", "DE"],
                "XLB": ["LIN", "APD", "SHW", "FCX", "NEM", "DD", "DOW", "PPG", "ECL"],
                "XLU": ["NEE", "DUK", "SO", "D", "AEP"],
                "XLRE": ["PLD", "AMT", "EQIX", "CCI", "SPG"],
                "XLC": ["GOOGL", "META", "NFLX", "DIS", "CMCSA", "VZ", "T", "TMUS", "CHTR"],
            }
            
            ticker_sector = None
            for etf, tickers in sector_etf_map.items():
                if ticker in tickers:
                    ticker_sector = etf
                    break
            
            if ticker_sector:
                try:
                    sector_tk = yf.Ticker(ticker_sector)
                    sector_hist = sector_tk.history(period="1y", interval="1d")["Close"]
                    sector_returns = sector_hist.pct_change().dropna()
                    
                    common_idx = returns.index.intersection(sector_returns.index)
                    if len(common_idx) >= 60:
                        stock_aligned = returns.loc[common_idx]
                        sector_aligned = sector_returns.loc[common_idx]
                        
                        for months, key in [(6, "sector_rel_6m"), (12, "sector_rel_12m")]:
                            days = int(months * 21)
                            if len(stock_aligned) >= days and len(sector_aligned) >= days:
                                stock_ret = (1 + stock_aligned.tail(days)).prod() - 1
                                sector_ret = (1 + sector_aligned.tail(days)).prod() - 1
                                sector_features[key] = float(stock_ret - sector_ret)
                            else:
                                sector_features[key] = np.nan
                    else:
                        sector_features["sector_rel_6m"] = np.nan
                        sector_features["sector_rel_12m"] = np.nan
                except Exception:
                    sector_features["sector_rel_6m"] = np.nan
                    sector_features["sector_rel_12m"] = np.nan
            else:
                sector_features["sector_rel_6m"] = np.nan
                sector_features["sector_rel_12m"] = np.nan
        
        return {**val_features, **mom_features, **vol_features, **beta_features, **sector_features}
        
    except Exception:
        return {k: np.nan for k in VALUATION_SCHEMA + MOMENTUM_SCHEMA + VOLATILITY_SCHEMA + BETA_SCHEMA + SECTOR_SCHEMA}


def build_feature_vector(kpis: Dict[str, Any], ticker: str | None = None, use_extended_features: bool = False) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Map KPI dict + ticker -> 1-row DataFrame.
    
    If use_extended_features=True, returns 20 features (5 KPI + 7 valuation + 3 momentum + 2 volatility + 1 beta + 2 sector).
    If use_extended_features=False (default), returns only 5 KPI features for backward compatibility with existing models.
    
    Missing KPIs are filled with cached column medians.
    Market features (valuation + momentum + volatility + beta + sector) are fetched from yfinance 
    (if ticker provided) and imputed with medians.
    
    Returns (features_df, metadata) where metadata includes:
        - missing_kpis: list of KPI names that were missing
        - missing_count: number of missing KPIs
        - confidence_penalty: float 0-1 indicating confidence reduction due to missing data
        - valuation_missing: list of valuation features that were missing/imputed
        - momentum_missing: list of momentum features that were missing/imputed
        - volatility_missing: list of volatility features that were missing/imputed
        - beta_missing: list of beta features that were missing/imputed
        - sector_missing: list of sector features that were missing/imputed
    """
    # KPIs
    row = {k: kpis.get(k) for k in KPI_SCHEMA}

    # current_ratio == 0.0 is an extraction artifact (banks don't report a
    # current/non-current split). Treat as missing so it gets median-imputed
    # like other nulls instead of dragging the model toward "Avoid".
    if row.get("current_ratio") == 0.0:
        row["current_ratio"] = None

    missing_kpis = [k for k, v in row.items() if v is None]
    missing_count = len(missing_kpis)

    medians = get_default_medians()
    row = {k: (v if v is not None else medians[k]) for k, v in row.items()}

    if use_extended_features:
        # Fetch ALL market features from yfinance (valuation + momentum + volatility + beta + sector)
        market_features = _fetch_market_features(ticker)
        row.update(market_features)
        
        # Impute missing market features with medians
        val_medians = get_valuation_medians()
        mom_medians = get_momentum_medians()
        vol_medians = get_volatility_medians()
        beta_medians = get_beta_medians()
        sector_medians = get_sector_medians()
        
        all_missing = []
        for schema, medians_dict in [
            (VALUATION_SCHEMA, val_medians),
            (MOMENTUM_SCHEMA, mom_medians),
            (VOLATILITY_SCHEMA, vol_medians),
            (BETA_SCHEMA, beta_medians),
            (SECTOR_SCHEMA, sector_medians),
        ]:
            for k in schema:
                v = market_features.get(k)
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    all_missing.append(k)
                    row[k] = medians_dict[k]
                else:
                    row[k] = v

        # Confidence penalty: 0.1 per missing KPI, max 0.5
        confidence_penalty = min(missing_count * 0.1, 0.5)
    else:
        # Backward compatibility: only 5 KPI features
        confidence_penalty = min(missing_count * 0.1, 0.5)
    
    # Select only the features we need
    if use_extended_features:
        feature_cols = ALL_FEATURES
    else:
        feature_cols = KPI_SCHEMA
    
    df = pd.DataFrame([row], columns=feature_cols)

    metadata = {
        "missing_kpis": missing_kpis,
        "missing_count": missing_count,
        "confidence_penalty": confidence_penalty,
    }
    return df, metadata

    metadata = {
        "missing_kpis": missing_kpis,
        "missing_count": missing_count,
        "confidence_penalty": confidence_penalty,
        "warning": f"{missing_count} KPI(s) missing, filled with training medians" if missing_count else None,
        "valuation_missing": [k for k in VALUATION_SCHEMA if k in all_missing],
        "momentum_missing": [k for k in MOMENTUM_SCHEMA if k in all_missing],
        "volatility_missing": [k for k in VOLATILITY_SCHEMA if k in all_missing],
        "beta_missing": [k for k in BETA_SCHEMA if k in all_missing],
        "sector_missing": [k for k in SECTOR_SCHEMA if k in all_missing],
    }

    return pd.DataFrame([row], columns=ALL_FEATURES), metadata


# ---------------------------------------------------------------------------
# Brief generation (Call #2)
# ---------------------------------------------------------------------------
def generate_brief(
    kpis: Dict[str, Any],
    score: float,
    label: str,
    confidence: float,
    shap: Dict[str, float],
    ood_info: Dict[str, Any] | None = None,
) -> str:
    """Ask Nemotron for a ~200-word investment brief.

    If ood_info is provided and indicates OOD, the brief will note that the
    model's confidence is unreliable due to out-of-distribution input.
    """
    client = make_client()
    ood_note = ""
    if ood_info and ood_info.get("is_ood"):
        reason = ood_info.get("ood_reason", "low training-data coverage")
        ood_note = (
            f"\n\nIMPORTANT: This analysis is marked OUT-OF-DISTRIBUTION ({reason}). "
            f"The model's confidence (and score) should be interpreted with extreme "
            f"caution — the extracted KPI profile falls outside the training data "
            f"regimes the model was calibrated on. Treat as a structural screen only."
        )
    prompt = BRIEF_GENERATION_PROMPT.format(
        kpis=json.dumps(kpis, indent=2),
        score=round(score, 2),
        label=label,
        confidence=round(confidence, 2),
        shap=json.dumps(shap, indent=2),
    ) + ood_note
    return call_nemotron(client, prompt)


# ---------------------------------------------------------------------------
# Financial chunk filtering
# ---------------------------------------------------------------------------
# Strict keywords that ONLY appear in actual financial statement TABLES (Item 8)
FIN_STMT_KEYWORDS_STRICT = [
    "Consolidated Statements of Operations",
    "Consolidated Balance Sheets",
    "Consolidated Statements of Cash Flows",
    "Consolidated Statements of Shareholders",
    "Consolidated Statements of Shareholders' Equity",
    "Consolidated Balance Sheets as of",
    "Consolidated Statements of Operations for the years ended",
    "Consolidated Statements of Cash Flows for the years ended",
    "Consolidated Statements of Shareholders' Equity for the years ended",
]

# Table-structure keywords - require at least 5 of these (actual table rows in Item 8)
FIN_STMT_TABLE_KEYWORDS = [
    "Total current assets",
    "Total current liabilities",
    "Short-term debt",
    "Long-term debt",
    "Total shareholders' equity",
    "Total shareholders equity",
    "Total assets",
    "Total liabilities",
    "Cash and cash equivalents",
    "Accounts payable",
    "Accounts receivable",
    "Inventories",
    "Property and equipment",
    "Goodwill",
    "Intangible assets",
    "Accrued and other current liabilities",
    "Total shareholders equity",
]

# Negative keywords - chunks with these are NOT financial tables
FIN_STMT_NEGATIVE_KEYWORDS = [
    "Climate Change",
    "Security Ownership",
    "Executive Compensation",
    "Compensation Discussion",
    "Business Overview",
    "Risk Factors",
    "Market for Registrant",
    "Properties",
    "Legal Proceedings",
    "Mine Safety",
    "Corporate Governance",
    "Director Compensation",
    "Stock Performance",
    "Audit Committee",
    "Nominating",
    "Proxy Statement",
    "Annual Meeting",
    "Debora Shoquist",
    "EVP, Operations",
    "Target Pay",
    "Performance Metrics",
    "Variable Cash Plan",
    "SY PSUs",
    "MY PSUs",
]

# Require BOTH specific date columns (only in actual financial tables)
FIN_STMT_REQUIRED_COLUMNS = [
    "Jan 26, 2025",
    "Jan 28, 2024",
]

def is_financial_chunk(chunk: str) -> bool:
    """Check if chunk likely contains financial statement TABLES (Item 8)."""
    chunk_lower = chunk.lower()
    
    # 1. Strict header match
    if any(kw.lower() in chunk_lower for kw in FIN_STMT_KEYWORDS_STRICT):
        return True
    
    # 2. Exclude negative keywords (proxy, compensation, risk, etc.)
    if any(kw.lower() in chunk_lower for kw in FIN_STMT_NEGATIVE_KEYWORDS):
        return False
    
    # 3. MUST have BOTH specific date columns (only in actual financial tables)
    has_jan26 = "jan 26, 2025" in chunk_lower
    has_jan28 = "jan 28, 2024" in chunk_lower
    if not (has_jan26 and has_jan28):
        return False
    
    # 4. Table structure: require at least 6 table row keywords
    table_matches = sum(1 for kw in FIN_STMT_TABLE_KEYWORDS if kw.lower() in chunk_lower)
    return table_matches >= 6

def filter_financial_chunks(chunks: list[str]) -> list[str]:
    """Filter chunks to only those likely containing financial statement tables (Item 8)."""
    return [c for c in chunks if is_financial_chunk(c)]


# ---------------------------------------------------------------------------
# Top-level orchestrator (called by run_phase1.py)
# ---------------------------------------------------------------------------
def _extract_ticker_from_filename(pdf_path: str) -> str | None:
    """Extract ticker from filename like 'NVIDIA-2025-Annual-Report.pdf' -> 'NVDA'."""
    import re
    from pathlib import Path
    name = Path(pdf_path).stem.lower()
    name_to_ticker = {
        "nvidia": "NVDA", "apple": "AAPL", "microsoft": "MSFT", "amazon": "AMZN",
        "meta": "META", "tesla": "TSLA", "google": "GOOGL", "alphabet": "GOOGL",
        "berkshire": "BRK-B", "jpmorgan": "JPM", "johnson": "JNJ", "unitedhealth": "UNH",
        "visa": "V", "mastercard": "MA", "walmart": "WMT", "procter": "PG",
        "home": "HD", "disney": "DIS", "salesforce": "CRM", "adobe": "ADBE",
        "netflix": "NFLX", "intel": "INTC", "amd": "AMD", "qualcomm": "QCOM",
        "broadcom": "AVGO", "texas": "TXN", "intuit": "INTU", "servicenow": "NOW",
        "applied": "AMAT", "lam": "LRCX", "klac": "KLAC", "synopsys": "SNPS", "cadence": "CDNS",
    }
    for name, tkr in name_to_ticker.items():
        if name in name:
            return tkr
    # Fallback: try to extract ticker-like pattern
    match = re.search(r'\b([A-Z]{2,5})\b', Path(pdf_path).stem)
    if match:
        return match.group(1)
    return None


def _rank_financial_chunks(chunks: List[str]) -> List[str]:
    """Rank financial chunks by density of financial keywords (highest first)."""
    FIN_STMT_KEYWORDS = [
        "revenue", "net income", "total current assets", "total current liabilities",
        "short-term debt", "long-term debt", "total shareholders' equity",
        "total shareholders equity", "total assets", "total liabilities",
        "cash and cash equivalents", "accounts payable", "inventories",
        "property and equipment", "goodwill", "intangible assets",
        "retained earnings", "operating income", "eps", "earnings per share"
    ]
    
    def score_chunk(chunk: str) -> int:
        chunk_lower = chunk.lower()
        return sum(1 for kw in FIN_STMT_KEYWORDS if kw.lower() in chunk_lower)
    
    return sorted(chunks, key=score_chunk, reverse=True)


def _combine_chunks(chunks: List[str], max_chars: int = 8000) -> List[str]:
    """Combine ranked chunks into fewer prompts, each <= max_chars."""
    if not chunks:
        return []
    
    combined = []
    current = chunks[0]
    
    for chunk in chunks[1:]:
        if len(current) + len(chunk) + 2 <= max_chars:  # +2 for "\n\n"
            current = current + "\n\n" + chunk
        else:
            combined.append(current)
            current = chunk
    
    if current:
        combined.append(current)
    
    return combined


def run_pipeline(pdf_path: str, pipeline) -> Dict[str, Any]:
    """End-to-end: PDF -> KPIs -> score -> SHAP -> brief. Returns a result dict."""
    print(f"\n[1/5] Extracting text from {pdf_path}")
    
    # Read PDF bytes for caching
    pdf_bytes = Path(pdf_path).read_bytes()
    pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()[:16]
    
    # Check cache first
    cached = _get_cached_kpis(pdf_bytes)
    if cached:
        print(f"[1/5] Using cached KPIs (hash: {pdf_hash})")
        kpis = cached
    else:
        print(f"[1/5] Extracting text from {pdf_path}")
        text = extract_text(pdf_path)

        # Extract ticker from filename for market data
        ticker = _extract_ticker_from_filename(pdf_path)
        if ticker:
            print(f"      Detected ticker: {ticker}")

        print("[2/5] Chunking text")
        chunks = chunk_text(text)

        # Filter to financial statement chunks only
        fin_chunks = filter_financial_chunks(chunks)
        print(f"[2/5] Filtered to {len(fin_chunks)} financial chunks (from {len(chunks)} total)")

        # Rank chunks by financial relevance
        fin_chunks = _rank_financial_chunks(fin_chunks)

        # Combine chunks into fewer prompts (max ~8000 chars each)
        combined_chunks = _combine_chunks(fin_chunks, max_chars=8000)
        print(f"[3/5] Combined into {len(combined_chunks)} prompts (from {len(fin_chunks)} chunks)")

        print(f"[3/5] Extracting KPIs across {len(combined_chunks)} combined prompt(s)")
        kpi_list = []
        for i, c in enumerate(combined_chunks):
            print(f"      Processing combined chunk {i+1}/{len(combined_chunks)}...")
            kpis = extract_kpis(c)
            kpi_list.append(kpis)
            time.sleep(0.5)  # Small delay between API calls

        kpis = merge_kpi_results(kpi_list)
        print(f"      Merged KPIs: {kpis}")

        # Cache the result
        _set_cached_kpis(Path(pdf_path).read_bytes(), kpis)

    text = extract_text(pdf_path)

    # Extract ticker from filename for market data
    ticker = _extract_ticker_from_filename(pdf_path)
    if ticker:
        print(f"      Detected ticker: {ticker}")

    print("[4/5] Building feature vector + scoring")
    features, feature_metadata = build_feature_vector(kpis, ticker=ticker, use_extended_features=True)
    score, label, conf = _score_with_pipeline(pipeline, features)
    # Apply confidence penalty for missing KPIs
    conf = round(conf * (1.0 - feature_metadata.get("confidence_penalty", 0.0)), 2)
    shap = _shap_with_pipeline(pipeline, features)

    print(f"      Score={score}/100  Label={label}  Confidence={conf}%")
    print(f"      SHAP: {shap}")
    if feature_metadata.get("warning"):
        print(f"      WARNING: {feature_metadata['warning']}")

    print("[5/5] Generating investment brief")
    brief = generate_brief(kpis, score, label, conf, shap)

    return {
        "kpis": kpis,
        "score": score,
        "label": label,
        "confidence": conf,
        "shap": shap,
        "brief": brief,
        "feature_metadata": feature_metadata,
    }


def _score_with_pipeline(pipeline, features):
    from ml.ml import score_company, load_model
    import json
    from pathlib import Path
    from ml.config import settings

    # Load calibration parameters
    calibration = None
    calib_path = Path(settings.calibration_path)
    if calib_path.exists():
        with open(calib_path, 'r') as f:
            calibration = json.load(f)

    result = score_company(pipeline, features, calibration=calibration)
    # score_company now returns (score, label, confidence, ood_info) in v1.5
    if len(result) == 4:
        score, label, conf, _ = result
    else:
        score, label, conf = result
    return score, label, conf


def _shap_with_pipeline(pipeline, features):
    from ml.ml import get_shap_values
    return get_shap_values(pipeline, features)
