"""RAG pipeline: PDF text -> index chunks -> retrieve top-k -> 1 Nemotron KPI call -> brief."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root (ml package) is importable.
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import logging
import time
from typing import Any, Dict

from app.core.config import settings
from app.services.embedding_service import embed_query
from app.services.retrieval_service import index_chunks, retrieve_kpis_context
from openai import OpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

logger = logging.getLogger(__name__)


# KPI extraction query — target the financial statements section specifically.
# SEC 10-K financial statements are in "Item 8" / "Consolidated Statements" sections.
KPI_QUERY = (
    "Consolidated Statements of Operations revenue net income earnings per share "
    "Consolidated Balance Sheets assets liabilities stockholders equity "
    "current assets current liabilities cash equivalents "
    "Statement of Cash Flows operating activities investing financing "
    "revenue growth profit margin return on equity current ratio debt equity"
)
TOP_K = 6  # Retrieve more chunks to ensure we hit the financial tables


# --- Nemotron client with timeouts & retry logic ----------------------------------

# Custom exception for JSON parse failures
class KPIExtractionError(Exception):
    """Raised when KPI extraction from LLM response fails."""
    pass


# Custom exception for upstream API errors
class UpstreamAPIError(Exception):
    """Raised when the upstream API returns an error response."""
    def __init__(self, message: str, status_code: int | None = None, is_retryable: bool = False):
        super().__init__(message)
        self.status_code = status_code
        self.is_retryable = is_retryable


def _make_nemotron_client() -> OpenAI:
    """Create OpenAI client with explicit timeout for NVIDIA NIM."""
    return OpenAI(
        api_key=settings.nvidia_api_key,
        base_url=settings.nvidia_base_url,
        timeout=45.0,  # 45s - 4B model usually responds in <3s, but allow cold-start margin
    )


# Retry configuration: 2 attempts max, fast backoff - total must stay under 120s gateway timeout
_retry_config = {
    "wait": wait_exponential_jitter(initial=0.5, max=2),
    "stop": stop_after_attempt(2),
    "retry": retry_if_exception_type((TimeoutError, ConnectionError, UpstreamAPIError)),
    "reraise": True,
}


@retry(**_retry_config)
def _call_nemotron_with_retry(client: OpenAI, prompt: str) -> str:
    """Call Nemotron with retry logic for transient failures.

    Raises:
        UpstreamAPIError: For non-retryable errors (4xx) or after retries exhausted.
        KPIExtractionError: If response parsing fails.
    """
    try:
        resp = client.chat.completions.create(
            model=settings.nemotron_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=800,
            extra_body={"chat_template_kwargs": {"enable_thinking": False, "force_nonempty_content": True}},
        )
    except Exception as e:
        # Check if it's a timeout or connection error
        if "timeout" in str(e).lower() or "timed out" in str(e).lower():
            raise TimeoutError(f"Nemotron request timed out: {e}") from e
        if "connection" in str(e).lower() or "connect" in str(e).lower():
            raise ConnectionError(f"Nemotron connection error: {e}") from e
        # Check for HTTP status codes if available
        status_code = getattr(e, "status_code", None) or getattr(e, "code", None)
        if status_code and 500 <= status_code < 600:
            raise UpstreamAPIError(f"Nemotron 5xx error: {e}", status_code=status_code, is_retryable=True) from e
        if status_code and 400 <= status_code < 500:
            raise UpstreamAPIError(f"Nemotron 4xx error: {e}", status_code=status_code, is_retryable=False) from e
        # Unknown error - treat as potentially retryable
        raise UpstreamAPIError(f"Nemotron error: {e}", status_code=status_code, is_retryable=True) from e

    raw = (resp.choices[0].message.content or "").strip()
    if not raw:
        raise KPIExtractionError("Nemotron returned empty response")

    return raw


def _parse_kpi_json(raw: str) -> Dict[str, Any]:
    """Parse KPI JSON from LLM response, with detailed error logging.

    Raises:
        KPIExtractionError: If parsing fails, with the raw response logged.
    """
    import json, re

    # Strip markdown code fences if present
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.M)

    # First try direct parse
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Try to locate the first {...} block
    match = re.search(r"\{.*\}", cleaned, flags=re.S)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    # Log the raw response for debugging
    logger.error(f"KPI JSON parse failed. Raw response:\n{raw}")
    raise KPIExtractionError("Failed to parse KPI JSON from Nemotron response")


def _extract_kpis_from_context(context: str) -> Dict[str, Any]:
    """Single Nemotron call on retrieved financial-statement context -> strict JSON KPIs.

    Two preprocessing steps before the LLM call:
      1. `clean_pdf_text()` strips PyMuPDF artifacts that confuse the LLM.
      2. Context is trimmed to the first 4000 chars for speed.

    Raises:
        KPIExtractionError: If extraction or parsing fails (fail-fast, no silent fallback).
        UpstreamAPIError: If the upstream API is unavailable or returns an error.
    """
    cleaned_context = clean_pdf_text(context)[:4000]

    prompt = _build_kpi_prompt(cleaned_context)

    client = _make_nemotron_client()
    raw = _call_nemotron_with_retry(client, prompt)

    parsed = _parse_kpi_json(raw)

    # Be defensive: only keep the 5 KPIs we expect, coerce to float|null.
    return {
        "revenue_growth": _coerce(parsed.get("revenue_growth")),
        "net_margin":     _coerce(parsed.get("net_margin")),
        "debt_equity":    _coerce(parsed.get("debt_equity")),
        "roe":            _coerce(parsed.get("roe")),
        "current_ratio":  _coerce(parsed.get("current_ratio")),
    }


# --- KPI extraction helpers ---------------------------------------------------


def process_pdf(pdf_bytes: bytes, report_id, db, filename: str = "") -> Dict[str, Any]:
    """Full RAG flow. Returns dict with kpis, score, label, confidence, shap, brief, elapsed_ms."""
    from ml.pipeline import extract_text, build_feature_vector, generate_brief
    from app.services.ml_scorer import MLScorer

    start = time.perf_counter()

    # Extract ticker from filename (e.g., "NVIDIA-2025-Annual-Report.pdf" -> "NVDA")
    ticker = _extract_ticker_from_filename(filename)

    # Write to temp file for PyMuPDF.
    # Use unique temp file per request to avoid conflicts when same PDF processed concurrently.
    import hashlib, tempfile, uuid
    pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()
    tmp_dir = Path(tempfile.gettempdir()) / "invest_reports"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    unique_suffix = uuid.uuid4().hex[:8]
    tmp_path = tmp_dir / f"upload_{pdf_hash[:12]}_{unique_suffix}.pdf"
    
    # Write with retry for Windows file locking
    for attempt in range(3):
        try:
            tmp_path.write_bytes(pdf_bytes)
            break
        except PermissionError:
            if attempt == 2:
                raise
            time.sleep(0.1 * (attempt + 1))

    try:
        print(f"[pipeline] [1/4] Extracting text from {tmp_path.name}")
        text = extract_text(str(tmp_path))

        print(f"[pipeline] [2/4] Indexing chunks (embed + pgvector)")
        chunk_count = index_chunks(db, report_id, text)
        print(f"[pipeline]       Indexed {chunk_count} chunks")

        print("[pipeline] [3/4] Retrieving financial-statement chunks + extracting KPIs (1 Nemotron call)")
        retrieved = retrieve_kpis_context(db, report_id, k=TOP_K)
        context = "\n\n---\n\n".join([r[0] for r in retrieved])
        print(f"[pipeline]       Retrieved {len(retrieved)} chunks ({len(context)} chars)")

        kpis = _extract_kpis_from_context(context)
        print(f"[pipeline]       KPIs: {kpis}")

        print("[pipeline] [4/4] Scoring + SHAP + brief")
        print(f"[pipeline]       Using ticker: {ticker}")
        features, feature_metadata = build_feature_vector(kpis, ticker=ticker)
        score, label, conf, ood_info = MLScorer.score(features, kpis)
        # Apply confidence penalty for missing KPIs
        conf = round(conf * (1.0 - feature_metadata.get("confidence_penalty", 0.0)), 2)
        shap = MLScorer.get_shap(features)
        brief = generate_brief(kpis, score, label, conf, shap, ood_info)
        print(f"[pipeline]       Score={score}/100 Label={label} Conf={conf}%")
        if ood_info and ood_info.get("is_ood"):
            print(f"[pipeline]       OOD: {ood_info.get('ood_reason')}")
        if feature_metadata.get("warning"):
            print(f"[pipeline]       WARNING: {feature_metadata['warning']}")
    finally:
        # Delete with retry for Windows file locking
        for attempt in range(3):
            try:
                tmp_path.unlink(missing_ok=True)
                break
            except PermissionError:
                if attempt == 2:
                    print(f"[pipeline] WARNING: Could not delete temp file {tmp_path}")
                time.sleep(0.1 * (attempt + 1))

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    return {
        "kpis": kpis, "score": score, "label": label,
        "confidence": conf, "shap": shap, "brief": brief,
        "elapsed_ms": elapsed_ms, "chunk_count": chunk_count,
        "ood_info": ood_info,
        "feature_metadata": feature_metadata,
    }


def _null_kpis() -> Dict[str, Any]:
    return {
        "revenue_growth": None, "net_margin": None,
        "debt_equity": None, "roe": None, "current_ratio": None,
    }


def _extract_ticker_from_filename(filename: str) -> str | None:
    """Extract ticker symbol from filename like 'NVIDIA-2025-Annual-Report.pdf' -> 'NVDA'."""
    if not filename:
        return None
    
    # Known company name -> ticker mapping
    name_to_ticker = {
        "nvidia": "NVDA",
        "apple": "AAPL",
        "microsoft": "MSFT",
        "amazon": "AMZN",
        "meta": "META",
        "tesla": "TSLA",
        "google": "GOOGL",
        "alphabet": "GOOGL",
        "nvidia": "NVDA",
        "berkshire": "BRK-B",
        "jpmorgan": "JPM",
        "johnson": "JNJ",
        "unitedhealth": "UNH",
        "visa": "V",
        "mastercard": "MA",
        "walmart": "WMT",
        "procter": "PG",
        "home": "HD",
        "disney": "DIS",
        "salesforce": "CRM",
        "adobe": "ADBE",
        "netflix": "NFLX",
        "intel": "INTC",
        "amd": "AMD",
        "qualcomm": "QCOM",
        "broadcom": "AVGO",
        "texas": "TXN",
        "intuit": "INTU",
        "servicenow": "NOW",
        "applied": "AMAT",
        "lam": "LRCX",
        "klac": "KLAC",
        "synopsys": "SNPS",
        "cadence": "CDNS",
    }
    
    filename_lower = filename.lower()
    for name, tkr in name_to_ticker.items():
        if name in filename_lower:
            return tkr
    
    # Fallback: try to extract ticker-like pattern (2-5 uppercase letters at start or after separator)
    import re
    match = re.search(r'(?:^|[-_])([A-Z]{2,5})(?:[-_\.]|$)', filename)
    if match:
        return match.group(1)
    
    return None


def _coerce(val: Any) -> Any:
    """Coerce a KPI value to float or None. Strings like "0.12" parse; bad -> None."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        s = val.strip().strip("%").strip()
        if not s or s.lower() in ("null", "n/a", "none", "-"):
            return None
        try:
            return float(s)
        except ValueError:
            return None
    return None


def clean_pdf_text(text: str) -> str:
    """Repair PyMuPDF/PDF-extraction artefacts that confuse the LLM.

    Specific fixes (driven by inspection of NVIDIA 10-K extractions):
      * "$\n215,938" — dollar sign and amount on consecutive lines.
      * "120,067?" — stray "?" appended to numbers (PyMuPDF emits "?" for
        certain Unicode chars like the "no-break space \xa0" or non-printable
        markers in tables).
      * "\xa0" — unicode no-break space — replace with regular space.
      * "?" or "\ufffd" (replacement char) — drop entirely.
      * Page footer noise like "51\n7/18/26, 12:23 AM\nnvda-...htm\n53/87\n"
        appears between tables. We keep them but the LLM prompt tells it to
        ignore anything that's clearly a page footer.
    """
    import re

    # \xa0 / U+FFFD -> space
    text = text.replace("\xa0", " ").replace("\ufffd", " ")

    # "$\n215,938" -> "$215,938" : collapse a "$" line followed by a number.
    text = re.sub(r"\$\s*\n\s*(\d[\d,]*)", r"$\1", text)

    # "120,067?" -> "120,067": drop trailing "?" right after a digit/comma run.
    text = re.sub(r"(\d[\d,\.]*)\s*\?(?=\s|\n|$)", r"\1", text)

    # "(259)" style negatives are fine — leave them.
    # Blanks "??\nJan 25, 2026" -> "Jan 25, 2026": sequences of "?" + spaces alone
    # on a line are stand-ins for empty cells. Replace with empty string.
    text = re.sub(r"(?m)^\s*\?+\s*$", "", text)

    return text


def _build_kpi_prompt(context: str) -> str:
    """Construct a human-analyst-style prompt for KPI extraction.

    Nemotron is instructed to act like a senior investment analyst performing
    due diligence: locate each line item in the text, show the arithmetic,
    validate plausibility, and only return null when a component is genuinely
    absent — never because of uncertainty.
    """
    return f"""You are a senior investment analyst performing due diligence on a
10-K annual report. Your job is to extract five financial KPIs from the
financial-statement excerpt below with the same rigor you would apply to an
investment committee memo.

Context may include page-footer noise ("N/87", URLs, "7/18/26", page numbers).
Ignore those lines. Focus only on the Consolidated Income Statement, Balance
Sheet, and Cash Flow Statement.

STEP 1 — Locate each line item in the excerpt.
For each of the following, quote the exact row label and the dollar amount you
find (most recent fiscal year = "current", prior year = "prior"). If a row
label appears multiple times, use the primary statement value, not a note or
MD&A restatement.

Required line items:
- Revenue (current, prior)
- Net income (current, prior)
- Short-term debt (current)
- Long-term debt (current)
- Total shareholders equity (current)
- Total current assets (current)
- Total current liabilities (current)

STEP 2 — Compute exactly these five KPIs using the formulas below. All money
figures are in millions of USD unless noted otherwise.

1. revenue_growth  = (Revenue_current - Revenue_prior) / Revenue_prior
   Example: 0.12 for +12%
2. net_margin      = Net income_current / Revenue_current
   Example: 0.08 for 8%
3. debt_equity     = (Short-term debt + Long-term debt) / Total shareholders equity
   Unitless ratio, e.g. 1.5. Use 0 if there is no debt.
4. roe             = Net income_current / Total shareholders equity
   Example: 0.18 for 18%
5. current_ratio   = Total current assets / Total current liabilities
   Unitless ratio, e.g. 2.1

STEP 3 — Sanity-check each result:
- revenue_growth should be between -1.0 and 2.0 for most companies
- net_margin should be between -0.5 and 0.5
- debt_equity should be between 0 and 5
- roe should be between -0.5 and 1.0
- current_ratio should be between 0 and 10
If a computed value falls outside these ranges, re-check your arithmetic and
the source numbers before accepting it.

STEP 4 — Confidence and null handling:
- If ALL components for a KPI are present in the excerpt, compute it. Do NOT
  return null just because you are uncertain — your arithmetic is the answer.
- If a component value is genuinely absent from the excerpt, return null for
  that KPI only. Do not fabricate numbers.
- Assign a confidence (0.0 to 1.0) to each KPI based on how clearly the
  source numbers are stated.

Output STRICT JSON only — no markdown fences, no prose before or after.
Round all numeric values to 4 decimal places.

Output schema (keys must match exactly):
{{
  "revenue_growth": 0.1234,
  "net_margin":     0.0850,
  "debt_equity":    0.4500,
  "roe":            0.1500,
  "current_ratio":  2.1000,
  "_reasoning": [
    "Revenue current: $100,000 (Consolidated Statements of Operations), Revenue prior: $90,000; revenue_growth = (100000-90000)/90000 = 0.1111",
    "Net income current: $12,000; Revenue current: $100,000; net_margin = 12000/100000 = 0.1200",
    ...
  ],
  "_confidence": {{
    "revenue_growth": 0.95,
    "net_margin":     0.92,
    "debt_equity":    0.88,
    "roe":            0.90,
    "current_ratio":  0.94
  }}
}}

Financial-statement excerpt (parse dollar amounts and row labels carefully,
treating dollar signs and "?" characters as formatting noise):
\"\"\"{context}\"\"\"
"""
