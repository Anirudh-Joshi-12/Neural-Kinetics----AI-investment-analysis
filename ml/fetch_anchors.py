"""Fetch the real-anchor 10-K filings from SEC EDGAR and extract their KPIs
using the backend's full pipeline (so the anchors see the same extraction
path real uploads go through).

Strategy:
  1. For each ticker, look up the company's CIK via the SEC ticker->CIK map.
  2. Query EDGAR's filings JSON API for that CIK, form type "10-K", sorted
     by filing date descending — take the most recent annual report.
  3. The 10-K's "primary document" is an HTML file (e.g. nvda-20260125.htm);
     EDGAR serves it alongside the full filing. We download the .htm, then
     use BeautifulSoup to strip HTML to plain text, then run our existing
     chunk -> classify -> retrieve -> extract pipeline to get the 5 KPIs.
  4. Cache results to ml/data/real_anchors.json so repeated runs don't hit
     EDGAR. Skip an anchor if its KPIs are null (extraction failed).

Usage:
    python -m ml.fetch_anchors                # fetch & extract all
    python -m ml.fetch_anchors --list          # show cached anchors
    python -m ml.fetch_anchors --ticker NVDA  # force re-fetch one

Settings: rate-limits itself to 1 request/sec to EDGAR (their fair-use
policy). Set FAIR_USE_DELAY_SEC in env to override.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


# --- SEC EDGAR endpoints ----------------------------------------------------
# SEC splits its API across two hosts; keep them separate to avoid routing bugs.
SEC_WWW_BASE = "https://www.sec.gov"        # ticker map + filing archives
SEC_DATA_BASE = "https://data.sec.gov"      # submissions + facts JSON APIs

TICKER_MAP_URL = f"{SEC_WWW_BASE}/files/company_tickers.json"
FAIR_USE_DELAY_SEC = 1.0   # SEC fair-use: 10 req/sec max; we stay well under.

# User-Agent is REQUIRED by SEC fair-use policy. Use a descriptive one with a
# reachable email — SEC reputation-checks UAs and blocks generic placeholders.
# NOTE: `requests` derives the Host header automatically from the URL, so we do
# NOT set it manually. Hardcoding Host was the root cause of an earlier silent
# 404 (stale Host vs actual subdomain mismatched).
HEADERS = {
    "User-Agent": "InvestmentAnalysisPipeline/1.0 (research; anirudhjoshi7895@gmail.com)",
    "Accept-Encoding": "gzip, deflate",
}


def pad_cik(cik) -> str:
    """Pad a CIK (int or string) to 10 digits. SEC API requires this format."""
    return str(cik).zfill(10)


def submissions_url(cik) -> str:
    """Submissions JSON for a CIK (recent filings metadata)."""
    return f"{SEC_DATA_BASE}/submissions/CIK{pad_cik(cik)}.json"


def company_facts_url(cik) -> str:
    """XBRL company-facts JSON (all historical financial facts). Currently
    unused but kept for future structured-data extraction (v2)."""
    return f"{SEC_DATA_BASE}/api/xbrl/companyfacts/CIK{pad_cik(cik)}.json"


def archives_url(cik, accession_no_spaces: str, primary_doc: str) -> str:
    """Direct archive URL for a filing's primary document."""
    accession_nohyphen = accession_no_spaces.replace("-", "")
    return (
        f"{SEC_WWW_BASE}/Archives/edgar/data/"
        f"{int(cik)}/{accession_nohyphen}/{primary_doc}"
    )


CACHE_PATH = Path(__file__).resolve().parent / "data" / "real_anchors.json"


# =============================================================================
# CIK lookup
# =============================================================================
def _load_ticker_to_cik() -> Dict[str, str]:
    """Fetch SEC's master ticker → CIK map. Returns a dict ticker -> CIK string."""
    resp = requests.get(
        TICKER_MAP_URL,
        headers=HEADERS,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    # The map is {n: {"cik_str": 789019, "ticker": "MSFT", "title": "MICROSOFT CORP"}, ...}
    # cik_str arrives as an int (despite the name); str() handles both types.
    out: Dict[str, str] = {}
    for entry in data.values():
        out[entry["ticker"].upper()] = str(entry["cik_str"])
    return out


# =============================================================================
# Find 10-K filings for a CIK
# =============================================================================
def _find_10k_filings(cik: str) -> List[Dict[str, str]]:
    """Return list of {'accession', 'primary_doc', 'filing_date', 'url'} for 10-K filings."""
    resp = requests.get(submissions_url(cik), headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])

    filings = []
    for i, form in enumerate(forms):
        if form == "10-K":
            accession = accessions[i]
            primary_doc = primary_docs[i]
            filing_date = dates[i]
            url = archives_url(cik, accession, primary_doc)
            filings.append({
                "accession": accession,
                "primary_doc": primary_doc,
                "filing_date": filing_date,
                "url": url,
            })

    # Sort by filing_date descending (newest first)
    filings.sort(key=lambda x: x["filing_date"], reverse=True)
    return filings


# =============================================================================
# Download + extract text
# =============================================================================
def _fetch_filing_html(url: str) -> str:
    """Download (HTML) and return as text. Rate-limited."""
    time.sleep(FAIR_USE_DELAY_SEC)
    resp = requests.get(url, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    # Decode defensively — EDGAR typically serves 10-Ks as latin-1 or utf-8.
    return resp.content.decode(resp.encoding or "utf-8", errors="replace")


def _html_to_text(html: str) -> str:
    """Strip HTML to plain text via BeautifulSoup. Dense paragraph-collapse —
    double newlines between block elements so the chunker splits on paragraphs."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    # Strip script/style.
    for tag in soup(["script", "style", "head"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    # Collapse runs of whitespace within a line but keep paragraph breaks.
    lines = []
    for line in text.split("\n"):
        collapsed = " ".join(line.split())
        lines.append(collapsed)
    return "\n\n".join(l for l in lines if l)


# =============================================================================
# Run the backend's full extraction pipeline on the HTML-derived text
# =============================================================================
def _extract_kpis_from_text(text: str) -> Dict[str, Optional[float]]:
    """Use the backend's chunker + retrieval + Nemotron extraction.

    Calls into app.services.retrieval_service and app.services.pdf_pipeline
    — must run with both backend/ and the project root on sys.path. We add
    those here.
    """
    # Make the backend package importable.
    root = Path(__file__).resolve().parent.parent
    backend = root / "backend"
    for p in (str(root), str(backend)):
        if p not in sys.path:
            sys.path.insert(0, p)

    from app.services.chunker import semantic_chunk, classify_chunk
    from app.services.embedding_service import embed_texts, embed_query
    from app.services.retrieval_service import retrieve_kpis_context
    from app.services.pdf_pipeline import _extract_kpis_from_context, clean_pdf_text
    from app.core.database import SessionLocal
    from app.models.chunk import Chunk
    from app.models.report import Report, ProcessingStatus
    from app.core.config import settings, MODEL_VERSION
    from sqlalchemy import select
    import uuid

    # We persist into the DB as a "synthetic" report so that retrieve_kpis_context
    # can find the chunks. Then immediately delete it to keep DB clean.
    db = SessionLocal()
    try:
        # Create a temporary report row (will be deleted below).
        report = Report(
            pdf_hash=f"synthetic-{uuid.uuid4().hex}",
            pdf_path="(embedded)",
            original_filename="real_anchor.html",
            model_version="anchor_only",
            status=ProcessingStatus.processing,
            llm_model=settings.nemotron_model,
            llm_prompt_version="anchor_only_v0",
        )
        db.add(report)
        db.commit()
        db.refresh(report)
        report_id = report.id

        chunks = semantic_chunk(text)
        texts = [t for t, _ in chunks]
        if not texts:
            return _null_kpis()
        vectors = embed_texts(texts)
        rows = [
            Chunk(
                report_id=report_id,
                chunk_index=i,
                chunk_type=ct,
                text=ct_text,
                embedding=vec,
            )
            for i, (ct_text, ct), vec in zip(
                range(len(texts)), chunks, vectors
            )
        ]
        db.bulk_save_objects(rows)
        db.commit()

        retrieved = retrieve_kpis_context(db, report_id, k=6)
        context = "\n\n---\n\n".join([r[0] for r in retrieved])
        kpis = _extract_kpis_from_context(context)
        return kpis
    finally:
        # Cleanup the temp report + its chunks (cascade will handle chunks
        # because of ondelete="CASCADE" on the FK).
        if 'report_id' in locals():
            db.query(Chunk).filter(Chunk.report_id == report_id).delete()
            db.query(Report).filter(Report.id == report_id).delete()
            db.commit()
        db.close()


def _null_kpis() -> Dict[str, Optional[float]]:
    return {
        "revenue_growth": None, "net_margin": None,
        "debt_equity": None, "roe": None, "current_ratio": None,
    }


# =============================================================================
# Top-level fetch + cache
# =============================================================================
DEFAULT_TICKERS = [
    # Mega-cap tech
    "NVDA", "AAPL", "MSFT", "AMZN", "GOOG", "META", "TSLA",
    # Semiconductors & hardware
    "AMD", "INTC", "AVGO", "QCOM", "TXN", "ADI", "MU", "LRCX", "KLAC",
    # Software & internet
    "CRM", "ORCL", "ADBE", "INTU", "NFLX", "PYPL", "SQ", "SHOP", "SNOW", "PLTR", "UBER", "ABNB",
    # Hypergrowth tech (additional)
    "SMCI", "ASAN", "NET", "DDOG", "CRWD", "PANW", "ZS", "FTNT",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "C", "USB", "PNC", "TD", "BMO", "AXP", "BLK", "SCHW", "CME", "ICE", "SPGI", "MCO",
    # Healthcare
    "UNH", "JNJ", "PFE", "MRK", "ABBV", "LLY", "BMY", "TMO", "DHR", "ABT", "MDT", "ISRG", "ZTS", "ELV", "HUM", "CI",
    # Industrials & aerospace
    "CAT", "BA", "HON", "GE", "UPS", "RTX", "LMT", "NOC", "DE", "CMI", "FDX", "NSC", "CSX", "EMR", "ETN",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "PXD", "OXY", "DVN", "FANG", "MPC", "VLO", "PSX", "HAL",
    # Consumer staples
    "PG", "KO", "PEP", "WMT", "COST", "WBA", "CL", "CLX", "KMB", "GIS", "K", "CAG", "HSY", "MO", "PM",
    # Consumer discretionary
    "HD", "MCD", "NKE", "SBUX", "TGT", "LOW", "TJX", "ROST", "CMG", "DPZ", "YUM", "MAR", "HLT", "LVS", "WYNN",
    # Materials
    "LIN", "APD", "FCX", "NUE", "MLM", "VMC", "SHW", "ECL", "DOW", "DD",
    # Utilities
    "NEE", "DUK", "SO", "EXC", "XEL", "SRE", "WEC", "AEP", "D", "ED", "PEG",
    # Telecommunications
    "VZ", "T", "CMCSA", "CHTR",
    # Real estate
    "PLD", "EQIX", "WELL", "AVB", "EQH",
]


def fetch_anchors(tickers: List[str], force: bool = False) -> Dict[str, Dict[str, Any]]:
    """Fetch + extract KPIs for each ticker. Cached to CACHE_PATH.

    Returns a dict ticker -> {kpis: {...}, filing_url: ..., filing_date: ..., ticker: ...}.
    """
    cache = _load_cache()

    print(f"[fetch_anchors] Loading SEC ticker->CIK map...")
    t2c = _load_ticker_to_cik()

    for ticker in tickers:
        if ticker not in t2c:
            print(f"  ! {ticker}: ticker not found on EDGAR — skipping")
            continue
        cik = t2c[ticker]
        filings = _find_10k_filings(cik)
        if not filings:
            print(f"  ! {ticker}: no 10-K filings found")
            continue
        filing = filings[0]  # most recent
        print(f"\n>>> {ticker} (CIK {cik}) - {filing['filing_date']}")
        if filing["url"] in [v.get("filing_url") for v in cache.values()] and not force:
            print(f"  Already cached; skipping (use --force to re-fetch)")
            continue
        try:
            html = _fetch_filing_html(filing["url"])
            print(f"  Downloaded {len(html)} chars. Extracting KPIs...")
            kpis = _extract_kpis_from_text(html)
            print(f"  KPIs: {kpis}")
            cache[ticker] = {
                "ticker": ticker,
                "cik": cik,
                "filing_date": filing["filing_date"],
                "filing_url": filing["url"],
                "kpis": kpis,
            }
            _save_cache(cache)
        except Exception as exc:
            print(f"  ! {type(exc).__name__}: {exc}")
            cache[ticker] = {"error": f"{type(exc).__name__}: {exc}"}
            _save_cache(cache)
    return cache


def _load_cache() -> Dict[str, Dict[str, Any]]:
    if CACHE_PATH.exists():
        with CACHE_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_cache(data: Dict[str, Dict[str, Any]]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_PATH.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# =============================================================================
# CLI
# =============================================================================
def _cli() -> int:
    parser = argparse.ArgumentParser(description="Real-anchor KPI fetcher.")
    parser.add_argument("--ticker", type=str, default=None,
                        help="Single ticker to (re)fetch.")
    parser.add_argument("--list", action="store_true",
                        help="Print the cached anchor table.")
    parser.add_argument("--fetch", action="store_true",
                        help="Fetch the default 12 anchors (caches results).")
    args = parser.parse_args()

    if not (args.ticker or args.list or args.fetch):
        parser.print_help()
        return 1

    if args.list:
        cache = _load_cache()
        if not cache:
            print("(empty cache)")
            return 0
        print(f"{'ticker':<8} {'filing_date':<12} {'revenue_growth':>15} "
              f"{'net_margin':>11} {'debt_equity':>11} {'roe':>9} "
              f"{'current_ratio':>14}")
        print("-" * 80)
        for t, info in sorted(cache.items()):
            kpis = info.get("kpis", {}) or {}
            print(f"{t:<8} {info.get('filing_date','?'):<12} "
                  f"{kpis.get('revenue_growth','?'):>15} "
                  f"{kpis.get('net_margin','?'):>11} "
                  f"{kpis.get('debt_equity','?'):>11} "
                  f"{kpis.get('roe','?'):>9} "
                  f"{kpis.get('current_ratio','?'):>14}")
        return 0

    ticker = args.ticker.upper() if args.ticker else None
    if args.fetch:
        fetch_anchors(DEFAULT_TICKERS)
    elif ticker:
        fetch_anchors([ticker], force=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())