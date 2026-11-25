"""Semantic chunker — splits long text into ~N-token chunks with overlap.

Splits on paragraphs/blank lines first, grams chunks up to roughly
CHUNK_SIZE tokens, with CHUNK_OVERLAP tokens of overlap. Avoids mid-sentence
cuts and produces far fewer chunks than char-splitting (fewer embed calls).

Each chunk is also tagged with a `chunk_type`:
  fin_stmt  -> structured financial-statement tables (Income / Balance / Cash Flow)
  notes     -> accounting policy notes (Level 3, warranties, revenue recognition)
  risk      -> Risk Factors section
  toc       -> table of contents / boilerplate headers
  other     -> narrative / MD&A / exhibits / signatures / anything else
"""
from __future__ import annotations

import re
from typing import List, Tuple

import tiktoken


CHUNK_SIZE = 800       # tokens per chunk — ~600 words, perfect for KPI context
CHUNK_OVERLAP = 100    # tokens overlap with the previous chunk


# --- Tier 1 signals: structural anchors ---------------------------------------
# Primary-statement headers as they appear as a *chapter title* in 10-K PDFs:
# the line above is typically "Company Name and Subsidiaries" (or blank) and the
# line below is the canonical prelude "(In millions...)" or "Year Ended".
# Capturing these context lines is what rejects inline narrative references
# ("...included in Accrued and Other Current Liabilities on our\nConsolidated
# Balance Sheets. The goodwill is tax deductible...") where the title appears
# mid-paragraph after a sentence wrap.
# The company line pattern matches: "ANY COMPANY NAME and Subsidiaries" or
# "ANY COMPANY NAME, INC. and Subsidiaries" or similar SEC-standard formats.
_FIN_STMT_HEADER_BLOCK = re.compile(
    # Optional "Company Name and Subsidiaries" line directly above.
    r"(?:"
    r"(?:^|\n)\s*"
    r"[A-Z][A-Z0-9&\.\,\-\s]{2,80}?\s+and\s+Subsidiaries\s*\n"
    r")?"
    # The chapter title itself, on its own line.
    r"(?:^|\n)\s*"
    r"(?:"
    r"Consolidated\s+Statements?\s+of\s+(?:Income|Operations|Earnings|Cash\s+Flows)"
    r"|Consolidated\s+Balance\s+Sheets?"
    r")\s*\n"
    # Canonical prelude line below: "(In millions...)" or "Year Ended".
    # Allow blank lines / page breaks between title and prelude (\n+).
    r"(?:\s*\n)*\s*(?:"
    r"\(\s*In\s+(?:millions|thousands)[^)]*\)"
    r"|Year\s+Ended"
    r")",
    re.I,
)

# Hard exclusion: a chunk that *starts* with the Notes-section header is a
# Notes sub-table, not a primary financial statement — no matter how many
# dollar amounts it contains. This catches the Note section tables that quote
# row labels like "Revenue" / "Operating income" in their reconciliation grids.
_NOTES_SECTION_HEADER = re.compile(
    r"^\s*Notes?\s+to\s+the\s+Consolidated\s+Financial\s+Statements\s*\n\s*\(Continued\)",
    re.I,
)

# --- Tier 2 signals: unique co-occurrence patterns ---------------------------
# These row labels co-occur ONLY inside the primary statement chunk (or its
# continuation), never all together in a Note schedule. Used as a fallback
# when Tier 1 (header block) wasn't found — covers continuation chunks like
# chunk 95 which starts mid-balance-sheet (no header) but contains the
# bottom rows: "Total shareholders equity" + "Total liabilities and shareholders
# equity" + "See accompanying Notes to the Consolidated Financial Statements."
_BS_TOTAL_EQUITY = re.compile(
    r"\bTotal\s+(?:shareholders['’]?\s+equity"
    r"|liabilities\s+and\s+shareholders['’]?\s+equity)\b",
    re.I,
)
_BS_TOTAL_ASSETS = re.compile(r"\bTotal\s+assets\b", re.I)
_BS_TOTAL_CA = re.compile(r"\bTotal\s+current\s+assets\b", re.I)
_BS_TOTAL_CL = re.compile(r"\bTotal\s+current\s+liabilities\b", re.I)
_CF_NET_INCOME = re.compile(r"^\s*Net\s+income\s*$", re.I | re.M)
_CF_NCOA = re.compile(
    r"\bNet\s+cash\s+provided\s+by\s+operating\s+activities\b", re.I)
_IS_REVENUE = re.compile(r"^\s*Revenue\s*$", re.I | re.M)
_IS_GP = re.compile(r"^\s*Gross\s+profit\s*$", re.I | re.M)
_IS_OPINC = re.compile(r"^\s*Operating\s+income\s*$", re.I | re.M)
_IS_NET_INCOME_PER_SHARE = re.compile(
    r"\bNet\s+income\s+per\s+share\b", re.I)

# Dollar run: 3+ amounts like $215,938 — confirms a real column-structured table.
_DOLLAR_RUN = re.compile(r"\$\s?\d[\d,]{2,}")

# Disqualifier: MD&A "expressed as a percentage of revenue" tables reuse the
# Income-Statement row labels as a percent-of-revenue breakdown — these aren't
# the primary statement (no absolute dollars, only percentages). Reject them so
# Tier 2 Rules C/D don't fire.
_PCT_OF_REVENUE_TABLE = re.compile(
    r"\bexpressed\s+as\s+a\s+percentage\s+of\s+revenue\b",
    re.I,
)


def classify_chunk(text: str) -> str:
    """Return one of: fin_stmt | notes | risk | toc | other.

    Two-tier detection:

    Tier 1 (hard structural anchor — primary signal):
        The chunk contains a real *chapter block*: the titled line
        "Consolidated Statements of {Income|Cash Flows|...}" or
        "Consolidated Balance Sheets" appearing as a titled paragraph whose
        immediate neighbour lines are the canonical NVIDIA 10-K prelude lines
        ("NVIDIA Corporation and Subsidiaries" above, "(In millions...)" or
        "Year Ended" below). This pattern only occurs in primary statements.

    Tier 2 (soft unique co-occurrence — fallback when Tier 1 misses):
        Continuation chunks that lost the header mid-chunk still qualify if
        they contain a row-label set that *only* the primary statement has:
          - Balance Sheet trio: Total current assets + Total current
            liabilities + (Total shareholders equity OR Total liabilities and
            shareholders equity).
          - Cash Flow signature: bare "Net income" line + Net cash provided by
            operating activities.
          - Income Stmt quartet: Revenue + Gross profit + Operating income +
            Net income, all as their own lines.

    Hard exclusion (overrides Tier 1+2):
        Chunk body literally starts with the Notes section header
        "Notes to the Consolidated Financial Statements (Continued)" — these
        are Notes sub-tables (debt schedules, segment tables, tax tables).
        Reconciliation tables that re-quote Income-Statement row labels live
        under this Notes umbrella and must be filtered out.
    """
    if not text:
        return "other"

    # --- Hard exclusions first ------------------------------------------------
    # Always classify as notes if the chunk opens with the Notes header.
    if _NOTES_SECTION_HEADER.search(text):
        return "notes"

    # Disqualifier for Tier 2: MD&A's "expressed as a percentage of revenue"
    # table reuses IS row labels but only contains percentages. Tier 1 cannot
    # fire on this chunk (no real chapter mask block), and Tier 2 must skip too.
    is_pct_of_rev = bool(_PCT_OF_REVENUE_TABLE.search(text))

    # --- Tier 1 ---------------------------------------------------------------
    if _FIN_STMT_HEADER_BLOCK.search(text):
        return "fin_stmt"

    dollar_count = len(_DOLLAR_RUN.findall(text))

    # --- Tier 2 (fallback) ----------------------------------------------------
    # Skip Tier 2 entirely on disqualifying patterns.
    if not is_pct_of_rev:
        # Rule B: Balance Sheet continuation — trio of totals.
        if (
            _BS_TOTAL_CA.search(text)
            and _BS_TOTAL_CL.search(text)
            and _BS_TOTAL_EQUITY.search(text)
            and dollar_count >= 3
        ):
            return "fin_stmt"

        # Rule C: Cash Flow signature.
        if (
            _CF_NET_INCOME.search(text)
            and _CF_NCOA.search(text)
            and dollar_count >= 3
        ):
            return "fin_stmt"

        # Rule D: Income Stmt quartet of row labels.
        if (
            _IS_REVENUE.search(text)
            and _IS_GP.search(text)
            and _IS_OPINC.search(text)
            and _CF_NET_INCOME.search(text)
            and dollar_count >= 3
        ):
            return "fin_stmt"

        # Rule E: Balance-Sheet bottom-lines — chunk 95 case (no header, no
        # current lines, but has the two total-equity lines + "See accompanying
        # Notes" footer).
        if (
            _BS_TOTAL_EQUITY.search(text)
            and bool(re.search(
                r"\bTotal\s+liabilities\s+and\s+shareholders['’]?\s+equity\b",
                text, re.I))
            and dollar_count >= 3
            and bool(re.search(r"See\s+accompanying\s+Notes", text, re.I))
        ):
            return "fin_stmt"

    # --- Tier 3: heuristic fallback ---------------------------------------------
    # Any chunk with >5 dollar runs AND >3 known financial row labels.
    # Catches continuation chunks that lost the header but still have dense
    # financial table content.
    if not is_pct_of_rev:
        row_label_hits = 0
        # Common financial statement row labels
        label_patterns = [
            _IS_REVENUE, _IS_GP, _IS_OPINC, _CF_NET_INCOME,
            _BS_TOTAL_ASSETS, _BS_TOTAL_CA, _BS_TOTAL_CL,
            _BS_TOTAL_EQUITY, _CF_NCOA,
        ]
        for pat in label_patterns:
            if pat.search(text):
                row_label_hits += 1
        if dollar_count >= 6 and row_label_hits >= 4:
            return "fin_stmt"

    # --- Non-financial classifications ----------------------------------------
    if re.search(r"Notes?\s+to\s+the\s+Consolidated\s+Financial\s+Statements",
                 text, re.I):
        return "notes"
    if re.search(r"\bRisk\s+Factors\b", text, re.I):
        return "risk"
    if len(text) < 600 and text.count("...") >= 3:
        return "toc"
    return "other"


def get_encoder():
    return tiktoken.get_encoding("cl100k_base")  # approximate for most modern LLMs


def _count_tokens(text: str, enc) -> int:
    return len(enc.encode(text))


def semantic_chunk(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP,
) -> List[Tuple[str, str]]:
    """Split text and tag each chunk. Returns list of (text, chunk_type) tuples."""
    raw_chunks = _split_into_raw(text, chunk_size, overlap)
    return [(c, classify_chunk(c)) for c in raw_chunks]


def semantic_chunk_text(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """Back-compat wrapper: returns text-only list (old callers)."""
    return [c for c, _ in semantic_chunk(text, chunk_size, overlap)]


def _split_into_raw(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Original greedy-paragraph splitter. Produces text-only list."""
    text = (text or "").strip()
    if not text:
        return []

    enc = get_encoder()

    # Initial split on double-newline paragraphs.
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: List[str] = []
    current_paras: List[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = _count_tokens(para, enc)
        if para_tokens > chunk_size:
            if current_paras:
                chunks.append("\n\n".join(current_paras))
                current_paras = []
                current_tokens = 0
            chunks.extend(_split_long_para(para, chunk_size, overlap, enc))
            continue
        if current_tokens + para_tokens > chunk_size and current_paras:
            chunks.append("\n\n".join(current_paras))
            tail = _tail_tokens("\n\n".join(current_paras), overlap, enc)
            current_paras = [tail] if tail else []
            current_tokens = _count_tokens(tail, enc) if tail else 0
        current_paras.append(para)
        current_tokens += para_tokens

    if current_paras:
        chunks.append("\n\n".join(current_paras))

    return chunks


def _split_long_para(para: str, chunk_size: int, overlap: int, enc) -> List[str]:
    """Split a paragraph that is longer than chunk_size by sentences."""
    import re
    sentences = re.split(r"(?<=[.!?])\s+", para)
    chunks: List[str] = []
    buf = ""
    buf_tokens = 0
    for sent in sentences:
        sent_tokens = _count_tokens(sent, enc)
        if buf_tokens + sent_tokens > chunk_size and buf:
            chunks.append(buf.strip())
            tail = _tail_tokens(buf, overlap, enc)
            buf = tail + " " if tail else ""
            buf_tokens = _count_tokens(buf, enc) if buf else 0
        buf = f"{buf} {sent}".strip() if buf else sent
        buf_tokens += sent_tokens
    if buf:
        chunks.append(buf.strip())
    return chunks


def _tail_tokens(text: str, n: int, enc) -> str:
    """Return the trailing ~n tokens of `text` decoded back to a string."""
    tokens = enc.encode(text)
    if len(tokens) <= n:
        return text
    return enc.decode(tokens[-n:])
