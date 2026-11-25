"""Retrieval service — stores chunk embeddings in pgvector, retrieves top-k chunks.

Retrieval is now filter-first: chunks are tagged with a `chunk_type` at index
time (see `chunker.classify_chunk`). At query time we restrict semantic search
to `chunk_type = 'fin_stmt'` chunks, with `chunk_type = 'notes'` as a soft
fallback only if no fin_stmt chunks exist. This eliminates the dominant
failure mode where semantic search returns Risk Factors / Notes noise.
"""
from __future__ import annotations

import uuid
from typing import List, Tuple

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.report import Report  # noqa: F401 - ensure Report table is registered for FK
from app.models.chunk import Chunk
from app.services.embedding_service import embed_query, embed_texts
from app.services.chunker import classify_chunk, semantic_chunk


def index_chunks(db: Session, report_id: uuid.UUID, text: str) -> int:
    """Chunk text, classify each chunk, embed in one batch, persist to DB.

    Returns the number of chunks persisted.
    """
    raw_chunks = semantic_chunk(text)  # list of (text, chunk_type)
    if not raw_chunks:
        return 0

    texts = [t for t, _ in raw_chunks]
    types = [ct for _, ct in raw_chunks]

    print(f"[retrieval] Chunking -> {len(raw_chunks)} chunks")
    vectors = embed_texts(texts)
    print(
        f"[retrieval] Embedded {len(vectors)} vectors "
        f"(dim={len(vectors[0]) if vectors else 0})"
    )

    # Print type distribution — visibility into classifier behaviour.
    from collections import Counter
    type_counts = Counter(types)
    print(f"[retrieval] Classifier distribution: {dict(type_counts)}")

    rows = [
        Chunk(
            report_id=report_id,
            chunk_index=i,
            chunk_type=ct,
            text=chunk_text,
            embedding=vec,
        )
        for i, (chunk_text, ct, vec) in enumerate(zip(texts, types, vectors))
    ]
    db.bulk_save_objects(rows)
    db.commit()
    return len(rows)


def retrieve_kpis_context(
    db: Session, report_id: uuid.UUID, k: int = 6
) -> List[Tuple[str, float]]:
    """Retrieve the most relevant chunks for KPI extraction.

    Strategy (filter-first):
      1. Pull every chunk tagged `chunk_type='fin_stmt'` for this report,
         ordered by chunk_index (preserves Income -> BS -> Cash Flow order).
      2. Embed the KPI query once and compute semantic distance for *all*
         chunks; this gives us a ranking for the fin_stmt chunks and a
         candidate pool for fallback.
      3. If we have < k fin_stmt chunks (rare for well-formed 10-Ks — typically
         4–6), pad the result with the next-best `notes` chunks by semantic
         distance.
      4. Sort: fin_stmt first in chunk_index order, notes fallback after.
    """
    # Pull all fin_stmt chunks in their natural document order.
    stmt_rows = db.execute(
        select(Chunk.text, Chunk.chunk_index)
        .where(Chunk.report_id == report_id)
        .where(Chunk.chunk_type == "fin_stmt")
        .order_by(Chunk.chunk_index)
    ).all()

    fin_stmt_items = [(text, idx) for text, idx in stmt_rows]  # (text, chunk_index)

    # If fin_stmt already filled the budget, return them in document order.
    if len(fin_stmt_items) >= k:
        return [(text, 0.0) for text, _idx in fin_stmt_items[:k]]

    if not fin_stmt_items:
        raise ValueError("No financial-statement chunks identified — classifier returned 0 fin_stmt")

    # Otherwise: pad with notes/other chunks ranked by semantic similarity.
    query_str = (
        "Revenue cost of revenue gross profit operating income net income "
        "earnings per share Total current assets Total current liabilities "
        "Total shareholders equity Total liabilities Cash and cash equivalents "
        "Net cash provided by operating activities "
        "revenue growth net margin return on equity debt equity current ratio"
    )
    query_vec = embed_query(query_str)

    # Fallback: only search notes/other chunks, ranked by cosine distance.
    stmt_fallback = (
        select(
            Chunk.text,
            Chunk.embedding.cosine_distance(query_vec).label("dist"),
            Chunk.chunk_index,
        )
        .where(Chunk.report_id == report_id)
        .where(Chunk.chunk_type.in_(["notes", "other"]))
        .order_by("dist")
        .limit(k - len(fin_stmt_items))
    )
    fallback_rows = db.execute(stmt_fallback).all()

    result = [(text, 0.0) for text, _idx in fin_stmt_items]
    result.extend([(text, float(dist)) for text, dist, _idx in fallback_rows])
    return result