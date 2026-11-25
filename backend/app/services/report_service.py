"""Report service — orchestrates DB write + GenAI/ML processing + DB update."""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import List, Optional

from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from app.core.config import settings, MODEL_VERSION
from app.models.report import ProcessingStatus, Report
from app.services.pdf_pipeline import process_pdf


DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_report_by_id(db: Session, report_id: uuid.UUID) -> Optional[Report]:
    return db.get(Report, report_id)


def get_report_by_hash(db: Session, pdf_hash: str) -> Optional[Report]:
    stmt = select(Report).where(Report.pdf_hash == pdf_hash)
    return db.scalars(stmt).first()


def get_report_status(db: Session, report_id: uuid.UUID) -> dict:
    """Get report status for polling."""
    report = get_report_by_id(db, report_id)
    if report is None:
        return {"status": "not_found", "progress_pct": 0}
    
    # Map status to progress percentage
    status_progress = {
        "pending": 5,
        "processing": 50,
        "completed": 100,
        "failed": 100,
    }
    return {
        "status": report.status.value,
        "progress_pct": status_progress.get(report.status.value, 0),
        "error_message": report.error_message,
    }


def list_reports(
    db: Session,
    page: int = 1,
    limit: int = 20,
    min_score: Optional[float] = None,
    model_version: Optional[str] = None,
    status: Optional[ProcessingStatus] = None,
) -> tuple[List[Report], int]:
    """Return (reports, total_count) with simple filters + pagination."""
    stmt = select(Report)
    count_stmt = select(func.count()).select_from(select(Report))

    if min_score is not None:
        stmt = stmt.where(Report.score >= min_score)
        count_stmt = count_stmt.where(Report.score >= min_score)
    if model_version is not None:
        stmt = stmt.where(Report.model_version == model_version)
        count_stmt = count_stmt.where(Report.model_version == model_version)
    if status is not None:
        stmt = stmt.where(Report.status == status)
        count_stmt = count_stmt.where(Report.status == status)

    total = db.execute(count_stmt).scalar() or 0
    offset = (max(1, page) - 1) * limit
    stmt = stmt.order_by(desc(Report.created_at)).offset(offset).limit(limit)
    return list(db.scalars(stmt)), total


def create_pending_report(db: Session, pdf_bytes: bytes, filename: str) -> Report:
    """Create a pending report row immediately (no pipeline execution).
    
    Idempotent:
      - If a completed row exists for this hash -> return it (cached).
      - If a failed/pending row exists -> reset to pending, return it.
    """
    import hashlib
    from sqlalchemy.dialects.postgresql import insert
    pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()

    # Persist PDF to disk (audit artifact).
    saved_path = DATA_DIR / f"{pdf_hash[:16]}.pdf"
    saved_path.write_bytes(pdf_bytes)

    # Upsert: insert new row, or do nothing if pdf_hash already exists.
    stmt = insert(Report.__table__).values(
        pdf_hash=pdf_hash,
        pdf_path=str(saved_path),
        original_filename=filename,
        model_version=MODEL_VERSION,
        status=ProcessingStatus.pending,
        llm_model=settings.nemotron_model,
        llm_prompt_version="kpi_extract_v1",
    ).on_conflict_do_nothing(index_elements=['pdf_hash'])
    db.execute(stmt)
    db.commit()

    # Re-fetch the report (either the one we just inserted, or the existing one)
    report = get_report_by_hash(db, pdf_hash)

    if report.status == ProcessingStatus.completed:
        # Cached result: return as-is, skip re-processing.
        return report

    # Reuse the existing (failed/pending) row — reset to pending.
    report.pdf_path = str(saved_path)
    report.original_filename = filename
    report.kpis = None
    report.score = None
    report.label = None
    report.confidence = None
    report.shap_values = None
    report.brief = None
    report.is_ood = None
    report.is_exceptional = None
    report.ood_info = None
    report.feature_metadata = None
    report.error_message = None
    report.processing_time_ms = None
    report.status = ProcessingStatus.pending

    return report


def process_upload(db: Session, pdf_bytes: bytes, filename: str) -> Report:
    """Main entry: create pending report -> run pipeline -> update -> return Report.
    
    This is the synchronous version for backward compatibility / direct calls.
    """
    report = create_pending_report(db, pdf_bytes, filename)
    
    if report.status == ProcessingStatus.completed:
        return report
    
    return _run_pipeline_sync(db, report, pdf_bytes, filename)


def _run_pipeline_sync(db: Session, report: Report, pdf_bytes: bytes, filename: str) -> Report:
    """Internal: run the heavy pipeline synchronously."""
    # Mark as processing
    report.status = ProcessingStatus.processing
    db.commit()
    db.refresh(report)

    # Wipe any stale chunks from a previous failed run.
    from app.models.chunk import Chunk
    db.query(Chunk).filter(Chunk.report_id == report.id).delete()
    db.commit()

    try:
        result = process_pdf(pdf_bytes, report.id, db, filename)
        report.kpis = result["kpis"]
        report.score = result["score"]
        report.label = result["label"]
        report.confidence = result["confidence"]
        report.shap_values = result["shap"]
        report.brief = result["brief"]
        report.processing_time_ms = result["elapsed_ms"]
        report.is_ood = (result.get("ood_info") or {}).get("is_ood")
        report.is_exceptional = (result.get("ood_info") or {}).get("is_exceptional")
        report.ood_info = result.get("ood_info")
        report.feature_metadata = result.get("feature_metadata")
        report.status = ProcessingStatus.completed
        report.error_message = None
    except Exception as exc:
        report.status = ProcessingStatus.failed
        report.error_message = f"{type(exc).__name__}: {exc}"
        print(f"[report_service] pipeline failed: {exc}")

    db.commit()
    db.refresh(report)
    return report


def run_pipeline_background(pdf_bytes: bytes, report_id: uuid.UUID, filename: str) -> None:
    """Background task to run the full pipeline (for BackgroundTasks)."""
    import sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[3]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from app.core.config import settings
    
    # Create new DB session for background task
    engine = create_engine(settings.database_url, pool_pre_ping=True)
    with Session(engine) as db:
        report = get_report_by_id(db, report_id)
        if report is None:
            print(f"[background] Report {report_id} not found")
            return
        
        try:
            _run_pipeline_sync(db, report, pdf_bytes, filename)
        except Exception as e:
            print(f"[background] Pipeline failed for {report_id}: {e}")


def delete_report(db: Session, report_id: uuid.UUID) -> bool:
    """Delete a report, its associated chunks, and the PDF file from disk.
    Returns True if deleted, False if not found.
    """
    report = get_report_by_id(db, report_id)
    if report is None:
        return False

    # Delete PDF file from disk (best effort - don't fail DB deletion if file issues)
    from pathlib import Path
    try:
        Path(report.pdf_path).unlink(missing_ok=True)
    except Exception as e:
        print(f"[delete_report] Warning: Could not delete PDF {report.pdf_path}: {e}")

    # Delete associated chunks first (foreign key constraint)
    from app.models.chunk import Chunk
    db.query(Chunk).filter(Chunk.report_id == report_id).delete()
    # Delete the report
    db.delete(report)
    db.commit()
    return True


def reprocess_report(db: Session, report_id: uuid.UUID, pdf_bytes: bytes, filename: str) -> Report:
    """Re-process an existing report with the same PDF bytes.

    This resets the report fields and re-runs the full pipeline.
    Unlike process_upload, this does NOT check for duplicates by hash
    — it always re-processes the given report_id.
    """
    report = get_report_by_id(db, report_id)
    if report is None:
        raise ValueError(f"Report {report_id} not found")

    # Reset all analysis fields
    report.kpis = None
    report.score = None
    report.label = None
    report.confidence = None
    report.shap_values = None
    report.brief = None
    report.is_ood = None
    report.is_exceptional = None
    report.ood_info = None
    report.feature_metadata = None
    report.error_message = None
    report.processing_time_ms = None
    report.status = ProcessingStatus.processing
    db.commit()
    db.refresh(report)

    # Wipe any stale chunks from a previous failed run.
    from app.models.chunk import Chunk
    db.query(Chunk).filter(Chunk.report_id == report.id).delete()
    db.commit()

    try:
        result = process_pdf(pdf_bytes, report.id, db, filename)
        report.kpis = result["kpis"]
        report.score = result["score"]
        report.label = result["label"]
        report.confidence = result["confidence"]
        report.shap_values = result["shap"]
        report.brief = result["brief"]
        report.processing_time_ms = result["elapsed_ms"]
        report.is_ood = (result.get("ood_info") or {}).get("is_ood")
        report.is_exceptional = (result.get("ood_info") or {}).get("is_exceptional")
        report.ood_info = result.get("ood_info")
        report.feature_metadata = result.get("feature_metadata")
        report.status = ProcessingStatus.completed
        report.error_message = None
    except Exception as exc:
        report.status = ProcessingStatus.failed
        report.error_message = f"{type(exc).__name__}: {exc}"
        print(f"[report_service] reprocess failed: {exc}")

    db.commit()
    db.refresh(report)
    return report
