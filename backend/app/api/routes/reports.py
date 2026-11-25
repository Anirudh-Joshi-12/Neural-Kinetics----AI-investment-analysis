"""Report routes — upload, list, get-by-id, PDF serving, reprocess."""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.api.deps import get_session
from app.models.report import ProcessingStatus
from app.schemas.report import ReportListResponse, ReportResponse
from app.services.report_service import (
    get_report_by_id,
    list_reports,
    create_pending_report,
    delete_report as delete_report_svc,
    get_report_status,
    reprocess_report,
    run_pipeline_background,
)


router = APIRouter(prefix="/reports", tags=["reports"])


def is_valid_pdf(filename: str | None, content_type: str | None) -> bool:
    """Validate PDF by extension (primary) or MIME type (fallback)."""
    name_ok = (filename or "").lower().endswith(".pdf")
    type_ok = content_type in ("application/pdf", "application/x-pdf", "application/octet-stream", None)
    return name_ok or type_ok


@router.post(
    "/upload",
    response_model=ReportResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a PDF and run the full ML+GenAI pipeline in background.",
)
async def upload_report(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Annual report PDF"),
    db: Session = Depends(get_session),
):
    if not is_valid_pdf(file.filename, file.content_type):
        raise HTTPException(400, "Only PDF uploads are accepted.")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(400, "Uploaded PDF is empty.")

    filename = file.filename or "upload.pdf"
    
    # Create pending report immediately (don't wait for pipeline)
    report = create_pending_report(db, pdf_bytes, filename)
    
    # Schedule pipeline in background
    background_tasks.add_task(run_pipeline_background, pdf_bytes, report.id, filename)
    
    return report


@router.get(
    "",
    response_model=ReportListResponse,
    summary="List reports with pagination + filters.",
)
def list_reports_route(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    min_score: Optional[float] = Query(None, ge=0, le=100),
    model_version: Optional[str] = None,
    status_filter: Optional[ProcessingStatus] = Query(None, alias="status"),
    db: Session = Depends(get_session),
):
    items, total = list_reports(
        db, page=page, limit=limit, min_score=min_score,
        model_version=model_version, status=status_filter,
    )
    return {
        "total": total,
        "page": page,
        "limit": limit,
        "items": items,
    }


@router.get(
    "/{report_id}",
    response_model=ReportResponse,
    summary="Fetch a single report by ID.",
)
def get_report(report_id: uuid.UUID, db: Session = Depends(get_session)):
    report = get_report_by_id(db, report_id)
    if report is None:
        raise HTTPException(404, f"Report {report_id} not found.")
    return report


@router.get(
    "/{report_id}/pdf",
    summary="Download the original PDF file for a report.",
)
def download_report_pdf(report_id: uuid.UUID, db: Session = Depends(get_session)):
    """Stream the original PDF file from backend/data/{hash}.pdf."""
    report = get_report_by_id(db, report_id)
    if report is None:
        raise HTTPException(404, f"Report {report_id} not found.")

    pdf_path = Path(report.pdf_path)
    if not pdf_path.exists():
        raise HTTPException(404, f"PDF file not found on disk for report {report_id}.")

    # Use original filename if available, otherwise generate one
    filename = report.original_filename or f"report_{report_id}.pdf"
    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=filename,
    )


@router.get(
    "/{report_id}/status",
    summary="Get report processing status for polling.",
)
def get_report_status_route(report_id: uuid.UUID, db: Session = Depends(get_session)):
    return get_report_status(db, report_id)


@router.post(
    "/{report_id}/reprocess",
    response_model=ReportResponse,
    summary="Re-run the analysis pipeline for an existing report.",
)
def reprocess_report_route(report_id: uuid.UUID, db: Session = Depends(get_session)):
    """Re-trigger the full pipeline (extraction + scoring) for an existing report.

    This is useful when:
    - The model has been retrained and you want fresh scores
    - A transient error occurred during the original processing
    - You want to re-extract KPIs with an updated prompt
    """
    report = get_report_by_id(db, report_id)
    if report is None:
        raise HTTPException(404, f"Report {report_id} not found.")

    # Read the PDF bytes from disk
    pdf_path = Path(report.pdf_path)
    if not pdf_path.exists():
        raise HTTPException(404, f"PDF file not found on disk for report {report_id}.")

    pdf_bytes = pdf_path.read_bytes()

    # Re-process using the existing report ID (will reset and re-run)
    report = reprocess_report(db, report_id, pdf_bytes, report.original_filename or "reprocess.pdf")
    return report


@router.delete(
    "/{report_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a report by ID.",
)
def delete_report(report_id: uuid.UUID, db: Session = Depends(get_session)):
    report = get_report_by_id(db, report_id)
    if report is None:
        raise HTTPException(404, f"Report {report_id} not found.")
    delete_report_svc(db, report_id)
    return None
