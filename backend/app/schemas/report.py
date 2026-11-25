"""Pydantic schemas for Report API validation + serialization."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.models.report import ProcessingStatus


class KPISchema(BaseModel):
    """Strict KPI structure — used internally to validate Nemotron JSON output."""
    revenue_growth: Optional[float] = None
    net_margin: Optional[float] = None
    debt_equity: Optional[float] = None
    roe: Optional[float] = None
    current_ratio: Optional[float] = None


class ReportBase(BaseModel):
    pdf_hash: str
    original_filename: Optional[str] = None
    model_version: str = "rf_v1"


class ReportResponse(BaseModel):
    """Full report returned by GET /reports/{id} and POST /reports/upload."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    pdf_hash: str
    pdf_path: str
    original_filename: Optional[str] = None
    kpis: Optional[Dict[str, Any]] = None
    score: Optional[float] = None
    label: Optional[str] = None
    confidence: Optional[float] = None
    shap_values: Optional[Dict[str, Any]] = None
    brief: Optional[str] = None
    model_version: str
    llm_model: Optional[str] = None
    llm_prompt_version: Optional[str] = None
    status: ProcessingStatus
    error_message: Optional[str] = None
    processing_time_ms: Optional[int] = None
    created_at: datetime
    updated_at: datetime
    is_ood: Optional[bool] = None
    is_exceptional: Optional[bool] = None
    ood_info: Optional[Dict[str, Any]] = None
    feature_metadata: Optional[Dict[str, Any]] = None


class ReportListItem(BaseModel):
    """Compact row returned by GET /reports (paginated list)."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    original_filename: Optional[str] = None
    score: Optional[float] = None
    label: Optional[str] = None
    confidence: Optional[float] = None
    status: ProcessingStatus
    model_version: str
    created_at: datetime


class ReportListResponse(BaseModel):
    """Paginated response envelope for GET /reports."""
    total: int
    page: int
    limit: int
    items: List[ReportListItem]


class ErrorResponse(BaseModel):
    detail: str
    status: Optional[str] = None
