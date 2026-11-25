"""Report SQLAlchemy model — one row per analyzed PDF."""
from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, Enum, Float, Index, String, Text
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class ProcessingStatus(str, enum.Enum):
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class Report(Base):
    __tablename__ = "reports"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    # Idempotency: SHA256 of the PDF bytes — prevents duplicate re-processing.
    pdf_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    pdf_path: Mapped[str] = mapped_column(String(512), nullable=False)
    original_filename: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # KPIs extracted by Nemotron (validated JSON matching KPI_SCHEMA)
    kpis: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # ML scorer output (frozen model)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    label: Mapped[str | None] = mapped_column(String(20), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    shap_values: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # OOD (out-of-distribution) detection — v1.5 honesty layer.
    # is_ood: True when the extracted KPI profile has low training-data coverage,
    # meaning the model's confidence cannot be trusted at face value.
    # ood_info: full diagnostics dict from ml.ood.score() (knn_pct, coverage, etc).
    # is_exceptional: True when the company is exceptionally good — all KPIs are
    # top-tier AND ML score is Strong Buy or better.
    is_ood: Mapped[bool | None] = mapped_column(nullable=True)
    is_exceptional: Mapped[bool | None] = mapped_column(nullable=True)
    ood_info: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # GenAI brief
    brief: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Model provenance (critical for retraining audit)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False, default="rf_v1.1_12feat")
    llm_model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    llm_prompt_version: Mapped[str | None] = mapped_column(String(30), nullable=True)

    # Processing metadata
    status: Mapped[ProcessingStatus] = mapped_column(
        Enum(ProcessingStatus, name="processing_status"),
        default=ProcessingStatus.pending,
        nullable=False,
        index=True,
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    processing_time_ms: Mapped[int | None] = mapped_column(nullable=True)
    feature_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Future: human labels for retraining pipeline
    human_label: Mapped[str | None] = mapped_column(String(20), nullable=True)
    labeled_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    labeled_by: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    __table_args__ = (
        Index("ix_reports_score_created", "score", "created_at"),
        Index("ix_reports_model_version", "model_version"),
    )

    def __repr__(self) -> str:
        return (
            f"<Report(id={self.id}, status={self.status}, "
            f"score={self.score}, label={self.label})>"
        )
