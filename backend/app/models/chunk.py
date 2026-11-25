"""Chunk SQLAlchemy model — one row per text chunk + its embedding."""
from __future__ import annotations

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base
from app.core.config import settings


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    report_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("reports.id", ondelete="CASCADE"), nullable=False
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_type: Mapped[str] = mapped_column(String(20), nullable=False, default="other")
    text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding = mapped_column(Vector(4096), nullable=True)  # nemotron-3.5-embed-8b dim

    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("ix_chunks_report_id", "report_id"),
        Index("ix_chunks_report_type", "report_id", "chunk_type"),
    )

    def __repr__(self) -> str:
        return f"<Chunk(id={self.id}, report_id={self.report_id}, idx={self.chunk_index}, len={len(self.text)})>"
