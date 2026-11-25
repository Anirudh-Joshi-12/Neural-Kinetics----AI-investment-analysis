"""FastAPI shared dependencies."""
from __future__ import annotations

from fastapi import Depends
from sqlalchemy.orm import Session

from app.core.database import get_db


def get_session(db: Session = Depends(get_db)) -> Session:
    """Alias for get_db — use wherever a route needs a DB session."""
    return db
