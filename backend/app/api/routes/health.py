"""Health-check route."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.api.deps import get_session
from app.core.config import settings
from app.services.ml_scorer import MLScorer


router = APIRouter(prefix="/health", tags=["health"])


@router.get("", status_code=status.HTTP_200_OK)
def health_check(db: Session = Depends(get_session)):
    """Returns: {'status': 'ok', 'db': True/False, 'ml': True/False, 'nvidia_api': True/False}."""
    db_ok = False
    try:
        db.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False

    # Check NVIDIA API key validity (fail fast in production)
    nvidia_api_ok = True
    try:
        settings.validate_nvidia_api_key()
    except RuntimeError:
        nvidia_api_ok = False

    ml_loaded = MLScorer.is_loaded()

    if not (db_ok and ml_loaded and nvidia_api_ok):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "degraded",
                "db": db_ok,
                "ml_model_loaded": ml_loaded,
                "ml_model_version": MLScorer.get_version() if ml_loaded else None,
                "ml_ood_loaded": MLScorer.has_ood() if ml_loaded else False,
                "nvidia_api_key_valid": nvidia_api_ok,
            },
        )

    return {
        "status": "ok",
        "db": db_ok,
        "ml_model_loaded": ml_loaded,
        "ml_model_version": MLScorer.get_version(),
        "ml_ood_loaded": MLScorer.has_ood(),
        "nvidia_api_key_valid": nvidia_api_ok,
    }
