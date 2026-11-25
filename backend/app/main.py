"""FastAPI application entrypoint.

Run:
    cd backend && uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure project root (containing the `ml` package) is importable from any CWD.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import health, reports
from app.core.database import Base, engine
from app.services.ml_scorer import MLScorer


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure pgvector extension exists (defensive, idempotent).
    print("[startup] Ensuring pgvector extension...")
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
        print("[startup] pgvector extension ready")
    except Exception as e:
        logger.error(f"FATAL: Failed to create pgvector extension: {e}")
        raise RuntimeError("Database pgvector extension unavailable") from e

    # Startup: load ML model once (frozen inference).
    # FAIL FAST: if model or OOD distribution is missing, crash at startup
    # so we don't silently serve 500s later.
    print("[startup] Loading ML model...")
    try:
        MLScorer.load()
        print(f"[startup] Model loaded: v{MLScorer.get_version()}")
        print(f"[startup] OOD distribution: {'loaded' if MLScorer.has_ood() else 'MISSING'}")
    except FileNotFoundError as e:
        logger.error(f"FATAL: {e}")
        raise RuntimeError(
            "ML model or OOD distribution not found. "
            "Train the model first: python -m ml.train_v15"
        ) from e

    # Optional: ensure tables exist (no-op if already there).
    # For first-time bootstrap run `python init_db.py` instead.
    yield

    # Shutdown: nothing to do.
    print("[shutdown] stopped.")


app = FastAPI(
    title="Investment Analysis API",
    description="ML + Nemotron pipeline for scoring annual report PDFs.",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow Vite dev server to call this API (adjust origin in production).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers.
app.include_router(health.router)
app.include_router(reports.router)


@app.get("/", tags=["root"])
def root():
    return {
        "service": "Investment Analysis API",
        "docs": "/docs",
        "version": app.version,
    }
