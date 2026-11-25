"""Phase 1 entry point — end-to-end test of the full backend logic.

Run from the project root:  python -m ml.run_phase1

Or from inside the ml/ folder:  python run_phase1.py

Does:
    1. Generate synthetic data, train the Random Forest scorer, evaluate, save.
    2. Load the saved model, run the full PDF pipeline on the sample PDF.
    3. Print the extracted KPIs, score, label, confidence, SHAP, and brief.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is importable when run as a script (not -m).
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.config import settings
from ml.ml import (
    evaluate,
    generate_data,
    load_model,
    save_model,
    train,
)
from ml.pipeline import run_pipeline


def main() -> int:
    print("=" * 70)
    print("PHASE 1 — End-to-end ML + GenAI pipeline test")
    print("=" * 70)

    # ------------------ STEP 1: TRAIN ML MODEL ------------------
    print("\n>>> STEP 1: ML MODEL TRAINING")
    X, y = generate_data(settings.n_samples)
    print(f"Generated {len(X)} synthetic rows | positive class ratio: {y.mean():.2%}")

    pipeline = train(X, y)
    save_model(pipeline, settings.model_save_path)

    # ------------------ STEP 2: RUN PDF PIPELINE ------------------
    print("\n>>> STEP 2: FULL GEN AI BACKEND PIPELINE")
    pdf_path = settings.sample_pdf_path
    if not Path(pdf_path).exists():
        print(f"\n  ! Sample PDF not found at '{pdf_path}'.")
        print("    Set SAMPLE_PDF_PATH in your .env to a real annual report PDF,")
        print("    then re-run:  python -m ml.run_phase1")
        print("\nPhase 1 ML part complete. Re-run with a PDF to exercise the GenAI pipeline.")
        return 0

    scorer = load_model(settings.model_save_path)
    result = run_pipeline(pdf_path, scorer)

    print("\n" + "=" * 70)
    print("FINAL INVESTMENT REPORT")
    print("=" * 70)
    print(f"KPIs       : {result['kpis']}")
    print(f"Score      : {result['score']}/100  ({result['label']})")
    print(f"Confidence : {result['confidence']}%")
    print(f"SHAP       : {result['shap']}")
    print("\nINVESTMENT BRIEF:")
    print("-" * 70)
    # Handle Windows console encoding safely
    brief = result["brief"]
    try:
        print(brief)
    except UnicodeEncodeError:
        print(brief.encode('ascii', 'replace').decode('ascii'))
    print("-" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
