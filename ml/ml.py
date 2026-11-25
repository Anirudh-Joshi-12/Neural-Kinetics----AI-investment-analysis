"""ML scorer: synthetic data generation, training, evaluation, scoring, SHAP.

Functional style: pure functions where practical, explicit return types,
no hidden side effects except documented file I/O (save_model, evaluate PNG).

v1.5: OOD (out-of-distribution) detection wired into score_company():
the function now returns (score, label, confidence, ood_info). Callers that
ignore the 4th return value retain backward compat. The OOD distribution
is loaded from `train_kpi_distribution.joblib` alongside the model.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import math
import matplotlib
matplotlib.use("Agg")  # non-interactive backend; safe for headless runs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler

from ml.common import KPI_SCHEMA, MODEL_VERSION
from ml.config import settings
from ml.ood import score as ood_score


DISTRIBUTION_PATH = Path(settings.model_save_path).parent / "train_kpi_distribution.joblib"


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------
def generate_data(n: int = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Generate n synthetic rows with 5 KPI features + binary labels.

    Labeling rules (deterministic + small noise) model a conceptual
    "good investment" classifier:
        positive if  growth > 0.05  AND  margin > 0.03  AND  debt_equity < 2
                          AND  roe > 0.08  AND  current_ratio > 1.0
    """
    n = n or settings.n_samples
    rng = np.random.default_rng(seed=42)

    revenue_growth = rng.uniform(-0.2, 0.4, n)
    net_margin = rng.normal(0.05, 0.05, n)
    debt_equity = rng.uniform(0.1, 3.0, n)
    roe = rng.normal(0.1, 0.06, n)
    current_ratio = rng.uniform(0.5, 3.5, n)

    X = pd.DataFrame({
        "revenue_growth": revenue_growth,
        "net_margin": net_margin,
        "debt_equity": debt_equity,
        "roe": roe,
        "current_ratio": current_ratio,
    })

    rule = (
        (X["revenue_growth"] > 0.05) &
        (X["net_margin"] > 0.03) &
        (X["debt_equity"] < 2.0) &
        (X["roe"] > 0.08) &
        (X["current_ratio"] > 1.0)
    ).astype(int).values

    noise = rng.binomial(1, 0.05, n)
    y = pd.Series(rule ^ noise, name="label")

    return X, y


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(X: pd.DataFrame, y: pd.Series) -> SkPipeline:
    """Fit a StandardScaler + RandomForest pipeline on an 80/20 split.

    Returns the fitted pipeline (trained on the full X/y after evaluation
    isolation is handled by the caller via the split). The returned pipeline
    is ready for scoring and SHAP.
    """
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=settings.test_size, random_state=42, stratify=y
    )

    pipeline = SkPipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])
    pipeline.fit(X_tr, y_tr)
    # Evaluate on the held-out split for honest metrics.
    evaluate(pipeline, X_te, y_te)
    # Refit on ALL data for the deployment-ready model.
    pipeline.fit(X, y)
    return pipeline


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(pipeline: SkPipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
    """Print metrics and save SHAP summary PNG. Returns a metrics dict."""
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    acc = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    cm = confusion_matrix(y_test, y_pred).tolist()
    cv = cross_val_score(pipeline, X_test, y_test, cv=5, scoring="f1", n_jobs=-1)
    cv_mean, cv_std = float(cv.mean()), float(cv.std())
    conf_mean, conf_std = float(y_proba.mean()), float(y_proba.std())

    print("\n" + "=" * 60)
    print("ML MODEL EVALUATION")
    print("=" * 60)
    print(f"Accuracy            : {acc:.4f}")
    print(f"F1 Score            : {f1:.4f}")
    print(f"Cross-Val F1 (5-fold): {cv_mean:.4f} +/- {cv_std:.4f}")
    print(f"Confusion Matrix    : {cm}")
    print(f"Confidence (mean)   : {conf_mean:.4f} +/- {conf_std:.4f}")
    print(f"Thresholds          : MIN_ACCURACY={settings.min_accuracy}, "
          f"MIN_CONFIDENCE={settings.min_confidence}")

    if acc < settings.min_accuracy:
        print(f"  ! WARNING: accuracy {acc:.4f} < {settings.min_accuracy}")
    if conf_mean < settings.min_confidence:
        print(f"  ! WARNING: mean confidence {conf_mean:.4f} < {settings.min_confidence}")

    _save_shap_summary(pipeline, X_test)
    return {
        "accuracy": acc, "f1": f1, "confusion_matrix": cm,
        "cv_mean": cv_mean, "cv_std": cv_std,
        "confidence_mean": conf_mean, "confidence_std": conf_std,
    }


def _save_shap_summary(pipeline: SkPipeline, X_test: pd.DataFrame) -> None:
    """Internal: render and save SHAP summary plot for the test set."""
    out_dir = settings.shap_dir_obj
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "shap_summary.png"

    rf = pipeline.named_steps["clf"]
    # SHAP expects raw (unscaled) features for TreeExplainer; StandardScaler
    # is a no-op for tree splits, so we pass the scaled values the classifier
    # actually saw. We reconstruct via pipeline transform for correctness.
    X_scaled = pipeline.named_steps["scaler"].transform(X_test)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X_test.columns)

    try:
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_scaled_df)

        if isinstance(shap_values, list):
            shap_arr = shap_values[1]
        else:
            shap_arr = np.asarray(shap_values)
            if shap_arr.ndim == 3:
                shap_arr = shap_arr[:, :, 1]

        plt.figure(figsize=(8, 5))
        shap.summary_plot(shap_arr, X_scaled_df, show=False)
        plt.tight_layout()
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"SHAP summary saved -> {out_path}")
    except Exception as exc:
        print(f"  ! SHAP plot skipped: {exc}")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score_company(
    pipeline: SkPipeline, features: pd.DataFrame,
    distribution: Optional[dict] = None,
    calibration: Optional[dict] = None,
) -> Tuple[float, str, float, Optional[dict]]:
    """Score a single company from a feature DataFrame row.

    Supports both 5-feature (KPI only) and 20-feature (extended) models.

    Returns (score_0_100, label, confidence_pct, ood_info).
    ood_info is None if no distribution is supplied, otherwise the dict from
    ml.ood.score().
    """
    expected_features = getattr(pipeline.named_steps['clf'], 'feature_names_in_', None)
    if expected_features is not None:
        if list(features.columns) != list(expected_features):
            raise ValueError(
                f"Feature columns {list(features.columns)} != expected schema {list(expected_features)}"
            )
    elif list(features.columns) != KPI_SCHEMA:
        # Fallback for older models without feature_names_in_
        raise ValueError(
            f"Feature columns {list(features.columns)} != expected schema {KPI_SCHEMA}"
        )

    proba = float(pipeline.predict_proba(features)[0, 1])

    # Apply Platt calibration if available
    if calibration is not None:
        calib_a = calibration.get("platt_A")
        calib_b = calibration.get("platt_B")
        if calib_a is not None and calib_b is not None:
            z = calib_a * proba + calib_b
            proba = 1.0 / (1.0 + math.exp(-z))
            proba = min(max(proba, 0.001), 0.999)

    # Apply calibrated threshold if available
    threshold = calibration.get("optimal_threshold", 0.5) if calibration else 0.5
    neg_threshold = 1.0 - threshold

    if proba >= threshold:
        label = "Strong Buy" if proba >= 0.75 else "Buy"
    elif proba <= neg_threshold:
        label = "Strong Avoid" if proba <= 0.30 else "Avoid"
    else:
        label = "Hold"

    score = round(proba * 100.0, 2)
    confidence = round(abs(proba - 0.5) * 200.0, 2)  # 0% at 0.5, 100% at 0/1

    ood_info = None
    if distribution is None:
        # Try to load OOD distribution automatically
        distribution = load_ood_distribution()
    if distribution is not None:
        try:
            ood_info = ood_score(features, distribution)
        except Exception:
            ood_info = None  # Fail silently on OOD errors

    return score, label, confidence, ood_info


def _label_from_score(score: float) -> str:
    """Map a 0-100 score to a discrete investment label."""
    if score >= 75:
        return "Strong Buy"
    if score >= 60:
        return "Buy"
    if score >= 45:
        return "Hold"
    if score >= 30:
        return "Avoid"
    return "Strong Avoid"


# ---------------------------------------------------------------------------
# SHAP for single prediction
# ---------------------------------------------------------------------------
def get_shap_values(pipeline: SkPipeline, features: pd.DataFrame) -> Dict[str, float]:
    """Return ranked feature -> SHAP value dict for a single-company prediction.

    Robust to both legacy SHAP output shapes:
      - list[class_0_array, class_1_array]   (binary classifiers, older shap)
      - ndarray of shape (n_samples, n_features)                (newer shap)
      - ndarray of shape (n_samples, n_features, n_classes)     (some versions)
    """
    rf = pipeline.named_steps["clf"]
    scaler = pipeline.named_steps["scaler"]
    X_scaled = pd.DataFrame(scaler.transform(features), columns=features.columns)

    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_scaled)

    if isinstance(shap_values, list):
        shap_arr = shap_values[1]
    else:
        shap_arr = shap_values

    shap_arr = np.asarray(shap_arr)

    # Handle (n_samples, n_features, n_classes): pick class 1 slice.
    if shap_arr.ndim == 3:
        shap_arr = shap_arr[:, :, 1]

    row = shap_arr[0]

    pairs = [
        (str(col), float(np.atleast_1d(v)[0]))
        for col, v in zip(features.columns, row)
    ]
    pairs.sort(key=lambda kv: -abs(kv[1]))
    return dict(pairs)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_model(pipeline: SkPipeline, path: str = None) -> Path:
    """Persist the pipeline to joblib. Ensures parent dir exists."""
    target = Path(path or settings.model_save_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, target)
    print(f"Model saved -> {target}")
    return target


def load_model(path: str = None) -> SkPipeline:
    """Load a previously saved pipeline. Errors loudly if missing."""
    target = Path(path or settings.model_save_path)
    if not target.exists():
        raise FileNotFoundError(f"Model file not found: {target}")
    return joblib.load(target)


def load_ood_distribution() -> Optional[dict]:
    """Load the kNN OOD distribution. Returns None if not available (back-compat
    for callers that don't care about OOD; score_company() then returns null
    ood_info instead of raising)."""
    target = DISTRIBUTION_PATH
    if not target.exists():
        return None
    return joblib.load(target)
