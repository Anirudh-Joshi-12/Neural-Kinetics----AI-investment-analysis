"""Thin wrapper around the existing ml.ml package — load + score + SHAP.

v1.5: OOD (out-of-distribution) detection wired into score(); also loads
`train_kpi_distribution.joblib` alongside the model.

v1.6: Calibration + threshold tuning applied to raw pipeline probabilities.
   - Platt calibration (a, b) fitted on out-of-fold predictions in train_v15.py
   - Optimal threshold tuned on OOF probabilities to maximize F1
   - Both are loaded from calibration.json at startup and applied in score()
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root (containing `ml` package) is importable.
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threading import Lock
from typing import Dict, Optional, Tuple

import joblib
import json
import math
import pandas as pd

from backend.app.core.config import settings, MODEL_VERSION, KPI_SCHEMA, ALL_FEATURES
from ml.ood import score as ood_score, EXCEPTIONAL_THRESHOLDS


class ModelNotReadyError(RuntimeError):
    """Raised when ML model is not loaded."""
    pass


class MLScorer:
    """Singleton-style scorer — loads the frozen model once at startup."""

    _pipeline = None
    _ood_distribution = None
    _calib_a: Optional[float] = None
    _calib_b: Optional[float] = None
    _calib_threshold: Optional[float] = None
    _lock = Lock()
    _initialized = False

    @classmethod
    def load(cls) -> None:
        """Load the joblib model + OOD distribution into memory. Thread-safe.

        Raises:
            FileNotFoundError: If model file or OOD distribution file is missing.
            RuntimeError: If NVIDIA API key is still placeholder (needed for brief generation).
        """
        with cls._lock:
            if cls._initialized:
                return

            # Validate NVIDIA API key early — fail fast if placeholder
            settings.validate_nvidia_api_key()

            path = settings.model_path_obj
            if not path.exists():
                raise FileNotFoundError(
                    f"Model file not found: {path}. "
                    f"Train the model first with: python -m ml.train_v15"
                )
            cls._pipeline = joblib.load(path)

            # OOD distribution is REQUIRED in v1.5 — fail fast if missing
            dist_path = Path(path).parent / "train_kpi_distribution.joblib"
            if not dist_path.exists():
                raise FileNotFoundError(
                    f"OOD distribution file not found: {dist_path}. "
                    f"Train the model first with: python -m ml.train_v15"
                )
            cls._ood_distribution = joblib.load(dist_path)

            # Optional Platt calibration (written by ml.train_v15). If present,
            # raw predict_proba is mapped through a logistic fix that un-bimodalizes
            # the RF output. Absent file -> raw probabilities are used as-is.
            calib_path = Path(path).parent / "calibration.json"
            cls._calib_a = cls._calib_b = None
            cls._calib_threshold = None
            if calib_path.exists():
                try:
                    calib = json.loads(calib_path.read_text(encoding="utf-8"))
                    # Support both key formats: legacy "a"/"b" and current "platt_A"/"platt_B"
                    cls._calib_a = float(calib.get("platt_A", calib.get("a", 0)))
                    cls._calib_b = float(calib.get("platt_B", calib.get("b", 0)))
                    cls._calib_threshold = float(calib.get("optimal_threshold", calib.get("threshold", 0.5)))
                except (ValueError, KeyError, OSError):
                    cls._calib_a = cls._calib_b = None
                    cls._calib_threshold = None

            cls._initialized = True

    @classmethod
    def is_loaded(cls) -> bool:
        return cls._pipeline is not None and cls._ood_distribution is not None

    @classmethod
    def score(
        cls, features: pd.DataFrame, raw_kpis: Optional[Dict[str, Any]] = None
    ) -> Tuple[float, str, float, Optional[dict]]:
        """Returns (score_0_100, label, confidence_pct, ood_info)."""
        cls._ensure_loaded()
        proba = float(cls._pipeline.predict_proba(features)[0, 1])

        # Platt calibration: maps raw RF probability through a logistic fix
        # fitted on held-out data. Softens the bimodal 0/1 output so real
        # companies get intermediate scores (Hold/Avoid ranges exist again).
        if cls._calib_a is not None and cls._calib_b is not None:
            z = cls._calib_a * proba + cls._calib_b
            proba = 1.0 / (1.0 + math.exp(-z))
            proba = min(max(proba, 0.001), 0.999)

        # Apply calibrated threshold for buy vs avoid decision.
        # Found by maximizing F1 on out-of-fold predictions during training.
        threshold = cls._calib_threshold if cls._calib_threshold is not None else 0.5
        neg_threshold = 1.0 - threshold

        if proba >= threshold:
            # Positive side
            label = "Strong Buy" if proba >= 0.75 else "Buy"
        elif proba <= neg_threshold:
            # Negative side (symmetric)
            label = "Strong Avoid" if proba <= 0.30 else "Avoid"
        else:
            # Middle ground between threshold and 1-threshold
            label = "Hold"

        score = round(proba * 100.0, 2)
        confidence = round(abs(proba - 0.5) * 200.0, 2)

        ood_info = None
        if cls._ood_distribution is not None:
            ood_info = ood_score(features, cls._ood_distribution)
            if raw_kpis is not None:
                ood_info["is_ood"] = ood_info.get("is_ood", False)
                ood_info["is_exceptional"] = _is_exceptional(raw_kpis, score)

        return score, label, confidence, ood_info

    @classmethod
    def has_ood(cls) -> bool:
        """True if the OOD distribution is loaded (lets backend gate OOD features)."""
        return cls._ood_distribution is not None

    @classmethod
    def get_shap(cls, features: pd.DataFrame) -> Dict[str, float]:
        """Ranked {feature: shap_value} dict for a single row."""
        cls._ensure_loaded()
        import shap
        import numpy as np

        rf = cls._pipeline.named_steps["clf"]
        scaler = cls._pipeline.named_steps["scaler"]
        X_scaled = pd.DataFrame(scaler.transform(features), columns=features.columns)

        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_scaled)

        if isinstance(shap_values, list):
            shap_arr = shap_values[1]
        else:
            shap_arr = np.asarray(shap_values)
            if shap_arr.ndim == 3:
                shap_arr = shap_arr[:, :, 1]

        row = shap_arr[0]
        pairs = [
            (str(col), float(np.atleast_1d(v)[0]))
            for col, v in zip(features.columns, row)
        ]
        pairs.sort(key=lambda kv: -abs(kv[1]))
        return dict(pairs)

    @classmethod
    def _ensure_loaded(cls) -> None:
        if cls._pipeline is None or cls._ood_distribution is None:
            raise ModelNotReadyError(
                "ML model or OOD distribution not loaded — "
                "train the model first (python -m ml.train_v15) and restart the application"
            )

    @staticmethod
    def _label_from_score(score: float) -> str:
        """Legacy mapping kept for backward compatibility.
        New code paths use the calibrated threshold in score() instead."""
        if score >= 75: return "Strong Buy"
        if score >= 60: return "Buy"
        if score >= 45: return "Hold"
        if score >= 30: return "Avoid"
        return "Strong Avoid"

    @classmethod
    def get_version(cls) -> str:
        return MODEL_VERSION

    @classmethod
    def get_kpi_schema(cls) -> list:
        return list(ALL_FEATURES)


def _is_exceptional(kpis: Dict[str, Any], score: float) -> bool:
    """Flag as exceptional if all KPIs are top-tier AND ML score is strong."""
    if score < EXCEPTIONAL_THRESHOLDS["score"]:
        return False
    if kpis.get("revenue_growth", 0) < EXCEPTIONAL_THRESHOLDS["revenue_growth"]:
        return False
    if kpis.get("net_margin", 0) < EXCEPTIONAL_THRESHOLDS["net_margin"]:
        return False
    if kpis.get("debt_equity", 999) > EXCEPTIONAL_THRESHOLDS["debt_equity"]:
        return False
    if kpis.get("roe", 0) < EXCEPTIONAL_THRESHOLDS["roe"]:
        return False
    if kpis.get("current_ratio", 0) < EXCEPTIONAL_THRESHOLDS["current_ratio"]:
        return False
    return True