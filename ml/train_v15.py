"""v1.5 training orchestrator.

Two-layer training:
  Layer 1 (6m): All samples with 6m data
  Layer 2 (12m): All samples with 12m data (subset of 6m)

Run:
    python -m ml.train_v15
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.config import (KPI_SCHEMA, VALUATION_SCHEMA, ALL_FEATURES, settings,
                       MOMENTUM_SCHEMA, VOLATILITY_SCHEMA, BETA_SCHEMA, SECTOR_SCHEMA)
from ml.generate_dataset import load_config
from ml.ood import compute_distribution
from ml.ml import save_model, _save_shap_summary
from ml.labeler import generate_labels
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.calibration import CalibratedClassifierCV


ANCHORS_PATH = Path(__file__).resolve().parent / "data" / "real_anchors.json"
MARKET_PATH = Path(__file__).resolve().parent / "data" / "market_data.json"
DISTRIBUTION_PATH = Path(settings.model_save_path).parent / "train_kpi_distribution.joblib"


def _load_market_data() -> dict:
    if not MARKET_PATH.exists():
        print(f"[train_v15] ! market data cache not found at {MARKET_PATH}")
        print("             Run `python -m ml.fetch_market_data` first.")
        return {}
    return json.loads(MARKET_PATH.read_text(encoding="utf-8"))


def _extract_market_features(market_info: dict) -> dict:
    """Extract all market features: valuation + momentum + volatility + beta + sector."""
    if not market_info or "error" in market_info:
        return {k: np.nan for k in VALUATION_SCHEMA + MOMENTUM_SCHEMA + VOLATILITY_SCHEMA + BETA_SCHEMA + SECTOR_SCHEMA}

    def _log1p(x):
        return np.log1p(x) if x is not None and x > 0 else np.nan

    def _log10(x):
        return np.log10(x) if x is not None and x > 0 else np.nan

    # Valuation features
    pe = market_info.get("trailing_pe")
    fpe = market_info.get("forward_pe")
    ps = market_info.get("price_to_sales")
    peg = market_info.get("peg")
    pb = market_info.get("price_to_book")
    mcap = market_info.get("market_cap")
    div = market_info.get("dividend_yield")

    val_features = {
        "trailing_pe_log": _log1p(pe),
        "forward_pe_log": _log1p(fpe),
        "price_to_sales_log": _log1p(ps),
        "peg_clipped": np.clip(peg, -5, 5) if peg is not None else np.nan,
        "price_to_book_log": _log1p(pb),
        "market_cap_log": _log10(mcap),
        "dividend_yield": div if div is not None else 0.0,
    }

    # Momentum features (raw returns, no transformation)
    mom_features = {
        "mom_3m": market_info.get("mom_3m"),
        "mom_6m": market_info.get("mom_6m"),
        "mom_12m": market_info.get("mom_12m"),
    }

    # Volatility features
    vol_features = {
        "vol_6m": market_info.get("vol_6m"),
        "vol_12m": market_info.get("vol_12m"),
    }

    # Beta feature
    beta_features = {
        "beta_12m": market_info.get("beta_12m"),
    }

    # Sector-relative features
    sector_features = {
        "sector_rel_6m": market_info.get("sector_rel_6m"),
        "sector_rel_12m": market_info.get("sector_rel_12m"),
    }

    return {**val_features, **mom_features, **vol_features, **beta_features, **sector_features}


def _impute_valuation_features(X: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values in ALL columns with column medians."""
    X = X.copy()
    for col in X.columns:
        if X[col].isna().any():
            median_val = X[col].median()
            if np.isnan(median_val):
                median_val = 0.0
            X[col] = X[col].fillna(median_val)
    return X


def _load_dual_layer_anchors(cfg: dict, weight: int) -> dict:
    if not ANCHORS_PATH.exists():
        print(f"[train_v15] ! anchor cache not found at {ANCHORS_PATH}")
        print("             Run `python -m ml.fetch_anchors --fetch` first.")
        print("             Continuing with synthetic-only training.")
        return {"X_6m": pd.DataFrame(), "y_6m": pd.Series([], dtype=int, name="label"),
                "groups_6m": pd.Series([], dtype=str, name="ticker"),
                "X_12m": pd.DataFrame(), "y_12m": pd.Series([], dtype=int, name="label"),
                "groups_12m": pd.Series([], dtype=str, name="ticker")}

    anchors_data = json.loads(ANCHORS_PATH.read_text(encoding="utf-8"))
    market_data = json.loads(MARKET_PATH.read_text(encoding="utf-8"))

    valid_crs = [
        info.get("kpis", {}).get("current_ratio")
        for info in anchors_data.values()
        if (info.get("kpis") or {}).get("current_ratio") not in (None, 0.0)
    ]
    cr_median = float(np.median(valid_crs)) if valid_crs else 1.2

    # Collect raw data first
    rows_X, excess_6m_list, excess_12m_list, tickers_list, filing_dates = [], [], [], [], []
    skipped = []

    for ticker, info in anchors_data.items():
        kpis = info.get("kpis") or {}
        if kpis.get("current_ratio") == 0.0:
            kpis = dict(kpis)
            kpis["current_ratio"] = cr_median
        if any(kpis.get(k) is None for k in KPI_SCHEMA):
            skipped.append(f"{ticker}: partial KPIs")
            continue

        market_info = market_data.get(ticker, {})
        val_features = _extract_market_features(market_info)
        row = [kpis[k] for k in KPI_SCHEMA] + [val_features.get(k, np.nan) for k in VALUATION_SCHEMA + MOMENTUM_SCHEMA + VOLATILITY_SCHEMA + BETA_SCHEMA + SECTOR_SCHEMA]

        excess_6m = market_info.get("excess_6m")
        excess_12m = market_info.get("excess_12m")
        filing_date = info.get("filing_date")

        rows_X.append(row)
        excess_6m_list.append(excess_6m)
        excess_12m_list.append(excess_12m)
        tickers_list.append(ticker)
        filing_dates.append(filing_date)

    if skipped:
        print(f"[train_v15] ! skipping anchors: {skipped}")

    # Compute labels using labeler module
    label_method = cfg.get("training", {}).get("label_method", "fixed")
    quantile_low = cfg.get("training", {}).get("quantile_low", 0.30)
    quantile_high = cfg.get("training", {}).get("quantile_high", 0.70)
    fixed_buy = cfg.get("training", {}).get("fixed_buy_threshold", 0.05)
    fixed_avoid = cfg.get("training", {}).get("fixed_avoid_threshold", -0.05)

    # 6m labels
    labels_6m = []
    for i, ex in enumerate(excess_6m_list):
        if ex is None:
            labels_6m.append(None)
        else:
            labels_6m.append(ex)

    # 12m labels
    labels_12m = []
    for i, ex in enumerate(excess_12m_list):
        if ex is None:
            labels_12m.append(None)
        else:
            labels_12m.append(ex)

    # Convert to Series for labeler
    excess_6m_series = pd.Series(labels_6m, dtype=float)
    excess_12m_series = pd.Series(labels_12m, dtype=float)
    tickers_series = pd.Series(tickers_list, dtype=str)
    filing_dates_series = pd.Series(filing_dates, dtype=str)

    # Generate labels for 6m
    label_result_6m = generate_labels(
        excess_returns=excess_6m_series,
        filing_dates=filing_dates_series,
        tickers=tickers_series,
        method=label_method,
        low_quantile=quantile_low,
        high_quantile=quantile_high,
        buy_threshold=fixed_buy,
        avoid_threshold=fixed_avoid,
    )
    labels_6m_final = label_result_6m.labels

    # Generate labels for 12m
    label_result_12m = generate_labels(
        excess_returns=excess_12m_series,
        filing_dates=filing_dates_series,
        tickers=tickers_series,
        method=label_method,
        low_quantile=quantile_low,
        high_quantile=quantile_high,
        buy_threshold=fixed_buy,
        avoid_threshold=fixed_avoid,
    )
    labels_12m_final = label_result_12m.labels

    print(f"[train_v15] Label method: {label_method}")
    print(f"[train_v15] 6m labels: buy={(labels_6m_final==1).sum()}, avoid={(labels_6m_final==0).sum()}, hold={(labels_6m_final==-1).sum()}")
    print(f"[train_v15] 12m labels: buy={(labels_12m_final==1).sum()}, avoid={(labels_12m_final==0).sum()}, hold={(labels_12m_final==-1).sum()}")

    # Build final datasets (only keep buy/avoid, drop hold)
    rows_6m_X, rows_6m_y, groups_6m = [], [], []
    rows_12m_X, rows_12m_y, groups_12m = [], [], []

    for i, row in enumerate(rows_X):
        # 6m
        if labels_6m_final.iloc[i] in (0, 1):
            rows_6m_X.append(row)
            rows_6m_y.append(int(labels_6m_final.iloc[i]))
            groups_6m.append(tickers_list[i])
        # 12m
        if labels_12m_final.iloc[i] in (0, 1):
            rows_12m_X.append(row)
            rows_12m_y.append(int(labels_12m_final.iloc[i]))
            groups_12m.append(tickers_list[i])

    X_6m = pd.DataFrame(rows_6m_X, columns=ALL_FEATURES) if rows_6m_X else pd.DataFrame()
    y_6m = pd.Series(rows_6m_y, name="label", dtype=int) if rows_6m_y else pd.Series([], dtype=int)
    groups_6m = pd.Series(groups_6m, name="ticker", dtype=str) if groups_6m else pd.Series([], dtype=str)

    X_12m = pd.DataFrame(rows_12m_X, columns=ALL_FEATURES) if rows_12m_X else pd.DataFrame()
    y_12m = pd.Series(rows_12m_y, name="label", dtype=int) if rows_12m_y else pd.Series([], dtype=int)
    groups_12m = pd.Series(groups_12m, name="ticker", dtype=str) if groups_12m else pd.Series([], dtype=str)

    # Replicate by weight FIRST (using raw data with NaNs)
    def _replicate(X, y, groups, weight):
        if len(X) == 0:
            return X, y, groups
        X_w = pd.concat([X] * weight, ignore_index=True)
        y_w = pd.concat([y] * weight, ignore_index=True)
        g_w = pd.Series([t for t in groups for _ in range(weight)], name="ticker")
        return X_w, y_w, g_w

    X_6m = pd.DataFrame(rows_6m_X, columns=ALL_FEATURES) if rows_6m_X else pd.DataFrame()
    y_6m = pd.Series(rows_6m_y, name="label", dtype=int) if rows_6m_y else pd.Series([], dtype=int)
    groups_6m = pd.Series(groups_6m, name="ticker", dtype=str) if len(groups_6m) > 0 else pd.Series([], dtype=str)

    X_12m = pd.DataFrame(rows_12m_X, columns=ALL_FEATURES) if rows_12m_X else pd.DataFrame()
    y_12m = pd.Series(rows_12m_y, name="label", dtype=int) if rows_12m_y else pd.Series([], dtype=int)
    groups_12m = pd.Series(groups_12m, name="ticker", dtype=str) if len(groups_12m) > 0 else pd.Series([], dtype=str)

    # Replicate by weight
    X_6m_w, y_6m_w, g_6m_w = _replicate(X_6m, y_6m, groups_6m, weight)
    X_12m_w, y_12m_w, g_12m_w = _replicate(X_12m, y_12m, groups_12m, weight)

    # NOW impute missing values on the replicated data
    X_6m_w = _impute_valuation_features(X_6m_w) if len(X_6m_w) else X_6m_w
    X_12m_w = _impute_valuation_features(X_12m_w) if len(X_12m_w) else X_12m_w

    print(f"[train_v15] Layer 1 (6m): {len(rows_6m_X)} unique, weighted x{weight} = {len(X_6m_w)} rows, pos={y_6m_w.mean():.2%}")
    print(f"[train_v15] Layer 2 (12m): {len(rows_12m_X)} unique, weighted x{weight} = {len(X_12m_w)} rows, pos={np.mean(y_12m_w):.2%}")

    return {
        "X_6m": X_6m_w,
        "y_6m": y_6m_w,
        "groups_6m": g_6m_w,
        "X_12m": X_12m_w,
        "y_12m": y_12m_w,
        "groups_12m": g_12m_w,
    }


def main() -> int:
    print("=" * 70)
    print("v1.5 -- Two-Layer Training: 6m (all) + 12m (override)")
    print("=" * 70)

    cfg = load_config()
    print(f"[1/5] Loading dual-layer anchors (weight {cfg['training']['anchor_weight']}x)...")
    data = _load_dual_layer_anchors(cfg, int(cfg["training"]["anchor_weight"]))

    if len(data["X_6m"]) == 0:
        print("      No 6m data found. Exiting.")
        return 1

    # ---- LAYER 1: 6m Model (all samples with 6m data) ----
    print(f"[2/5] Training Layer 1 (6m) on {len(data['X_6m'])} samples...")
    X_6m = pd.DataFrame(data["X_6m"], columns=ALL_FEATURES)
    y_6m = pd.Series(data["y_6m"], name="label")
    groups_6m = data["groups_6m"]

    # ---- LAYER 2: 12m Model (subset with 12m data) ----
    print(f"[2/5] Training Layer 2 (12m) on {len(data['X_12m'])} samples...")
    X_12m = pd.DataFrame(data["X_12m"], columns=ALL_FEATURES)
    y_12m = pd.Series(data["y_12m"], name="label")
    groups_12m = data["groups_12m"]

    # Deployment model = Layer 1 (6m) for general use
    X_all = pd.DataFrame(data["X_6m"], columns=ALL_FEATURES)
    y_all = pd.Series(data["y_6m"], name="label")
    groups_all = data["groups_6m"]

    print("[3/5] Ticker-grouped evaluation + deployment training...")

    def _make_pipeline():
        return SkPipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=500,
                max_depth=6,
                min_samples_leaf=8,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )),
        ])

    # ---- LAYER 1: 6m Model Evaluation ----
    print("\n      Layer 1 (6m) GroupKFold (5 folds, grouped by ticker):")
    N_FOLDS = 5
    gkf = GroupKFold(n_splits=5)

    X_6m_df = pd.DataFrame(data["X_6m"], columns=ALL_FEATURES)
    y_6m_s = pd.Series(data["y_6m"], name="label")
    groups_6m_s = pd.Series(data["groups_6m"], name="ticker")

    print(f"\n      Layer 1 (6m) GroupKFold (5 folds, grouped by ticker):")
    for fold, (tr_idx, te_idx) in enumerate(gkf.split(data["X_6m"], data["y_6m"], groups=data["groups_6m"]), 1):
        Xf_tr = pd.DataFrame(data["X_6m"].iloc[tr_idx], columns=ALL_FEATURES)
        yf_tr = pd.Series(data["y_6m"].iloc[tr_idx], name="label")
        Xf_te = pd.DataFrame(data["X_6m"].iloc[te_idx], columns=ALL_FEATURES)
        yf_te = pd.Series(data["y_6m"].iloc[te_idx], name="label")

        pf = _make_pipeline()
        pf.fit(Xf_tr, yf_tr)
        proba = pf.predict_proba(Xf_te)[:, 1]

        pred = (proba >= 0.5).astype(int)
        yt = yf_te.values
        tp = int(((pred == 1) & (yt == 1)).sum())
        fp = int(((pred == 1) & (yt == 0)).sum())
        fn = int(((pred == 0) & (yt == 1)).sum())
        tn = int(((pred == 0) & (yt == 0)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        acc = float((pred == yt).mean())
        n_tickers = len(set(data["groups_6m"].iloc[te_idx]))
        print(f"        fold {fold}: tickers={n_tickers:>2}  acc={acc:.3f}  F1={f1:.3f}  "
              f"cm=[[{tn},{fp}],[{fn},{tp}]]")

    # Layer 2 CV
    print(f"\n      Layer 2 (12m) GroupKFold (5 folds, grouped by ticker):")
    for fold, (tr_idx, te_idx) in enumerate(gkf.split(data["X_12m"], data["y_12m"], groups=data["groups_12m"]), 1):
        Xf_tr = pd.DataFrame(data["X_12m"].iloc[tr_idx], columns=ALL_FEATURES)
        yf_tr = pd.Series(data["y_12m"].iloc[tr_idx], name="label")
        Xf_te = pd.DataFrame(data["X_12m"].iloc[te_idx], columns=ALL_FEATURES)
        yf_te = pd.Series(data["y_12m"].iloc[te_idx], name="label")

        pf = _make_pipeline()
        pf.fit(Xf_tr, yf_tr)
        proba = pf.predict_proba(Xf_te)[:, 1]

        pred = (proba >= 0.5).astype(int)
        yt = yf_te.values
        tp = int(((pred == 1) & (yt == 1)).sum())
        fp = int(((pred == 1) & (yt == 0)).sum())
        fn = int(((pred == 0) & (yt == 1)).sum())
        tn = int(((pred == 0) & (yt == 0)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        acc = float((pred == yt).mean())
        n_tickers = len(set(data["groups_12m"].iloc[te_idx]))
        print(f"        fold {fold}: tickers={n_tickers:>2}  acc={acc:.3f}  F1={f1:.3f}  "
              f"cm=[[{tn},{fp}],[{fn},{tp}]]")

    # Train final models
    print("\n      Training final Layer 1 (6m) model...")
    pipeline_6m = _make_pipeline()
    X_6m_df = pd.DataFrame(data["X_6m"], columns=ALL_FEATURES)
    y_6m_s = pd.Series(data["y_6m"], name="label")
    pipeline_6m.fit(X_6m_df, y_6m_s)
    # Ensure feature names are stored for validation
    pipeline_6m.named_steps['clf'].feature_names_in_ = np.array(ALL_FEATURES)

    print("      Training Layer 2 (12m) model...")
    pipeline_12m = _make_pipeline()
    pipeline_12m.fit(pd.DataFrame(data["X_12m"], columns=ALL_FEATURES), pd.Series(data["y_12m"], name="label"))
    pipeline_12m.named_steps['clf'].feature_names_in_ = np.array(ALL_FEATURES)

    # Deployment model = Layer 1 (6m) for general use
    pipeline = _make_pipeline()
    pipeline.fit(pd.DataFrame(data["X_6m"], columns=ALL_FEATURES), pd.Series(data["y_6m"], name="label"))
    pipeline.named_steps['clf'].feature_names_in_ = np.array(ALL_FEATURES)

    # Platt calibration and threshold tuning on 6m model
    print("\n      Platt calibration & threshold tuning (Layer 1)...")
    calibrated = CalibratedClassifierCV(pipeline, method='sigmoid', cv=5, ensemble=False)
    X_6m_df = pd.DataFrame(data["X_6m"], columns=ALL_FEATURES)
    y_6m_s = pd.Series(data["y_6m"], name="label")
    calibrated.fit(X_6m_df, y_6m_s)

    # Find optimal threshold on calibrated probabilities
    proba_cal = calibrated.predict_proba(X_6m_df)[:, 1]
    thresholds = np.linspace(0.1, 0.9, 81)
    best_f1 = 0
    best_thresh = 0.5
    for thresh in thresholds:
        pred = (proba_cal >= thresh).astype(int)
        tp = int(((pred == 1) & (y_6m_s == 1)).sum())
        fp = int(((pred == 1) & (y_6m_s == 0)).sum())
        fn = int(((pred == 0) & (y_6m_s == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    print(f"      Optimal threshold: {best_thresh:.3f} (F1={best_f1:.3f})")

    # Save calibration params
    import json as _json
    calibration_data = {
        "platt_A": float(calibrated.calibrated_classifiers_[0].calibrators[0].a_),
        "platt_B": float(calibrated.calibrated_classifiers_[0].calibrators[0].b_),
        "optimal_threshold": float(best_thresh),
        "best_f1": float(best_f1),
        "classes": calibrated.classes_.tolist()
    }
    with open(settings.calibration_path, 'w') as f:
        _json.dump(calibration_data, f, indent=2)
    print(f"      Calibration saved -> {settings.calibration_path}")

    # Save both models
    save_model(pipeline, settings.model_save_path)  # 6m model (primary)
    save_model(pipeline_12m, Path(settings.model_save_path).parent / "scorer_12m.joblib")  # 12m model

    # OOD distribution on all 6m data
    print("\n[4/5] Computing kNN OOD distribution...")
    distribution = compute_distribution(pd.DataFrame(data["X_6m"], columns=ALL_FEATURES))
    joblib.dump(distribution, DISTRIBUTION_PATH)
    print(f"      OOD distribution -> {DISTRIBUTION_PATH}")

    print("\n[5/5] Saving models...")
    print(f"      Layer 1 (6m) -> {settings.model_save_path}")
    print(f"      Layer 2 (12m) -> {Path(settings.model_save_path).parent / 'scorer_12m.joblib'}")

    # Sanity check
    print()
    print("=" * 70)
    print("SANITY: Scoring NVIDIA...")
    print("=" * 70)
    from ml.ood import score as ood_score
    nvda_kpis = [0.6547, 0.556, 0.0538, 0.7633, 3.9049]
    nvda_val = [np.log1p(20), np.log1p(25), np.log1p(13), 0.2, np.log1p(15), np.log10(2.5e12), 0.01]
    nvda_mom = [0.15, 0.25, 0.45]  # mom_3m, mom_6m, mom_12m
    nvda_vol = [0.35, 0.30]  # vol_6m, vol_12m
    nvda_beta = [1.2]  # beta_12m
    nvda_sector = [0.05, 0.03]  # sector_rel_6m, sector_rel_12m
    nvda_all = nvda_kpis + nvda_val + nvda_mom + nvda_vol + nvda_beta + nvda_sector
    nvda_row = pd.DataFrame([dict(zip(ALL_FEATURES, nvda_all))])
    s = ood_score(nvda_row, compute_distribution(pd.DataFrame(data["X_6m"], columns=ALL_FEATURES)))
    print(f"  is_ood: {s['is_ood']}, coverage={s['coverage_score']:.4f}, pct={s['knn_percentile']:.4f}")

    print()
    print("=" * 70)
    print("v1.5 TRAINING COMPLETE - Two-Layer Model")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())