"""kNN-based out-of-distribution scoring for incoming KPI vectors.

Single-layer distance score computed on the training dataset at training time
and serialized alongside the model as `train_kpi_distribution.joblib`.

Method:
    D_kNN(x) = distance to k-th nearest neighbor in standardized KPI space.

    Standardization uses the per-feature stddev of the training set.
    kNN distance is independent of any distribution-shape assumption.

Coverage score:
    coverage_score = 1 - percentile(D_kNN(x), training_knn_distances)
    range: [0, 1]. 1.0 means x is at the centroid of training density;
    ~0.0 means x is more extreme than any training sample.

OOD flag fires when:
    is_ood = coverage_score < 0.05   OR   knn_percentile > 0.95
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


KNN_K = 10
PCT_LOW = 0.05
PCT_HIGH = 0.95

# Exceptional company thresholds — flag when ALL KPIs are top-tier AND ML score is high
EXCEPTIONAL_THRESHOLDS = {
    "revenue_growth": 0.20,   # >20% YoY growth
    "net_margin": 0.20,        # >20% margin
    "debt_equity": 1.00,       # <1.0 debt/equity
    "roe": 0.20,               # >20% ROE
    "current_ratio": 1.50,     # >1.5 liquidity
    "score": 75.0,             # Strong Buy or better
}


def _knn_distance(tree: cKDTree, point: np.ndarray, k: int, is_in_tree: bool) -> float:
    """Return the true k-th neighbor distance for `point` against `tree`.

    `point` must be 1D (shape (5,)). If `is_in_tree`, the point is a member of
    the tree's data (self at index 0): we query k+1 and take the last element.
    If not in tree, we query k and take the last element.
    """
    point = np.asarray(point).reshape(-1)
    n_query = k + 1 if is_in_tree else k
    dists, _ = tree.query(point, k=n_query)
    return float(dists[-1])


def compute_distribution(X: pd.DataFrame) -> Dict[str, Any]:
    """Compute the distribution statistics from training data X.

    Returns a dict suitable for joblib.dump and consumption by `score()`.
    """
    if X.shape[0] < 2 * KNN_K:
        raise ValueError(
            f"Need at least {2 * KNN_K} training samples for stable kNN; "
            f"got {X.shape[0]}"
        )

    X_arr = X.to_numpy(dtype=float)

    # Standardize by training stddev (consistent with kNN).
    scale = X_arr.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    X_std = (X_arr - X_arr.mean(axis=0)) / scale

    # Build KD-tree on standardized training points.
    tree = cKDTree(X_std)

    # Per-training-sample distance to its k-th nearest neighbour (skipping self).
    knn_train = np.empty(X_std.shape[0], dtype=float)
    for i in range(X_std.shape[0]):
        knn_train[i] = _knn_distance(tree, X_std[i], KNN_K, is_in_tree=True)
    knn_sorted = np.sort(knn_train)

    return {
        "mean": X_arr.mean(axis=0),
        "scale": scale,
        "X_train_std": X_std,
        "knn_sorted": knn_sorted,
        "knn_k": KNN_K,
        "pct_low": PCT_LOW,
        "pct_high": PCT_HIGH,
        "n_train": X_arr.shape[0],
    }


def _percentile_of(value: float, sorted_arr: np.ndarray) -> float:
    """Percentile rank of value within sorted_arr ∈ [0, 1] (linear interp)."""
    n = len(sorted_arr)
    if n == 0:
        return 0.0
    idx = int(np.searchsorted(sorted_arr, value, side="left"))
    if idx == 0:
        return 0.0
    if idx >= n:
        return 1.0
    lo_val = float(sorted_arr[idx - 1])
    hi_val = float(sorted_arr[idx])
    if hi_val == lo_val:
        return float(idx - 1) / max(1, n - 1)
    frac = (value - lo_val) / (hi_val - lo_val)
    return (float(idx - 1) + frac) / max(1, n - 1)


def score(features: pd.DataFrame, distribution: Dict[str, Any]) -> Dict[str, Any]:
    """Score a single KPI feature row. Returns dict with coverage/knn/ood fields."""
    # Flatten to 1D regardless of input shape.
    x = features.to_numpy(dtype=float).reshape(-1)

    # Standardize.
    mean = distribution["mean"]
    scale = distribution["scale"]
    x_std = ((x - mean) / scale).reshape(-1)

    # kNN distance to training set (using shared logic, point NOT in tree).
    tree = cKDTree(distribution["X_train_std"])
    knn_distance = _knn_distance(tree, x_std, distribution["knn_k"], is_in_tree=False)

    # Percentile against training kNN distances.
    knn_pct = float(_percentile_of(knn_distance, distribution["knn_sorted"]))
    coverage_score = float(1.0 - knn_pct)

    # OOD decision.
    is_ood_knn = knn_pct > PCT_HIGH
    is_ood_cov = coverage_score < PCT_LOW
    is_ood = bool(is_ood_knn or is_ood_cov)

    if is_ood_knn and is_ood_cov:
        reason = (
            f"Both layers flag OOD: coverage_score={coverage_score:.3f} < "
            f"{PCT_LOW} and knn_pct={knn_pct:.3f} > {PCT_HIGH}"
        )
    elif is_ood_knn:
        reason = f"kNN OOD: knn_pct={knn_pct:.3f} (> {PCT_HIGH})"
    elif is_ood_cov:
        reason = f"Coverage OOD: coverage_score={coverage_score:.3f} (< {PCT_LOW})"
    else:
        reason = None

    return {
        "coverage_score": coverage_score,
        "knn_percentile": knn_pct,
        "knn_distance": knn_distance,
        "is_ood": is_ood,
        "ood_reason": reason,
    }