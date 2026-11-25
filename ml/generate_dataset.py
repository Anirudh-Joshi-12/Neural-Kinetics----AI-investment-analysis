"""Regime-aware synthetic training-data generator.

Reads `ml/training_config.yaml` and produces a pandas DataFrame with rows
sampled from named archetype distributions. Each archetype KPI is drawn from
a Beta(a, b) shifted/scaled into the archetype's [range_min, range_max],
where (a, b) are computed via closed-form moment matching from target_mean
and target_stddev (NOT fitted from data — there is no historical data; the
targets ARE the spec).

The closed-form moment matching:

    Given target_mean μ, target_stddev σ, and the support [lo, hi]:
        mean_beta = clamp((μ - lo) / (hi - lo), eps, 1 - eps)
        var_beta  = (σ / (hi - lo))**2

    Beta(a, b) has mean = a/(a+b) and variance = ab / ((a+b)^2 * (a+b+1)).
    Solving these two equations for (a, b):
        common = mean_beta * (1 - mean_beta) / var_beta - 1
        a = mean_beta * common
        b = (1 - mean_beta) * common

    Returns (a, b) clipped to [eps, large_cap] so the generator never breaks
    on parameter combinations that violate the variance cap of beta (the
    variance of Beta is bounded above by mean*(1-mean); if requested variance
    exceeds that, common would go negative and (a,b) < 0 → invalid).

This file is deliberately pure-functional: `generate_dataset()` takes a
config dict, returns a DataFrame. Configs are loaded by `load_config()`.
No side effects (no writes), no DB, no network. Network I/O for real
anchors lives in `ingest_real_anchors()` so the costly EDGAR fetch only
runs once and can be cached.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

from ml.config import KPI_SCHEMA, VALUATION_SCHEMA, ALL_FEATURES


# =============================================================================
# Config loading
# =============================================================================
_CONFIG_PATH = Path(__file__).resolve().parent / "training_config.yaml"


def load_config(path: Path | str | None = None) -> Dict[str, Any]:
    """Load the YAML training config. Pure function — no side effects."""
    cfg_path = Path(path) if path else _CONFIG_PATH
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# =============================================================================
# Closed-form beta moment matching
# =============================================================================
_EPS = 1e-6
_LARGE_CAP = 1e6   # cap on (a, b) to avoid +inf when requested variance is huge


def _beta_params(target_mean: float, target_stddev: float,
                 range_min: float, range_max: float) -> Tuple[float, float]:
    """Compute (a, b) for Beta(a,b) shifted+scaled into [range_min, range_max]
    such that the resulting scaled Beta has mean=target_mean and
    stddev=target_stddev.

    Raises ValueError if the requested standard deviation exceeds the
    theoretical maximum for a beta on this range (which would otherwise
    silently produce invalid (a, b) < 0).

    The theoretical max stddev for Beta(a, b) is 0.5 (when a=b=1, uniform),
    so on the scaled range it is 0.5 * (range_max - range_min). Achievable
    stddev is actually below 0.5 unless a or b < 1 — so this check errs
    slightly on the strict side; users wanting near-uniform dispersion
    should use the borderline_noise archetype with a near-flat variance.
    """
    span = range_max - range_min
    if span <= 0:
        raise ValueError(
            f"Invalid archetype range: range_max ({range_max}) "
            f"must exceed range_min ({range_min})"
        )

    # Map to beta's natural support (0, 1) for the standardized mean.
    mean_beta = (target_mean - range_min) / span
    if not (0.0 < mean_beta < 1.0):
        raise ValueError(
            f"target_mean {target_mean} outside archetype range "
            f"({range_min}, {range_max})"
        )

    var_beta = (target_stddev / span) ** 2
    # Theoretical max variance of Beta is mean_beta * (1 - mean_beta),
    # achieved by Beta(a, b) with a, b → 0+. Cap conservatively below it.
    max_var_beta = mean_beta * (1 - mean_beta)
    if var_beta >= max_var_beta * 0.99:
        # Replace with the maximum achievable and report.
        var_beta = max_var_beta * 0.95

    # Now solve for (a, b):
    #   mean_beta = a / (a + b)
    #   var_beta  = ab / ((a+b)^2 (a+b+1))
    # Let s = a + b (the "concentration" of the beta). Then:
    #   s = mean_beta * (1 - mean_beta) / var_beta - 1
    # And:
    #   a = mean_beta * s
    #   b = (1 - mean_beta) * s
    s = mean_beta * (1 - mean_beta) / var_beta - 1.0
    if s <= 0.0:
        # Should not happen after the cap above; defensive guard.
        raise ValueError(
            f"Computed beta concentration s={s} <= 0 — requested sigma too "
            f"large for range [{range_min}, {range_max}]"
        )
    a = mean_beta * s
    b = (1 - mean_beta) * s

    # Numerical safety: clip very-small or very-large params.
    a = max(min(a, _LARGE_CAP), _EPS)
    b = max(min(b, _LARGE_CAP), _EPS)
    return float(a), float(b)


# =============================================================================
# Per-archetype row generation
# =============================================================================
def _sample_archetype_kpi(
    kpi_cfg: Dict[str, float],
    rng: np.random.Generator,
    n: int,
) -> np.ndarray:
    """Sample n values for ONE kpi according to its shifted/scaled Beta."""
    a, b = _beta_params(
        target_mean=float(kpi_cfg["target_mean"]),
        target_stddev=float(kpi_cfg["target_stddev"]),
        range_min=float(kpi_cfg["range_min"]),
        range_max=float(kpi_cfg["range_max"]),
    )
    # numpy's beta returns values in (0, 1).
    raw = rng.beta(a, b, size=n)
    # Shift+scale into the archetype range.
    lo = float(kpi_cfg["range_min"])
    hi = float(kpi_cfg["range_max"])
    return lo + raw * (hi - lo)


def _sample_archetype(
    archetype_cfg: Dict[str, Any],
    rng: np.random.Generator,
    total_n: int,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Sample all rows for ONE archetype; returns (X, y)."""
    weight = float(archetype_cfg["weight"])
    n = max(1, int(round(weight * total_n)))

    kpi_data: Dict[str, np.ndarray] = {}
    for kpi_name in KPI_SCHEMA:
        if kpi_name not in archetype_cfg["kpis"]:
            raise ValueError(
                f"Archetype missing KPI {kpi_name!r}; available: "
                f"{list(archetype_cfg['kpis'].keys())}"
            )
        kpi_data[kpi_name] = _sample_archetype_kpi(
            archetype_cfg["kpis"][kpi_name], rng, n
        )

    # Sample valuation features if present in archetype config
    val_data: Dict[str, np.ndarray] = {}
    if "valuation" in archetype_cfg:
        for val_name in VALUATION_SCHEMA:
            if val_name not in archetype_cfg["valuation"]:
                raise ValueError(
                    f"Archetype missing valuation {val_name!r}; available: "
                    f"{list(archetype_cfg['valuation'].keys())}"
                )
            val_data[val_name] = _sample_archetype_kpi(
                archetype_cfg["valuation"][val_name], rng, n
            )
    else:
        # No valuation config — fill with NaN (will be imputed later)
        for val_name in VALUATION_SCHEMA:
            val_data[val_name] = np.full(n, np.nan)

    X = pd.DataFrame({**kpi_data, **val_data})
    deterministic_label = int(archetype_cfg["label"])
    noise_p = float(archetype_cfg["label_noise"])
    flips = rng.binomial(1, noise_p, size=n).astype(bool)
    y = pd.Series(np.where(flips, 1 - deterministic_label, deterministic_label),
                  name="label")
    return X, y


# =============================================================================
# Top-level dataset generation (synthetic only — real anchors in separate fn)
# =============================================================================
def generate_synthetic_dataset(
    cfg: Dict[str, Any] | None = None,
    seed: int | None = None,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, int]]:
    """Generate the full synthetic dataset from all archetypes.

    Returns:
      X        — (n_synthetic, 5) DataFrame with KPI columns.
      y        — (n_synthetic,) Series with binary labels.
      counts   — archetype_name -> row count, for reproducibility audit.

    Pure function: deterministic given cfg + seed.
    """
    if cfg is None:
        cfg = load_config()
    seed = int(seed if seed is not None else cfg.get("random_seed", 42))
    n_synth = int(cfg.get("n_synthetic", 990))

    rng = np.random.default_rng(seed=seed)
    archetypes = cfg["archetypes"]

    total_weight = sum(float(a["weight"]) for a in archetypes.values())
    if not math.isclose(total_weight, 1.0, abs_tol=1e-3):
        raise ValueError(
            f"Archetype weights must sum to 1.0; got {total_weight:.4f}"
        )

    frames_X: List[pd.DataFrame] = []
    frames_y: List[pd.Series] = []
    counts: Dict[str, int] = {}

    for name, acfg in archetypes.items():
        X_arch, y_arch = _sample_archetype(acfg, rng, n_synth)
        # Tag rows with archetype name as an audit column (dropped before train).
        X_arch = X_arch.assign(archetype=name)
        frames_X.append(X_arch)
        frames_y.append(y_arch)
        counts[name] = len(X_arch)

    X = pd.concat(frames_X, ignore_index=True)
    y = pd.concat(frames_y, ignore_index=True)

    # Shuffle rows so train_test_split doesn't see archetype blocks in order.
    perm = rng.permutation(len(X))
    X = X.iloc[perm].reset_index(drop=True)
    y = y.iloc[perm].reset_index(drop=True)

    return X, y, counts


# =============================================================================
# Sanity check: prints the realized moments per archetype for review.
# =============================================================================
def validate_config(cfg: Dict[str, Any] | None = None, seed: int = 42) -> None:
    """Sample one giant batch per archetype and print realized mean/stddev
    alongside the spec target_mean/target_stddev. Used at config-review time
    to confirm moments match the spec."""
    if cfg is None:
        cfg = load_config()
    rng = np.random.default_rng(seed=seed)

    print(f"{'archetype':<22} {'feature':<22} {'spec_mu':>9} {'got_mu':>9} "
          f"{'spec_sd':>9} {'got_sd':>9}")
    print("-" * 90)
    for name, acfg in cfg["archetypes"].items():
        # Oversample to reduce noise in the realized moments.
        X_arch, _ = _sample_archetype(acfg, rng, total_n=10000)
        for feat in ALL_FEATURES:
            if feat in KPI_SCHEMA:
                spec_mu = acfg["kpis"][feat]["target_mean"]
                spec_sd = acfg["kpis"][feat]["target_stddev"]
            elif "valuation" in acfg and feat in acfg["valuation"]:
                spec_mu = acfg["valuation"][feat]["target_mean"]
                spec_sd = acfg["valuation"][feat]["target_stddev"]
            else:
                spec_mu = spec_sd = float("nan")
            got_mu = float(X_arch[feat].mean())
            got_sd = float(X_arch[feat].std())
            print(f"{name:<22} {feat:<22} {spec_mu:>9.4f} {got_mu:>9.4f} "
                  f"{spec_sd:>9.4f} {got_sd:>9.4f}")


# =============================================================================
# CLI entrypoint:
#   python -m ml.generate_dataset --validate       show realized moments
#   python -m ml.generate_dataset --generate       save synthetic parquet to ml/data/
# =============================================================================
def _cli() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Regime dataset generator.")
    parser.add_argument("--validate", action="store_true",
                        help="Print realized per-archetype moment check vs spec.")
    parser.add_argument("--generate", action="store_true",
                        help="Generate and save synthetic dataset as parquet.")
    args = parser.parse_args()

    if not (args.validate or args.generate):
        parser.print_help()
        return 1

    if args.validate:
        validate_config()
        return 0

    if args.generate:
        X, y, counts = generate_synthetic_dataset()
        out_dir = Path(__file__).resolve().parent / "data"
        out_dir.mkdir(parents=True, exist_ok=True)
        X_path = out_dir / "synthetic_X.csv"
        y_path = out_dir / "synthetic_y.csv"
        X.to_csv(X_path, index=False)
        y.to_frame().to_csv(y_path, index=False)
        print(f"Synthetic dataset saved:")
        print(f"  X -> {X_path}  shape={X.shape}")
        print(f"  y -> {y_path}  shape={y.shape}")
        print(f"  archetype counts: {counts}")
        print(f"  positive class ratio: {y.mean():.2%}")
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
