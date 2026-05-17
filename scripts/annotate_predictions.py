"""Annotate a clustbench run directory with theoretical predictions.

Reads ``runs/<name>/results.parquet`` (or .csv fallback), looks up each
row's :class:`AlgorithmCard`, and writes three new columns:

- ``theoretical_wall_time_s`` — predicted wall time, per-algorithm
  calibrated from the same benchmark's empirical wall times.
- ``theoretical_rss_mb`` — predicted memory delta, same calibration.
- ``theoretical_ari_upper_bound`` — predicted ARI ceiling from
  ``predict_ari_upper_bound`` against a task-derived fingerprint.

A row's "fingerprint" is *approximated* from its task identifiers
without re-generating the dataset — see ``_fingerprint_from_row`` below.
Coarser than a real data fingerprint but free at annotation time.

Usage::

    python scripts/annotate_predictions.py --run runs/full

Writes back into the same ``results.parquet`` / ``results.csv`` and
prints a quick predicted-vs-actual diagnostic.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any, Dict

import numpy as np
import pandas as pd

from clustbench.algorithm_cards import (
    ALGORITHM_CARDS,
    calibrate_from_benchmark,
    predict_ari_upper_bound,
    predict_performance,
)


# Coarse fingerprints derived from dataset IDs — used when we don't want
# to re-generate the data just to score the cards.
_DATASET_PRIORS = {
    # name             conv_cv   eff_dim_ratio   notes
    "mdcgen":             (0.30,  1.00),
    "blobs":              (0.30,  1.00),
    "mixed":              (0.45,  0.50),
    "anisotropic":        (0.50,  1.00),
    "moons":              (0.85,  0.50),
    "circles":            (0.90,  0.50),
    "spiral":             (0.85,  0.50),
    "swiss_roll":         (0.80,  0.50),
    "s_curve":            (0.70,  0.50),
    "rings":              (0.85,  0.50),
    "varying_density":    (0.60,  1.00),
    "imbalanced":         (0.55,  1.00),
    "mixed_shapes":       (0.75,  0.70),
    "extreme_outliers":   (0.55,  1.00),
    "inverse_pca":        (0.40,  0.10),  # genuine low-rank
    "iris":               (0.50,  1.00),
    "wine":               (0.60,  0.85),
    "breast_cancer":      (0.50,  0.50),
    "digits":             (0.65,  0.40),
    "olivetti_faces":     (0.75,  0.05),  # low-rank face manifold
    "glass":              (0.65,  0.80),
    "vehicle":            (0.60,  0.80),
    "segment":            (0.55,  0.50),
    "yeast":              (0.65,  1.00),
    "ecoli":              (0.60,  0.85),
    "text20news":         (0.60,  0.20),
}


def _fingerprint_from_row(row: Dict[str, Any]) -> Dict[str, float]:
    """Approximate the 7-feature fingerprint from a task row alone.

    Real fingerprints (from ``learned_router._fingerprint``) need the
    dataset itself. This function uses the dataset id + the row's
    parameters as a coarser-but-zero-cost stand-in.
    """
    ds = row.get("dataset_id")
    n = float(row.get("n_samples", 0) or 0)
    d = float(row.get("n_features", 0) or 1)
    k = float(row.get("k_target", 0) or 1)
    outliers = float(row.get("outliers", 0) or 0)
    noise = float(row.get("noise", 0) or 0)
    density = float(row.get("density", 1.0) or 1.0)

    conv_cv, eff_dim_ratio = _DATASET_PRIORS.get(ds, (0.50, 1.0))
    fp = {
        "log_n": float(np.log10(max(2.0, n))),
        "d": d,
        "k": k,
        "eff_dim": d * eff_dim_ratio,
        "conv_cv": conv_cv,
        "outlier_frac": (outliers / max(1.0, n + outliers)),
        "density_skew": min(1.0, 1.0 - density) + 0.1 * (noise > 0),
    }
    return fp


def annotate(run_dir: pathlib.Path) -> Dict[str, Any]:
    """Load ``run_dir`` results, fit per-algo calibrations, write back."""
    if (run_dir / "results.parquet").exists():
        try:
            df = pd.read_parquet(run_dir / "results.parquet")
        except Exception:
            df = pd.read_csv(run_dir / "results.csv")
    else:
        df = pd.read_csv(run_dir / "results.csv")

    print(f"loaded {len(df)} rows, {df.algo.nunique()} algorithms")

    # Calibrate from this run's empirical numbers.
    calib = calibrate_from_benchmark(df.to_dict("records"))
    print(f"calibrated {len(calib)} algorithms")

    rows: list[Dict[str, Any]] = []
    for _, row in df.iterrows():
        algo = row["algo"]
        card = ALGORITHM_CARDS.get(algo)
        if card is None:
            rows.append({
                "theoretical_wall_time_s": None,
                "theoretical_rss_mb": None,
                "theoretical_ari_upper_bound": None,
            })
            continue
        c = calib.get(algo, {"time": 1e-7, "mem": 8e-6})
        fp = _fingerprint_from_row(row.to_dict())
        pred = predict_performance(
            card,
            int(row.get("n_samples", 0) or 1),
            int(row.get("n_features", 0) or 1),
            int(row.get("k_target", 0) or 1),
            fingerprint=fp,
            time_calibration=c["time"],
            mem_calibration=c["mem"],
        )
        rows.append(pred)

    pred_df = pd.DataFrame(rows)
    for col in pred_df.columns:
        df[col] = pred_df[col].values

    # Save back. Parquet may fail on the extra dict column — fall back.
    df.to_csv(run_dir / "results.csv", index=False)
    try:
        df.to_parquet(run_dir / "results.parquet")
    except Exception as e:
        print(f"parquet write skipped: {e}")

    # Quick predicted-vs-actual diagnostic per algo.
    diag = {}
    if {"theoretical_wall_time_s", "wall_time_s"}.issubset(df.columns):
        for algo, sub in df.groupby("algo"):
            actual = sub["wall_time_s"].dropna()
            pred = sub["theoretical_wall_time_s"].dropna()
            if len(actual) == 0 or len(pred) == 0:
                continue
            m_actual = float(actual.median())
            m_pred = float(pred.median())
            diag[algo] = {
                "median_actual_wall_s": round(m_actual, 4),
                "median_predicted_wall_s": round(m_pred, 4),
                "ratio_actual_to_predicted": round(m_actual / max(1e-9, m_pred), 2),
            }
    print("\npredicted-vs-actual median wall time (top 10 by ratio):")
    sorted_diag = sorted(diag.items(), key=lambda kv: kv[1]["ratio_actual_to_predicted"])
    for algo, d in sorted_diag[:5] + sorted_diag[-5:]:
        print(f"  {algo:<22} actual={d['median_actual_wall_s']:.3f}s "
              f"pred={d['median_predicted_wall_s']:.3f}s "
              f"ratio={d['ratio_actual_to_predicted']:.2f}")

    # Also save a per-algo prediction summary.
    (run_dir / "predictions_summary.json").write_text(
        json.dumps({"calibration": calib, "diagnostic": diag}, indent=2, default=float)
    )
    return diag


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=pathlib.Path, required=True)
    args = ap.parse_args()
    annotate(args.run)


if __name__ == "__main__":
    main()
