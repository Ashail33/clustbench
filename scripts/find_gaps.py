"""Find gap regimes in algorithm-space and propose synthetic datasets.

A *gap fingerprint* is a point in the 7-feature task fingerprint space
where the best :class:`clustbench.algorithm_cards.AlgorithmCard` scores
below ``--threshold`` under :func:`predict_ari_upper_bound`. Those are
the hardest tasks in algorithm-space; benchmarking new algorithms on
them is maximally discriminative.

This script:

1. Sweeps a fingerprint grid (capped at ``--max-grid`` points).
2. Scores each fingerprint against every algorithm card.
3. Keeps gaps (max score < threshold), ranked by (1 - max_score) *
   exp(-distance_to_nearest_benchmark_task).
4. For each top gap, finds the closest existing benchmark task and
   proposes a :class:`DataSpec` override that would land on the gap.
5. Writes ``runs/gaps_report.json`` and prints a smoke-test summary.

Usage::

    python scripts/find_gaps.py
    python scripts/find_gaps.py --threshold 0.7 --max-grid 1000 --top 30
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, Dict, List

# Make sure the in-repo package is importable when run as a script.
_HERE = pathlib.Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from clustbench.algorithm_cards import ALGORITHM_CARDS  # noqa: E402
from clustbench.gap_finder import (  # noqa: E402
    build_fingerprint_grid,
    gap_to_dict,
    rank_gaps,
    unique_existing_fingerprints,
)

# Import _DATASET_PRIORS from the annotation script — single source of truth
# for the inverse fingerprint -> dataset_id mapping.
from annotate_predictions import _DATASET_PRIORS  # noqa: E402


def _load_results(path: pathlib.Path) -> List[Dict[str, Any]]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"expected a list of rows in {path}, got {type(data).__name__}")
    return data


def _format_fp(fp: Dict[str, float]) -> str:
    keys = ("log_n", "d", "k", "eff_dim_ratio", "conv_cv", "outlier_frac", "density_skew")
    return ", ".join(f"{k}={fp.get(k, 0):.2f}" for k in keys)


def _format_overrides(ov: Dict[str, Any]) -> str:
    fields = ("n_samples", "n_features", "centers", "compactness",
              "outliers", "noise", "density", "outlier_extremity")
    parts = [f"{k}={ov[k]}" for k in fields if k in ov]
    return ", ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=pathlib.Path,
                    default=_REPO / "docs/data/results.json",
                    help="benchmark results JSON (4332-row array).")
    ap.add_argument("--out", type=pathlib.Path,
                    default=_REPO / "docs/reports/gaps_report.json",
                    help="output report path.")
    ap.add_argument("--threshold", type=float, default=0.75,
                    help="max-card-score threshold below which a fingerprint is a gap. "
                         "predict_ari_upper_bound floors at 0.5 and meta cards always "
                         "add ~0.1, so the lowest achievable max score across the 35 "
                         "current cards is ~0.65; 0.75 surfaces the genuinely-hardest "
                         "regimes for the registry.")
    ap.add_argument("--max-grid", type=int, default=500,
                    help="cap on number of grid fingerprints to evaluate.")
    ap.add_argument("--top", type=int, default=20,
                    help="number of top gaps to keep in the report.")
    args = ap.parse_args()

    print(f"loading benchmark from {args.results} ...")
    rows = _load_results(args.results)
    existing = unique_existing_fingerprints(rows, _DATASET_PRIORS)
    print(f"  {len(rows)} rows -> {len(existing)} unique tasks")

    print(f"building fingerprint grid (cap={args.max_grid}) ...")
    grid = build_fingerprint_grid(max_grid=args.max_grid)
    print(f"  {len(grid)} fingerprints in grid")

    print(f"ranking gaps (threshold={args.threshold}, top={args.top}) ...")
    gaps, n_below = rank_gaps(
        grid, existing, _DATASET_PRIORS,
        threshold=args.threshold, top_n=args.top,
    )

    # ---- Smoke-test summary ----------------------------------------------
    print()
    print("=" * 72)
    print("GAP FINDER — SMOKE TEST")
    print("=" * 72)
    print(f"total fingerprints searched:    {len(grid)}")
    print(f"gaps below threshold ({args.threshold:.2f}):  {n_below}")
    print(f"top gaps kept:                  {len(gaps)}")
    print(f"unique benchmark tasks:         {len(existing)}")
    print(f"algorithm cards in registry:    {len(ALGORITHM_CARDS)}")
    print()
    print(f"Top {min(5, len(gaps))} gap fingerprints (by interest):")
    print("-" * 72)
    for i, g in enumerate(gaps[:5], 1):
        print(f"#{i}  interest={g.interest:.3f}  max_existing_score={g.max_existing_score:.3f}")
        print(f"    target: {_format_fp(g.target_fingerprint)}")
        print(f"    top-3 algos:")
        for algo, sc in g.top_algorithms:
            print(f"      {algo:<22} predicted_ari_upper_bound={sc:.3f}")
        print(f"    nearest task: {g.nearest_dataset_id}  (distance={g.nearest_distance:.3f})")
        print(f"    suggested DataSpec overrides:")
        print(f"      {_format_overrides(g.suggested_overrides)}")
        delta = g.suggested_overrides.get("_delta_from_nearest")
        if delta:
            delta_str = ", ".join(f"{k}: {v['from']}->{v['to']}" for k, v in delta.items())
            print(f"    delta from nearest:")
            print(f"      {delta_str}")
        print()

    # ---- Write report -----------------------------------------------------
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "threshold": args.threshold,
        "max_grid": args.max_grid,
        "n_fingerprints_searched": len(grid),
        "n_gaps_below_threshold": n_below,
        "n_unique_benchmark_tasks": len(existing),
        "n_algorithm_cards": len(ALGORITHM_CARDS),
        "top_gaps": [gap_to_dict(g) for g in gaps],
    }
    args.out.write_text(json.dumps(payload, indent=2, default=float))
    print(f"wrote report to {args.out}")


if __name__ == "__main__":
    main()
