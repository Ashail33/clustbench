#!/usr/bin/env python3
"""CLI entrypoint for the algorithm mutator.

Usage:
    python scripts/mutate_algorithms.py --iter 200 --top 10 \
        --out runs/mutator_report.json

Generates novel clustering AlgorithmCards by mutating existing cards in
:mod:`clustbench.algorithm_cards`, scores each candidate's predicted
ARI across ~50 fingerprints sampled from ``docs/data/results.json``,
and saves the top-N candidates that beat the best existing card by at
least the configured margin.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _ensure_repo_on_path() -> None:
    """Allow running as ``python scripts/mutate_algorithms.py`` without
    installing the package."""
    here = Path(__file__).resolve()
    repo = here.parent.parent
    src = repo / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


def main(argv=None) -> int:
    _ensure_repo_on_path()

    # Imported after path fix so the script works in-tree.
    from clustbench.algorithm_mutator import random_search

    parser = argparse.ArgumentParser(
        description="Mutate clustering AlgorithmCards and rank candidates "
                    "by predicted ARI upper bound."
    )
    parser.add_argument("--iter", type=int, default=200,
                        help="Number of random mutations to evaluate (default: 200).")
    parser.add_argument("--top", type=int, default=10,
                        help="Keep this many top candidates (default: 10).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0).")
    parser.add_argument("--margin", type=float, default=0.02,
                        help="Min predicted-ARI delta vs best existing "
                             "card to keep a candidate (default: 0.02).")
    parser.add_argument("--n-fingerprints", type=int, default=50,
                        help="Distinct fingerprints sampled from "
                             "results.json (default: 50).")
    parser.add_argument("--results", type=str, default=None,
                        help="Path to docs/data/results.json (default: "
                             "auto-discover relative to repo root).")
    parser.add_argument("--out", type=str, default="docs/reports/mutator_report.json",
                        help="Output JSON report path "
                             "(default: docs/reports/mutator_report.json).")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress stdout printing of top 5 candidates.")
    args = parser.parse_args(argv)

    report = random_search(
        n_iter=args.iter,
        top_n=args.top,
        seed=args.seed,
        margin=args.margin,
        results_path=args.results,
        n_fingerprints=args.n_fingerprints,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    if not args.quiet:
        print(f"=== Algorithm Mutator Report ===")
        print(f"iterations: {report['n_evaluated']}")
        print(f"fingerprints: {report['fingerprint_sample_size']}")
        print(
            f"best existing card: {report['best_existing_card']['name']} "
            f"(mean predicted ARI={report['best_existing_card']['predicted_mean_ari']:.4f})"
        )
        print(f"candidates beating best existing card "
              f"by margin {report['margin']:+.2f}: {report['n_beat_existing']}")
        print()
        print("Top 5 mutated candidates:")
        for i, rec in enumerate(report["top"][:5], 1):
            parents = " + ".join(rec["parents"])
            print(f" {i}. {rec['name']}")
            print(f"      parent(s): {parents}")
            print(f"      op:        {rec['op']}")
            print(
                f"      predicted_mean_ari: {rec['predicted_mean_ari']:.4f} "
                f"(delta={rec['delta_vs_best_existing']:+.4f})"
            )
            print(f"      next step: {rec['next_implementation_step']}")
        print()
        print(f"Saved full report to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
