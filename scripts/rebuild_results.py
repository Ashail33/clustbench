"""Rebuild a run directory's results tables from its per-run JSON artifacts.

Every (algorithm, task) run writes ``artifacts/metrics_<algo>__<task>.json``
holding the full Record. If ``results.parquet`` / ``results.csv`` /
``summary.json`` are missing or stale (e.g. an interrupted sweep, or the
Parquet writer failed), this regenerates them without re-running anything.

    python scripts/rebuild_results.py runs/my_run [runs/other_run ...]
"""
from __future__ import annotations

import argparse
import json
import pathlib

from clustbench.cli.run import results_frame, write_results, write_summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=pathlib.Path)
    args = ap.parse_args()
    for run in args.runs:
        rows = [json.loads(p.read_text()) for p in sorted((run / "artifacts").glob("metrics_*.json"))]
        df = results_frame(rows)
        write_results(df, run)
        write_summary(df, run)
        print(f"{run}: {len(df)} rows")


if __name__ == "__main__":
    main()
