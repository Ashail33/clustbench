#!/usr/bin/env python3
"""Generate summary charts from a clustbench results parquet or CSV.

Example:
```
python scripts/report_summary.py --results runs/demo/results.parquet
```
This will produce PNG bar charts for each metric by algorithm.
"""

from __future__ import annotations

import argparse
import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Results Parquet or CSV file")
    parser.add_argument("--outdir", default=".", help="Directory to write charts")
    args = parser.parse_args()
    if args.results.endswith(".parquet"):
        df = pd.read_parquet(args.results)
    else:
        df = pd.read_csv(args.results)
    metrics = [c for c in ["ari", "nmi", "silhouette", "davies_bouldin", "wall_time_s", "rss_delta_mb"] if c in df.columns]
    for m in metrics:
        plt.figure()
        df.groupby("algo")[m].mean().sort_values(ascending=False).plot(kind="bar")
        plt.title(f"Mean {m} by algorithm")
        plt.tight_layout()
        fname = f"{args.outdir}/{m}_by_algo.png"
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()

if __name__ == "__main__":
    main()
