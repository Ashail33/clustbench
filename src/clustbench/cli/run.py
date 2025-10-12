"""CLI entrypoint for clustbench."""

from __future__ import annotations
import argparse
import json
import pathlib
import platform
import time
from typing import List

import yaml
import pandas as pd

from ..benchmark import run_task, AlgoCfg, average_ranks, friedman


def main() -> None:
    parser = argparse.ArgumentParser(description="Run clustbench benchmark")
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    outdir = args.out
    outdir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "config": cfg,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "env": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Prepare algorithm configurations
    algos: List[AlgoCfg] = [
        AlgoCfg(name=a["name"], kind=a.get("kind", "python"), entry=a["entry"], params=a.get("params", {}))
        for a in cfg["algorithms"]
    ]

    rows: List[dict] = []
    for ds in cfg["datasets"]:
        dataset_id = ds["id"]
        for n in ds["n_samples"]:
            for d in ds["n_features"]:
                for k in ds["k_targets"]:
                    for c in ds["compactness"]:
                        for seed in cfg.get("seeds", [42]):
                            rows.extend(run_task(dataset_id, n, d, k, c, seed, algos, outdir))

    # Flatten metrics dict so metrics appear top‑level in the DataFrame
    def flatten(r: dict) -> dict:
        m = r["metrics"]
        base = {k: v for k, v in r.items() if k != "metrics"}
        base.update(m)
        return base

    flat_rows = [flatten(r) for r in rows]
    df = pd.DataFrame(flat_rows)

    # Save results; attempt Parquet, fallback to CSV
    try:
        df.to_parquet(outdir / "results.parquet")
    except Exception:
        pass
    df.to_csv(outdir / "results.csv", index=False)

    metrics_list = [
        ("ari", True),
        ("nmi", True),
        ("silhouette", True),
        ("davies_bouldin", False),
        ("wall_time_s", False),
        ("rss_delta_mb", False),
    ]
    summary = {}
    for metric, higher in metrics_list:
        if metric not in df.columns:
            continue
        try:
            ranks = average_ranks(df, metric, higher).to_dict(orient="records")
            fr = friedman(df, metric, higher)
            summary[metric] = {"average_ranks": ranks, "friedman": fr}
        except Exception as e:
            summary[metric] = {"error": str(e)}
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote results to {outdir}")
