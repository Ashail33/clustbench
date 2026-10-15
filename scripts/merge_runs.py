"""Merge sharded clustbench run directories into one run directory.

Lets a large sweep be split across processes (e.g. one config per dataset
group, each with ``timing.threads: 1``) and recombined before
``scripts/build_site.py``. Rows are concatenated, exact duplicate
(algo, task) rows are rejected, and ``summary.json`` is recomputed over the
merged table with the same code the CLI uses.

    python scripts/merge_runs.py --runs runs/shard_a runs/shard_b --out runs/merged
"""
from __future__ import annotations

import argparse
import json
import pathlib

import pandas as pd

from clustbench.cli.run import write_summary
from clustbench.stats import TASK_KEYS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()

    frames, manifests = [], []
    for r in args.runs:
        frames.append(pd.read_parquet(r / "results.parquet"))
        manifests.append(json.loads((r / "manifest.json").read_text()))
    df = pd.concat(frames, ignore_index=True)
    keys = [k for k in TASK_KEYS if k in df.columns] + ["algo"]
    dup = df.duplicated(subset=keys, keep=False)
    if dup.any():
        raise SystemExit(f"{int(dup.sum())} duplicate (task, algo) rows across shards")

    args.out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out / "results.parquet")
    df.to_csv(args.out / "results.csv", index=False)
    manifest = dict(manifests[0])
    manifest["shards"] = [
        {"run_dir": str(r), "config_sha256": m.get("config_sha256"), "generated_at": m.get("generated_at"),
         "datasets": [d["id"] for d in m["config"].get("datasets", [])]}
        for r, m in zip(args.runs, manifests)
    ]
    manifest["config"] = dict(manifest["config"])
    manifest["config"]["datasets"] = [d for m in manifests for d in m["config"].get("datasets", [])]
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    write_summary(df, args.out)
    print(f"merged {len(df)} rows from {len(args.runs)} shards into {args.out}")


if __name__ == "__main__":
    main()
