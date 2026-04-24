"""Convert a clustbench run directory into JSON the static dashboard can load.

Usage
-----
    python scripts/build_site.py --run runs/dashboard --out docs/data

Writes two files under ``--out``:
- ``results.json`` — one record per (algo, task) with key metrics
- ``trajectories.json`` — list of per-step cost curves keyed by the same
  task identifiers (only for algorithms that emit a trajectory)

The JSON is intentionally small and denormalized so the browser can render
it with no further processing.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib

import pandas as pd


RESULT_COLS = [
    "algo",
    "dataset_id",
    "n_samples",
    "n_features",
    "k_target",
    "seed",
    "ari",
    "nmi",
    "silhouette",
    "davies_bouldin",
    "dunn",
    "wall_time_s",
    "rss_delta_mb",
    "n_clusters_found",
    "n_steps",
]


def _clean(v):
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def build(run_dir: pathlib.Path, out_dir: pathlib.Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(run_dir / "results.parquet")
    df = df.rename(columns={"compactness": "compactness_metric"})
    # Recover the configured compactness from the trajectory filename suffix
    # so the dashboard can slice by experiment setting instead of the output
    # metric of the same name.
    def _cfg_compactness(row):
        if pd.isna(row.get("trajectory_path")):
            lp = row.get("labels_path")
            if lp:
                import re
                m = re.search(r"_c([0-9.]+)_s", lp)
                if m:
                    return float(m.group(1))
            return None
        import re
        m = re.search(r"_c([0-9.]+)_s", row["trajectory_path"])
        return float(m.group(1)) if m else None

    df["compactness_cfg"] = df.apply(_cfg_compactness, axis=1)

    results = []
    for _, row in df.iterrows():
        rec = {c: _clean(row.get(c)) for c in RESULT_COLS if c in df.columns}
        rec["compactness"] = _clean(row.get("compactness_cfg"))
        results.append(rec)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))

    trajectories = []
    for _, row in df.iterrows():
        tp = row.get("trajectory_path")
        if not tp or pd.isna(tp):
            continue
        traj_path = pathlib.Path(tp)
        if not traj_path.is_absolute():
            traj_path = run_dir.parent / traj_path if not traj_path.exists() else traj_path
        if not traj_path.exists():
            # Try resolving relative to cwd
            traj_path = pathlib.Path(tp)
        traj_df = pd.read_parquet(traj_path)
        steps = [
            {
                "step_idx": int(s["step_idx"]),
                "cost": float(s["cost"]),
                "delta_cost": _clean(s.get("delta_cost")),
                "accepted": bool(s.get("accepted", True)),
            }
            for _, s in traj_df.iterrows()
        ]
        trajectories.append(
            {
                "algo": row["algo"],
                "dataset_id": row["dataset_id"],
                "n_samples": int(row["n_samples"]),
                "n_features": int(row["n_features"]),
                "k_target": int(row["k_target"]),
                "compactness": _clean(row.get("compactness_cfg")),
                "seed": int(row["seed"]),
                "n_steps": len(steps),
                "final_cost": steps[-1]["cost"] if steps else None,
                "steps": steps,
            }
        )
    (out_dir / "trajectories.json").write_text(json.dumps(trajectories, default=str))

    manifest = json.loads((run_dir / "manifest.json").read_text())
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"Wrote {len(results)} results and {len(trajectories)} trajectories to {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=pathlib.Path, required=True, help="clustbench run directory")
    ap.add_argument("--out", type=pathlib.Path, required=True, help="output directory for JSON")
    args = ap.parse_args()
    build(args.run, args.out)


if __name__ == "__main__":
    main()
