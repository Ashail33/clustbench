"""Append a clustbench run's results into the live dashboard JSON.

Usage
-----
    python scripts/append_run_to_dashboard.py \\
        --run runs/<new_run_dir> \\
        --dashboard docs/data

Reads the new run's ``results.parquet`` (and per-algo trajectory
parquets) and appends them to the existing ``docs/data/results.json``
and ``docs/data/trajectories.json``, updating ``manifest.json``'s
``merged_from`` list. This lets us add algorithms to the dashboard
without re-running every prior algorithm.

Safe to run repeatedly — duplicate ``(algo, dataset_id, n_samples,
n_features, k_target, compactness, outliers, noise, density, seed)``
rows from the new run replace any existing dashboard rows with the
same key (so re-running a backfill is idempotent).
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
    "outliers",
    "noise",
    "density",
    "ari",
    "nmi",
    "silhouette",
    "davies_bouldin",
    "dunn",
    "wall_time_s",
    "rss_delta_mb",
    "cpu_user_s",
    "cpu_system_s",
    "n_clusters_found",
    "n_steps",
]

TASK_KEY = (
    "algo", "dataset_id", "n_samples", "n_features", "k_target",
    "compactness", "outliers", "noise", "density", "seed",
)


def _clean(v):
    if v is None:
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def _row_key(rec: dict) -> tuple:
    return tuple(rec.get(k) for k in TASK_KEY)


def append(run_dir: pathlib.Path, dashboard_dir: pathlib.Path) -> None:
    # --- Load the new run.
    df = pd.read_parquet(run_dir / "results.parquet")
    if "compactness" in df.columns and df["compactness"].dtype == object:
        df["compactness"] = pd.to_numeric(df["compactness"], errors="coerce")

    new_results: list[dict] = []
    for _, row in df.iterrows():
        rec = {c: _clean(row.get(c)) for c in RESULT_COLS if c in df.columns}
        rec["compactness"] = _clean(row.get("compactness"))
        new_results.append(rec)

    new_trajectories: list[dict] = []
    for _, row in df.iterrows():
        tp = row.get("trajectory_path")
        if not tp or pd.isna(tp):
            continue
        traj_path = pathlib.Path(tp)
        if not traj_path.exists():
            traj_path = run_dir / "artifacts" / traj_path.name
        if not traj_path.exists():
            continue
        traj_df = (
            pd.read_csv(traj_path) if traj_path.suffix == ".csv"
            else pd.read_parquet(traj_path)
        )
        steps = [
            {
                "step_idx": int(s["step_idx"]),
                "cost": float(s["cost"]),
                "delta_cost": _clean(s.get("delta_cost")),
                "accepted": bool(s.get("accepted", True)),
            }
            for _, s in traj_df.iterrows()
        ]
        new_trajectories.append(
            {
                "algo": row["algo"],
                "dataset_id": row["dataset_id"],
                "n_samples": int(row["n_samples"]),
                "n_features": int(row["n_features"]),
                "k_target": int(row["k_target"]),
                "compactness": _clean(row.get("compactness")),
                "outliers": _clean(row.get("outliers")),
                "noise": _clean(row.get("noise")),
                "density": _clean(row.get("density")),
                "seed": int(row["seed"]),
                "n_steps": len(steps),
                "final_cost": steps[-1]["cost"] if steps else None,
                "steps": steps,
            }
        )

    # --- Load existing dashboard data.
    res_path = dashboard_dir / "results.json"
    traj_path = dashboard_dir / "trajectories.json"
    existing_results = json.loads(res_path.read_text()) if res_path.exists() else []
    existing_trajectories = (
        json.loads(traj_path.read_text()) if traj_path.exists() else []
    )

    new_keys = {_row_key(r) for r in new_results}
    merged_results = [r for r in existing_results if _row_key(r) not in new_keys]
    merged_results.extend(new_results)
    new_traj_keys = {_row_key(t) for t in new_trajectories}
    merged_trajectories = [
        t for t in existing_trajectories if _row_key(t) not in new_traj_keys
    ]
    merged_trajectories.extend(new_trajectories)

    res_path.write_text(json.dumps(merged_results, indent=2, default=str))
    traj_path.write_text(json.dumps(merged_trajectories, default=str))

    manifest_path = dashboard_dir / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    )
    merged_from = manifest.get("merged_from", [])
    run_label = str(run_dir.resolve().relative_to(pathlib.Path.cwd().resolve())
                    if run_dir.resolve().is_relative_to(pathlib.Path.cwd().resolve())
                    else run_dir)
    if run_label not in merged_from:
        merged_from.append(run_label)
    manifest["merged_from"] = merged_from
    manifest["row_count"] = len(merged_results)
    manifest["algorithm_count"] = len({r["algo"] for r in merged_results})
    manifest["task_count"] = len(
        {
            (r["dataset_id"], r["n_samples"], r["n_features"], r["k_target"],
             r.get("compactness"), r.get("outliers"), r.get("noise"), r.get("density"))
            for r in merged_results
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(
        f"Appended {len(new_results)} new rows from {run_dir} "
        f"(now {len(merged_results)} total; "
        f"{manifest['algorithm_count']} algos, {manifest['task_count']} tasks)."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=pathlib.Path, required=True)
    ap.add_argument("--dashboard", type=pathlib.Path, required=True)
    args = ap.parse_args()
    append(args.run, args.dashboard)


if __name__ == "__main__":
    main()
