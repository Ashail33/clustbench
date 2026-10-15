"""Leave-one-dataset-family-out (LOFO) evaluation of the learned routers.

The routers train on the benchmark's own results. Their in-distribution
scores only drop an exact fingerprint self-match, so sibling seeds of the
same configuration stay in the training set and leak. LOFO removes the whole
dataset family: for each family F, every router is retrained on all rows
*except* F's and evaluated on F's tasks. This is the honest estimate of how a
router does on a kind of data it has never seen.

Each fold runs in a fresh subprocess because routers cache their training
set per process (see ``clustbench.training_data``).

Usage
-----
    python scripts/router_lofo.py --results docs/data/results.json \\
        --out docs/data/router_lofo.json --jobs 4
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

ROUTERS = [
    "learned_router",
    "learned_router_v2",
    "learned_router_v3",
    "learned_router_v4",
    "learned_router_v5",
    "learned_router_v7",
    "learned_router_v8",
]
TASK_FIELDS = ["dataset_id", "n_samples", "n_features", "k_target", "compactness",
               "outliers", "noise", "density", "outlier_extremity", "seed"]

FOLD_CODE = r"""
import json, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.metrics import adjusted_rand_score
from clustbench.algorithms.base import ALGO_REGISTRY
import clustbench.algorithms  # noqa: F401  (registers everything)
from clustbench.benchmark import generate_with_provenance
from clustbench.datasets import DataSpec
from clustbench.metrics import inlier_scores

tasks = json.loads(sys.argv[1]); routers = json.loads(sys.argv[2])
for t in tasks:
    spec = DataSpec(n_samples=t["n_samples"], n_features=t["n_features"], centers=t["k_target"],
                    compactness=t.get("compactness") or 1.0, seed=t["seed"],
                    outliers=t.get("outliers") or 0, noise=t.get("noise") or 0,
                    density=t.get("density") or 1.0,
                    outlier_extremity=t.get("outlier_extremity") or 1.0)
    X, y, src = generate_with_provenance(t["dataset_id"], spec)
    for r in routers:
        rec = dict(t, algo=r, dataset_source=src)
        try:
            res = ALGO_REGISTRY[r]().fit_predict(X, k=t["k_target"])
            rec["ari"] = float(adjusted_rand_score(y, res.labels))
            rec.update(inlier_scores(res.labels, y))
            rec["chose"] = (res.extra or {}).get("chose")
        except Exception as e:  # keep going; record the failure
            rec["error"] = f"{type(e).__name__}: {e}"[:200]
        print(json.dumps(rec), flush=True)
"""


def _tasks_by_family(rows: list[dict]) -> dict[str, list[dict]]:
    seen = set()
    fam: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        t = {k: r.get(k) for k in TASK_FIELDS}
        key = tuple(t.values())
        if key in seen:
            continue
        seen.add(key)
        fam[t["dataset_id"]].append(t)
    return fam


def run_fold(family: str, tasks: list[dict], results: pathlib.Path, routers: list[str]) -> list[dict]:
    env = {**os.environ,
           "CLUSTBENCH_ROUTER_TRAINING": str(results.resolve()),
           "CLUSTBENCH_ROUTER_EXCLUDE_DATASETS": family,
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    proc = subprocess.run([sys.executable, "-c", FOLD_CODE, json.dumps(tasks), json.dumps(routers)],
                          env=env, capture_output=True, text=True)
    out = [json.loads(line) for line in proc.stdout.splitlines() if line.startswith("{")]
    if proc.returncode != 0:
        print(f"[{family}] exit {proc.returncode}: {proc.stderr[-500:]}", file=sys.stderr)
    print(f"[{family}] {len(out)} rows", file=sys.stderr, flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=pathlib.Path, default=pathlib.Path("docs/data/results.json"))
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("docs/data/router_lofo.json"))
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--routers", nargs="*", default=ROUTERS)
    args = ap.parse_args()

    rows = json.loads(args.results.read_text())
    fams = _tasks_by_family(rows)
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(run_fold, f, ts, args.results, args.routers) for f, ts in sorted(fams.items())]
        lofo = [r for fu in futs for r in fu.result()]

    # Compare with the in-distribution (leave-one-out) numbers for the same tasks.
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None

    indist = defaultdict(list)
    for r in rows:
        if r.get("algo") in args.routers and r.get("ari") is not None:
            indist[r["algo"]].append(r["ari"])
    summary = {}
    for a in args.routers:
        mine = [r for r in lofo if r["algo"] == a]
        summary[a] = {
            "lofo_mean_ari": mean([r.get("ari") for r in mine]),
            "indist_mean_ari": mean(indist[a]),
            "n_tasks": sum(1 for r in mine if r.get("ari") is not None),
            "n_errors": sum(1 for r in mine if "error" in r),
            "per_family": {f: mean([r.get("ari") for r in mine if r["dataset_id"] == f]) for f in sorted(fams)},
        }
    args.out.write_text(json.dumps({"summary": summary, "rows": lofo}, indent=1))
    for a, s in summary.items():
        print(f"{a:20s} in-dist {s['indist_mean_ari'] or float('nan'):.3f}  "
              f"LOFO {s['lofo_mean_ari'] or float('nan'):.3f}  (n={s['n_tasks']}, errors={s['n_errors']})")


if __name__ == "__main__":
    main()
