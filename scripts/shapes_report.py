"""Report for the all-datasets run (configs/benchmark.shapes.yaml).

Writes to ``--out`` (default ``docs/data_shapes``):

* ``results.json`` / ``failures.json``: every run, and every run that hit the
  time or memory limit;
* ``summary.md``: per dataset group (non-convex shapes, mixtures,
  density / imbalance / outliers, convex blobs, real data), every algorithm's
  mean ARI and rank, with runs that did not finish counted as ARI 0;
* ``heatmap.pdf``: mean ARI for every algorithm on every dataset;
* ``gallery.pdf``: the 2-D shape datasets (seed 1) coloured by the ground
  truth and by a few algorithms' labels (recomputed, same data and seeds).

    python scripts/shapes_report.py runs/shapes/s0 runs/shapes/s1 ... --out docs/data_shapes
"""
from __future__ import annotations

import argparse
import json
import pathlib
import warnings

import numpy as np
import pandas as pd

GROUPS = {
    "non-convex shapes": ["moons", "circles", "rings", "spiral", "s_curve", "swiss_roll"],
    "mixtures": ["mixed_shapes", "mixed", "VariableDensityBridges", "hierarchical_nested",
                 "heavy_tailed_mixture", "PowerLawStudentT"],
    "density / imbalance / outliers": ["varying_density", "imbalanced", "imbalanced_blobs", "extreme_outliers"],
    "convex blobs": ["blobs", "anisotropic", "mdcgen", "inverse_pca"],
    "real data": ["iris", "wine", "breast_cancer", "digits", "ecoli", "glass", "yeast", "vehicle", "segment",
                  "olivetti_faces", "text20news", "ts_ecg200", "ts_trace", "graph_karate"],
    "other synthetic": ["graph_sbm", "ts_synth"],
}
FAMILY_GROUP = {f: g for g, fs in GROUPS.items() for f in fs}
GALLERY_ALGOS = ["kmeans", "gmm", "dbscan_auto", "agglomerative", "spectral", "lmm", "mcg_cf", "lmm_mcg",
                 "learned_router_v3"]
GALLERY_SETS = [("moons", 2, 2), ("circles", 2, 2), ("rings", 2, 5), ("spiral", 2, 3), ("s_curve", 3, 4),
                ("swiss_roll", 3, 4), ("mixed_shapes", 2, 5), ("mixed", 2, 4), ("VariableDensityBridges", 2, 4),
                ("hierarchical_nested", 2, 4), ("heavy_tailed_mixture", 2, 4), ("varying_density", 2, 4),
                ("imbalanced_blobs", 2, 4), ("anisotropic", 2, 4)]
KEYS = ["dataset_id", "n_samples", "n_features", "k_target", "seed"]


def task_label(r) -> str:
    k = f" k={r.k_target}" if r.dataset_id in ("rings", "spiral") else ""
    return f"{r.dataset_id}{k} d={r.n_features}"


def load(runs):
    res = pd.concat([pd.read_parquet(r / "results.parquet") for r in runs], ignore_index=True)
    res["ari"] = pd.to_numeric(res["ari"], errors="coerce")
    fails = []
    for r in runs:
        for p in (r / "artifacts").glob("error_*.json"):
            fails.append(json.loads(p.read_text()))
        for p in (r / "artifacts").glob("skipped_*.json"):
            fails.append({**json.loads(p.read_text()), "error_type": "capped"})
    return res, pd.DataFrame(fails)


def scored(res: pd.DataFrame, fails: pd.DataFrame) -> pd.DataFrame:
    """One row per (algorithm, task); runs that did not finish score ARI 0."""
    a = res[["algo"] + KEYS + ["ari", "wall_time_s"]].copy()
    a["finished"] = True
    if len(fails):
        f = fails[["algo"] + KEYS].copy()
        f["ari"], f["wall_time_s"], f["finished"] = 0.0, np.nan, False
        a = pd.concat([a, f], ignore_index=True)
    a["ari"] = a["ari"].fillna(0.0)
    a["group"] = a.dataset_id.map(FAMILY_GROUP).fillna("other synthetic")
    a["task"] = a.apply(task_label, axis=1)
    return a


def summary_md(a: pd.DataFrame) -> str:
    lines = ["# Every algorithm on every dataset",
             "",
             "Mean ARI over tasks and seeds. Runs that hit the 120 s / 3 GB limit count as ARI 0",
             "(the DNF column says how many). Rank is within the group.",
             ""]
    overall = a.groupby("algo").ari.mean().sort_values(ascending=False)
    groups = [g for g in GROUPS if g in set(a.group)]
    per = {g: a[a.group == g].groupby("algo").agg(ari=("ari", "mean"), dnf=("finished", lambda s: int((~s).sum())))
           for g in groups}
    lines += ["## Overview (mean ARI per group)", "",
              "| rank | algorithm | all | " + " | ".join(groups) + " | DNF | median time (s) |",
              "|---" * (len(groups) + 5) + "|"]
    med_t = a[a.finished].groupby("algo").wall_time_s.median()
    dnf = a.groupby("algo").finished.apply(lambda s: int((~s).sum()))
    for i, (algo, v) in enumerate(overall.items(), 1):
        cells = []
        for g in groups:
            sub = per[g].ari.sort_values(ascending=False)
            r = list(sub.index).index(algo) + 1 if algo in sub.index else None
            cells.append(f"{sub.get(algo, float('nan')):.3f} ({r})")
        lines.append(f"| {i} | `{algo}` | {v:.3f} | " + " | ".join(cells)
                     + f" | {dnf.get(algo, 0)} | {med_t.get(algo, float('nan')):.2f} |")
    for g in groups:
        sub = a[a.group == g]
        piv = sub.pivot_table(index="algo", columns="dataset_id", values="ari", aggfunc="mean")
        piv["mean"] = piv.mean(axis=1)
        piv = piv.sort_values("mean", ascending=False)
        fams = [f for f in GROUPS[g] if f in piv.columns]
        lines += ["", f"## {g}", "", "| rank | algorithm | mean | " + " | ".join(fams) + " |",
                  "|---" * (len(fams) + 3) + "|"]
        for i, (algo, r) in enumerate(piv.iterrows(), 1):
            lines.append(f"| {i} | `{algo}` | {r['mean']:.3f} | " + " | ".join(f"{r[f]:.2f}" for f in fams) + " |")
    return "\n".join(lines) + "\n"


def heatmap(a: pd.DataFrame, path: pathlib.Path) -> None:
    import matplotlib.pyplot as plt

    order_algo = a.groupby("algo").ari.mean().sort_values(ascending=False).index
    tasks = a[["task", "group", "dataset_id", "n_features", "k_target"]].drop_duplicates()
    tasks["g"] = tasks.group.map({g: i for i, g in enumerate(GROUPS)})
    tasks = tasks.sort_values(["g", "dataset_id", "k_target", "n_features"])
    piv = a.pivot_table(index="algo", columns="task", values="ari", aggfunc="mean").reindex(
        index=order_algo, columns=tasks.task)
    fig, ax = plt.subplots(figsize=(0.19 * piv.shape[1] + 3, 0.16 * piv.shape[0] + 2))
    im = ax.imshow(piv.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels(piv.columns, rotation=75, ha="right", fontsize=5)
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels(piv.index, fontsize=5)
    # group separators
    gs = tasks.group.tolist()
    for j in range(1, len(gs)):
        if gs[j] != gs[j - 1]:
            ax.axvline(j - 0.5, color="white", lw=1.2)
    fig.colorbar(im, ax=ax, fraction=0.02).set_label("mean ARI (DNF = 0)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def gallery(path: pathlib.Path) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import adjusted_rand_score
    from threadpoolctl import threadpool_limits

    from clustbench.algorithms.base import ALGO_REGISTRY
    from clustbench.datasets import DATASETS, DataSpec

    cols = ["truth"] + GALLERY_ALGOS
    fig, axes = plt.subplots(len(GALLERY_SETS), len(cols), figsize=(1.45 * len(cols), 1.45 * len(GALLERY_SETS)))
    with warnings.catch_warnings(), threadpool_limits(1):
        warnings.simplefilter("ignore")
        for i, (name, d, k) in enumerate(GALLERY_SETS):
            X, y = DATASETS[name](DataSpec(n_samples=2000, n_features=d, centers=k, compactness=1.0, seed=1))
            X = np.asarray(X, float)
            P = X[:, [0, 2]] if X.shape[1] == 3 else X[:, :2]  # S-curve / Swiss roll: the unrolled view
            for j, c in enumerate(cols):
                ax = axes[i, j]
                if c == "truth":
                    lab, title = y, "truth"
                else:
                    try:
                        lab = ALGO_REGISTRY[c]().fit_predict(X, k=k).labels
                        title = f"{c}\n{adjusted_rand_score(y, lab):.2f}"
                    except Exception as e:  # noqa: BLE001
                        lab, title = np.zeros(len(X), int), f"{c}\n{type(e).__name__}"
                ax.scatter(P[:, 0], P[:, 1], c=lab, s=0.6, cmap="tab10", linewidths=0, rasterized=True)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(title, fontsize=5, pad=2)
                if j == 0:
                    ax.set_ylabel(f"{name}\nk={k}", fontsize=5.5)
    fig.tight_layout(pad=0.3)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("docs/data_shapes"))
    ap.add_argument("--no-gallery", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    res, fails = load(args.runs)
    cols = ["algo"] + KEYS + ["dataset_source", "ari", "nmi", "wall_time_s", "n_clusters_found"]
    res[[c for c in cols if c in res.columns]].to_json(args.out / "results.json", orient="records", indent=1)
    if len(fails):
        fails.to_json(args.out / "failures.json", orient="records", indent=1)
    a = scored(res, fails)
    (args.out / "summary.md").write_text(summary_md(a))
    heatmap(a, args.out / "heatmap.pdf")
    if not args.no_gallery:
        gallery(args.out / "gallery.pdf")
    print(f"wrote {args.out}: {len(res)} runs, {len(fails)} did not finish")


if __name__ == "__main__":
    main()
