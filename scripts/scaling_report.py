"""Scaling report for a big-data sweep (configs/benchmark.bigdata.yaml).

Reads one or more run directories (results + error/skip artifacts) and writes

* ``scaling_table.csv`` / ``scaling_table.md``: per algorithm, the fitted
  log-log growth exponent of wall time in n, d and k (each from the
  one-factor-at-a-time axis through the base point n=10k, d=20, k=10), wall
  time and ARI at the base and at the largest n, and how many runs did not
  finish (timeout / memory error / capped);
* ``scaling_axes.pdf``: wall time against n, d and k for every algorithm;
* ``scaling_grid.pdf``: wall time per algorithm x task, with runs that did
  not finish marked.

    python scripts/scaling_report.py runs/bigdata --out docs/data_bigdata
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
from collections import defaultdict

import numpy as np
import pandas as pd

BASE = {"n_samples": 10000, "n_features": 20, "k_target": 10}
AXES = {"n": "n_samples", "d": "n_features", "k": "k_target"}
# Representative algorithms drawn in colour in scaling_axes.pdf.
HIGHLIGHT = ["minibatch_kmeans", "kmeans", "gmm", "bico_stream", "trident", "fmm", "lmm",
             "spectral", "agglomerative", "birch_algo", "louvain_knn", "learned_router_v3",
             "dbscan_auto", "tbmm_fl"]


def load(runs: list[pathlib.Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    res = pd.concat([pd.read_parquet(r / "results.parquet") for r in runs], ignore_index=True)
    fails = []
    for r in runs:
        for kind, pat in (("error", "error_*.json"), ("skipped", "skipped_*.json")):
            for p in (r / "artifacts").glob(pat):
                e = json.loads(p.read_text())
                e["status"] = e.get("error_type", "capped") if kind == "error" else "capped"
                fails.append(e)
    return res, pd.DataFrame(fails)


def axis_rows(df: pd.DataFrame, axis: str) -> pd.DataFrame:
    col = AXES[axis]
    others = {c: v for c, v in BASE.items() if c != col}
    m = np.ones(len(df), dtype=bool)
    for c, v in others.items():
        m &= df[c].values == v
    return df[m]


def slope(x: np.ndarray, y: np.ndarray) -> float:
    ok = (x > 0) & (y > 0)
    if ok.sum() < 2:
        return float("nan")
    lx, ly = np.log(x[ok]), np.log(y[ok])
    return float(np.polyfit(lx, ly, 1)[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("docs/data_bigdata"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    res, fails = load(args.runs)
    res["ari"] = pd.to_numeric(res["ari"], errors="coerce")

    rows = []
    for algo, g in res.groupby("algo"):
        row = {"algo": algo}
        for ax, col in AXES.items():
            a = axis_rows(g, ax).groupby(col)["wall_time_s"].median()
            row[f"exp_{ax}"] = slope(a.index.values.astype(float), a.values)
            row[f"max_{ax}_done"] = int(a.index.max()) if len(a) else None
        base = g[(g.n_samples == 10000) & (g.n_features == 20) & (g.k_target == 10)]
        row["wall_base_s"] = float(base.wall_time_s.median()) if len(base) else float("nan")
        row["ari_base"] = float(base.ari.mean()) if len(base) else float("nan")
        big = g[(g.n_samples == g.n_samples.max())]
        row["largest_n_done"] = int(g.n_samples.max())
        row["wall_largest_n_s"] = float(big.wall_time_s.median())
        row["ari_largest_n"] = float(big.ari.mean())
        f = fails[fails.algo == algo] if len(fails) else fails
        for st in ("Timeout", "MemoryError", "capped"):
            row[f"dnf_{st.lower()}"] = int((f.status == st).sum()) if len(f) else 0
        row["dnf_other"] = int((~f.status.isin(["Timeout", "MemoryError", "capped"])).sum()) if len(f) else 0
        row["n_done"] = int(len(g))
        rows.append(row)
    # algorithms that never finished anything
    if len(fails):
        for algo in sorted(set(fails.algo) - set(res.algo)):
            f = fails[fails.algo == algo]
            rows.append({"algo": algo, "n_done": 0,
                         **{f"dnf_{st.lower()}": int((f.status == st).sum()) for st in ("Timeout", "MemoryError", "capped")}})
    tab = pd.DataFrame(rows).sort_values(["n_done", "wall_base_s"], ascending=[False, True])
    tab.to_csv(args.out / "scaling_table.csv", index=False)

    def fmt(v, p=2):
        return "" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{p}f}"

    lines = ["| algorithm | exp. n | exp. d | exp. k | wall @ base (s) | ARI @ base | largest n done | wall there (s) | ARI there | DNF (timeout / memory / capped) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in tab.iterrows():
        lines.append(f"| `{r.algo}` | {fmt(r.get('exp_n'))} | {fmt(r.get('exp_d'))} | {fmt(r.get('exp_k'))} | "
                     f"{fmt(r.get('wall_base_s'), 3)} | {fmt(r.get('ari_base'))} | "
                     f"{'' if pd.isna(r.get('largest_n_done')) else int(r.get('largest_n_done')):,} | "
                     f"{fmt(r.get('wall_largest_n_s'), 1)} | {fmt(r.get('ari_largest_n'))} | "
                     f"{int(r.get('dnf_timeout', 0) or 0)} / {int(r.get('dnf_memoryerror', 0) or 0)} / {int(r.get('dnf_capped', 0) or 0)} |")
    (args.out / "scaling_table.md").write_text("\n".join(lines) + "\n")

    # JSON copies for the dashboard / papers
    cols = ["algo", "dataset_id", "n_samples", "n_features", "k_target", "seed", "ari", "nmi",
            "wall_time_s", "rss_delta_mb", "n_clusters_found"]
    res[[c for c in cols if c in res.columns]].to_json(args.out / "results.json", orient="records", indent=1)
    if len(fails):
        fails.to_json(args.out / "failures.json", orient="records", indent=1)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    algos = list(tab.algo)
    cmap = plt.get_cmap("tab20")
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), sharey=True)
    for ax_i, (ax, col) in enumerate(AXES.items()):
        a = axes[ax_i]
        hi = [h for h in HIGHLIGHT if h in algos]
        for algo in algos:
            if algo in hi:
                continue
            s = axis_rows(res[res.algo == algo], ax).groupby(col)["wall_time_s"].median()
            if len(s) >= 2:
                a.plot(s.index, s.values, "-", lw=0.4, color="0.85", zorder=1)
        for i, algo in enumerate(hi):
            s = axis_rows(res[res.algo == algo], ax).groupby(col)["wall_time_s"].median()
            if len(s) >= 2:
                a.plot(s.index, s.values, "o-", ms=2.5, lw=1.0, color=cmap(i % 20), label=algo, zorder=3)
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_xlabel({"n": "points n", "d": "features d", "k": "clusters k"}[ax])
        a.axhline(150, color="grey", lw=0.6, ls="--")
    axes[0].set_ylabel("wall time (s), fit_predict")
    axes[2].legend(fontsize=6, frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left",
                   title="grey: other algorithms", title_fontsize=6)
    fig.tight_layout()
    fig.savefig(args.out / "scaling_axes.pdf")
    plt.close(fig)

    # grid: algorithm x task, log10 wall time, DNF marked
    res["task"] = res.apply(lambda r: f"n={r.n_samples:,} d={r.n_features} k={r.k_target}", axis=1)
    order = (res[["task", "n_samples", "n_features", "k_target"]].drop_duplicates()
             .sort_values(["n_samples", "n_features", "k_target"]).task.tolist())
    if len(fails):
        fails["task"] = fails.apply(lambda r: f"n={r.n_samples:,} d={r.n_features} k={r.k_target}", axis=1)
        order = sorted(set(order) | set(fails.task), key=lambda t: [int(x.split("=")[1].replace(",", "")) for x in t.split()])
    M = np.full((len(algos), len(order)), np.nan)
    D = np.zeros_like(M, dtype=object)
    piv = res.groupby(["algo", "task"]).wall_time_s.median()
    for (a, t), v in piv.items():
        if a in algos and t in order:
            M[algos.index(a), order.index(t)] = math.log10(max(v, 1e-3))
    if len(fails):
        for _, f in fails.iterrows():
            if f.algo in algos and f.task in order:
                D[algos.index(f.algo), order.index(f.task)] = {"Timeout": "T", "MemoryError": "M", "capped": "c"}.get(f.status, "E")
    fig, ax = plt.subplots(figsize=(7, 0.16 * len(algos) + 1.5))
    im = ax.imshow(M, aspect="auto", cmap="magma_r", vmin=-2.5, vmax=math.log10(150))
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if D[i, j]:
                ax.text(j, i, D[i, j], ha="center", va="center", fontsize=5, color="tab:blue", weight="bold")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=60, ha="right", fontsize=5.5)
    ax.set_yticks(range(len(algos)))
    ax.set_yticklabels(algos, fontsize=5)
    cb = fig.colorbar(im, ax=ax, fraction=0.03)
    cb.set_label("log10 wall time (s); T=timeout M=memory c=capped")
    fig.tight_layout()
    fig.savefig(args.out / "scaling_grid.pdf")
    plt.close(fig)
    print(f"wrote {args.out}/scaling_table.md, scaling_axes.pdf, scaling_grid.pdf ({len(res)} runs, {len(fails)} DNF)")


if __name__ == "__main__":
    main()
