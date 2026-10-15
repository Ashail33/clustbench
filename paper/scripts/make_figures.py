"""Regenerate every figure used by the two papers in ``paper/``.

All numbers come from the committed dashboard data, so the figures are
reproducible from a clean checkout without re-running the benchmark:

    python paper/scripts/make_figures.py

Inputs
------
docs/data/results.json          in-distribution benchmark (47 algorithms)
docs/data_unseen/results.json   out-of-distribution benchmark (42 algorithms)
docs/data/summary.json          average ranks + Friedman test
docs/data/trajectories.json     per-step trajectory records

Outputs (PDF) go to ``paper/figures/``.
"""
from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

ROUTERS = {
    "learned_router",
    "learned_router_v2",
    "learned_router_v3",
    "learned_router_v4",
    "learned_router_v5",
    "learned_router_v6",
    "learned_router_v6b",
    "learned_router_v6c",
    "learned_router_v7",
    "learned_router_v8",
}


def load(path: str) -> list[dict]:
    return json.load(open(ROOT / path))


def mean_ari(rows: list[dict]) -> dict[str, float]:
    by = defaultdict(list)
    for r in rows:
        if r.get("ari") is not None:
            by[r["algo"]].append(r["ari"])
    return {a: st.mean(v) for a, v in by.items()}


def per_dataset(rows: list[dict]) -> dict[tuple[str, str], float]:
    by = defaultdict(list)
    for r in rows:
        if r.get("ari") is not None:
            by[(r["dataset_id"], r["algo"])].append(r["ari"])
    return {k: st.mean(v) for k, v in by.items()}


def fig_rank_bars(summary: dict) -> None:
    ranks = summary["ari"]["average_ranks"]
    names = [r["algo"] for r in ranks]
    vals = [r["rank"] for r in ranks]
    colors = ["#c0504d" if n in ROUTERS else "#4f81bd" for n in names]
    fig, ax = plt.subplots(figsize=(3.4, 6.2))
    y = np.arange(len(names))[::-1]
    ax.barh(y, vals, color=colors, height=0.75)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=5.5)
    ax.set_xlabel("average ARI rank (lower is better)")
    fr = summary["ari"]["friedman"]
    ax.set_title(
        f"Friedman $\\chi^2$={fr['statistic']:.0f}, p={fr['p_value']:.1e}\n"
        f"{fr['k']} algorithms, {fr['n_tasks']} complete tasks"
    )
    from matplotlib.patches import Patch

    ax.legend(
        handles=[Patch(color="#c0504d", label="learned router"), Patch(color="#4f81bd", label="other")],
        loc="upper right",
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(OUT / "rank_bars.pdf")
    plt.close(fig)


def fig_heatmap(rows: list[dict]) -> None:
    pd_ = per_dataset(rows)
    algos = [
        "kmeans",
        "agglomerative",
        "birch_algo",
        "gmm",
        "spectral",
        "dbscan_auto",
        "meanshift_robust",
        "pwcc_diverse",
        "lmm",
        "fmm",
        "louvain_knn",
        "rapid",
        "aura_v3",
        "trident",
        "learned_router",
        "learned_router_v3",
    ]
    datasets = sorted({d for d, _ in pd_}, key=str.lower)
    M = np.array([[pd_.get((d, a), np.nan) for a in algos] for d in datasets])
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(algos)))
    ax.set_xticklabels(algos, rotation=60, ha="right")
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(datasets)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=4.6,
                    color="white" if v < 0.55 else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cb.set_label("mean ARI over seeds")
    fig.tight_layout()
    fig.savefig(OUT / "heatmap.pdf")
    plt.close(fig)


def fig_ood(rows_in: list[dict], rows_ood: list[dict]) -> None:
    a_in, a_ood = mean_ari(rows_in), mean_ari(rows_ood)
    common = sorted(set(a_in) & set(a_ood))
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    for a in common:
        is_r = a in ROUTERS
        ax.scatter(a_in[a], a_ood[a], s=14, c="#c0504d" if is_r else "#4f81bd",
                   marker="s" if is_r else "o", zorder=3)
    label = {
        "learned_router": ("router v1", (6, 0)),
        "learned_router_v3": ("v3/v5/v7", (-6, 8)),
        "learned_router_v4": ("v4", (5, -8)),
        "learned_router_v2": ("v2", (5, -3)),
        "lmm": ("lmm", (-4, 6)),
        "louvain_knn": ("louvain_knn", (-50, -4)),
        "aura_v3": ("aura_v2/v3", (-46, 5)),
        "spectral": ("spectral", (5, -3)),
        "meta_clusterer_v2": ("meta_clusterer_v2/v3", (5, -3)),
        "agglomerative": ("agglomerative", (-52, 3)),
    }
    for a, (txt, off) in label.items():
        if a in a_in and a in a_ood:
            ax.annotate(txt, (a_in[a], a_ood[a]), fontsize=5.5, xytext=off,
                        textcoords="offset points")
    n_out = sum(1 for a in common if min(a_in[a], a_ood[a]) < 0.5)
    ax.text(0.51, 0.505, f"{n_out} algorithms below 0.5 not shown", fontsize=5.5, color="grey")
    lim = (0.5, 0.95)
    ax.plot(lim, lim, ls="--", lw=0.6, c="grey")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("mean ARI, in-distribution benchmark")
    ax.set_ylabel("mean ARI, unseen benchmark")
    fig.tight_layout()
    fig.savefig(OUT / "ood_scatter.pdf")
    plt.close(fig)


def fig_trajectories(traj: list[dict]) -> None:
    want = [
        ("kmeans", "mdcgen", 5, 100),
        ("clarans", "mdcgen", 5, 100),
        ("clarans_pp", "mdcgen", 5, 100),
    ]
    fig, axes = plt.subplots(1, len(want), figsize=(6.8, 1.8))
    for ax, (algo, ds, k, outl) in zip(axes, want):
        cands = [
            t for t in traj
            if t["algo"] == algo and t["dataset_id"] == ds and t.get("k_target") == k
            and (t.get("outliers") or 0) == outl and t.get("seed") == 1 and t["steps"]
        ]
        if not cands:
            cands = [t for t in traj if t["algo"] == algo and t["steps"]]
        if not cands:
            ax.set_visible(False)
            continue
        t = cands[0]
        s = [p["step_idx"] for p in t["steps"]]
        c = [p["cost"] for p in t["steps"]]
        acc = [p.get("accepted", True) for p in t["steps"]]
        ax.plot(s, c, lw=0.8, c="#4f81bd")
        rej = [(si, ci) for si, ci, a in zip(s, c, acc) if not a]
        if rej:
            ax.scatter(*zip(*rej), s=3, c="#c0504d", zorder=3, label="rejected")
        ax.set_title(f"{algo} on {t['dataset_id']} (k={t.get('k_target')})")
        ax.set_xlabel("step")
    axes[0].set_ylabel("objective (cost)")
    fig.tight_layout()
    fig.savefig(OUT / "trajectories.pdf")
    plt.close(fig)


def fig_router_lineage() -> None:
    # Mean ARI of each router generation at the round it was introduced and in
    # the final snapshot, plus the unseen benchmark. Sources: the in-tree
    # docs/ALGORITHM_ANALYSIS.md tables (rounds 4-12) and the committed data.
    rows_in = load("docs/data/results.json")
    rows_ood = load("docs/data_unseen/results.json")
    a_in, a_ood = mean_ari(rows_in), mean_ari(rows_ood)
    gens = ["learned_router", "learned_router_v2", "learned_router_v3", "learned_router_v4",
            "learned_router_v5", "learned_router_v6", "learned_router_v6b",
            "learned_router_v6c", "learned_router_v7", "learned_router_v8"]
    short = ["v1", "v2", "v3", "v4", "v5", "v6", "v6b", "v6c", "v7", "v8"]
    x = np.arange(len(gens))
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    ax.plot(x, [a_in.get(g, np.nan) for g in gens], "o-", ms=3, lw=0.8, c="#c0504d",
            label="in-distribution")
    ax.plot(x, [a_ood.get(g, np.nan) for g in gens], "s--", ms=3, lw=0.8, c="#4f81bd",
            label="unseen")
    for ref, col in (("lmm", "#9bbb59"),):
        ax.axhline(a_in[ref], c=col, lw=0.6, ls=":")
        ax.axhline(a_ood[ref], c=col, lw=0.6, ls="-.")
        ax.text(len(gens) - 0.6, a_ood[ref] + 0.005, "lmm (unseen)", fontsize=5.5, color=col, ha="right")
        ax.text(len(gens) - 0.6, a_in[ref] - 0.017, "lmm (in-dist.)", fontsize=5.5, color=col, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(short)
    ax.set_ylabel("mean ARI")
    ax.set_ylim(0.6, 0.95)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "router_lineage.pdf")
    plt.close(fig)


def main() -> None:
    rows_in = load("docs/data/results.json")
    rows_ood = load("docs/data_unseen/results.json")
    summary = load("docs/data/summary.json")
    traj = load("docs/data/trajectories.json")
    fig_rank_bars(summary)
    fig_heatmap(rows_in)
    fig_ood(rows_in, rows_ood)
    fig_trajectories(traj)
    fig_router_lineage()
    print("wrote", sorted(p.name for p in OUT.glob("*.pdf")))


if __name__ == "__main__":
    main()
