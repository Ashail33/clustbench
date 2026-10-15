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
        "lmm": ("lmm / spectral", (-58, 5)),
        "learned_router": ("v1", (5, -2)),
        "learned_router_v3": ("v3/v5/v7", (5, 2)),
        "learned_router_v4": ("v4", (5, -5)),
        "learned_router_v2": ("v2", (5, -6)),
        "louvain_knn": ("louvain_knn", (-50, 3)),
        "aura_v3": ("aura_v2/v3", (5, -9)),
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
    """Router generations on three evaluations: in-distribution (task-level
    leave-one-out), leave-one-dataset-family-out, and the unseen split."""
    rows_in = load("docs/data/results.json")
    rows_ood = load("docs/data_unseen/results.json")
    a_in, a_ood = mean_ari(rows_in), mean_ari(rows_ood)
    lofo_path = ROOT / "docs" / "data" / "router_lofo.json"
    lofo = json.load(open(lofo_path))["summary"] if lofo_path.exists() else {}
    gens = ["learned_router", "learned_router_v2", "learned_router_v3", "learned_router_v4",
            "learned_router_v5", "learned_router_v6", "learned_router_v6b",
            "learned_router_v6c", "learned_router_v7", "learned_router_v8"]
    short = ["v1", "v2", "v3", "v4", "v5", "v6", "v6b", "v6c", "v7", "v8"]
    x = np.arange(len(gens))
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.plot(x, [a_in.get(g, np.nan) for g in gens], "o-", ms=3, lw=0.8, c="#c0504d",
            label="in-distribution")
    ax.plot(x, [(lofo.get(g) or {}).get("lofo_mean_ari", np.nan) for g in gens], "^:", ms=3,
            lw=0.8, c="#8064a2", label="leave-one-family-out")
    ax.plot(x, [a_ood.get(g, np.nan) for g in gens], "s--", ms=3, lw=0.8, c="#4f81bd",
            label="unseen split")
    col = "#9bbb59"
    ax.axhline(a_in["lmm"], c=col, lw=0.6, ls=":")
    ax.axhline(a_ood["lmm"], c=col, lw=0.6, ls="-.")
    ax.text(-0.4, a_ood["lmm"] + 0.004, "lmm (unseen)", fontsize=5.5, color=col)
    ax.text(-0.4, a_in["lmm"] + 0.004, "lmm (in-dist.)", fontsize=5.5, color=col)
    ax.set_xticks(x)
    ax.set_xticklabels(short)
    ax.set_ylabel("mean ARI")
    ax.set_ylim(0.55, 0.9)
    ax.legend(frameon=False, loc="lower right", fontsize=5.5)
    fig.tight_layout()
    fig.savefig(OUT / "router_lineage.pdf")
    plt.close(fig)


def fig_ci_forest(summary: dict, top: int = 25) -> None:
    """Mean ARI with configuration-level bootstrap 95% CIs, plus the
    Nemenyi critical difference on the average-rank axis."""
    post = summary.get("ari", {}).get("posthoc", {})
    ci = post.get("bootstrap_ci")
    if not ci:
        return
    ranks = summary["ari"]["average_ranks"]
    order = [r["algo"] for r in ranks][:top]
    rk = {r["algo"]: r["rank"] for r in ranks}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 4.6), sharey=True,
                                   gridspec_kw={"width_ratios": [1.1, 1]})
    y = np.arange(len(order))[::-1]
    for yi, a in zip(y, order):
        c = ci.get(a)
        if not c:
            continue
        col = "#c0504d" if a in ROUTERS else "#4f81bd"
        ax1.plot([c["ci_low"], c["ci_high"]], [yi, yi], c=col, lw=1.2)
        ax1.plot(c["mean"], yi, "o", c=col, ms=3)
        ax2.plot(rk[a], yi, "s", c=col, ms=3)
    ax1.set_yticks(y)
    ax1.set_yticklabels(order, fontsize=5.8)
    ax1.set_xlabel("mean ARI (95% bootstrap CI over configurations)")
    cd = post.get("nemenyi_cd")
    best = rk[order[0]]
    if cd:
        ax2.axvspan(best, best + cd, color="grey", alpha=0.15, lw=0)
        ax2.text(best + cd / 2, y[0] + 0.9, f"Nemenyi CD = {cd:.1f}", ha="center", fontsize=6)
    ax2.set_xlabel("average ARI rank")
    fig.tight_layout()
    fig.savefig(OUT / "ci_forest.pdf")
    plt.close(fig)


def fig_scaling(path: str = "docs/data_scaling/results.json") -> None:
    p = ROOT / path
    if not p.exists():
        return
    rows = json.load(open(p))
    by = defaultdict(list)
    for r in rows:
        if r.get("wall_time_s") is not None:
            by[(r["algo"], r["dataset_id"], r["n_samples"])].append(r["wall_time_s"])
    algos = sorted({a for a, _, _ in by})
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8), sharey=True)
    cmap = plt.get_cmap("tab20")
    for ax, ds in zip(axes, ("mdcgen", "moons")):
        for i, a in enumerate(algos):
            pts = sorted((n, st.median(v)) for (aa, d, n), v in by.items() if aa == a and d == ds)
            if len(pts) < 2:
                continue
            ax.plot(*zip(*pts), "o-", ms=2, lw=0.8, color=cmap(i % 20), label=a)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("n")
        ax.set_title(ds)
    axes[0].set_ylabel("wall time (s)")
    axes[1].legend(fontsize=5, frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "scaling.pdf")
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
    fig_ci_forest(summary)
    fig_scaling()
    print("wrote", sorted(p.name for p in OUT.glob("*.pdf")))


if __name__ == "__main__":
    main()
