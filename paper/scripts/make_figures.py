"""Regenerate every figure used by the two papers in ``paper/``.

All numbers come from the committed dashboard data, so the figures are
reproducible from a clean checkout without re-running the benchmark:

    python paper/scripts/make_figures.py

Inputs
------
docs/data/results.json          in-distribution benchmark (61 algorithms)
docs/data_unseen/results.json   out-of-distribution benchmark (62 algorithms)
docs/data/summary.json          average ranks + Friedman test
docs/data/trajectories.json     per-step trajectory records
docs/data_bigdata/              big-data scaling sweep (scaling_axes.pdf)
docs/data_shapes/               every algorithm on every dataset
docs/data_fastgraph/            graph clustering at n = 20k / 100k

The manifold and gallery figures recompute labels / eigenvectors from the
generators (same data and seeds as the benchmark).

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
MCG = {"mcg", "mcg_cf", "lmm_mcg", "lmm_mcg_dual"}


def colour(a: str) -> str:
    return "#c0504d" if a in ROUTERS else "#2e8b57" if a in MCG else "#4f81bd"


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
    colors = [colour(n) for n in names]
    fig, ax = plt.subplots(figsize=(3.4, 7.2))
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
        handles=[Patch(color="#c0504d", label="learned router"), Patch(color="#2e8b57", label="MCG family"),
                 Patch(color="#4f81bd", label="other")],
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
        "tbmm_fl",
        "mcg_cf",
        "lmm_mcg",
        "lmm_mcg_dual",
        "learned_router",
        "learned_router_v3",
    ]
    datasets = sorted({d for d, _ in pd_}, key=str.lower)
    M = np.array([[pd_.get((d, a), np.nan) for a in algos] for d in datasets])
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
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
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=4.0,
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
        ax.scatter(a_in[a], a_ood[a], s=14, c=colour(a),
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
        "lmm_mcg": ("lmm_mcg", (-34, 3)),
        "mcg_cf": ("mcg_cf", (5, -2)),
        "lmm_mcg_dual": ("lmm_mcg_dual", (5, -3)),
        "mcg": ("mcg", (-20, -7)),
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
        col = colour(a)
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


def fig_bigdata_scaling() -> None:
    """The big-data sweep's wall time against n, d and k (from scaling_report.py)."""
    import shutil

    src = ROOT / "docs/data_bigdata/scaling_axes.pdf"
    if src.exists():
        shutil.copy(src, OUT / "bigdata_axes.pdf")


SHAPE_GROUPS = {
    "curved shapes": ["moons", "circles", "rings", "spiral", "s_curve", "swiss_roll"],
    "mixtures": ["mixed_shapes", "mixed", "VariableDensityBridges", "hierarchical_nested",
                 "heavy_tailed_mixture", "PowerLawStudentT"],
    "density /\nimbalance /\noutliers": ["varying_density", "imbalanced", "imbalanced_blobs", "extreme_outliers"],
    "convex blobs": ["blobs", "anisotropic", "mdcgen", "inverse_pca"],
    "real data": ["iris", "wine", "breast_cancer", "digits", "ecoli", "glass", "yeast", "vehicle", "segment",
                  "olivetti_faces", "text20news", "ts_ecg200", "ts_trace", "graph_karate"],
}


def fig_shapes_groups() -> None:
    """Mean ARI per dataset group for selected algorithms (DNF counts as 0)."""
    p = ROOT / "docs/data_shapes/results.json"
    if not p.exists():
        return
    rows = json.load(open(p))
    fails = json.load(open(ROOT / "docs/data_shapes/failures.json")) if (ROOT / "docs/data_shapes/failures.json").exists() else []
    vals = defaultdict(list)
    fam = {f: g for g, fs in SHAPE_GROUPS.items() for f in fs}
    for r in rows:
        if r["dataset_id"] in fam:
            vals[(r["algo"], fam[r["dataset_id"]])].append(r["ari"] if r["ari"] is not None else 0.0)
    for f in fails:
        if f["dataset_id"] in fam:
            vals[(f["algo"], fam[f["dataset_id"]])].append(0.0)
    algos = ["kmeans", "gmm", "agglomerative", "birch_algo", "dbscan_auto", "spectral", "lmm", "tbmm_fl",
             "mcg", "mcg_cf", "lmm_mcg", "lmm_mcg_dual", "learned_router_v3"]
    groups = list(SHAPE_GROUPS)
    M = np.array([[np.mean(vals[(a, g)]) if vals[(a, g)] else np.nan for g in groups] for a in algos])
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, fontsize=6.5)
    ax.set_yticks(range(len(algos)))
    ax.set_yticklabels(algos, fontsize=6.5)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.5,
                    color="white" if v < 0.55 else "black",
                    weight="bold" if v >= np.nanmax(M[:, j]) - 1e-9 else "normal")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label("mean ARI")
    fig.tight_layout()
    fig.savefig(OUT / "shapes_groups.pdf")
    plt.close(fig)


def fig_shapes_gallery() -> None:
    """Six 2-D shapes coloured by the truth and by seven algorithms (seed 1)."""
    import warnings

    from sklearn.metrics import adjusted_rand_score
    from threadpoolctl import threadpool_limits

    from clustbench.algorithms.base import ALGO_REGISTRY
    from clustbench.datasets import DATASETS, DataSpec

    sets = [("moons", 2, 2), ("rings", 2, 5), ("spiral", 2, 3), ("swiss_roll", 3, 4),
            ("mixed_shapes", 2, 5), ("VariableDensityBridges", 2, 4)]
    cols = ["truth", "kmeans", "gmm", "spectral", "lmm", "mcg", "lmm_mcg_dual", "learned_router_v3"]
    fig, axes = plt.subplots(len(sets), len(cols), figsize=(7.0, 5.6))
    with warnings.catch_warnings(), threadpool_limits(1):
        warnings.simplefilter("ignore")
        for i, (name, d, k) in enumerate(sets):
            X, y = DATASETS[name](DataSpec(n_samples=2000, n_features=d, centers=k, compactness=1.0, seed=1))
            X = np.asarray(X, float)
            P = X[:, [0, 2]] if X.shape[1] == 3 else X[:, :2]
            lo, hi = np.percentile(P, 1, axis=0), np.percentile(P, 99, axis=0)
            pad = 0.08 * (hi - lo)
            for j, c in enumerate(cols):
                ax = axes[i, j]
                if c == "truth":
                    lab, title = y, "truth"
                else:
                    lab = ALGO_REGISTRY[c]().fit_predict(X, k=k).labels
                    title = f"{c.replace('learned_router_v3', 'router v3')}\n{adjusted_rand_score(y, lab):.2f}"
                ax.scatter(P[:, 0], P[:, 1], c=lab, s=0.4, cmap="tab10", linewidths=0, rasterized=True)
                ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
                ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
                ax.set_xticks([])
                ax.set_yticks([])
                for sp_ in ax.spines.values():
                    sp_.set_visible(True)
                    sp_.set_linewidth(0.3)
                ax.set_title(title, fontsize=5.5, pad=1.5)
                if j == 0:
                    ax.set_ylabel(name.replace("VariableDensityBridges", "density bridges"), fontsize=6)
    fig.tight_layout(pad=0.3, h_pad=0.4, w_pad=0.2)
    fig.savefig(OUT / "shapes_gallery.pdf", dpi=220)
    plt.close(fig)


def fig_manifold() -> None:
    """Swiss roll: the first three non-trivial eigenvectors of each basis
    against position along the roll, with the cosine modes cos(m pi s)."""
    import warnings

    from sklearn.datasets import make_swiss_roll
    from sklearn.neighbors import NearestNeighbors

    from clustbench.algorithms.base import ALGO_REGISTRY
    from clustbench.datasets import DATASETS, DataSpec

    X, _ = DATASETS["swiss_roll"](DataSpec(n_samples=2000, n_features=3, centers=4, compactness=1.0, seed=1))
    X = np.asarray(X, float)
    X3, t0 = make_swiss_roll(n_samples=2000, noise=0.05, random_state=1)
    _, idx = NearestNeighbors(n_neighbors=1).fit(X3).kneighbors(X)
    s = np.argsort(np.argsort(t0[idx[:, 0]])) / len(X)
    bases = [("LMM (graph on all points)", ALGO_REGISTRY["lmm"](row_normalize=False)),
             ("lmm_mcg (CF-weighted micro-cluster graph)", ALGO_REGISTRY["lmm_mcg"](row_normalize=False)),
             ("count basis (coarse-grained Laplacian)", ALGO_REGISTRY["lmm_mcg"](basis="count", row_normalize=False))]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2), sharey=True)
    order = np.argsort(s)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for ax, (title, alg) in zip(axes, bases):
            alg._k_hint, alg.n_frequencies = 4, 4
            _, V, _ = alg._build_basis(X, np.random.default_rng(42))
            for m, colr in zip((1, 2, 3), ("#1f77b4", "#ff7f0e", "#2ca02c")):
                v = V[:, m]
                cm = np.cos(m * np.pi * s)
                v = v * np.sign(np.corrcoef(v, cm)[0, 1])
                v = (v - v.mean()) / (v.std() + 1e-12) * cm.std()
                r = abs(np.corrcoef(V[:, m], cm)[0, 1])
                ax.plot(s[order], v[order], ".", ms=0.6, color=colr, alpha=0.5, rasterized=True)
                ax.plot(s[order], cm[order], "-", lw=0.6, color="k", alpha=0.6)
                ax.text(0.03, 0.02 + 0.1 * (3 - m), f"mode {m}: |r| = {r:.2f}", transform=ax.transAxes,
                        ha="left", va="bottom", fontsize=5.2, color=colr,
                        bbox=dict(facecolor="white", edgecolor="none", pad=0.5, alpha=0.85))
            ax.set_title(title, fontsize=6.5)
            ax.set_xlabel("position along the roll, s")
    axes[0].set_ylabel("eigenvector (scaled)")
    fig.tight_layout()
    fig.savefig(OUT / "manifold_eigvecs.pdf", dpi=220)
    plt.close(fig)


def fig_largen() -> None:
    """Graph clustering at n = 100k: mean ARI against median wall time."""
    p = ROOT / "docs/data_fastgraph/results_all.json"
    if not p.exists():
        return
    rows = json.load(open(p))
    acc, tim = defaultdict(list), defaultdict(list)
    for r in rows:
        if r["n_samples"] == 100000 and not r["algo"].endswith("_lobpcg"):
            acc[r["algo"]].append(r["ari"])
            tim[r["algo"]].append(r["wall_time_s"])
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    off = {"lmm_mcg": (-12, 7), "mcg_cf": (-8, -10), "mcg": (-4, -10), "lmm_mcg_dual": (4, -9),
           "lmm_fast": (4, 2), "spectral_fast": (4, -8), "lmm": (-10, -9), "gmm": (-4, -9),
           "minibatch_kmeans": (2, 5), "trident": (4, 2)}
    for a in acc:
        x, y = st.median(tim[a]), float(np.mean(acc[a]))
        ax.scatter(x, y, s=16, c=colour(a), zorder=3)
        ax.annotate(a, (x, y), fontsize=5.5, xytext=off.get(a, (4, 2)), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("median wall time at n = 100,000 (s)")
    ax.set_ylabel("mean ARI (11 families, 2 seeds)")
    fig.tight_layout()
    fig.savefig(OUT / "largen.pdf")
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
    fig_bigdata_scaling()
    fig_shapes_groups()
    fig_largen()
    fig_manifold()
    fig_shapes_gallery()
    print("wrote", sorted(p.name for p in OUT.glob("*.pdf")))


if __name__ == "__main__":
    main()
