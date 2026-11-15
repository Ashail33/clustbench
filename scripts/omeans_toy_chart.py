"""Recreate the k-means / e-means / o-means toy chart from clustbench's own runs.

Top: which centroids one point is compared with under each method. Bottom: mean squared error against distance
computations (1 = one full k-means pass over the data). 12,000 2-D points, k = 64, batch 256, 3 epochs.

    python scripts/omeans_toy_chart.py [--out docs/data_omeans/toy_chart.png]
"""
from __future__ import annotations

import argparse

import numpy as np
from sklearn.datasets import make_blobs

from clustbench.algorithms import o_means as om

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"           # categorical slots 1-3 (validated all-pairs)


def lloyd(X, k, passes, seed=0):
    """Lloyd's k-means from random rows; error after each full pass, and the centroids after each."""
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)].copy()
    mse = lambda C: float(((X - C[om.exact_assign(X, C)]) ** 2).sum(1).mean())
    errs, states = [mse(C)], [C.copy()]
    for _ in range(passes):
        lab = om.exact_assign(X, C)
        for j in range(k):
            if (lab == j).any():
                C[j] = X[lab == j].mean(0)
        errs.append(mse(C)); states.append(C.copy())
    return errs, states


def panel(ax, X, C, p, targets, colour, title, sub, foot, cells=None):
    ax.set_facecolor("#f3f2ee")
    ax.scatter(X[:, 0], X[:, 1], s=0.6, c="#b9b8b2", linewidths=0, rasterized=True)
    if cells is not None:                                        # o-means: lines to the cell centres first
        for c in cells:
            ax.plot([p[0], c[0]], [p[1], c[1]], color=MUTED, lw=0.6, ls=(0, (2, 2)), zorder=2)
        ax.scatter(cells[:, 0], cells[:, 1], s=22, marker="s", facecolor=SURFACE, edgecolor=MUTED, lw=1, zorder=3)
    for c in targets:
        ax.plot([p[0], c[0]], [p[1], c[1]], color=INK, lw=0.5, alpha=0.75, zorder=2)
    ax.scatter(C[:, 0], C[:, 1], s=26, c=colour, edgecolor=SURFACE, lw=1.2, zorder=4)
    ax.scatter([p[0]], [p[1]], s=60, c=INK, edgecolor=SURFACE, lw=1.5, zorder=5)
    n_checks = len(targets) + (0 if cells is None else len(cells))
    ax.text(0.03, 0.03, f"one point, {n_checks} checks", transform=ax.transAxes, fontsize=8.5, color=INK2)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold", color=INK, pad=34)
    ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=8.5, color=INK2, va="bottom", linespacing=1.4)
    ax.text(0, -0.07, foot, transform=ax.transAxes, fontsize=9.5, color=INK, va="top")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/data_omeans/toy_chart.png")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    X, _ = make_blobs(12000, centers=12, cluster_std=1.2, center_box=(-15, 15), random_state=3)
    n, k, b, epochs = len(X), 64, 256, 3
    l_err, l_states = lloyd(X, k, 6)
    e_tr, o_tr = [], []
    e = om.omeans(X, k, epochs=epochs, batch_size=b, two_level=False, random_state=0, trace=e_tr)
    o = om.omeans(X, k, epochs=epochs, batch_size=b, two_level=True, two_level_min_k=0, nprobe=1,
                  random_state=0, trace=o_tr)
    updates = int(np.ceil(n / b)) * epochs
    cells = om._CellIndex(o["centroids"], 16, 1, np.random.default_rng(0))

    p = X[np.argmin(((X - X.mean(0)) ** 2).sum(1))]               # the highlighted point: nearest the data centre
    home = int(np.argmin(((cells.centres - p) ** 2).sum(1)))
    o_targets = o["centroids"][cells.members[home]]

    fig = plt.figure(figsize=(13, 11), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.9], hspace=0.42, wspace=0.12, left=0.05, right=0.97, top=0.83, bottom=0.15)
    fig.text(0.05, 0.955, "k-means vs e-means vs o-means", fontsize=22, fontweight="bold", color=INK)
    fig.text(0.05, 0.928, "Same data, same k, same compute budget. Who gets to a good answer first?", fontsize=12, color=INK2)
    panel(fig.add_subplot(gs[0, 0]), X, l_states[3], p, l_states[3], BLUE, "k-means",
          "One update per full pass.\nEach point checks all 64 centroids.",
          f"After 3 passes: 3 updates, error {l_err[3]:.3f}")
    panel(fig.add_subplot(gs[0, 1]), X, e["centroids"], p, e["centroids"], ORANGE, "e-means",
          "Updates after every mini-batch.\nEach point checks all 64 centroids.",
          f"Done: {epochs} passes, {updates} updates, error {e['objective']:.3f}")
    panel(fig.add_subplot(gs[0, 2]), X, o["centroids"], p, o_targets, AQUA, "o-means",
          "Same updates as e-means.\nEach point checks 16 cell centres, then one cell.",
          f"Done: {o['exact_pass_equivalent']:.2f} passes, {updates} updates, error {o['objective']:.3f}",
          cells=cells.centres)

    ax = fig.add_subplot(gs[1, :])
    ax.set_facecolor(SURFACE)
    ax.step(range(7), l_err, where="post", color=BLUE, lw=2)
    ex, ey = np.array(e_tr).T; ox, oy = np.array(o_tr).T
    ax.plot(ex, ey, color=ORANGE, lw=2); ax.plot(ox, oy, color=AQUA, lw=2)
    for (x, y, c, name, dy) in ((6, l_err[6], BLUE, "k-means", 0.012), (ex[-1], ey[-1], ORANGE, "e-means", -0.03),
                                (ox[-1], oy[-1], AQUA, "o-means", -0.03)):
        ax.scatter([x], [y], s=40, c=c, edgecolor=SURFACE, lw=1.5, zorder=5)
        ax.text(x, y + dy, name, color=INK, fontsize=10, fontweight="bold", ha="center")
    ax.set_xlim(-0.05, 6.2); ax.set_ylim(0.65, 1.25)
    ax.set_xlabel("Distance calculations (1 = one full k-means pass over the data)", color=INK2, fontsize=10)
    ax.set_ylabel("Mean squared error", color=INK2, fontsize=10)
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_title("Clustering error vs compute spent", loc="left", fontsize=12, fontweight="bold", color=INK)
    ax.text(1, 1.02, "toy 2-D demo: 12,000 points, k = 64", transform=ax.transAxes, ha="right", fontsize=9, color=MUTED)
    # footnote numbers: both methods over 3 seeds, each curve read at exactly 1 pass of compute
    at1 = {"e": [], "o": []}
    for seed in range(3):
        for key, kw in (("e", dict(two_level=False)), ("o", dict(two_level=True, two_level_min_k=0, nprobe=1))):
            tr = []
            om.omeans(X, k, epochs=epochs, batch_size=b, random_state=seed, trace=tr, **kw)
            tx, ty = np.array(tr).T
            at1[key].append(np.interp(min(1.0, tx[-1]), tx, ty))
    l25 = lloyd(X, k, 25)[0][-1]
    fig.text(0.05, 0.025, "Reproduced with clustbench's CPU re-implementation of o-means (scripts/omeans_toy_chart.py): "
             f"batch {b}, {epochs} epochs; k-means is Lloyd from random rows; curves are seed 0.\n"
             f"e-means passes k-means' 25-pass error ({l25:.3f}) within {epochs} passes. At 1 pass of compute, mean of 3 seeds: "
             f"e-means {np.mean(at1['e']):.3f}, o-means {np.mean(at1['o']):.3f}, k-means {l_err[1]:.3f}.",
             fontsize=9, color=INK2, linespacing=1.5)
    fig.savefig(a.out, dpi=150, facecolor=SURFACE)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
