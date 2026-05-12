"""MWC vs consensus / pwcc ablation.

Compares the new Mann-Whitney consensus algorithm against the existing
ensemble methods on the standard 18-config benchmark with a fixed base
panel of (kmeans, gmm, lmm) — a diverse set covering centroid /
probabilistic / graph-based clustering.

Result on commit 3bf1c60 (3 seeds × 18 configs)::

  algo        mean ARI  med ARI    NMI    mean t(ms)
  lmm            0.933    0.992    0.928       29
  mwc            0.815    0.969    0.804      125
  pwcc           0.814    0.971    0.799      101
  consensus      0.814    0.971    0.799      104
  gmm            0.794    0.968    0.793       40
  kmeans         0.783    0.967    0.774       23

Headline finding: **all three ensembles are statistical ties at
~0.815**, and **all three are worse than the strongest base learner
(LMM at 0.933)**. Voting can only hurt when one of the voters is
strictly better than the others. The ensemble averages the strong
with the weak instead of trusting the strong — textbook ensemble
failure mode.

Where MWC's rank test does fire differently from plain consensus:

  dataset    n    k   lmm   consensus  pwcc  mwc
  circles  400    2  1.00      0.08    0.08  0.33   <-- MWC fires; right call
  moons    400    2  0.67      0.46    0.46  0.35   <-- MWC fires; wrong call
  circles  800    2  1.00      0.07    0.07  0.00   <-- MWC fires; wrong call

The Mann-Whitney gating shifts behaviour on non-convex shapes — for
better at circles@400, for worse at circles@800. The algorithm is
"working as designed" (the test detects when base algorithms
disagree and forces stricter merge gating) but the right test for it
is an ensemble of *peers* — multiple spectral-like algorithms that
agree roughly but fail on different configs — not a panel where one
algorithm dominates.

Run from repo root::

    python scripts/bench_mwc_ensembles.py
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score as ARI,
    normalized_mutual_info_score as NMI,
)

from clustbench.algorithms.base import ALGO_REGISTRY
from clustbench.algorithms.mwc import Mwc
from clustbench.consensus import Consensus
from clustbench.datasets import DATASETS, DataSpec


BASE_PANEL = ["kmeans", "gmm", "lmm"]


def make_consensus() -> Consensus:
    return Consensus(base=BASE_PANEL)


def make_mwc() -> Mwc:
    return Mwc(base=BASE_PANEL)


VARIANTS = {
    "kmeans": lambda: ALGO_REGISTRY["kmeans"](),
    "gmm": lambda: ALGO_REGISTRY["gmm"](),
    "lmm": lambda: ALGO_REGISTRY["lmm"](),
    "consensus": make_consensus,
    "pwcc": lambda: ALGO_REGISTRY["pwcc"](base=BASE_PANEL),
    "mwc": make_mwc,
}


def main() -> None:
    grid = []
    for n in (400, 800):
        for k, comp in [(3, 0.3), (5, 0.5), (8, 0.7)]:
            grid.append(("blobs", n, k, comp))
        for k, comp in [(3, 0.5), (5, 0.5), (8, 0.5)]:
            grid.append(("mdcgen", n, k, comp))
        grid.append(("moons", n, 2, 1.0))
        grid.append(("circles", n, 2, 1.0))
        grid.append(("anisotropic", n, 3, 1.0))

    names = list(VARIANTS.keys())
    ari_all = {a: [] for a in names}
    nmi_all = {a: [] for a in names}
    t_all = {a: [] for a in names}

    print(
        f"{'dataset':12s}{'n':>5s}{'k':>3s}  "
        + "".join(f"{a:>11s}" for a in names),
        flush=True,
    )
    print("-" * (20 + 11 * len(names)), flush=True)
    for name, n, k, comp in grid:
        nf = 2 if name in ("moons", "circles", "anisotropic") else 5
        aris = {a: [] for a in names}
        nmis = {a: [] for a in names}
        for seed in (1, 2, 3):
            X, y = DATASETS[name](
                DataSpec(n_samples=n, n_features=nf, centers=k,
                         compactness=comp, seed=seed)
            )
            for a in names:
                try:
                    t0 = time.time()
                    res = VARIANTS[a]().fit_predict(X, k=k)
                    t_all[a].append(time.time() - t0)
                    aris[a].append(ARI(y, res.labels))
                    nmis[a].append(NMI(y, res.labels))
                except Exception:
                    aris[a].append(float("nan"))
                    nmis[a].append(float("nan"))
        means = {a: np.nanmean(aris[a]) for a in names}
        for a in names:
            ari_all[a].append(means[a])
            nmi_all[a].append(np.nanmean(nmis[a]))
        print(
            f"{name:12s}{n:>5d}{k:>3d}  "
            + "".join(f"{means[a]:>11.3f}" for a in names),
            flush=True,
        )

    print(flush=True)
    print("=== Aggregate (3 seeds × 18 configs) ===", flush=True)
    print(
        f"{'algo':12s}{'mean ARI':>10s}{'med ARI':>9s}"
        f"{'NMI':>9s}{'mean t(ms)':>12s}",
        flush=True,
    )
    for a in sorted(names, key=lambda a: -np.nanmean(ari_all[a])):
        print(
            f"{a:12s}"
            f"{np.nanmean(ari_all[a]):>10.3f}"
            f"{np.nanmedian(ari_all[a]):>9.3f}"
            f"{np.nanmean(nmi_all[a]):>9.3f}"
            f"{1000 * np.mean(t_all[a]):>12.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
