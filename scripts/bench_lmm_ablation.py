"""LMM novelty ablation against Sharma 2009 (SMM).

Compares LMM (exp-family + self-normalised EM + Newton + heat-kernel τ)
against SMM (GMM on the same Laplacian eigenvector basis, the closest
published prior art — Sharma et al. 2009) and plain spectral clustering
across the standard 18-config benchmark.

Run from repo root::

    python scripts/bench_lmm_ablation.py

Outcome on commit 3a84a1a (5 seeds × 18 configs)::

  algo        mean ARI  med ARI    NMI    mean t(ms)
  smm            0.924    0.984    0.921       50
  lmm            0.918    0.985    0.918       40
  spectral       0.868    0.911    0.897      166
  gmm            0.781    0.878    0.783       42
  kmeans         0.772    0.967    0.767       26

  LMM minus SMM:
    mean delta: -0.006  (SMM marginally ahead)
    LMM > SMM by >0.01:   0 / 18 configs
    SMM > LMM by >0.01:   2 / 18 configs
    tied (|Δ| ≤ 0.01):   16 / 18 configs

Reads as: the LMM advantage over plain ``spectral`` (k-means on the
Ng-Jordan-Weiss embedding) is fully explained by "EM > k-means on the
embedding" — a result already in Sharma 2009. The specific LMM design
choices (exp-family components, self-normalised log-partition, Newton
M-step, heat-kernel τ on Laplacian eigenvalues) match but do not beat
that baseline. LMM's novelty is small; the win over ``spectral`` is
real but published.
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score as ARI,
    normalized_mutual_info_score as NMI,
)

from clustbench.algorithms.base import ALGO_REGISTRY
from clustbench.datasets import DATASETS, DataSpec


ALGOS = ["kmeans", "gmm", "spectral", "smm", "lmm"]


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

    ari_all = {a: [] for a in ALGOS}
    nmi_all = {a: [] for a in ALGOS}
    t_all = {a: [] for a in ALGOS}

    print(
        f"{'dataset':12s}{'n':>5s}{'k':>3s}  "
        + "".join(f"{a:>9s}" for a in ALGOS),
        flush=True,
    )
    print("-" * (18 + 9 * len(ALGOS)), flush=True)
    for name, n, k, comp in grid:
        nf = 2 if name in ("moons", "circles", "anisotropic") else 5
        aris = {a: [] for a in ALGOS}
        nmis = {a: [] for a in ALGOS}
        for seed in (1, 2, 3, 4, 5):
            X, y = DATASETS[name](
                DataSpec(n_samples=n, n_features=nf, centers=k,
                         compactness=comp, seed=seed)
            )
            for a in ALGOS:
                try:
                    t0 = time.time()
                    res = ALGO_REGISTRY[a]().fit_predict(X, k=k)
                    t_all[a].append(time.time() - t0)
                    aris[a].append(ARI(y, res.labels))
                    nmis[a].append(NMI(y, res.labels))
                except Exception:
                    aris[a].append(float("nan"))
                    nmis[a].append(float("nan"))
        means = {a: np.nanmean(aris[a]) for a in ALGOS}
        for a in ALGOS:
            ari_all[a].append(means[a])
            nmi_all[a].append(np.nanmean(nmis[a]))
        print(
            f"{name:12s}{n:>5d}{k:>3d}  "
            + "".join(f"{means[a]:>9.3f}" for a in ALGOS),
            flush=True,
        )

    print(flush=True)
    print("=== Aggregate (5 seeds × 18 configs) ===", flush=True)
    print(
        f"{'algo':10s}{'mean ARI':>10s}{'med ARI':>9s}{'NMI':>9s}{'mean t(ms)':>12s}",
        flush=True,
    )
    for a in sorted(ALGOS, key=lambda a: -np.nanmean(ari_all[a])):
        print(
            f"{a:10s}{np.nanmean(ari_all[a]):>10.3f}"
            f"{np.nanmedian(ari_all[a]):>9.3f}"
            f"{np.nanmean(nmi_all[a]):>9.3f}"
            f"{1000 * np.mean(t_all[a]):>12.1f}",
            flush=True,
        )

    print(flush=True)
    print("=== LMM minus SMM (positive = LMM advantage) ===", flush=True)
    deltas = np.array(ari_all["lmm"]) - np.array(ari_all["smm"])
    print(
        f"mean delta: {deltas.mean():+.3f}, "
        f"median delta: {np.median(deltas):+.3f}",
        flush=True,
    )
    print(
        f"configs where LMM > SMM (delta > 0.01): "
        f"{int((deltas > 0.01).sum())}/{len(deltas)}",
        flush=True,
    )
    print(
        f"configs where SMM > LMM (delta < -0.01): "
        f"{int((deltas < -0.01).sum())}/{len(deltas)}",
        flush=True,
    )
    print(
        f"configs where tied (|delta| <= 0.01): "
        f"{int((np.abs(deltas) <= 0.01).sum())}/{len(deltas)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
