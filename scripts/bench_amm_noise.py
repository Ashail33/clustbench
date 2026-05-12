"""Sweep the noise-robustness of every clustering algo on high-d data.

Tests the AMM hypothesis: when most feature dimensions are noise,
the autoencoder bottleneck learns to project away the irrelevant axes
while k-NN graph methods (LMM, spectral) drown in them.

Run from repo root:

    python scripts/bench_amm_noise.py

Reports a table of mean ARI (over 3 seeds) per (algorithm, n_informative)
plus mean fit times at the hardest noise level. Used to validate AMM
on a regime the in-sandbox 20-newsgroups fetch couldn't reach.
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import adjusted_rand_score as ARI

from clustbench.algorithms.base import ALGO_REGISTRY


ALGOS = ["kmeans", "gmm", "agglomerative", "spectral", "fmm", "lmm", "amm"]


def main() -> None:
    configs = [
        # (n_samples, n_features, n_classes, n_informative)
        (2000, 200, 4, 50),
        (2000, 200, 4, 20),
        (2000, 200, 4, 10),
        (2000, 200, 4, 5),
    ]

    print(f"{'n':>5s}{'d':>5s}{'k':>3s}{'inf':>5s}  " +
          "  ".join(f"{a:>9s}" for a in ALGOS), flush=True)
    last_times: dict[str, list[float]] = {a: [] for a in ALGOS}
    for n, d, k, n_inf in configs:
        aris: dict[str, list[float]] = {a: [] for a in ALGOS}
        ts: dict[str, list[float]] = {a: [] for a in ALGOS}
        for seed in (1, 2, 3):
            X, y = make_classification(
                n_samples=n, n_features=d, n_informative=n_inf,
                n_redundant=0, n_repeated=0,
                n_classes=k, n_clusters_per_class=1,
                class_sep=1.5, flip_y=0.0, random_state=seed,
            )
            X = X.astype(np.float32)
            for a in ALGOS:
                try:
                    t0 = time.time()
                    res = ALGO_REGISTRY[a]().fit_predict(X, k=k)
                    elapsed = time.time() - t0
                    aris[a].append(ARI(y, res.labels))
                    ts[a].append(elapsed)
                except Exception:
                    aris[a].append(float("nan"))
                    ts[a].append(float("nan"))
        print(
            f"{n:>5d}{d:>5d}{k:>3d}{n_inf:>5d}  " +
            "  ".join(f"{np.nanmean(aris[a]):>9.3f}" for a in ALGOS),
            flush=True,
        )
        last_times = ts  # keep the times from the hardest config

    print("\n=== mean fit time (ms, n_informative=5 — hardest config) ===", flush=True)
    for a in ALGOS:
        print(f"  {a:>10s}  {np.nanmean(last_times[a]) * 1000:>7.0f} ms", flush=True)


if __name__ == "__main__":
    main()
