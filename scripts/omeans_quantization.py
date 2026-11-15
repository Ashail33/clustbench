"""Check o-means' claims on a vector-quantisation task (large k), on CPU.

Claims from the o-means write-up (GPU, RTX 5090): it reaches Lloyd's 25-iteration k-means error with 0.15-0.45
passes' worth of distance computations, its two-level assignment needs ~5-20% of exact comparisons, and after enough
epochs it ends 0.7-1.7% below 25-iteration Lloyd error. Wall-clock speed-ups are GPU-specific and not checked here.

Data: an anisotropic Gaussian mixture with many more modes than k, so the quantisation error is not trivially zero.

    python scripts/omeans_quantization.py [--n 100000] [--d 64] [--k 1024 4096] [--batch 256 1024 4096] [--epochs 1 3]

Without --batch, each run uses o-means' default batch size; with it, every listed batch size is run.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from threadpoolctl import threadpool_limits

from clustbench.algorithms.o_means import exact_assign, omeans


def make_data(n, d, modes, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(modes, d)) * 3
    scales = rng.uniform(0.3, 1.5, size=(modes, d))
    z = rng.integers(0, modes, n)
    return (centres[z] + rng.normal(size=(n, d)) * scales[z]).astype(np.float64)


def objective(X, C):
    lab = exact_assign(X, C)
    return float(((X - C[lab]) ** 2).sum(1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--k", type=int, nargs="+", default=[1024, 4096])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--batch", type=int, nargs="*", default=[None])
    ap.add_argument("--epochs", type=float, nargs="+", default=[1, 3, 6])
    a = ap.parse_args()
    X = make_data(a.n, a.d, modes=4 * max(a.k))
    rows = []
    with threadpool_limits(a.threads):
        for k in a.k:
            t = time.time()
            km = KMeans(k, n_init=1, max_iter=25, tol=0, init="random", random_state=0).fit(X)
            ref = objective(X, km.cluster_centers_)
            rows.append((k, "Lloyd, 25 iterations", ref, 25.0, time.time() - t))
            t = time.time()
            mb = MiniBatchKMeans(k, batch_size=max(4096, 2 * k), max_iter=3, n_init=1, random_state=0).fit(X)
            rows.append((k, "MiniBatchKMeans, 3 epochs", objective(X, mb.cluster_centers_), np.nan, time.time() - t))
            for b, ep, two in [(b, ep, two) for b in a.batch for ep in a.epochs for two in (False, True)]:
                t = time.time()
                r = omeans(X, k, epochs=ep, batch_size=b, two_level=two, random_state=0)
                bs = f", batch {b}" if b else ""
                name = f"{'o-means' if two else 'e-means (exact assign)'}{bs}, {ep:g} epoch{'s' if ep != 1 else ''}"
                rows.append((k, name, r["objective"], r["exact_pass_equivalent"], time.time() - t))
            print(f"k={k} done", flush=True)
    print(f"\nn={a.n}, d={a.d}; error relative to 25-iteration Lloyd (lower is better); 'passes' = distance "
          f"computations / (n k)\n")
    print("| k | method | error vs Lloyd | passes | time (s, CPU) |\n|---|---|---|---|---|")
    for k, name, obj, passes, sec in rows:
        ref = next(o for kk, nm, o, *_ in rows if kk == k and nm.startswith("Lloyd"))
        p = "" if np.isnan(passes) else f"{passes:.2f}"
        print(f"| {k} | {name} | {100 * (obj / ref - 1):+.2f}% | {p} | {sec:.1f} |")


if __name__ == "__main__":
    main()
