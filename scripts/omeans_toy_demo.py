"""Reproduce the k-means / e-means / o-means toy chart: 12,000 2-D points, k = 64, error vs distance computations.

    python scripts/omeans_toy_demo.py
"""
import numpy as np
from sklearn.datasets import make_blobs
from clustbench.algorithms.o_means import omeans, exact_assign

X, _ = make_blobs(12000, centers=12, cluster_std=1.2, center_box=(-15, 15), random_state=3)
n, k = len(X), 64
mse = lambda C: float(((X - C[exact_assign(X, C)]) ** 2).sum(1).mean())

# Lloyd from random data rows, one pass = one iteration (n*k distance computations)
rng = np.random.default_rng(0); C = X[rng.choice(n, k, replace=False)].copy(); lloyd = [mse(C)]
for it in range(25):
    lab = exact_assign(X, C)
    for j in range(k):
        if (lab == j).any(): C[j] = X[lab == j].mean(0)
    lloyd.append(mse(C))
print("Lloyd (passes: error)", {p: round(lloyd[p], 3) for p in (1, 2, 3, 4, 6, 10, 25)})

print("\n| method | epochs | passes used | error | vs Lloyd@6 | vs Lloyd@25 |\n|---|---|---|---|---|---|")
for name, kw in [(f"{m}, batch {b or 'default'}", dict(kw0, batch_size=b)) for b in (256, None)
                 for m, kw0 in (("e-means (exact)", dict(two_level=False)),
                                ("o-means (16 cells, probe 1 ~ 20 checks/pt)", dict(two_level=True, two_level_min_k=0, nprobe=1)))]:
    for ep in (0.25, 0.5, 1, 2, 3):
        errs = []; passes = []
        for s in range(3):
            r = omeans(X, k, epochs=ep, random_state=s, **kw)
            errs.append(r["objective"]); passes.append(r["exact_pass_equivalent"])
        e = np.mean(errs)
        print(f"| {name} | {ep} | {np.mean(passes):.2f} | {e:.3f} | {100*(e/lloyd[6]-1):+.1f}% | {100*(e/lloyd[25]-1):+.1f}% |")
