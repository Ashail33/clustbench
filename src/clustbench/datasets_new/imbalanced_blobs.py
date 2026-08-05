from __future__ import annotations
import numpy as np

from clustbench.datasets import DataSpec


def gen_imbalanced_blobs(spec: DataSpec) -> tuple[np.ndarray, np.ndarray]:
    """Isotropic Gaussian clusters with geometrically decaying sizes.

    Draws ``spec.centers`` cluster means on the surface of a sphere of radius
    ``5 * spec.compactness`` in R^{spec.n_features}. Cluster sizes follow a
    geometric schedule ``n_k proportional to r^k`` with
    ``r = (1/10)**(1/(centers-1))``, giving a fixed 10:1 largest-to-smallest
    ratio. Sizes are renormalized to ``spec.n_samples`` via rounding with any
    rounding drift absorbed by the largest cluster, then a floor of
    ``max(2, spec.n_features + 1)`` samples per cluster is enforced by
    borrowing from the largest bucket if needed. Each cluster is drawn from
    ``Normal(mean_k, spec.compactness * I)``. Returns ``X`` (float32) and
    integer labels in ``[0, centers)`` after shuffling with the same RNG.
    """
    rng = np.random.default_rng(spec.seed)

    d = int(spec.n_features)
    n = int(spec.n_samples)
    c = max(1, int(spec.centers))
    compactness = float(spec.compactness)

    r_sphere = 5.0 * compactness

    # Draw c centers on the surface of a sphere of radius r_sphere.
    directions = rng.standard_normal(size=(c, d))
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    centers = (directions / norms) * r_sphere

    # Geometric schedule: n_k proportional to r^k with r = (1/10)^(1/(c-1)).
    if c == 1:
        weights = np.ones(1, dtype=np.float64)
    else:
        ratio = (1.0 / 10.0) ** (1.0 / (c - 1))
        weights = ratio ** np.arange(c, dtype=np.float64)

    # Renormalize to sum to n (rounded), absorb drift into the largest cluster.
    raw = weights * (n / weights.sum())
    counts = np.rint(raw).astype(np.int64)
    drift = n - int(counts.sum())
    # Largest cluster is index 0 by construction (r < 1 for c > 1).
    largest_idx = int(np.argmax(counts))
    counts[largest_idx] += drift

    # Enforce a per-cluster floor of max(2, d + 1), borrowing from the largest.
    floor = max(2, d + 1)
    # Only enforce if feasible; otherwise borrow as much as possible.
    for k in range(c):
        if k == largest_idx:
            continue
        if counts[k] < floor:
            need = floor - int(counts[k])
            available = int(counts[largest_idx]) - floor
            take = max(0, min(need, available))
            counts[k] += take
            counts[largest_idx] -= take
    # Recompute largest after borrowing, in case the ordering changed.
    largest_idx = int(np.argmax(counts))
    # Final safety: clip negative counts (shouldn't happen with sane inputs).
    counts = np.clip(counts, 0, None)
    # Re-absorb any leftover drift from clipping.
    drift = n - int(counts.sum())
    counts[largest_idx] += drift

    # Sample each cluster from Normal(mean_k, compactness * I).
    total = int(counts.sum())
    X = np.empty((total, d), dtype=np.float64)
    y = np.empty(total, dtype=np.int64)
    offset = 0
    for k in range(c):
        cnt = int(counts[k])
        if cnt <= 0:
            continue
        pts = rng.standard_normal(size=(cnt, d)) * compactness + centers[k]
        X[offset : offset + cnt] = pts
        y[offset : offset + cnt] = k
        offset += cnt

    # Shuffle with the same RNG.
    perm = rng.permutation(total)
    X = X[perm]
    y = y[perm]

    return X.astype(np.float32), y
