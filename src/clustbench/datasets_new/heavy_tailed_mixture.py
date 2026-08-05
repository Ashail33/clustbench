from __future__ import annotations
import numpy as np

from clustbench.datasets import DataSpec


def gen_heavy_tailed_mixture(spec: DataSpec) -> tuple[np.ndarray, np.ndarray]:
    """Mixture of multivariate Student-t components with low degrees of freedom.

    Cluster means are drawn on a sphere of radius ``6 * spec.compactness`` in
    ``R^{spec.n_features}``. ``spec.n_samples`` are split as evenly as possible
    across ``spec.centers`` clusters (remainder assigned to the first buckets).
    Offsets from each mean are drawn from a multivariate Student-t distribution
    with ``df = 3`` and scale matrix ``(spec.compactness ** 2) * I`` using the
    normal / chi-square construction: ``z ~ N(0, scale)``,
    ``u ~ ChiSquare(df) / df``, ``offset = z / sqrt(u)``. The low degrees of
    freedom produce intrinsic in-cluster outliers, stressing SSE-based
    objectives and the Mahalanobis assumptions of Gaussian mixture models.

    Points and integer labels ``0..spec.centers - 1`` are shuffled jointly
    before being returned. Features are cast to ``float32``.
    """
    rng = np.random.default_rng(spec.seed)

    d = int(spec.n_features)
    n = int(spec.n_samples)
    k = max(1, int(spec.centers))
    compactness = float(spec.compactness)

    df = 3.0
    r_center = 6.0 * compactness

    # Draw k cluster means on a sphere of radius r_center.
    directions = rng.standard_normal(size=(k, d))
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    centers = (directions / norms) * r_center

    # Balanced partition of n across k buckets, remainder to first buckets.
    base = n // k
    rem = n % k
    counts = np.full(k, base, dtype=np.int64)
    counts[:rem] += 1

    X = np.empty((n, d), dtype=np.float64)
    y = np.empty(n, dtype=np.int64)
    offset = 0
    for j in range(k):
        cnt = int(counts[j])
        if cnt == 0:
            continue
        # Multivariate Student-t via normal / sqrt(chi2/df).
        # scale matrix = (compactness ** 2) * I, so z has std = compactness.
        z = rng.standard_normal(size=(cnt, d)) * compactness
        u = rng.chisquare(df, size=cnt) / df
        # Guard against pathological zeros from the chi-square draw.
        u = np.where(u <= 0.0, np.finfo(np.float64).tiny, u)
        t_offsets = z / np.sqrt(u)[:, None]
        X[offset : offset + cnt] = centers[j] + t_offsets
        y[offset : offset + cnt] = j
        offset += cnt

    # Joint shuffle of points and labels.
    perm = rng.permutation(n)
    X = X[perm]
    y = y[perm]

    return X.astype(np.float32), y
