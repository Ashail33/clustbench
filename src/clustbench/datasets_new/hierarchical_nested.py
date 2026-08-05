from __future__ import annotations
import math
import numpy as np

from clustbench.datasets import DataSpec


def gen_hierarchical_nested(spec: DataSpec) -> tuple[np.ndarray, np.ndarray]:
    """Two-level nested Gaussians.

    Splits ``spec.centers`` into K super-clusters and M sub-clusters, with
    K = max(2, ceil(sqrt(spec.centers))) and M = ceil(spec.centers / K).
    Super-centers are drawn uniformly in [-1, 1]^d and rescaled to radius
    R_super = 10 * spec.compactness. For each super-cluster, M sub-centers are
    placed on a sphere of radius R_sub = 2 * spec.compactness around the
    super-center. ``spec.n_samples`` are partitioned as evenly as possible
    across the K*M sub-clusters (with remainder assigned to the first buckets)
    and drawn from an isotropic Normal with std ``spec.compactness``.
    Ground-truth labels are integer sub-cluster ids in [0, K*M).
    """
    rng = np.random.default_rng(spec.seed)

    d = int(spec.n_features)
    n = int(spec.n_samples)
    c = int(spec.centers)
    compactness = float(spec.compactness)

    # Determine hierarchy sizes.
    k_super = max(2, int(math.ceil(math.sqrt(max(1, c)))))
    m_sub = int(math.ceil(max(1, c) / k_super))
    total_sub = k_super * m_sub

    r_super = 10.0 * compactness
    r_sub = 2.0 * compactness

    # Draw K super-centers uniformly in [-1, 1]^d and rescale to r_super.
    super_centers = rng.uniform(-1.0, 1.0, size=(k_super, d)) * r_super

    # For each super-cluster, draw M sub-centers on a sphere of radius r_sub.
    sub_centers = np.empty((total_sub, d), dtype=np.float64)
    for i in range(k_super):
        directions = rng.standard_normal(size=(m_sub, d))
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        # Avoid division by zero (astronomically unlikely for reasonable d).
        norms = np.where(norms == 0.0, 1.0, norms)
        directions = directions / norms
        sub_centers[i * m_sub : (i + 1) * m_sub] = (
            super_centers[i] + directions * r_sub
        )

    # Partition n_samples across total_sub buckets: remainder to first buckets.
    base = n // total_sub
    rem = n % total_sub
    counts = np.full(total_sub, base, dtype=np.int64)
    counts[:rem] += 1

    # Sample points and assign labels.
    X = np.empty((n, d), dtype=np.float64)
    y = np.empty(n, dtype=np.int64)
    offset = 0
    for j in range(total_sub):
        cnt = int(counts[j])
        if cnt == 0:
            continue
        pts = rng.standard_normal(size=(cnt, d)) * compactness + sub_centers[j]
        X[offset : offset + cnt] = pts
        y[offset : offset + cnt] = j
        offset += cnt

    return X.astype(np.float32), y
