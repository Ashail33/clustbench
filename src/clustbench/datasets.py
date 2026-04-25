"""Dataset generators for the benchmark harness.

The :class:`DataSpec` carries the core knobs every generator understands —
``n_samples``, ``n_features``, ``centers``, ``compactness``, ``seed`` — plus
optional MDCGen-style parameters (``outliers``, ``noise``, ``density``) that
let us reproduce the experimental design in
`Maharaj (2024) <paper/clustering-review-maharaj-2024.pdf>`_.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_classification, make_moons


@dataclass
class DataSpec:
    n_samples: int
    n_features: int
    centers: int
    compactness: float
    seed: int
    outliers: int = 0
    noise: int = 0
    density: float = 1.0


def _inject_outliers(X: np.ndarray, y: np.ndarray, n_outliers: int, rng: np.random.Generator):
    """Append uniform-random outliers far from the cluster envelope.

    Outliers are drawn from a uniform box that sits at 1.5x the data range,
    so they're well outside the cluster cloud. Their ground-truth label is
    ``-1`` (the de-facto noise label used by DBSCAN-style algorithms).
    """
    if n_outliers <= 0:
        return X, y
    lo, hi = X.min(axis=0), X.max(axis=0)
    span = (hi - lo) + 1e-9
    box_lo = lo - 0.5 * span
    box_hi = hi + 0.5 * span
    extra = rng.uniform(box_lo, box_hi, size=(n_outliers, X.shape[1])).astype(X.dtype)
    extra_y = -np.ones(n_outliers, dtype=y.dtype)
    return np.vstack([X, extra]), np.concatenate([y, extra_y])


def _inject_noise(X: np.ndarray, y: np.ndarray, n_noise: int, rng: np.random.Generator):
    """Append Gaussian noise points centred on the data centroid.

    Distinct from outliers: noise points sit *inside* the cluster cloud and
    are harder to separate. Label ``-1``.
    """
    if n_noise <= 0:
        return X, y
    mu = X.mean(axis=0)
    sigma = X.std(axis=0) + 1e-9
    extra = (mu + rng.normal(size=(n_noise, X.shape[1])) * sigma).astype(X.dtype)
    extra_y = -np.ones(n_noise, dtype=y.dtype)
    return np.vstack([X, extra]), np.concatenate([y, extra_y])


def gen_blobs(spec: DataSpec):
    X, y = make_blobs(
        n_samples=spec.n_samples,
        n_features=spec.n_features,
        centers=spec.centers,
        cluster_std=spec.compactness,
        shuffle=True,
        random_state=spec.seed,
    )
    return X.astype(np.float32), y


def gen_mixed(spec: DataSpec):
    X, y = make_classification(
        n_samples=spec.n_samples,
        n_features=spec.n_features,
        n_informative=max(2, spec.n_features // 2),
        n_redundant=spec.n_features // 4,
        n_clusters_per_class=1,
        n_classes=min(8, spec.centers),
        class_sep=1.0 / max(0.2, spec.compactness),
        random_state=spec.seed,
    )
    return X.astype(np.float32), y


def gen_mdcgen(spec: DataSpec):
    """MDCGen-style synthetic dataset generator (Lopez et al., reproduced).

    Generates ``centers`` Gaussian clusters whose spread is modulated by
    ``density`` (lower density => looser clusters), then optionally injects
    ``noise`` in-cloud points and ``outliers`` out-of-cloud points. Cluster
    sizes are drawn from a Dirichlet so they're not exactly equal — a
    closer match to the kind of imbalance MDCGen produces.

    The ``compactness`` knob is multiplied into the per-cluster covariance
    so existing configs that tune compactness keep behaving the same way.
    """
    rng = np.random.default_rng(spec.seed)
    k = max(1, spec.centers)
    d = spec.n_features

    # Cluster centres in a unit cube scaled by sqrt(d) so per-feature spacing is constant.
    centers = rng.uniform(-1.0, 1.0, size=(k, d)) * np.sqrt(d)
    proportions = rng.dirichlet(np.ones(k) * 4.0)
    sizes = (proportions * spec.n_samples).astype(int)
    sizes[-1] = spec.n_samples - sizes[:-1].sum()

    # density in (0, 1] -> std multiplier; lower density = bigger std.
    density = max(0.01, float(spec.density))
    base_std = max(0.05, float(spec.compactness)) / np.sqrt(density)

    X_parts = []
    y_parts = []
    for j in range(k):
        cov_diag = (base_std * (0.5 + rng.random(d))).astype(np.float32)
        pts = rng.normal(loc=centers[j], scale=cov_diag, size=(sizes[j], d)).astype(np.float32)
        X_parts.append(pts)
        y_parts.append(np.full(sizes[j], j, dtype=np.int64))
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)

    # Shuffle the cluster points.
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]

    X, y = _inject_noise(X, y, spec.noise, rng)
    X, y = _inject_outliers(X, y, spec.outliers, rng)

    # Final shuffle so cluster, noise and outlier rows are interleaved.
    perm = rng.permutation(len(X))
    return X[perm].astype(np.float32), y[perm]


def gen_moons(spec: DataSpec):
    """Two interleaving half-moons. Non-convex; trips up centroid-based methods."""
    X, y = make_moons(n_samples=spec.n_samples, noise=0.1 * float(spec.compactness), random_state=spec.seed)
    if spec.n_features > 2:
        rng = np.random.default_rng(spec.seed)
        pad = rng.normal(scale=0.05 * float(spec.compactness), size=(X.shape[0], spec.n_features - 2))
        X = np.hstack([X, pad])
    return X.astype(np.float32), y.astype(np.int64)


def gen_circles(spec: DataSpec):
    """Concentric circles. ``compactness`` controls noise; ``centers`` is forced to 2."""
    X, y = make_circles(
        n_samples=spec.n_samples,
        noise=0.05 * float(spec.compactness),
        factor=0.5,
        random_state=spec.seed,
    )
    if spec.n_features > 2:
        rng = np.random.default_rng(spec.seed)
        pad = rng.normal(scale=0.05 * float(spec.compactness), size=(X.shape[0], spec.n_features - 2))
        X = np.hstack([X, pad])
    return X.astype(np.float32), y.astype(np.int64)


def gen_anisotropic(spec: DataSpec):
    """Sheared blobs. Tests robustness to non-spherical cluster shape."""
    X, y = make_blobs(
        n_samples=spec.n_samples,
        n_features=spec.n_features,
        centers=spec.centers,
        cluster_std=spec.compactness,
        random_state=spec.seed,
    )
    rng = np.random.default_rng(spec.seed)
    transform = rng.normal(size=(spec.n_features, spec.n_features))
    transform = transform / np.linalg.norm(transform, axis=0, keepdims=True)
    return (X @ transform).astype(np.float32), y.astype(np.int64)


DATASETS = {
    "blobs": gen_blobs,
    "mixed": gen_mixed,
    "mdcgen": gen_mdcgen,
    "moons": gen_moons,
    "circles": gen_circles,
    "anisotropic": gen_anisotropic,
}
