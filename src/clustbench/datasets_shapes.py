"""Synthetic cluster-shape generators that extend ``datasets.py``.

Each ``gen_*`` follows the same ``DataSpec -> (X, y)`` contract as the
generators in :mod:`clustbench.datasets`:

- ``X`` is ``np.float32`` of shape ``(n_samples + outliers + noise,
  n_features)``.
- ``y`` is ``np.int64`` with cluster IDs in ``[0, k)`` and ``-1`` for any
  injected outlier / noise points.

The intent is to enrich the benchmark grid with shapes a single algorithm
family won't dominate: non-convex manifolds (spiral, swiss_roll, s_curve,
rings) where spectral methods should shine, and stress-cases on the
convex side (varying density, imbalance, extreme outliers) where
centroid-based methods should still hold up. ``gen_mixed_shapes`` mixes a
few of the above into one dataset.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .datasets import DataSpec, _inject_noise, _inject_outliers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pad_features(
    X: np.ndarray,
    n_features: int,
    rng: np.random.Generator,
    scale: float,
) -> np.ndarray:
    """Append Gaussian-noise columns so X has ``n_features`` columns total."""
    if n_features <= X.shape[1]:
        return X
    pad = rng.normal(scale=scale, size=(X.shape[0], n_features - X.shape[1]))
    return np.hstack([X, pad])


def _split_sizes(n_samples: int, k: int, rng: np.random.Generator,
                 alpha: float = 4.0) -> np.ndarray:
    """Dirichlet-balanced cluster sizes summing exactly to ``n_samples``."""
    k = max(1, int(k))
    proportions = rng.dirichlet(np.ones(k) * alpha)
    sizes = (proportions * n_samples).astype(int)
    sizes[-1] = n_samples - sizes[:-1].sum()
    # Guard against zero-size clusters when n_samples is small.
    while (sizes <= 0).any():
        deficit = np.where(sizes <= 0)[0]
        donor = int(np.argmax(sizes))
        for j in deficit:
            sizes[donor] -= 1
            sizes[j] += 1
    return sizes


def _finalize(
    X: np.ndarray,
    y: np.ndarray,
    spec: DataSpec,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Common post-processing: noise/outlier injection and a final shuffle."""
    X = X.astype(np.float32, copy=False)
    y = y.astype(np.int64, copy=False)
    X, y = _inject_noise(X, y, spec.noise, rng)
    X, y = _inject_outliers(X, y, spec.outliers, rng)
    perm = rng.permutation(len(X))
    return X[perm].astype(np.float32), y[perm].astype(np.int64)


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def gen_spiral(spec: DataSpec):
    """K Archimedean spiral arms emanating from the origin.

    Each arm is the curve ``r = a * t`` rotated by ``2*pi*j / K``, sampled
    along ``t`` with small radial / angular Gaussian noise scaled by
    ``spec.compactness``. The first two columns hold the intrinsic 2-D
    structure; remaining columns are filled with small Gaussian padding.
    """
    rng = np.random.default_rng(spec.seed)
    k = max(1, int(spec.centers))
    n = int(spec.n_samples)
    sizes = _split_sizes(n, k, rng)
    noise_scale = 0.15 * float(spec.compactness)

    parts_X: List[np.ndarray] = []
    parts_y: List[np.ndarray] = []
    for j in range(k):
        m = int(sizes[j])
        # t in [0.5, 4.5] keeps arms from collapsing at the origin.
        t = rng.uniform(0.5, 4.5, size=m)
        t.sort()
        offset = 2.0 * np.pi * j / k
        r = t
        theta = t + offset
        x = r * np.cos(theta) + rng.normal(scale=noise_scale, size=m)
        z = r * np.sin(theta) + rng.normal(scale=noise_scale, size=m)
        parts_X.append(np.column_stack([x, z]))
        parts_y.append(np.full(m, j, dtype=np.int64))

    X2 = np.vstack(parts_X)
    y = np.concatenate(parts_y)
    X = _pad_features(X2, spec.n_features, rng,
                      scale=0.05 * float(spec.compactness))
    return _finalize(X, y, spec, rng)


def gen_swiss_roll(spec: DataSpec):
    """sklearn's Swiss-roll manifold, binned into K clusters by arc length.

    The manifold-parameter ``t`` increases monotonically along the roll, so
    binning ``t`` into K equal-mass slabs gives K bands of the roll that
    sit next to each other in the ambient 3-D embedding -- a textbook
    non-convex topology.
    """
    from sklearn.datasets import make_swiss_roll

    rng = np.random.default_rng(spec.seed)
    k = max(1, int(spec.centers))
    noise = 0.05 * float(spec.compactness)
    X3, t = make_swiss_roll(
        n_samples=spec.n_samples,
        noise=noise,
        random_state=spec.seed,
    )
    # Equal-mass bins so cluster sizes are roughly equal.
    edges = np.quantile(t, np.linspace(0.0, 1.0, k + 1))
    edges[-1] += 1e-9
    y = np.clip(np.digitize(t, edges[1:-1]), 0, k - 1).astype(np.int64)
    X = _pad_features(X3, spec.n_features, rng,
                      scale=0.05 * float(spec.compactness))
    return _finalize(X, y, spec, rng)


def gen_s_curve(spec: DataSpec):
    """sklearn's S-curve, binned into K clusters by the curve parameter.

    Like ``gen_swiss_roll`` but for the simpler S-shaped 2-D manifold
    embedded in 3-D. Local linear methods on the manifold should win;
    Euclidean centroid methods cluster across bends of the S.
    """
    from sklearn.datasets import make_s_curve

    rng = np.random.default_rng(spec.seed)
    k = max(1, int(spec.centers))
    noise = 0.05 * float(spec.compactness)
    X3, t = make_s_curve(
        n_samples=spec.n_samples,
        noise=noise,
        random_state=spec.seed,
    )
    edges = np.quantile(t, np.linspace(0.0, 1.0, k + 1))
    edges[-1] += 1e-9
    y = np.clip(np.digitize(t, edges[1:-1]), 0, k - 1).astype(np.int64)
    X = _pad_features(X3, spec.n_features, rng,
                      scale=0.05 * float(spec.compactness))
    return _finalize(X, y, spec, rng)


def gen_rings(spec: DataSpec):
    """K concentric rings in 2-D, padded to ``n_features``.

    Generalises ``make_circles`` (which fixes K=2) to arbitrary K. Each
    ring's radius is ``j+1`` (so they're well-separated). Algorithms that
    rely on convex compactness can't tell the rings apart; spectral or
    DBSCAN can.
    """
    rng = np.random.default_rng(spec.seed)
    k = max(1, int(spec.centers))
    n = int(spec.n_samples)
    sizes = _split_sizes(n, k, rng)
    noise_scale = 0.05 * float(spec.compactness)

    parts_X: List[np.ndarray] = []
    parts_y: List[np.ndarray] = []
    for j in range(k):
        m = int(sizes[j])
        radius = float(j + 1)
        theta = rng.uniform(0.0, 2.0 * np.pi, size=m)
        # Per-point radius jitter scales with the base radius so wider
        # rings don't have proportionally tighter bands.
        r = radius + rng.normal(scale=noise_scale, size=m)
        x = r * np.cos(theta)
        z = r * np.sin(theta)
        parts_X.append(np.column_stack([x, z]))
        parts_y.append(np.full(m, j, dtype=np.int64))

    X2 = np.vstack(parts_X)
    y = np.concatenate(parts_y)
    X = _pad_features(X2, spec.n_features, rng,
                      scale=0.05 * float(spec.compactness))
    return _finalize(X, y, spec, rng)


def gen_varying_density(spec: DataSpec):
    """K Gaussian blobs with heterogeneous per-cluster density.

    Each cluster gets a multiplier drawn from ``Uniform(0.3, 1.5)`` applied
    on top of ``spec.compactness``, so some clusters are tight and others
    diffuse. This is the canonical failure mode of *single-bandwidth*
    methods like DBSCAN -- kmeans, which doesn't share a bandwidth across
    clusters, should remain competitive.
    """
    rng = np.random.default_rng(spec.seed)
    k = max(1, int(spec.centers))
    d = max(1, int(spec.n_features))
    n = int(spec.n_samples)
    sizes = _split_sizes(n, k, rng)

    centers = rng.uniform(-1.0, 1.0, size=(k, d)) * np.sqrt(d) * 2.0
    parts_X: List[np.ndarray] = []
    parts_y: List[np.ndarray] = []
    base_std = max(0.05, float(spec.compactness))
    for j in range(k):
        m = int(sizes[j])
        multiplier = float(rng.uniform(0.3, 1.5))
        std = base_std * multiplier
        pts = rng.normal(loc=centers[j], scale=std, size=(m, d))
        parts_X.append(pts)
        parts_y.append(np.full(m, j, dtype=np.int64))

    X = np.vstack(parts_X)
    y = np.concatenate(parts_y)
    return _finalize(X, y, spec, rng)


def gen_imbalanced(spec: DataSpec):
    """K Gaussian blobs with extremely imbalanced sizes (Dirichlet alpha 0.5).

    Cluster proportions are drawn from ``Dirichlet(0.5)`` (vs the alpha=4
    used by ``gen_mdcgen``) -- typically one cluster takes the lion's
    share and the rest are tiny. Tests how robust an algorithm is to
    class imbalance.
    """
    rng = np.random.default_rng(spec.seed)
    k = max(1, int(spec.centers))
    d = max(1, int(spec.n_features))
    n = int(spec.n_samples)
    sizes = _split_sizes(n, k, rng, alpha=0.5)

    centers = rng.uniform(-1.0, 1.0, size=(k, d)) * np.sqrt(d) * 2.0
    parts_X: List[np.ndarray] = []
    parts_y: List[np.ndarray] = []
    std = max(0.05, float(spec.compactness))
    for j in range(k):
        m = int(sizes[j])
        pts = rng.normal(loc=centers[j], scale=std, size=(m, d))
        parts_X.append(pts)
        parts_y.append(np.full(m, j, dtype=np.int64))

    X = np.vstack(parts_X)
    y = np.concatenate(parts_y)
    return _finalize(X, y, spec, rng)


def gen_mixed_shapes(spec: DataSpec):
    """A single dataset built from 2-3 heterogeneous shape primitives.

    Cluster IDs are assigned in order, with each shape contributing a
    contiguous block of labels:

    - 2 Gaussian blobs (offset to the left)
    - 2 half-moons (centred, mid-right)
    - 1 spiral arm (offset further right)

    The total cluster count is fixed at 5 -- ``spec.centers`` is only used
    to seed cluster placement; the dataset's intrinsic K is 5 regardless
    of the spec. The dataset is intentionally 2-D, padded to
    ``spec.n_features``. Designed to break algorithms that assume
    homogeneous cluster shape.
    """
    rng = np.random.default_rng(spec.seed)
    n = int(spec.n_samples)
    # Allocate roughly equal points across 5 sub-clusters.
    per = n // 5
    sizes = [per] * 4 + [n - 4 * per]

    parts_X: List[np.ndarray] = []
    parts_y: List[np.ndarray] = []

    # --- two Gaussian blobs on the left ---
    blob_centers = np.array([[-8.0, -2.0], [-8.0, 2.0]])
    std = max(0.05, 0.3 * float(spec.compactness))
    for j in range(2):
        m = sizes[j]
        pts = rng.normal(loc=blob_centers[j], scale=std, size=(m, 2))
        parts_X.append(pts)
        parts_y.append(np.full(m, j, dtype=np.int64))

    # --- two half-moons centred at origin ---
    from sklearn.datasets import make_moons
    n_moons = sizes[2] + sizes[3]
    Xm, ym = make_moons(
        n_samples=n_moons,
        noise=0.08 * float(spec.compactness),
        random_state=spec.seed,
    )
    # Re-balance the per-moon counts to exactly match sizes[2], sizes[3].
    mask_a = ym == 0
    mask_b = ym == 1
    moon_a = Xm[mask_a][: sizes[2]]
    moon_b = Xm[mask_b][: sizes[3]]
    # If make_moons gave us fewer of one class than requested, pad with
    # duplicates of what we have -- preserves sizes without an extra draw.
    if len(moon_a) < sizes[2]:
        extra = rng.choice(len(moon_a), size=sizes[2] - len(moon_a))
        moon_a = np.vstack([moon_a, moon_a[extra]])
    if len(moon_b) < sizes[3]:
        extra = rng.choice(len(moon_b), size=sizes[3] - len(moon_b))
        moon_b = np.vstack([moon_b, moon_b[extra]])
    parts_X.append(moon_a)
    parts_y.append(np.full(sizes[2], 2, dtype=np.int64))
    parts_X.append(moon_b)
    parts_y.append(np.full(sizes[3], 3, dtype=np.int64))

    # --- one spiral arm offset to the right ---
    m = sizes[4]
    t = rng.uniform(0.5, 4.5, size=m)
    t.sort()
    noise_scale = 0.12 * float(spec.compactness)
    sx = t * np.cos(t) + rng.normal(scale=noise_scale, size=m) + 8.0
    sz = t * np.sin(t) + rng.normal(scale=noise_scale, size=m)
    parts_X.append(np.column_stack([sx, sz]))
    parts_y.append(np.full(m, 4, dtype=np.int64))

    X2 = np.vstack(parts_X)
    y = np.concatenate(parts_y)
    X = _pad_features(X2, spec.n_features, rng,
                      scale=0.05 * float(spec.compactness))
    return _finalize(X, y, spec, rng)


def gen_extreme_outliers(spec: DataSpec):
    """MDCGen-style blobs but with tunable outlier extremity.

    Reads ``outlier_extremity`` off ``spec`` via ``getattr`` (default 1.0,
    matching the existing ``_inject_outliers`` behaviour). Higher values
    place outliers further from the cluster cloud: outliers are drawn from
    a uniform box centred on the data, with each side scaled by
    ``outlier_extremity`` relative to the cluster envelope.

    All other behaviour mirrors ``gen_mdcgen``.
    """
    rng = np.random.default_rng(spec.seed)
    k = max(1, int(spec.centers))
    d = max(1, int(spec.n_features))
    extremity = float(getattr(spec, "outlier_extremity", 1.0))

    centers = rng.uniform(-1.0, 1.0, size=(k, d)) * np.sqrt(d)
    sizes = _split_sizes(spec.n_samples, k, rng, alpha=4.0)

    density = max(0.01, float(spec.density))
    base_std = max(0.05, float(spec.compactness)) / np.sqrt(density)

    parts_X: List[np.ndarray] = []
    parts_y: List[np.ndarray] = []
    for j in range(k):
        m = int(sizes[j])
        cov_diag = (base_std * (0.5 + rng.random(d))).astype(np.float32)
        pts = rng.normal(loc=centers[j], scale=cov_diag, size=(m, d))
        parts_X.append(pts)
        parts_y.append(np.full(m, j, dtype=np.int64))
    X = np.vstack(parts_X).astype(np.float32)
    y = np.concatenate(parts_y).astype(np.int64)

    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]

    # In-cloud noise behaves the same as in gen_mdcgen.
    X, y = _inject_noise(X, y, spec.noise, rng)

    # Custom out-of-cloud outlier injection with extremity scaling.
    if spec.outliers > 0:
        lo, hi = X.min(axis=0), X.max(axis=0)
        span = (hi - lo) + 1e-9
        # Default _inject_outliers uses a 0.5-span pad on each side; we
        # scale that pad by `extremity` so larger values push outliers
        # further from the cluster envelope.
        pad = 0.5 * span * extremity
        box_lo = lo - pad
        box_hi = hi + pad
        extra = rng.uniform(
            box_lo, box_hi, size=(spec.outliers, X.shape[1])
        ).astype(np.float32)
        extra_y = -np.ones(spec.outliers, dtype=np.int64)
        X = np.vstack([X, extra])
        y = np.concatenate([y, extra_y])

    perm = rng.permutation(len(X))
    return X[perm].astype(np.float32), y[perm].astype(np.int64)


# ---------------------------------------------------------------------------
# Registry — to be merged into ``datasets.DATASETS`` by the caller.
# ---------------------------------------------------------------------------

SHAPE_DATASETS = {
    "spiral": gen_spiral,
    "swiss_roll": gen_swiss_roll,
    "s_curve": gen_s_curve,
    "rings": gen_rings,
    "varying_density": gen_varying_density,
    "imbalanced": gen_imbalanced,
    "mixed_shapes": gen_mixed_shapes,
    "extreme_outliers": gen_extreme_outliers,
}
