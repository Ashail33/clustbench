from __future__ import annotations
import numpy as np

from clustbench.datasets import DataSpec


def gen_VariableDensityBridges(spec: DataSpec) -> tuple[np.ndarray, np.ndarray]:
    """Radially non-stationary clusters linked by low-density Poisson bridges.

    Each cluster has a dense isotropic Gaussian core (60% of its mass) and an
    isotropic radial Cauchy tail (40%), producing a two-orders-of-magnitude
    density gradient from core to tail edge. The ``spec.centers`` core centers
    are placed uniformly on a low-dimensional signal manifold (2D or 3D) that
    is embedded in the first axes of the ambient ``spec.n_features`` space; the
    remaining ambient dimensions carry only isotropic Gaussian noise at ~5% of
    the manifold amplitude. Adjacent center pairs (Delaunay neighbors on the
    manifold, or all pairs when there are too few centers for a triangulation)
    are joined by straight-line bridges. Each bridge is sampled as an
    inhomogeneous-Poisson-like process at ~2% of core density, with a thin
    transverse Gaussian jitter; every bridge point is labeled with the nearest
    of its two endpoint clusters (the ambiguity is intentional).

    The construction produces the classic single-link failure mode by design:
    any distance scale small enough to keep the Cauchy tails intact fragments
    the tails, while any scale large enough to keep them whole chains all
    clusters into one component through the bridges. All draws share a single
    ``np.random.default_rng(spec.seed)`` for reproducibility, and features are
    returned as ``float32``.
    """
    rng = np.random.default_rng(spec.seed)

    d = int(spec.n_features)
    n = int(spec.n_samples)
    k = max(1, int(spec.centers))
    compactness = float(spec.compactness)

    # Signal manifold dimensionality: 3D when possible, else 2D, else 1D.
    if d >= 3:
        signal_dim = 3
    elif d == 2:
        signal_dim = 2
    else:
        signal_dim = 1

    # Geometry / density parameters (scaled by compactness).
    core_sigma = 0.5 * compactness           # tight Gaussian core
    tail_scale = 1.0 * compactness           # Cauchy scale for the heavy tail
    r_centers = 8.0 * compactness            # manifold extent for center placement
    noise_amp = 0.05 * compactness           # 5% isotropic noise on ambient axes
    bridge_sigma = 0.15 * compactness        # thin transverse jitter on bridges
    r_tail_cap = 6.0 * r_centers             # radial cap for pathological Cauchy draws

    # Place k core centers on the low-dim manifold.
    if k == 1:
        centers_manifold = np.zeros((1, signal_dim), dtype=np.float64)
    elif signal_dim == 1:
        centers_manifold = np.linspace(
            -r_centers, r_centers, k, dtype=np.float64
        ).reshape(-1, 1)
    else:
        centers_manifold = rng.uniform(
            -r_centers, r_centers, size=(k, signal_dim)
        ).astype(np.float64)

    # Determine adjacency edges between centers.
    edges: list[tuple[int, int]] = []
    if k >= 2:
        if signal_dim == 1:
            # 1D: connect neighbors along the axis.
            order = np.argsort(centers_manifold[:, 0], kind="stable")
            for i in range(len(order) - 1):
                a, b = int(order[i]), int(order[i + 1])
                if a > b:
                    a, b = b, a
                edges.append((a, b))
        elif k <= signal_dim + 1:
            # Too few points for a proper triangulation: connect all pairs.
            for i in range(k):
                for j in range(i + 1, k):
                    edges.append((i, j))
        else:
            try:
                from scipy.spatial import Delaunay

                tri = Delaunay(centers_manifold)
                edge_set: set[tuple[int, int]] = set()
                for simplex in tri.simplices:
                    m = len(simplex)
                    for i in range(m):
                        for j in range(i + 1, m):
                            a, b = int(simplex[i]), int(simplex[j])
                            if a > b:
                                a, b = b, a
                            edge_set.add((a, b))
                edges = sorted(edge_set)
            except Exception:
                # Fallback (degenerate / coplanar centers): connect all pairs.
                for i in range(k):
                    for j in range(i + 1, k):
                        edges.append((i, j))

    # Budget: ~2% of samples go to bridges, remainder to clusters.
    n_bridge_total = int(round(0.02 * n)) if len(edges) > 0 else 0
    n_bridge_total = min(n_bridge_total, max(0, n - k))  # leave >=1 per cluster if possible
    n_cluster_total = n - n_bridge_total

    # Balanced partition across clusters (remainder to first buckets).
    if k > 0:
        base = n_cluster_total // k
        rem = n_cluster_total % k
        counts = np.full(k, base, dtype=np.int64)
        counts[:rem] += 1
    else:
        counts = np.zeros(0, dtype=np.int64)

    X_chunks: list[np.ndarray] = []
    y_chunks: list[np.ndarray] = []

    # Per-cluster: 60% Gaussian core + 40% isotropic radial Cauchy tail.
    for j in range(k):
        cnt = int(counts[j])
        if cnt <= 0:
            continue
        n_core = int(round(0.6 * cnt))
        n_tail = cnt - n_core

        pts = np.zeros((cnt, d), dtype=np.float64)

        # Core: Gaussian on manifold, isotropic noise on remaining axes.
        if n_core > 0:
            pts[:n_core, :signal_dim] = (
                rng.standard_normal(size=(n_core, signal_dim)) * core_sigma
                + centers_manifold[j]
            )
            if d > signal_dim:
                pts[:n_core, signal_dim:] = (
                    rng.standard_normal(size=(n_core, d - signal_dim)) * noise_amp
                )

        # Tail: uniform direction on sphere x |Cauchy| radius, centered at mu.
        if n_tail > 0:
            dirs = rng.standard_normal(size=(n_tail, signal_dim))
            dn = np.linalg.norm(dirs, axis=1, keepdims=True)
            dn = np.where(dn == 0.0, 1.0, dn)
            dirs = dirs / dn
            radii = np.abs(rng.standard_cauchy(size=n_tail)) * tail_scale
            radii = np.minimum(radii, r_tail_cap)  # guard against extreme outliers
            pts[n_core:, :signal_dim] = dirs * radii[:, None] + centers_manifold[j]
            if d > signal_dim:
                pts[n_core:, signal_dim:] = (
                    rng.standard_normal(size=(n_tail, d - signal_dim)) * noise_amp
                )

        X_chunks.append(pts)
        y_chunks.append(np.full(cnt, j, dtype=np.int64))

    # Bridges: straight-line segments between Delaunay-adjacent centers.
    if n_bridge_total > 0 and len(edges) > 0:
        n_edges = len(edges)
        per_edge = n_bridge_total // n_edges
        edge_rem = n_bridge_total % n_edges
        bridge_counts = np.full(n_edges, per_edge, dtype=np.int64)
        bridge_counts[:edge_rem] += 1

        b_pts = np.zeros((n_bridge_total, d), dtype=np.float64)
        b_lab = np.zeros(n_bridge_total, dtype=np.int64)
        off = 0
        for (a, b), bc in zip(edges, bridge_counts.tolist()):
            if bc <= 0:
                continue
            ts = rng.uniform(0.0, 1.0, size=bc)
            seg = (
                centers_manifold[a] * (1.0 - ts[:, None])
                + centers_manifold[b] * ts[:, None]
            )
            seg = seg + rng.standard_normal(size=(bc, signal_dim)) * bridge_sigma
            b_pts[off : off + bc, :signal_dim] = seg
            if d > signal_dim:
                b_pts[off : off + bc, signal_dim:] = (
                    rng.standard_normal(size=(bc, d - signal_dim)) * noise_amp
                )
            # Label = nearest endpoint by linear-interpolation midpoint.
            b_lab[off : off + bc] = np.where(ts < 0.5, a, b).astype(np.int64)
            off += bc

        X_chunks.append(b_pts)
        y_chunks.append(b_lab)

    # Assemble, shuffle, cast.
    if len(X_chunks) == 0:
        return np.zeros((0, d), dtype=np.float32), np.zeros(0, dtype=np.int64)

    X = np.vstack(X_chunks)
    y = np.concatenate(y_chunks)

    perm = rng.permutation(len(y))
    X = X[perm]
    y = y[perm]

    return X.astype(np.float32), y
