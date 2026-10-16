"""MCG: micro-cluster graph clustering.

What makes LMM and spectral clustering accurate is the k-NN graph: two points
belong together when a chain of close neighbours connects them, not when they
are close to the same centroid. That is what separates moons, circles and
rings, where centroid methods score ~0 ARI. But a graph on all ``n`` points
costs O(n^2 d) to build exactly and an n x n eigenproblem to cut.

MCG keeps the connectivity signal and drops the cost by building the graph on
``m`` micro-clusters (m ~ 1000, independent of n) instead of on the points:

1. **Micro-clusters**: mini-batch k-means with ``m`` centres, fitted on a
   subsample of ``fit_n`` points (a fixed cost).
2. **Edges by competitive Hebbian learning**: every point votes for an edge
   between its nearest and second-nearest centre (Martinetz & Schulten,
   1994). Points lying *between* two centres create the edge, so edges follow
   the data: two centres on the same ring are linked by the points between
   them, and centres on neighbouring rings are linked only if data bridges the
   gap. Edge counts are the CONN similarity of Tasdemir & Merenyi (2009).
3. **Gaussian edge weights**: each count is multiplied by
   ``exp(-|c_i - c_j|^2 / 2 sigma^2)``, with sigma^2 the median squared
   length of the existing edges. Long edges come from points that are far
   from every centre (noise, or a gap wider than the centre spacing), and
   down-weighting them is what separates close concentric rings: 0.86 to
   0.97 ARI on rings at n = 20k.
4. **Spectral clustering on the m x m graph**, and each point takes its
   nearest centre's label.

Cost is O(n m d) for the assignment plus a fixed O(fit_n m d + m^3), so it is
linear in n. Spectral clustering on k-means centroids is KASP (Yan, Huang &
Jordan, 2009), which uses a Gaussian kernel between centroids only; the
point-voted CHL edges are what make the centroid graph follow the data on
manifolds: spectral clustering on a k-NN graph of the same 1000 centroids
scores 0.27 ARI on rings at n = 20k, against 0.97 for MCG.

The resolution limit is the centre spacing: structures closer together than
the distance between neighbouring centres merge. ``n_micro`` trades that
against time.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import scipy.sparse as sp

from .base import AlgoResult, Algorithm, register


def _top2(X: np.ndarray, C: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """Indices of each row's nearest and second-nearest centre."""
    X32 = np.asarray(X, dtype=np.float32)
    C32 = np.asarray(C, dtype=np.float32)
    cn = (C32 ** 2).sum(axis=1)
    out = np.empty((len(X32), 2), dtype=np.int64)
    for s in range(0, len(X32), chunk):
        D = cn[None, :] - 2.0 * X32[s:s + chunk] @ C32.T
        rows = np.arange(len(D))
        i1 = D.argmin(axis=1)
        D[rows, i1] = np.inf
        out[s:s + chunk, 0] = i1
        out[s:s + chunk, 1] = D.argmin(axis=1)
    return out


@register
class Mcg(Algorithm):
    """Spectral clustering on a competitive-Hebbian micro-cluster graph."""

    def __init__(
        self,
        n_micro: Any = "auto",
        fit_n: int = 20000,
        edge_weight: str = "gauss",
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        self.name = "mcg"
        if edge_weight not in ("gauss", "count"):
            raise ValueError("edge_weight must be 'gauss' or 'count'")
        self.n_micro = n_micro
        self.fit_n = int(fit_n)
        self.edge_weight = edge_weight
        self.random_state = int(random_state)

    def _resolve_m(self, n: int, k: int) -> int:
        if isinstance(self.n_micro, int):
            m = self.n_micro
        else:  # "auto": 1000, or ~10 points per micro-cluster on small data
            m = min(1000, max(n // 10, 10 * k))
        return int(max(k + 1, min(m, n - 1)))

    def fit_predict(self, X: np.ndarray, k: Optional[int] = None) -> AlgoResult:
        from sklearn.cluster import MiniBatchKMeans, SpectralClustering

        assert k is not None, "k must be given"
        X = np.asarray(X, dtype=np.float64)
        n = len(X)
        m = self._resolve_m(n, k)
        rng = np.random.default_rng(self.random_state)
        S = X[rng.choice(n, size=min(n, self.fit_n), replace=False)]
        C = MiniBatchKMeans(n_clusters=m, n_init=1, random_state=self.random_state,
                            batch_size=2048, max_iter=50).fit(S).cluster_centers_

        T = _top2(X, C)
        a, b = T[:, 0], T[:, 1]
        W = sp.coo_matrix((np.ones(n), (a, b)), shape=(m, m)).tocsr()
        W = (W + W.T).toarray()
        if self.edge_weight == "gauss":
            sq = (C ** 2).sum(axis=1)
            D2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * C @ C.T, 0.0)
            s2 = float(np.median(D2[W > 0])) if (W > 0).any() else 1.0
            W = W * np.exp(-D2 / (2.0 * max(s2, 1e-12)))

        alive = W.sum(axis=1) > 0
        lab_m = np.zeros(m, dtype=np.int64)
        n_alive = int(alive.sum())
        if n_alive > k:
            Wa = W[np.ix_(alive, alive)] + 1e-9  # keep the graph connected for the eigensolver
            lab_m[alive] = SpectralClustering(
                n_clusters=k, affinity="precomputed", assign_labels="cluster_qr",
                random_state=self.random_state,
            ).fit_predict(Wa)
        labels = lab_m[a]
        return AlgoResult(labels=labels.astype(np.int64),
                          extra={"n_micro": m, "n_edges": int((W > 0).sum() // 2),
                                 "edge_weight": self.edge_weight})
