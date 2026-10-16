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


def cf_triples(X: np.ndarray, labels: np.ndarray, m: int):
    """BIRCH clustering-feature triples (N, LS, SS) for ``m`` clusters."""
    N = np.bincount(labels, minlength=m).astype(np.float64)
    LS = np.zeros((m, X.shape[1]))
    np.add.at(LS, labels, X)
    SS = np.bincount(labels, weights=np.einsum("ij,ij->i", X, X), minlength=m)
    return N, LS, SS


def cf_radius2(N: np.ndarray, LS: np.ndarray, SS: np.ndarray) -> np.ndarray:
    """Mean squared distance to the centroid, SS/N - |LS/N|^2, from CF triples."""
    Nn = np.maximum(N, 1.0)
    return np.maximum(SS / Nn - ((LS / Nn[:, None]) ** 2).sum(axis=1), 0.0)


@register
class Mcg(Algorithm):
    """Spectral clustering on a competitive-Hebbian micro-cluster graph.

    ``tiers=2`` builds the micro-clusters coarse-to-fine: a coarse k-means
    with ~2 sqrt(m) centres, then each coarse cluster is split locally into
    children in proportion to its point count (hierarchical k-means, as in a
    BIRCH CF tree). ``edge_weight="cf"`` scales each edge by the two
    micro-clusters' own spread, read from their CF triples:
    ``exp(-|c_i - c_j|^2 / (gamma (R_i^2 + R_j^2)))``, instead of one global
    bandwidth. See :class:`Mcg_cf`.
    """

    def __init__(
        self,
        n_micro: Any = "auto",
        fit_n: int = 20000,
        edge_weight: str = "gauss",
        tiers: int = 1,
        gamma: float = 1.0,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        self.name = "mcg"
        if edge_weight not in ("gauss", "count", "cf"):
            raise ValueError("edge_weight must be 'gauss', 'count' or 'cf'")
        if tiers not in (1, 2):
            raise ValueError("tiers must be 1 or 2")
        self.n_micro = n_micro
        self.fit_n = int(fit_n)
        self.edge_weight = edge_weight
        self.tiers = int(tiers)
        self.gamma = float(gamma)
        self.random_state = int(random_state)

    def _resolve_m(self, n: int, k: int) -> int:
        if isinstance(self.n_micro, int):
            m = self.n_micro
        else:  # "auto": 1000, or ~10 points per micro-cluster on small data
            m = min(1000, max(n // 10, 10 * k))
        return int(max(k + 1, min(m, n - 1)))

    def _centres_flat(self, S: np.ndarray, m: int) -> np.ndarray:
        from sklearn.cluster import MiniBatchKMeans

        return MiniBatchKMeans(n_clusters=m, n_init=1, random_state=self.random_state,
                               batch_size=2048, max_iter=50).fit(S).cluster_centers_

    def _centres_tiered(self, S: np.ndarray, m: int) -> np.ndarray:
        from sklearn.cluster import KMeans

        m0 = int(max(2, min(len(S), 2 * np.sqrt(m))))
        C0 = self._centres_flat(S, m0)
        l0 = _top2(S, C0)[:, 0]
        N0 = np.bincount(l0, minlength=m0).astype(np.float64)
        share = N0 / N0.sum() * m
        kids = np.maximum(1, np.floor(share)).astype(int)
        rem = m - int(kids.sum())
        if rem > 0:
            kids[np.argsort(-(share - np.floor(share)))[:rem]] += 1
        out = []
        for c in range(m0):
            P = S[l0 == c]
            if len(P) == 0:
                continue
            kc = int(min(kids[c], len(P)))
            if kc <= 1:
                out.append(P.mean(axis=0, keepdims=True))
            else:
                out.append(KMeans(n_clusters=kc, n_init=1, max_iter=20,
                                  random_state=self.random_state).fit(P).cluster_centers_)
        return np.vstack(out)

    def build_graph(self, X: np.ndarray, k: int) -> dict:
        """Micro-clusters and their weighted graph (shared with second stages).

        Returns centres ``C`` (m, d), each point's nearest and second-nearest
        centre ``a``, ``b``, the dense weighted graph ``W`` (m, m), and the CF
        statistics ``N`` (points per centre) and ``R2`` (mean squared radius).
        """
        X = np.asarray(X, dtype=np.float64)
        n = len(X)
        m = self._resolve_m(n, k)
        rng = np.random.default_rng(self.random_state)
        S = X[rng.choice(n, size=min(n, self.fit_n), replace=False)]
        C = self._centres_tiered(S, m) if self.tiers == 2 else self._centres_flat(S, m)
        m = len(C)

        T = _top2(X, C)
        a, b = T[:, 0], T[:, 1]
        W = sp.coo_matrix((np.ones(n), (a, b)), shape=(m, m)).tocsr()
        W = (W + W.T).toarray()
        N, LS, SS = cf_triples(X, a, m)
        R2 = cf_radius2(N, LS, SS)
        ok = N > 1
        # Singletons have no spread estimate; give them the typical one.
        R2 = np.where(ok, R2, float(np.median(R2[ok])) if ok.any() else 1.0)
        if self.edge_weight in ("gauss", "cf"):
            sq = (C ** 2).sum(axis=1)
            D2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * C @ C.T, 0.0)
        if self.edge_weight == "gauss":
            s2 = float(np.median(D2[W > 0])) if (W > 0).any() else 1.0
            W = W * np.exp(-D2 / (2.0 * max(s2, 1e-12)))
        elif self.edge_weight == "cf":
            W = W * np.exp(-D2 / (self.gamma * (R2[:, None] + R2[None, :]) + 1e-12))
        return {"C": C, "a": a, "b": b, "W": W, "N": N, "R2": R2}

    def fit_predict(self, X: np.ndarray, k: Optional[int] = None) -> AlgoResult:
        from sklearn.cluster import SpectralClustering

        assert k is not None, "k must be given"
        g = self.build_graph(X, k)
        W, a = g["W"], g["a"]
        m = len(W)
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
                                 "edge_weight": self.edge_weight, "tiers": self.tiers})


@register
class Mcg_cf(Mcg):
    """Tiered MCG with CF-triple edge weights.

    Two-tier (coarse-to-fine) micro-clusters, and each edge's bandwidth is
    the two micro-clusters' own spread from their BIRCH CF triples
    (N, LS, SS) rather than one global median. Tiers resolve small and
    unevenly sized clusters (breast cancer 0.24 -> 0.49, rings 0.97 -> 0.98)
    and are 2.5x faster at n = 20k. Local bandwidths keep sparse clusters
    and outliers from being cut apart or merged by a single global scale
    (extreme outliers 0.56 -> 0.72). ``gamma = 1`` was chosen on the unseen
    and n = 20k suites, with the 140 in-distribution tasks held out
    (round 21 in docs/ALGORITHM_ANALYSIS.md).
    """

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("tiers", 2)
        kwargs.setdefault("edge_weight", "cf")
        kwargs.setdefault("gamma", 1.0)
        super().__init__(**kwargs)
        self.name = "mcg_cf"


def _lmm_mcg_class():
    """Defined lazily so importing mcg does not pull in the LMM/FMM stack first."""
    from .lmm import Lmm

    @register
    class Lmm_mcg(Lmm):
        """LMM whose Laplacian basis comes from the MCG micro-cluster graph.

        LMM's accuracy comes from the k-NN graph, and its cost from building
        that graph and eigendecomposing it on all n points. Here the graph
        is ``mcg_cf``'s (tiered micro-clusters, CF-weighted competitive-
        Hebbian edges; m <= 1000 nodes), its bottom eigenvectors are a dense
        m x m problem, and each point takes the eigenvector row of its
        nearest micro-cluster (``interp="hard"``) or a CF-kernel blend of its
        two nearest (``"soft"``). LMM's EM then runs on the points as usual,
        so the soft assignments and the Yu-Shi initialisation are LMM's.
        ``interp="hard"`` was chosen on the unseen and n = 20k suites (round
        22); the two tie there.
        """

        def __init__(self, interp: str = "hard", mcg_params: Optional[dict] = None,
                     **kwargs: Any) -> None:
            super().__init__(**kwargs)
            if interp not in ("hard", "soft"):
                raise ValueError("interp must be 'hard' or 'soft'")
            self.name = "lmm_mcg"
            self.interp = interp
            self.mcg_params = dict(mcg_params or {})

        def _build_basis(self, X, rng):
            k = self._k_hint or 2
            mcg = Mcg_cf(random_state=self.random_state, **self.mcg_params)
            g = mcg.build_graph(X, k)
            W = g["W"] + 1e-9  # connect stray micro-clusters for the eigensolver
            dinv = 1.0 / np.sqrt(W.sum(axis=1))
            L = np.eye(len(W)) - dinv[:, None] * W * dinv[None, :]
            vals, V = np.linalg.eigh(0.5 * (L + L.T))
            me = int(min(self.n_frequencies, len(W) - 1))
            vals, V = np.clip(vals[:me], 0.0, None), V[:, :me]
            if self.interp == "hard":
                phi = V[g["a"]]
            else:
                X = np.asarray(X, dtype=np.float64)
                C = g["C"]
                d2 = np.stack([((X - C[g["a"]]) ** 2).sum(axis=1), ((X - C[g["b"]]) ** 2).sum(axis=1)], axis=1)
                idx = np.stack([g["a"], g["b"]], axis=1)
                w = np.exp(-(d2 - d2.min(axis=1, keepdims=True)) / (g["R2"][idx] + 1e-12))
                w /= w.sum(axis=1, keepdims=True)
                phi = (V[idx] * w[:, :, None]).sum(axis=1)
            rms = np.sqrt(np.mean(phi ** 2, axis=0)) + 1e-12
            phi = phi / rms[None, :]
            if self.row_normalize:
                phi = phi / np.maximum(np.linalg.norm(phi, axis=1, keepdims=True), 1e-12)
            return vals.reshape(-1, 1), phi, vals

        def _basis_action(self) -> dict:
            return {"type": "mcg_laplacian_basis", "n_eigvecs": int(self.n_frequencies),
                    "interp": self.interp}

    return Lmm_mcg


Lmm_mcg = _lmm_mcg_class()
