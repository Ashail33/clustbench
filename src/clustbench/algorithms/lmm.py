"""LMM: Laplacian-basis exponential-family mixture model.

Same EM machinery as :class:`~clustbench.algorithms.fmm.Fmm` but the
random Fourier basis is replaced by the bottom eigenvectors of the
normalised k-NN graph Laplacian. Each component's log-density becomes

    log p_k(x_i) ∝ alpha_k . psi(x_i),

where ``psi(x_i)`` is the ``i``-th row of the eigenvector matrix —
exactly the embedding spectral clustering builds before running
k-means in eigen-space, except here cluster assignments are soft and
the basis weights are EM-trained.

The heat kernel ``W_m(tau_k) = exp(-tau_k * lambda_m / 2)`` (where
``lambda_m`` is the ``m``-th smallest Laplacian eigenvalue) is then
literal *graph diffusion*: small ``tau_k`` keeps fine-grained
neighbourhood structure, large ``tau_k`` averages each cluster across
its connected component. This is the right notion of "bandwidth" for
non-convex data — moons, circles, manifolds — where Euclidean
distance is misleading.

The class inherits everything from :class:`Fmm` and only swaps in
:meth:`_build_basis` plus the trajectory action tag.
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, lobpcg
from sklearn.neighbors import kneighbors_graph

from .base import register
from .fmm import Fmm


@register
class Lmm(Fmm):
    """k-NN graph Laplacian basis mixture model.

    Parameters
    ----------
    n_eigvecs : int or "auto"
        Number of Laplacian eigenvectors to use as the basis. ``"auto"``
        sets it to ``k`` at fit time — the standard spectral-clustering
        choice, where the bottom-``k`` eigenvectors are the cluster
        indicators in the ideal-graph case. Extra eigenvectors past
        that just give EM room to overfit. Override with an int if you
        want explicit control.
    n_neighbors : int or "auto"
        k for the k-NN graph that defines the Laplacian. ``"auto"`` uses
        ``max(10, ⌊4·√(log2(n))⌋)`` — a gentle scaling that keeps the
        neighbourhood ~10 for moderate ``n`` and grows slowly with
        dataset size. Empirically the sweet spot: too few breaks
        within-cluster connectivity, too many lets the graph bridge
        clusters.
    affinity : {"binary", "heat"}
        Edge weighting:
          * ``"binary"`` — connect each point to its ``n_neighbors``
            nearest neighbours with weight 1; symmetrised by
            ``W <- max(W, W.T)`` (standard spectral-clustering choice).
          * ``"heat"`` — weights ``exp(-dist^2 / sigma^2)`` with sigma
            set to the median k-NN distance.
    row_normalize : bool
        If True, normalise each row of the eigenvector matrix to unit
        L2 norm (Ng-Jordan-Weiss spectral-clustering convention). This
        makes the per-point feature vectors directionally comparable
        and is what lets spectral methods cleanly separate
        non-convex shapes.
    Other parameters are inherited from :class:`Fmm` (``max_iter``,
    ``l2``, ``adaptive_l2``, ``damping``, ``learn_bandwidth``, etc.).
    """

    def __init__(
        self,
        n_eigvecs: Any = "auto",
        n_neighbors: Any = "auto",
        affinity: str = "binary",
        row_normalize: bool = True,
        **kwargs: Any,
    ) -> None:
        # ``n_frequencies`` in the Fmm parent is reused as the basis
        # dimensionality so feature_dim / param-count plumbing works.
        kwargs.setdefault("n_scales", 1)
        # Disabled by default for LMM: the basis is already eigenvalue-
        # weighted, so per-cluster heat-kernel scaling is redundant. The
        # tau line search is ~30% of LMM's fit time with no observed
        # quality benefit (identical ARI across the benchmark with
        # ``learn_bandwidth=True``).
        kwargs.setdefault("learn_bandwidth", False)
        kwargs.setdefault("tau_init", 0.0)
        # ``n_frequencies`` is set per-fit (in ``_build_basis``) because
        # "auto" depends on ``k``; the constructor seeds a safe default.
        seed_dim = n_eigvecs if isinstance(n_eigvecs, int) else 8
        super().__init__(n_frequencies=seed_dim, **kwargs)
        self.name = "lmm"
        self.n_eigvecs = n_eigvecs
        self.n_neighbors = n_neighbors
        self.affinity = affinity
        self.row_normalize = row_normalize
        self._k_hint: int | None = None

    def _resolve_n_eigvecs(self, k: int | None) -> int:
        if isinstance(self.n_eigvecs, int):
            return int(self.n_eigvecs)
        # "auto": ``k`` (spectral-clustering default).
        kk = k if k is not None else (self._k_hint or 2)
        return max(int(kk), 2)

    def _init_alpha(self, X, k, phi, rng):
        """Hand-rolled k-means++ + a few Lloyd iterations on ``phi``.

        ``phi`` has shape ``(n, k)`` for LMM and is already
        cluster-aligned (the bottom-k Laplacian eigenvectors are
        approximate cluster indicators), so the partition converges in
        a handful of Lloyd sweeps. Hand-rolling avoids sklearn's
        ~10 ms of dispatch / validation overhead per ``KMeans().fit``,
        which matters at our scale where the whole LMM fit is ~25-40 ms.
        """
        eps = 1e-12
        n, m2 = phi.shape

        # k-means++ seeding on phi.
        centers_idx = [int(rng.integers(n))]
        d2 = np.full(n, np.inf)
        for _ in range(k - 1):
            diff = phi - phi[centers_idx[-1]]
            d2 = np.minimum(d2, np.einsum("nm,nm->n", diff, diff))
            total = d2.sum()
            if total <= 0:
                centers_idx.append(int(rng.integers(n)))
            else:
                centers_idx.append(int(rng.choice(n, p=d2 / total)))
        centers = phi[np.asarray(centers_idx)].copy()                  # (k, m2)

        # Up to 10 Lloyd sweeps; usually converges in 3-5.
        labels = np.zeros(n, dtype=np.int64)
        for _ in range(10):
            # ||phi - centers||^2 = ||phi||^2 - 2 phi.centers^T + ||centers||^2
            phi_sq = np.einsum("nm,nm->n", phi, phi)
            c_sq = np.einsum("km,km->k", centers, centers)
            D = phi_sq[:, None] - 2.0 * phi @ centers.T + c_sq[None, :]  # (n, k)
            new_labels = D.argmin(axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            new_centers = centers.copy()
            for j in range(k):
                mask = labels == j
                if mask.any():
                    new_centers[j] = phi[mask].mean(axis=0)
            centers = new_centers

        global_mean = phi.mean(axis=0)
        phi_std = phi.std(axis=0) + eps
        init_strength = 1.0 / np.sqrt(m2)
        alpha = init_strength * (centers - global_mean) / phi_std
        alpha = alpha + rng.normal(scale=1e-3, size=alpha.shape)
        return (
            alpha,
            {"type": "kmeans_on_eigenvectors", "n_clusters": int(k)},
            {"cost": float(D[np.arange(n), labels].sum())},
        )

    def _resolve_n_neighbors(self, n: int) -> int:
        if isinstance(self.n_neighbors, int):
            return max(1, min(self.n_neighbors, n - 1))
        # "auto" — gentle scaling that keeps nn near 10 for moderate n.
        auto = int(4 * np.sqrt(np.log2(max(n, 2))))
        return max(10, min(auto, n - 1))

    def _basis_action(self) -> dict:
        return {
            "type": "laplacian_basis",
            "n_eigvecs": int(self.n_frequencies),
            "n_neighbors": self.n_neighbors,
            "affinity": str(self.affinity),
            "row_normalize": bool(self.row_normalize),
        }

    def fit_predict(self, X, k=None):
        # Stash k so ``_build_basis`` can size the basis to it.
        self._k_hint = k
        self.n_frequencies = self._resolve_n_eigvecs(k)
        result = super().fit_predict(X, k=k)
        if result.extra is not None:
            result.extra["n_neighbors_used"] = self._resolve_n_neighbors(X.shape[0])
            result.extra["n_eigvecs_used"] = self.n_frequencies
        return result

    def _build_basis(
        self, X: np.ndarray, rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the bottom ``n_eigvecs`` eigenvectors of the normalised
        k-NN graph Laplacian. Eigenvalues replace ``||omega||^2`` so the
        heat kernel becomes ``exp(-tau * lambda / 2)`` — graph diffusion.
        """
        n = X.shape[0]
        m_eig = int(self.n_frequencies)
        k_nn = self._resolve_n_neighbors(n)

        if self.affinity == "heat":
            knn = kneighbors_graph(X, n_neighbors=k_nn, mode="distance", include_self=False)
            knn_sym = knn.maximum(knn.T)
            sigma = float(np.median(knn.data)) if knn.data.size else 1.0
            W = knn_sym.copy()
            W.data = np.exp(-(W.data ** 2) / max(sigma ** 2, 1e-12))
        else:
            knn = kneighbors_graph(X, n_neighbors=k_nn, mode="connectivity", include_self=False)
            W = knn.maximum(knn.T)

        # Normalised Laplacian L = I - D^{-1/2} W D^{-1/2}.
        deg = np.asarray(W.sum(axis=1)).ravel()
        deg = np.maximum(deg, 1e-12)
        d_inv_sqrt = 1.0 / np.sqrt(deg)
        D_inv_sqrt = sp.diags(d_inv_sqrt)
        L = sp.eye(n) - D_inv_sqrt @ W @ D_inv_sqrt
        # Symmetrise to floating-point exactness.
        L = (L + L.T) * 0.5

        # Smallest ``m`` eigenvalues. LOBPCG is the practical choice
        # here: it operates on ``L`` purely through sparse matvecs,
        # whereas ARPACK shift-invert (``eigsh(L, sigma=1e-6)``) needs
        # to factor ``L - sigma*I`` which dominates the runtime for
        # n > a few thousand. At n=5000 LOBPCG is ~40x faster than
        # shift-invert ARPACK on these Laplacians; at n=10000 ~180x.
        m = min(m_eig, n - 1)
        # Seed LOBPCG with the constant eigenvector (always the bottom
        # of a connected-component Laplacian) plus random orthonormal
        # columns — convergence is markedly more stable than purely
        # random init.
        X0 = np.empty((n, m))
        X0[:, 0] = 1.0 / np.sqrt(n)
        if m > 1:
            rest = rng.standard_normal((n, m - 1))
            # Project out the constant direction so QR keeps it.
            rest -= rest.mean(axis=0, keepdims=True)
            q, _ = np.linalg.qr(rest)
            X0[:, 1:] = q
        try:
            vals, vecs = lobpcg(
                L, X0, largest=False, tol=1e-5, maxiter=200, verbosityLevel=0
            )
        except Exception:
            # Fallback paths: ARPACK shift-invert for small n, then dense.
            try:
                vals, vecs = eigsh(L, k=m, sigma=1e-6, which="LM")
            except Exception:
                dense = L.toarray()
                vals_all, vecs_all = np.linalg.eigh(dense)
                vals, vecs = vals_all[:m], vecs_all[:, :m]
        order = np.argsort(vals)
        vals = np.clip(vals[order], 0.0, None)
        vecs = vecs[:, order]

        # Scale eigenvectors so each column has unit RMS. This keeps the
        # softmax in its responsive regime (mirrors the std-floor scaling
        # used in Fmm's k-means warm start).
        rms = np.sqrt(np.mean(vecs ** 2, axis=0)) + 1e-12
        phi = vecs / rms[None, :]

        if self.row_normalize:
            # Ng-Jordan-Weiss normalisation: each row to unit L2.
            row_norms = np.linalg.norm(phi, axis=1, keepdims=True)
            phi = phi / np.maximum(row_norms, 1e-12)

        # ``omega`` is opaque to the EM loop; pack the eigenvalues into it
        # so trajectory introspection / debugging can see the spectrum.
        omega = vals.reshape(-1, 1)
        omega_norm_sq = vals  # heat kernel: exp(-tau * lambda / 2)
        return omega, phi, omega_norm_sq
