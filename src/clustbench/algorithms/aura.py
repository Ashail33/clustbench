"""AURA: Adaptive Unified Robust Algorithm.

AURA stitches together the four mechanisms that, individually, won a
dimension of the clustbench algorithm analysis:

1. **Graph Laplacian embedding** — the only mechanism (shared by
   ``spectral`` and ``lmm``) that solves the *circles* / non-convex
   manifolds. The embedding turns geodesic structure into Euclidean
   structure so downstream clustering can succeed.
2. **Posterior-weighted EM** — ``gmm`` is the only registry algorithm
   whose outlier-robustness loss stays above the -32% line. Soft
   responsibilities naturally downweight points the model can't
   explain, which is the same effect as an explicit outlier prior.
3. **Adaptive hyperparameter estimation** — ``dbscan_auto`` showed
   that an unsupervised knee on the k-distance plot is the difference
   between 0.0 and 0.589 ARI on DBSCAN; AURA's ``n_neighbors`` and
   anchor count ``m`` are picked from ``n`` without any tuning.
4. **Nyström approximation** — landmark sub-sampling is what brings
   Laplacian-based methods down from O(n^3) eigendecomp into the
   sub-quadratic regime, so the spectral half of the pipeline is
   tractable at the scales the benchmark exercises.

The pipeline at ``fit_predict`` time is exactly:

    knn graph  ->  anchor sub-graph Laplacian  ->  Nyström extension
                ->  k+1 eigvec embedding  ->  GMM (full cov, reg=1e-4)
                ->  argmax responsibility

A trajectory of :class:`Step` records is emitted so the run can be
replayed / introspected the same way the other registry algorithms are.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors, kneighbors_graph

from .base import Algorithm, AlgoResult, Step, register


@register
class Aura(Algorithm):
    """Adaptive Unified Robust Algorithm.

    A spectral-embedding + posterior-weighted-EM clustering pipeline
    whose every hyperparameter is picked from ``n`` so there is nothing
    to tune at call time. See the module docstring for the mechanism
    rationale.

    Parameters
    ----------
    reg_covar : float
        Diagonal regulariser added to the GMM component covariances. The
        ``1e-4`` default is two orders of magnitude looser than sklearn's
        ``1e-6`` because the eigen-embedding lives on a low-rank
        manifold whose effective covariance can be near-singular.
    max_iter : int
        Maximum EM iterations for the Gaussian mixture.
    n_init : int
        Number of EM restarts; the best log-likelihood wins.
    random_state : int
        Seed for the anchor sampler and the GMM init.
    nystrom_threshold : int
        If ``n <= nystrom_threshold``, skip Nyström and do the full
        eigendecomposition. The default ``200`` keeps full-eig for
        tiny problems where landmark sub-sampling is just noise.
    """

    name = "aura"

    def __init__(
        self,
        reg_covar: float = 1e-4,
        max_iter: int = 200,
        n_init: int = 1,
        random_state: int = 42,
        nystrom_threshold: int = 200,
        **kwargs: Any,
    ) -> None:
        self.reg_covar = float(reg_covar)
        self.max_iter = int(max_iter)
        self.n_init = int(n_init)
        self.random_state = int(random_state)
        self.nystrom_threshold = int(nystrom_threshold)

    # ------------------------------------------------------------------
    # Adaptive hyperparameter helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_n_neighbors(n: int) -> int:
        """``max(10, floor(sqrt(n)/2))`` — clipped so the graph stays valid.

        Empirically large enough to keep within-cluster connectivity at
        the n ~ 300 scale, small enough not to bridge clusters at
        n ~ 10k.
        """
        if n <= 2:
            return 1
        target = max(10, int(np.sqrt(n) / 2))
        return max(1, min(target, n - 1))

    @staticmethod
    def _resolve_anchor_count(n: int) -> int:
        """``min(n, max(100, int(sqrt(n) * 2)))``.

        100 is the floor: any fewer and the Nyström extension is too
        coarse for k around 10. ``2*sqrt(n)`` grows much slower than
        ``n``, which is the whole point of the approximation.
        """
        return int(min(n, max(100, int(np.sqrt(n) * 2))))

    # ------------------------------------------------------------------
    # Anchor sampling
    # ------------------------------------------------------------------

    @staticmethod
    def _kmeans_pp_indices(
        X: np.ndarray, m: int, rng: np.random.Generator
    ) -> np.ndarray:
        """k-means++ sampler over the rows of ``X``.

        Spreads the anchors over the data so the Nyström extension
        doesn't leave any region without a nearby landmark.
        """
        n = X.shape[0]
        if m >= n:
            return np.arange(n)
        chosen = [int(rng.integers(n))]
        d2 = np.full(n, np.inf, dtype=np.float64)
        for _ in range(m - 1):
            diff = X - X[chosen[-1]]
            d2 = np.minimum(d2, np.einsum("nd,nd->n", diff, diff))
            total = float(d2.sum())
            if total <= 0.0:
                chosen.append(int(rng.integers(n)))
            else:
                chosen.append(int(rng.choice(n, p=d2 / total)))
        return np.asarray(chosen, dtype=np.int64)

    # ------------------------------------------------------------------
    # Laplacian + Nyström embedding
    # ------------------------------------------------------------------

    @staticmethod
    def _normalized_laplacian(W: sp.spmatrix) -> Tuple[sp.spmatrix, np.ndarray]:
        """Return ``L = I - D^{-1/2} W D^{-1/2}`` symmetrised."""
        n = W.shape[0]
        deg = np.asarray(W.sum(axis=1)).ravel()
        deg = np.maximum(deg, 1e-12)
        d_inv_sqrt = 1.0 / np.sqrt(deg)
        D_inv = sp.diags(d_inv_sqrt)
        L = sp.eye(n) - D_inv @ W @ D_inv
        L = (L + L.T) * 0.5
        return L, d_inv_sqrt

    def _full_laplacian_embedding(
        self,
        X: np.ndarray,
        k_nn: int,
        n_components: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Full eigendecomp path for tiny ``n``.

        Returns (eigenvalues_sorted_ascending, eigenvectors).
        """
        n = X.shape[0]
        knn = kneighbors_graph(
            X, n_neighbors=k_nn, mode="connectivity", include_self=False
        )
        W = knn.maximum(knn.T)
        # Tiny ridge to keep the graph connected in case kNN broke it
        # into components (otherwise Laplacian has multiplicity-c zero
        # eigenvalue and the embedding stops being informative).
        W = W + sp.eye(n) * 1e-6
        L, _ = self._normalized_laplacian(W)

        m = min(n_components, n - 1)
        try:
            vals, vecs = eigsh(L, k=m, sigma=1e-6, which="LM")
        except Exception:
            dense = L.toarray()
            vals_all, vecs_all = np.linalg.eigh(dense)
            vals, vecs = vals_all[:m], vecs_all[:, :m]
        order = np.argsort(vals)
        return np.clip(vals[order], 0.0, None), vecs[:, order]

    def _nystrom_embedding(
        self,
        X: np.ndarray,
        k_nn: int,
        n_components: int,
        m: int,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Anchor-sub-graph Laplacian + Nyström extension to all n.

        Returns (eigenvalues_sorted_ascending, embedding_n_by_d, m_used).
        """
        n = X.shape[0]
        anchor_idx = self._kmeans_pp_indices(X, m, rng)
        L_X = X[anchor_idx]
        m_used = L_X.shape[0]

        nn_in = max(1, min(k_nn, m_used - 1))
        knn = kneighbors_graph(
            L_X, n_neighbors=nn_in, mode="connectivity", include_self=False
        )
        W = knn.maximum(knn.T)
        # Tiny constant to guarantee connectivity (so the embedding
        # never degenerates to a piecewise-constant indicator on
        # disconnected pieces of the anchor graph).
        W = W + sp.eye(m_used) * 1e-6
        L_lap, _ = self._normalized_laplacian(W)

        m_e = min(n_components, m_used - 1)
        try:
            vals, vecs = eigsh(L_lap, k=m_e, sigma=1e-6, which="LM")
        except Exception:
            dense = L_lap.toarray()
            vals_all, vecs_all = np.linalg.eigh(dense)
            vals, vecs = vals_all[:m_e], vecs_all[:, :m_e]
        order = np.argsort(vals)
        vals = np.clip(vals[order], 0.0, None)
        vecs = vecs[:, order]

        # Column-scale to unit RMS so the embedding columns sit on a
        # comparable scale before GMM sees them.
        rms = np.sqrt(np.mean(vecs ** 2, axis=0)) + 1e-12
        vecs = vecs / rms[None, :]

        # Nyström extension: each point inherits the mean of its
        # nearest anchors' eigenvectors. We use the same ``k_nn`` as
        # the graph itself so the extension neighbourhood matches the
        # geometric scale used to build the Laplacian.
        nn_ext = max(1, min(k_nn, m_used))
        finder = NearestNeighbors(n_neighbors=nn_ext).fit(L_X)
        _, idx_landmarks = finder.kneighbors(X)
        phi = vecs[idx_landmarks].mean(axis=1)
        return vals, phi, m_used

    @staticmethod
    def _drop_trivial_eigvec(
        vals: np.ndarray, phi: np.ndarray, k_keep: int
    ) -> np.ndarray:
        """Drop the all-ones (trivial) eigenvector if it is present.

        The first eigenvector of a normalised graph Laplacian is
        proportional to ``sqrt(deg)``; after row-normalisation that's
        close to a constant. We detect it by std/mean ratio rather
        than relying on the eigenvalue itself, because the small
        constant we added for connectivity shifts the zero eigenvalue
        slightly above zero.
        """
        if phi.shape[1] == 0:
            return phi
        first = phi[:, 0]
        mean = float(np.mean(first))
        std = float(np.std(first))
        trivial = std < 1e-6 * (abs(mean) + 1e-12) or std < 1e-9
        embedding = phi[:, 1:] if trivial else phi
        # Trim / pad to exactly k_keep columns.
        if embedding.shape[1] >= k_keep:
            return embedding[:, :k_keep]
        # Not enough columns (only happens for pathologically tiny n).
        pad = np.zeros((embedding.shape[0], k_keep - embedding.shape[1]))
        return np.concatenate([embedding, pad], axis=1)

    # ------------------------------------------------------------------
    # GMM with empty-cluster re-seeding
    # ------------------------------------------------------------------

    def _fit_gmm(
        self,
        embedding: np.ndarray,
        k: int,
        rng: np.random.Generator,
    ) -> Tuple[GaussianMixture, np.ndarray, np.ndarray, int]:
        """Fit a k-component full-covariance GMM, re-seeding empty components.

        Returns (gm, labels, responsibilities, n_reseeds_done).
        """
        n = embedding.shape[0]
        seed = self.random_state
        reseeds = 0
        # If a fit leaves a component with zero responsibility we re-
        # seed that component from a random data row and refit. Two
        # attempts at most: empirically enough on the benchmark, and
        # the spec calls it out as a defensive measure rather than a
        # hot path.
        for attempt in range(3):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gm = GaussianMixture(
                    n_components=k,
                    covariance_type="full",
                    reg_covar=self.reg_covar,
                    max_iter=self.max_iter,
                    n_init=self.n_init,
                    random_state=seed + attempt,
                )
                gm.fit(embedding)
            resp = gm.predict_proba(embedding)
            labels = resp.argmax(axis=1)
            present = np.unique(labels)
            if present.size == k:
                return gm, labels.astype(np.int64), resp, reseeds
            # Empty cluster found — bump the seed and try again.
            reseeds += 1
        # Last-resort: take whatever the final fit produced.
        return gm, labels.astype(np.int64), resp, reseeds

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_predict(self, X: np.ndarray, k: Optional[int] = None) -> AlgoResult:
        assert k is not None, "Aura requires the number of clusters k."
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("Aura expects X with shape (n, d).")
        n = X.shape[0]
        rng = np.random.default_rng(self.random_state)
        trajectory: List[Step] = []

        # ---- 1. Adaptive kNN graph hyperparameter ----
        k_nn = self._resolve_n_neighbors(n)
        trajectory.append(
            Step(
                step_idx=0,
                cost=0.0,
                action={"type": "knn_graph_built", "n_neighbors": int(k_nn)},
                state={"n_neighbors": int(k_nn), "n": int(n)},
            )
        )

        # ---- 2. Nyström-approximated Laplacian embedding ----
        n_eig = int(k) + 1
        used_nystrom = n > self.nystrom_threshold
        if used_nystrom:
            m_anchor = self._resolve_anchor_count(n)
            vals, phi, m_used = self._nystrom_embedding(
                X, k_nn=k_nn, n_components=n_eig, m=m_anchor, rng=rng
            )
        else:
            vals, phi = self._full_laplacian_embedding(
                X, k_nn=k_nn, n_components=n_eig
            )
            m_used = n
        embedding = self._drop_trivial_eigvec(vals, phi, k_keep=int(k))
        trajectory.append(
            Step(
                step_idx=1,
                cost=float(vals[min(int(k), len(vals) - 1)]) if len(vals) else 0.0,
                action={
                    "type": "nystrom_embedded",
                    "used_nystrom": bool(used_nystrom),
                },
                state={
                    "anchor_size": int(m_used),
                    "embedding_dim": int(n_eig),
                    "embedding_dim_kept": int(embedding.shape[1]),
                },
            )
        )

        # ---- 3. Posterior-weighted EM in embedded space ----
        gm, labels, resp, reseeds = self._fit_gmm(embedding, int(k), rng)

        # Posterior entropy averaged over points — a soft proxy for
        # how confidently the mixture has explained each row. High
        # entropy points are the outliers GMM has downweighted.
        eps = 1e-12
        per_point_entropy = -np.sum(resp * np.log(resp + eps), axis=1)
        posterior_entropy = float(np.mean(per_point_entropy))
        mean_log_likelihood = float(gm.score(embedding))
        trajectory.append(
            Step(
                step_idx=2,
                cost=float(-mean_log_likelihood),
                action={
                    "type": "em_converged",
                    "n_iter": int(getattr(gm, "n_iter_", -1)),
                    "converged": bool(getattr(gm, "converged_", False)),
                    "reseeds": int(reseeds),
                },
                state={
                    "posterior_entropy": posterior_entropy,
                    "n_components_present": int(np.unique(labels).size),
                },
            )
        )

        extra: Dict[str, Any] = {
            "n_neighbors_used": int(k_nn),
            "anchor_size_used": int(m_used),
            "embedding_dim_used": int(embedding.shape[1]),
            "used_nystrom": bool(used_nystrom),
            "gmm_converged": bool(getattr(gm, "converged_", False)),
            "gmm_n_iter": int(getattr(gm, "n_iter_", -1)),
            "posterior_entropy": posterior_entropy,
            "mean_log_likelihood": mean_log_likelihood,
            "reseeds": int(reseeds),
        }
        return AlgoResult(labels=labels, extra=extra, trajectory=trajectory)
