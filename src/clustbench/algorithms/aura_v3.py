"""AURA v3: route between v1's GMM and v2's z-scored k-means on effective rank.

Both v1 (``aura``) and v2 (``aura_v2``) share the same first stage — an
adaptive kNN graph, Nyström-approximated normalised Laplacian, and a
projection of every row into the leading eigen-embedding ``Y``. They
differ only in the post-embedding step:

* v1 fits a **full-covariance Gaussian mixture** on ``Y`` directly. This
  works well when ``Y`` is effectively one-dimensional plus noise (the
  GMM happily models curved structure along the one informative axis),
  but it collapses concentric rings into a single elongated Gaussian on
  ``circles`` (ARI 0.00) because two independent eigen-axes are both
  informative there.
* v2 z-scores ``Y`` column-wise and runs k-means. This recovers
  ``circles`` perfectly (ARI 1.00) — z-scoring restores the
  scale-equivariance spectral clustering's final k-means assumes —
  but on ``moons`` the embedding is one-dimensional plus near-noise
  low-variance columns; z-scoring blows up the noise, so the k-means
  partition becomes arbitrary (ARI 0.01).

v3 simply dispatches between the two post-steps based on the embedding's
**effective rank**: how many eigen-columns have a standard deviation at
least ``effective_rank_threshold * col_stds.max()``. With the default
threshold of 0.1:

* ``effective_rank >= 2`` -> z-scored k-means (v2 path). Multiple
  informative axes are present; z-scoring is safe and necessary.
* ``effective_rank < 2`` -> raw GMM (v1 path). One dominant direction;
  let GMM model structure along it rather than amplifying noise.

The embedding code is copied verbatim from v1 (not delegated, to avoid
recomputing the eigendecomposition).
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors, kneighbors_graph

from .base import Algorithm, AlgoResult, Step, register


@register
class Aura_v3(Algorithm):
    """AURA v3 — same embedding as v1/v2, post-step chosen by effective rank.

    Parameters
    ----------
    effective_rank_threshold : float
        Fraction of the largest per-column embedding std below which an
        eigen-direction is considered uninformative. ``effective_rank`` is
        the count of columns whose std exceeds this fraction times
        ``col_stds.max()``. The default ``0.1`` reproduces the dispatch
        described in ``docs/ALGORITHM_ANALYSIS.md``.
    reg_covar : float
        Diagonal regulariser for the GMM post-step (v1 path).
    max_iter : int
        Iteration cap for both the GMM (v1 path) and the k-means
        (v2 path) post-step.
    n_init : int
        Number of restarts for the chosen post-step.
    random_state : int
        Seed for the anchor sampler, the GMM init, and the k-means init.
    nystrom_threshold : int
        ``n > threshold`` triggers the Nyström extension; otherwise the
        full eigendecomposition runs. Default ``1000`` matches v2; below
        that, Nyström over a small anchor set was empirically smoothing
        away the non-convex (moons / circles) structure.
    """

    name = "aura_v3"

    def __init__(
        self,
        effective_rank_threshold: float = 0.1,
        reg_covar: float = 1e-4,
        max_iter: int = 200,
        n_init: int = 5,
        random_state: int = 42,
        nystrom_threshold: int = 1000,
        **kwargs: Any,
    ) -> None:
        self.effective_rank_threshold = float(effective_rank_threshold)
        self.reg_covar = float(reg_covar)
        self.max_iter = int(max_iter)
        self.n_init = int(n_init)
        self.random_state = int(random_state)
        self.nystrom_threshold = int(nystrom_threshold)

    # ------------------------------------------------------------------
    # Adaptive hyperparameter helpers (verbatim from v1)
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_n_neighbors(n: int) -> int:
        if n <= 2:
            return 1
        target = max(10, int(np.sqrt(n) / 2))
        return max(1, min(target, n - 1))

    @staticmethod
    def _resolve_anchor_count(n: int) -> int:
        return int(min(n, max(100, int(np.sqrt(n) * 2))))

    # ------------------------------------------------------------------
    # Anchor sampling (verbatim from v1)
    # ------------------------------------------------------------------

    @staticmethod
    def _kmeans_pp_indices(
        X: np.ndarray, m: int, rng: np.random.Generator
    ) -> np.ndarray:
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
    # Laplacian + Nyström embedding (verbatim from v1)
    # ------------------------------------------------------------------

    @staticmethod
    def _normalized_laplacian(W: sp.spmatrix) -> Tuple[sp.spmatrix, np.ndarray]:
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
        n = X.shape[0]
        knn = kneighbors_graph(
            X, n_neighbors=k_nn, mode="connectivity", include_self=False
        )
        W = knn.maximum(knn.T)
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
        n = X.shape[0]
        anchor_idx = self._kmeans_pp_indices(X, m, rng)
        L_X = X[anchor_idx]
        m_used = L_X.shape[0]

        nn_in = max(1, min(k_nn, m_used - 1))
        knn = kneighbors_graph(
            L_X, n_neighbors=nn_in, mode="connectivity", include_self=False
        )
        W = knn.maximum(knn.T)
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

        rms = np.sqrt(np.mean(vecs ** 2, axis=0)) + 1e-12
        vecs = vecs / rms[None, :]

        nn_ext = max(1, min(k_nn, m_used))
        finder = NearestNeighbors(n_neighbors=nn_ext).fit(L_X)
        _, idx_landmarks = finder.kneighbors(X)
        phi = vecs[idx_landmarks].mean(axis=1)
        return vals, phi, m_used

    @staticmethod
    def _drop_trivial_eigvec(
        vals: np.ndarray, phi: np.ndarray, k_keep: int
    ) -> np.ndarray:
        if phi.shape[1] == 0:
            return phi
        first = phi[:, 0]
        mean = float(np.mean(first))
        std = float(np.std(first))
        trivial = std < 1e-6 * (abs(mean) + 1e-12) or std < 1e-9
        embedding = phi[:, 1:] if trivial else phi
        if embedding.shape[1] >= k_keep:
            return embedding[:, :k_keep]
        pad = np.zeros((embedding.shape[0], k_keep - embedding.shape[1]))
        return np.concatenate([embedding, pad], axis=1)

    # ------------------------------------------------------------------
    # Embedding driver
    # ------------------------------------------------------------------

    def _build_embedding(
        self, X: np.ndarray, k: int, rng: np.random.Generator
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Run the full v1 embedding pipeline once; return embedding + info."""
        n = X.shape[0]
        k_nn = self._resolve_n_neighbors(n)
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
        info = {
            "n_neighbors": int(k_nn),
            "anchor_size": int(m_used),
            "embedding_dim": int(n_eig),
            "embedding_dim_kept": int(embedding.shape[1]),
            "used_nystrom": bool(used_nystrom),
            "leading_eigval": (
                float(vals[min(int(k), len(vals) - 1)]) if len(vals) else 0.0
            ),
        }
        return embedding, info

    # ------------------------------------------------------------------
    # Post-step A: GMM on raw embedding (v1 path)
    # ------------------------------------------------------------------

    def _fit_gmm(
        self, embedding: np.ndarray, k: int
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Full-covariance GMM with empty-cluster re-seeding (v1 recipe)."""
        seed = self.random_state
        reseeds = 0
        gm: Optional[GaussianMixture] = None
        labels = np.zeros(embedding.shape[0], dtype=np.int64)
        for attempt in range(3):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gm = GaussianMixture(
                    n_components=k,
                    covariance_type="full",
                    reg_covar=self.reg_covar,
                    max_iter=self.max_iter,
                    n_init=1,
                    random_state=seed + attempt,
                )
                gm.fit(embedding)
            resp = gm.predict_proba(embedding)
            labels = resp.argmax(axis=1).astype(np.int64)
            if np.unique(labels).size == k:
                break
            reseeds += 1
        assert gm is not None
        mean_log_likelihood = float(gm.score(embedding))
        info = {
            "post": "gmm",
            "converged": bool(getattr(gm, "converged_", False)),
            "n_iter": int(getattr(gm, "n_iter_", -1)),
            "reseeds": int(reseeds),
            "mean_log_likelihood": mean_log_likelihood,
            "cost": float(-mean_log_likelihood),
        }
        return labels, info

    # ------------------------------------------------------------------
    # Post-step B: k-means on z-scored embedding (v2 path)
    # ------------------------------------------------------------------

    @staticmethod
    def _zscore(X: np.ndarray) -> np.ndarray:
        mu = X.mean(axis=0, keepdims=True)
        sigma = X.std(axis=0, keepdims=True)
        sigma = np.where(sigma < 1e-12, 1.0, sigma)
        return (X - mu) / sigma

    def _fit_kmeans_z(
        self, embedding: np.ndarray, k: int
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Z-score columns then k-means (v2's circles-solving recipe)."""
        Z = self._zscore(embedding)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            km = KMeans(
                n_clusters=k,
                init="k-means++",
                n_init=self.n_init,
                max_iter=self.max_iter,
                random_state=self.random_state,
            )
            km.fit(Z)
        labels = km.labels_.astype(np.int64)
        info = {
            "post": "kmeans_z",
            "inertia": float(km.inertia_),
            "n_iter": int(km.n_iter_),
            "cost": float(km.inertia_),
        }
        return labels, info

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_predict(self, X: np.ndarray, k: Optional[int] = None) -> AlgoResult:
        assert k is not None, "Aura_v3 requires the number of clusters k."
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("Aura_v3 expects X with shape (n, d).")
        rng = np.random.default_rng(self.random_state)
        trajectory: List[Step] = []

        # ---- 1. Build the spectral embedding (shared v1/v2 stage) ----
        embedding, embed_info = self._build_embedding(X, int(k), rng)
        col_stds = embedding.std(axis=0)
        col_stds_list = [float(s) for s in col_stds]
        max_std = float(col_stds.max()) if col_stds.size else 0.0
        cutoff = self.effective_rank_threshold * max_std
        effective_rank = int(np.sum(col_stds > cutoff)) if col_stds.size else 0
        trajectory.append(
            Step(
                step_idx=0,
                cost=float(embed_info["leading_eigval"]),
                action={"type": "embedding_built"},
                state={
                    "n_neighbors": int(embed_info["n_neighbors"]),
                    "anchor_size": int(embed_info["anchor_size"]),
                    "embedding_dim": int(embed_info["embedding_dim_kept"]),
                    "col_stds": col_stds_list,
                    "effective_rank": int(effective_rank),
                },
            )
        )

        # ---- 2. Router: pick the post-step based on effective rank ----
        chose = "kmeans_z" if effective_rank >= 2 else "gmm"
        trajectory.append(
            Step(
                step_idx=1,
                cost=0.0,
                action={
                    "type": "router",
                    "effective_rank": int(effective_rank),
                    "chose": chose,
                },
                state={
                    "effective_rank": int(effective_rank),
                    "effective_rank_threshold": float(
                        self.effective_rank_threshold
                    ),
                    "chose": chose,
                },
            )
        )

        # ---- 3. Run the chosen post-step ----
        if chose == "gmm":
            labels, post_info = self._fit_gmm(embedding, int(k))
        else:
            labels, post_info = self._fit_kmeans_z(embedding, int(k))

        trajectory.append(
            Step(
                step_idx=2,
                cost=float(post_info["cost"]),
                action={
                    "type": "post_step_converged",
                    "post": post_info["post"],
                    "n_iter": int(post_info.get("n_iter", -1)),
                },
                state={
                    "n_components_present": int(np.unique(labels).size),
                    **{
                        kk: vv
                        for kk, vv in post_info.items()
                        if kk not in {"post", "cost"}
                    },
                },
            )
        )

        extra: Dict[str, Any] = {
            "effective_rank": int(effective_rank),
            "chose": chose,
            "col_stds": col_stds_list,
            "effective_rank_threshold": float(self.effective_rank_threshold),
            "n_neighbors_used": int(embed_info["n_neighbors"]),
            "anchor_size_used": int(embed_info["anchor_size"]),
            "embedding_dim_used": int(embed_info["embedding_dim_kept"]),
            "used_nystrom": bool(embed_info["used_nystrom"]),
            "post_step_info": post_info,
        }
        return AlgoResult(labels=labels, extra=extra, trajectory=trajectory)
