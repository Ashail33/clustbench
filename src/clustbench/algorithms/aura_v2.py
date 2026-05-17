"""AURA v2: keep v1's spectral embedding, replace GMM with a two-track
post-embedding strategy.

Motivation
----------
AURA v1 stitched a Laplacian-based spectral embedding (adaptive kNN graph +
Nyström-approximated normalised Laplacian) to a full-covariance Gaussian
mixture in the embedded space. On the 16-task benchmark v1 reached ARI rank
8/25, *solving* anisotropic (1.00) and mdcgen (0.87) but *failing* on moons
(0.37) and circles (0.00) — exactly where ``spectral`` and ``lmm`` win.

The post-mortem in ``docs/ALGORITHM_ANALYSIS.md`` traces the failure to the
GMM step: with full covariances, EM happily absorbs both concentric rings
into one elongated Gaussian. The Laplacian eigenvectors *do* separate the
loops into linearly separable clusters in eigen-space, the wheels just come
off when full-covariance EM is asked to label them.

v2 therefore keeps the embedding stage of v1 verbatim and swaps in a
post-embedding step that mimics what ``spectral`` does at the end:

- **Track A (always run, default)** — k-means with a *trimmed-mean centroid
  update* (drops the ``trim`` fraction of cluster members farthest from the
  centroid before averaging). This is the same recipe as
  :class:`~clustbench.algorithms.improvements.Kmeans_trimmed` and stays
  centroid-based, so it never falls into the "merge two rings into one
  elongated Gaussian" trap.
- **Track B (gated, optional)** — *tied* or *spherical* covariance Gaussian
  mixture in the embedded space. The constraint on covariance prevents the
  v1 collapse while still giving EM a chance to win on the cases where it
  was already winning. It is gated on the data's outlier signal: if LOF's
  outlier fraction on the *original* ``X`` exceeds ``track_b_lof_thresh``
  (default 0.05) Track B is run *and* compared against Track A via
  silhouette score on the embedded points; otherwise Track A is returned
  unchanged.

The trim fraction defaults to 0.10 so the effective contamination tolerated
caps at ~20% (the standard 2x rule for trimmed estimators).
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor

from .aura import Aura
from .base import Algorithm, AlgoResult, Step, register


@register
class Aura_v2(Algorithm):
    """AURA v2 — v1's spectral embedding + a two-track post-embedding stage.

    Parameters
    ----------
    trim : float
        Fraction of cluster members dropped (those farthest from the
        centroid) before averaging in the trimmed-mean k-means M-step.
        Default ``0.10`` — capped contamination ~20% (2x trim).
    max_iter : int
        Maximum k-means EM iterations on Track A.
    n_init : int
        Number of k-means restarts; the best inertia wins.
    track_b_lof_thresh : float
        Outlier-signal threshold above which Track B is also run and
        compared. Computed as the LOF outlier fraction on the original
        ``X`` (one-line with :class:`sklearn.neighbors.LocalOutlierFactor`).
    track_b_covariance_type : {"tied", "spherical"}
        Covariance constraint for the Track B GaussianMixture. The whole
        point of v2 is that *full* covariances broke v1, so we forbid
        it here.
    random_state : int
        Seed for the embedding sampler, k-means restarts, and the GMM init.
    nystrom_threshold : int
        Forwarded to the embedded :class:`Aura` instance. ``n > threshold``
        triggers the Nyström extension; otherwise the full eigendecomp is
        used.
    reg_covar : float
        Diagonal regulariser for the Track B GaussianMixture.
    """

    name = "aura_v2"

    def __init__(
        self,
        trim: float = 0.10,
        max_iter: int = 200,
        n_init: int = 5,
        track_b_lof_thresh: float = 0.05,
        track_b_covariance_type: str = "tied",
        random_state: int = 42,
        nystrom_threshold: int = 1000,
        reg_covar: float = 1e-4,
        **kwargs: Any,
    ) -> None:
        if track_b_covariance_type not in ("tied", "spherical"):
            raise ValueError(
                "track_b_covariance_type must be 'tied' or 'spherical' "
                f"(got {track_b_covariance_type!r})."
            )
        self.trim = float(trim)
        self.max_iter = int(max_iter)
        self.n_init = int(n_init)
        self.track_b_lof_thresh = float(track_b_lof_thresh)
        self.track_b_covariance_type = track_b_covariance_type
        self.random_state = int(random_state)
        # v2 lifts v1's ``nystrom_threshold`` default from 200 to 1000.
        # Diagnosis: at n~300 the Nyström extension over ~100 anchors
        # smooths the eigenvectors enough to wash out non-convex (rings,
        # moons) class structure. The embedding code is unchanged; we
        # just route benchmark-scale problems through the full-eig path
        # by default, which is the *same code path* spectral and lmm
        # exercise. Above n=1000 Nyström kicks in again to keep
        # asymptotic cost sub-quadratic.
        self.nystrom_threshold = int(nystrom_threshold)
        self.reg_covar = float(reg_covar)

    # ------------------------------------------------------------------
    # Embedding stage (delegated to v1)
    # ------------------------------------------------------------------

    def _embed(
        self, X: np.ndarray, k: int, rng: np.random.Generator
    ) -> Tuple[np.ndarray, int, int, bool, float]:
        """Build v1's adaptive-kNN + Nyström-Laplacian embedding.

        Returns (embedding, k_nn, m_used, used_nystrom, leading_eigval).
        """
        helper = Aura(
            random_state=self.random_state,
            nystrom_threshold=self.nystrom_threshold,
        )
        n = X.shape[0]
        k_nn = helper._resolve_n_neighbors(n)
        n_eig = int(k) + 1
        used_nystrom = n > self.nystrom_threshold
        if used_nystrom:
            m_anchor = helper._resolve_anchor_count(n)
            vals, phi, m_used = helper._nystrom_embedding(
                X, k_nn=k_nn, n_components=n_eig, m=m_anchor, rng=rng
            )
        else:
            vals, phi = helper._full_laplacian_embedding(
                X, k_nn=k_nn, n_components=n_eig
            )
            m_used = n
        embedding = helper._drop_trivial_eigvec(vals, phi, k_keep=int(k))
        leading = (
            float(vals[min(int(k), len(vals) - 1)]) if len(vals) else 0.0
        )
        return embedding, int(k_nn), int(m_used), bool(used_nystrom), leading

    # ------------------------------------------------------------------
    # Outlier signal
    # ------------------------------------------------------------------

    def _lof_outlier_fraction(self, X: np.ndarray) -> float:
        """Fraction of rows LOF flags as outliers on the original ``X``.

        One-line wrapper around :class:`LocalOutlierFactor`; the
        ``contamination='auto'`` decision_function threshold is exactly
        the inlier/outlier boundary we want. Falls back to 0.0 if LOF
        chokes on a degenerate input (tiny ``n``, etc.).
        """
        n = X.shape[0]
        if n < 5:
            return 0.0
        n_neighbors = min(20, max(2, n - 1))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lof = LocalOutlierFactor(
                    n_neighbors=n_neighbors, contamination="auto"
                )
                pred = lof.fit_predict(X)
            return float(np.mean(pred == -1))
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Track A — trimmed-mean k-means in the embedded space
    # ------------------------------------------------------------------

    @staticmethod
    def _kpp_init(
        X: np.ndarray, k: int, rng: np.random.Generator
    ) -> np.ndarray:
        """k-means++ init in the embedded space (same recipe as
        :mod:`improvements`)."""
        n = X.shape[0]
        if k <= 0:
            return np.zeros((0, X.shape[1]), dtype=X.dtype)
        centers = [X[int(rng.integers(n))]]
        for _ in range(1, k):
            stacked = np.stack(centers, axis=0)
            d2 = np.min(
                np.sum((X[:, None, :] - stacked[None, :, :]) ** 2, axis=2),
                axis=1,
            )
            total = float(d2.sum())
            if total <= 0.0:
                idx = int(rng.integers(n))
            else:
                idx = int(rng.choice(n, p=d2 / total))
            centers.append(X[idx])
        return np.stack(centers, axis=0)

    @staticmethod
    def _zscore(X: np.ndarray) -> np.ndarray:
        """Column z-score the embedding so kmeans is scale-invariant.

        v1 surfaced its Laplacian eigenvectors with wildly different
        per-column scales: the leading "near-trivial" column tends to
        sit at ``std ~ 0.05`` while the next eigenvector is at
        ``std ~ 0.9``. Euclidean k-means weighs those columns by their
        variance, so the small-scale-but-discriminative columns get
        ignored — which is exactly the failure mode we observed on the
        circles dataset, where the per-class signal lives in the
        small-variance column. A column z-score restores
        scale-equivariance, which is what spectral clustering's final
        kmeans implicitly relies on.
        """
        mu = X.mean(axis=0, keepdims=True)
        sigma = X.std(axis=0, keepdims=True)
        sigma = np.where(sigma < 1e-12, 1.0, sigma)
        return (X - mu) / sigma

    def _run_trimmed_kmeans_once(
        self,
        X: np.ndarray,
        k: int,
        rng: np.random.Generator,
        trim: float,
    ) -> Tuple[np.ndarray, np.ndarray, float, int]:
        """One restart of trimmed-mean k-means.

        Mirrors :meth:`Kmeans_trimmed._run` but operates on the embedded
        space and skips trajectory recording (we record one summary
        :class:`Step` per track instead). ``trim`` is passed in so the
        caller can adapt it to the LOF outlier signal.
        """
        centroids = self._kpp_init(X, k, rng).astype(np.float64)
        prev_inertia: Optional[float] = None
        n_iter = 0
        tol = 1e-4
        for step_idx in range(self.max_iter):
            n_iter = step_idx + 1
            D = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
            labels = D.argmin(axis=1)
            inertia = float(D[np.arange(X.shape[0]), labels].sum())
            new_centroids = centroids.copy()
            for j in range(k):
                members = np.where(labels == j)[0]
                if len(members) == 0:
                    # Re-seed empty cluster at the data point farthest
                    # from any existing centroid (worst-explained row).
                    worst = int(np.argmax(np.min(D, axis=1)))
                    new_centroids[j] = X[worst]
                    continue
                dists = D[members, j]
                keep_n = max(1, int(len(members) * (1.0 - trim)))
                keep = members[np.argsort(dists)[:keep_n]]
                new_centroids[j] = X[keep].mean(axis=0)
            shift = float(np.linalg.norm(new_centroids - centroids))
            centroids = new_centroids
            if prev_inertia is not None and abs(inertia - prev_inertia) < tol:
                break
            if shift < tol:
                break
            prev_inertia = inertia
        D = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        labels = D.argmin(axis=1).astype(np.int64)
        inertia = float(D[np.arange(X.shape[0]), labels].sum())
        return labels, centroids, inertia, n_iter

    def _track_a(
        self,
        embedding: np.ndarray,
        k: int,
        rng: np.random.Generator,
        lof_frac: float,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Run ``n_init`` restarts of trimmed-mean k-means; return best.

        Two adaptations vs a vanilla trimmed-kmeans on the raw embedding:

        - The embedding is **column-zscored** first (see
          :meth:`_zscore`). v1's eigenvectors come out with mismatched
          per-column scales; without rescaling, k-means' Euclidean
          metric ignores the lower-variance columns even when those
          columns carry the discriminative signal.
        - The trim fraction is **capped by 2x the LOF outlier
          fraction** (``min(self.trim, 2 * lof_frac)``). The spec
          phrases this as "Cap effective contamination tolerated at
          2x trim ~ 20%"; we read it as a contract on the *effective*
          trim — if the data has zero outlier signal there is nothing
          to trim, and aggressive trimming hurts non-convex clusters
          by yanking the centroid away from cluster extremities (e.g.
          the curve endpoints of a moon).
        """
        embedding = self._zscore(embedding)
        effective_trim = float(min(self.trim, 2.0 * lof_frac))
        best: Optional[
            Tuple[np.ndarray, np.ndarray, float, int, int]
        ] = None
        for init_idx in range(self.n_init):
            labels, centroids, inertia, n_iter = self._run_trimmed_kmeans_once(
                embedding, k, rng, effective_trim
            )
            if best is None or inertia < best[2]:
                best = (labels, centroids, inertia, n_iter, init_idx)
        assert best is not None
        labels, _centroids, inertia, n_iter, init_idx = best
        return labels, {
            "inertia": float(inertia),
            "n_iter": int(n_iter),
            "best_init": int(init_idx),
            "trim": float(effective_trim),
            "trim_configured": float(self.trim),
        }

    # ------------------------------------------------------------------
    # Track B — constrained-covariance GMM in the embedded space
    # ------------------------------------------------------------------

    def _track_b(
        self, embedding: np.ndarray, k: int
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Tied / spherical Gaussian mixture in the embedded space.

        The embedding is column-zscored first for the same reason as
        Track A: the constrained-covariance GMM still implicitly uses a
        Euclidean-ish metric (especially in the ``spherical`` case) and
        benefits from balanced column scales.
        """
        embedding = self._zscore(embedding)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gm = GaussianMixture(
                n_components=k,
                covariance_type=self.track_b_covariance_type,
                reg_covar=self.reg_covar,
                max_iter=self.max_iter,
                n_init=1,
                random_state=self.random_state,
            )
            gm.fit(embedding)
        labels = gm.predict(embedding).astype(np.int64)
        return labels, {
            "covariance_type": self.track_b_covariance_type,
            "converged": bool(getattr(gm, "converged_", False)),
            "n_iter": int(getattr(gm, "n_iter_", -1)),
            "lower_bound": float(getattr(gm, "lower_bound_", float("nan"))),
        }

    # ------------------------------------------------------------------
    # Silhouette helper (safe fallback)
    # ------------------------------------------------------------------

    @staticmethod
    def _silhouette(embedding: np.ndarray, labels: np.ndarray) -> float:
        """Silhouette score with a safe fallback.

        Returns ``-inf`` if the labelling is degenerate (single cluster
        or every cluster size 1), which loses cleanly to any valid
        partition during track selection.
        """
        unique = np.unique(labels)
        if len(unique) < 2 or len(unique) >= len(labels):
            return float("-inf")
        try:
            return float(silhouette_score(embedding, labels))
        except Exception:
            return float("-inf")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_predict(self, X: np.ndarray, k: Optional[int] = None) -> AlgoResult:
        assert k is not None, "Aura_v2 requires the number of clusters k."
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("Aura_v2 expects X with shape (n, d).")
        n = X.shape[0]
        rng = np.random.default_rng(self.random_state)
        trajectory: List[Step] = []

        # ---- 1. Adaptive kNN graph hyperparameter ----
        # (the actual graph is built inside ``_embed``; we surface the
        # n_neighbors choice here so the trajectory matches v1's shape).
        k_nn = Aura._resolve_n_neighbors(n)
        trajectory.append(
            Step(
                step_idx=0,
                cost=0.0,
                action={"type": "knn_graph_built", "n_neighbors": int(k_nn)},
                state={"n_neighbors": int(k_nn), "n": int(n)},
            )
        )

        # ---- 2. Nyström-approximated Laplacian embedding ----
        embedding, k_nn_used, m_used, used_nystrom, leading_val = self._embed(
            X, int(k), rng
        )
        trajectory.append(
            Step(
                step_idx=1,
                cost=leading_val,
                action={
                    "type": "nystrom_embedded",
                    "used_nystrom": bool(used_nystrom),
                },
                state={
                    "anchor_size": int(m_used),
                    "embedding_dim_kept": int(embedding.shape[1]),
                },
            )
        )

        # ---- 3. Outlier signal: LOF outlier fraction on the original X ----
        lof_frac = self._lof_outlier_fraction(X)
        run_b = lof_frac > self.track_b_lof_thresh

        # ---- 4. Track A — trimmed-mean k-means in the embedded space ----
        labels_a, info_a = self._track_a(embedding, int(k), rng, lof_frac)
        # Silhouettes are computed on the same z-scored embedding both
        # tracks actually consume, so Track A and Track B scores are
        # comparable on the metric the EM / k-means objective is using.
        scoring_embedding = self._zscore(embedding)
        sil_a = self._silhouette(scoring_embedding, labels_a)
        trajectory.append(
            Step(
                step_idx=len(trajectory),
                cost=info_a["inertia"],
                action={
                    "type": "track_a_trimmed_kmeans",
                    "trim": float(info_a.get("trim", self.trim)),
                    "trim_configured": float(self.trim),
                    "n_iter": int(info_a["n_iter"]),
                },
                state={
                    "silhouette": float(sil_a),
                    "n_components_present": int(np.unique(labels_a).size),
                },
            )
        )

        # ---- 5. Track B — gated tied / spherical GMM ----
        labels_b: Optional[np.ndarray] = None
        info_b: Optional[Dict[str, Any]] = None
        sil_b: Optional[float] = None
        if run_b:
            labels_b, info_b = self._track_b(embedding, int(k))
            sil_b = self._silhouette(scoring_embedding, labels_b)
            trajectory.append(
                Step(
                    step_idx=len(trajectory),
                    cost=-float(info_b.get("lower_bound", 0.0)),
                    action={
                        "type": "track_b_constrained_gmm",
                        "covariance_type": self.track_b_covariance_type,
                        "n_iter": int(info_b["n_iter"]),
                    },
                    state={
                        "silhouette": float(sil_b),
                        "n_components_present": int(np.unique(labels_b).size),
                    },
                )
            )

        # ---- 6. Track selection ----
        if run_b and sil_b is not None and sil_b > sil_a:
            selected = "B"
            labels = labels_b
        else:
            selected = "A"
            labels = labels_a
        trajectory.append(
            Step(
                step_idx=len(trajectory),
                cost=0.0,
                action={"type": "track_selected", "selected": selected},
                state={
                    "selected": selected,
                    "sil_A": float(sil_a),
                    "sil_B": None if sil_b is None else float(sil_b),
                },
            )
        )

        extra: Dict[str, Any] = {
            "selected_track": selected,
            "silhouette_A": float(sil_a),
            "silhouette_B": None if sil_b is None else float(sil_b),
            "outlier_signal_lof_frac": float(lof_frac),
            "track_b_lof_thresh": float(self.track_b_lof_thresh),
            "track_b_ran": bool(run_b),
            "n_neighbors_used": int(k_nn_used),
            "anchor_size_used": int(m_used),
            "embedding_dim_used": int(embedding.shape[1]),
            "used_nystrom": bool(used_nystrom),
            "trim": float(self.trim),
            "track_a_info": info_a,
            "track_b_info": info_b,
        }
        return AlgoResult(
            labels=np.asarray(labels, dtype=np.int64),
            extra=extra,
            trajectory=trajectory,
        )
