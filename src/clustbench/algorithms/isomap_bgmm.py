"""Isomap geodesic embedding followed by a Dirichlet-process Bayesian GMM.

Two-stage manifold-aware clustering:

1.  Build a symmetric kNN graph on X (bump k until it is a single connected
    component -- Isomap's classic failure mode), then unfold curved manifolds
    with :class:`sklearn.manifold.Isomap` to a small target dim d'.
2.  Fit :class:`sklearn.mixture.BayesianGaussianMixture` with a Dirichlet
    process (stick-breaking) prior on the embedded coordinates. Weakly used
    components collapse to near-zero weight, so the effective number of
    clusters is inferred from the variational posterior rather than fixed.

Trajectory: one :class:`Step` per warm-started VB stage
(cost = negative variational lower bound, state = current weight vector +
active count). ``extra`` returns ``k_hat``, ``k_nbrs_final``,
``isomap_reconstruction_error`` and per-point ambiguity flags.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple
import warnings

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.mixture import BayesianGaussianMixture
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import connected_components

from .base import Algorithm, AlgoResult, Step, register


def _find_connected_k(
    X: np.ndarray, start_k: int, max_k: int = 50, step: int = 5
) -> int:
    """Return the smallest k >= start_k (in ``step`` increments) whose symmetric
    kNN graph on ``X`` is a single connected component. Returns -1 if no k up
    to ``max_k`` works.
    """
    n = X.shape[0]
    k_cap = min(max_k, n - 1)
    k = max(1, min(start_k, k_cap))
    while k <= k_cap:
        G = kneighbors_graph(X, n_neighbors=k, mode="connectivity", include_self=False)
        G = G.maximum(G.T)  # symmetrize
        ncomp, _ = connected_components(G, directed=False)
        if ncomp == 1:
            return k
        k += step
    return -1


def _embed(
    X: np.ndarray, k_nbrs: int, target_dim: int, max_landmarks: int, seed: int
) -> Tuple[np.ndarray, int, float, bool]:
    n, p = X.shape
    d_prime = max(1, min(target_dim, p, n - 1))

    k_final = _find_connected_k(X, start_k=k_nbrs, max_k=50, step=5)
    if k_final < 0:
        # Fallback: PCA (keeps geometry roughly intact if graph fragments).
        Z = PCA(n_components=d_prime, random_state=seed).fit_transform(X)
        return Z.astype(np.float64), -1, float("nan"), True

    if n > max_landmarks:
        # Landmark L-Isomap: fit on subsample, transform all.
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_landmarks, replace=False)
        Xs = X[idx]
        n_nbr = min(k_final, max_landmarks - 1)
        iso = Isomap(n_neighbors=n_nbr, n_components=d_prime)
        iso.fit(Xs)
        Z = iso.transform(X)
    else:
        iso = Isomap(n_neighbors=min(k_final, n - 1), n_components=d_prime)
        Z = iso.fit_transform(X)

    try:
        err = float(iso.reconstruction_error())
    except Exception:
        err = float("nan")
    return np.asarray(Z, dtype=np.float64), int(k_final), err, False


@register
class Isomap_bgmm(Algorithm):
    """Isomap + Dirichlet-process Bayesian GMM.

    Manifold-aware clustering that also infers the effective number of
    clusters from the variational weights (does not require a hard k).
    """

    def __init__(
        self,
        k_nbrs: int = 10,
        target_dim: int = 10,
        max_landmarks: int = 5000,
        weight_prior_scale: float = 1.0,
        prune_weight: float = 0.01,
        ambiguity_threshold: float = 0.5,
        n_vb_stages: int = 3,
        max_iter_per_stage: int = 60,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        self.name = "isomap_bgmm"
        self.k_nbrs = k_nbrs
        self.target_dim = target_dim
        self.max_landmarks = max_landmarks
        self.weight_prior_scale = weight_prior_scale
        self.prune_weight = prune_weight
        self.ambiguity_threshold = ambiguity_threshold
        self.n_vb_stages = n_vb_stages
        self.max_iter_per_stage = max_iter_per_stage
        self.random_state = random_state

    def fit_predict(self, X: np.ndarray, k: Optional[int] = None) -> AlgoResult:
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]

        # DP truncation: 2*k if k known, else 25; clamp to a sane range.
        base_max = 2 * k if k is not None else 25
        max_components = max(2, min(base_max, max(2, n // 4)))

        # ---- Isomap embedding ------------------------------------------------
        Z, k_final, iso_err, used_fallback = _embed(
            X,
            k_nbrs=self.k_nbrs,
            target_dim=self.target_dim,
            max_landmarks=self.max_landmarks,
            seed=self.random_state,
        )
        d_prime = Z.shape[1]

        # ---- Bayesian GMM ----------------------------------------------------
        # 'full' covariance in d'>=10 needs a decent n; else drop to 'diag'.
        cov_type = "full"
        if n < 20 * max_components or d_prime >= n:
            cov_type = "diag"

        weight_prior = self.weight_prior_scale / max_components

        bgmm = BayesianGaussianMixture(
            n_components=max_components,
            covariance_type=cov_type,
            weight_concentration_prior_type="dirichlet_process",
            weight_concentration_prior=weight_prior,
            max_iter=self.max_iter_per_stage,
            n_init=1,
            warm_start=True,
            init_params="kmeans",
            random_state=self.random_state,
            reg_covar=1e-5,
        )

        trajectory: list[Step] = []
        prev_cost: Optional[float] = None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for stage_i in range(self.n_vb_stages):
                bgmm.fit(Z)
                lb = getattr(bgmm, "lower_bound_", float("nan"))
                cost = float(-lb) if np.isfinite(lb) else float("nan")
                n_active = int(np.sum(bgmm.weights_ > self.prune_weight))
                delta = None if prev_cost is None else cost - prev_cost
                trajectory.append(
                    Step(
                        step_idx=stage_i,
                        cost=cost,
                        delta_cost=delta,
                        accepted=True,
                        action={
                            "type": "vb_stage",
                            "n_iter_budget": self.max_iter_per_stage,
                        },
                        state={
                            "weights": bgmm.weights_.tolist(),
                            "n_active": n_active,
                        },
                    )
                )
                prev_cost = cost

        # ---- Prune & assign --------------------------------------------------
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            resp = bgmm.predict_proba(Z)  # (n, max_components)

        weights = bgmm.weights_
        active_mask = weights > self.prune_weight
        if not active_mask.any():
            active_mask = np.zeros_like(weights, dtype=bool)
            active_mask[int(np.argmax(weights))] = True
        active_idx = np.where(active_mask)[0]

        active_resp = resp[:, active_idx]
        # Renormalize among surviving components so the ambiguity threshold
        # is meaningful even after pruning tiny mass.
        row_sum = active_resp.sum(axis=1, keepdims=True)
        row_sum = np.where(row_sum > 0, row_sum, 1.0)
        active_resp = active_resp / row_sum

        labels = active_resp.argmax(axis=1).astype(np.int64)
        max_post = active_resp.max(axis=1)
        ambiguous = (max_post < self.ambiguity_threshold).astype(np.int64)

        k_hat = int(active_mask.sum())

        extra: dict[str, Any] = {
            "k_hat": k_hat,
            "k_nbrs_final": int(k_final),
            "isomap_reconstruction_error": float(iso_err),
            "covariance_type": cov_type,
            "used_isomap_fallback": bool(used_fallback),
            "embedding_dim": int(d_prime),
            "max_components": int(max_components),
            "weight_concentration_prior": float(weight_prior),
            "final_weights": weights.tolist(),
            "ambiguous_mask": ambiguous.tolist(),
            "mean_max_posterior": float(max_post.mean()),
        }

        return AlgoResult(labels=labels, extra=extra, trajectory=trajectory)
