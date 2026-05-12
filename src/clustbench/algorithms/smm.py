"""SMM: Spectral Mixture Model — Sharma 2009 baseline.

Reproduction of the closest prior art to LMM:

    Sharma, Horaud, Knossow & von Lavante (2009).
    "Mesh Segmentation Using Laplacian Eigenvectors and Gaussian Mixtures."
    AAAI Fall Symposium on Manifold Learning.

Algorithm:
  1. Compute the bottom-k eigenvectors of the normalised k-NN graph
     Laplacian (the standard Ng-Jordan-Weiss spectral embedding,
     identical to LMM's basis).
  2. Fit a Gaussian mixture model directly on the (n, k) embedding
     matrix via standard EM with closed-form Gaussian updates.
  3. Argmax responsibility → labels.

Used as an ablation: if LMM still beats Smm on the benchmark, the
specific LMM contributions (exp-family components, self-normalised EM,
Newton M-step with line search) are doing real work. If Smm matches
LMM, the win is "EM > k-means on the spectral embedding" — a result
already published in Sharma 2009 — and LMM's novelty would be small.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .base import Algorithm, AlgoResult, Step, register
from .lmm import Lmm


@register
class Smm(Algorithm):
    """Gaussian-mixture-on-Laplacian-eigenvectors baseline (Sharma 2009).

    Inherits LMM's basis construction (k-NN normalised Laplacian +
    LOBPCG + row normalisation) so the comparison is on the same
    embedding. The clustering step is plain sklearn ``GaussianMixture``
    on the embedded ``(n, k)`` matrix.

    Parameters
    ----------
    n_eigvecs, n_neighbors, affinity, row_normalize, nystrom,
    n_landmarks, nystrom_refine_iter :
        Passed through to a private ``Lmm`` instance for basis construction.
        Defaults match LMM so the embedding is identical.
    covariance_type : str
        sklearn GaussianMixture covariance type — ``"full"`` (default)
        gives Sharma 2009 the best shot.
    gmm_max_iter : int
        EM iteration cap for the Gaussian mixture step.
    gmm_n_init : int
        Number of random restarts of the GMM. Same default as our
        ``gmm`` algorithm in the registry.
    random_state : int
        RNG seed (for both basis sampling and GMM init).
    """

    def __init__(
        self,
        n_eigvecs: Any = "auto",
        n_neighbors: Any = "auto",
        affinity: str = "binary",
        row_normalize: bool = True,
        nystrom: Any = False,
        n_landmarks: int = 200,
        nystrom_refine_iter: int = 10,
        covariance_type: str = "full",
        gmm_max_iter: int = 100,
        gmm_n_init: int = 1,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        self.name = "smm"
        self.n_eigvecs = n_eigvecs
        self.n_neighbors = n_neighbors
        self.affinity = affinity
        self.row_normalize = row_normalize
        self.nystrom = nystrom
        self.n_landmarks = int(n_landmarks)
        self.nystrom_refine_iter = int(nystrom_refine_iter)
        self.covariance_type = covariance_type
        self.gmm_max_iter = int(gmm_max_iter)
        self.gmm_n_init = int(gmm_n_init)
        self.random_state = random_state

    def fit_predict(self, X: np.ndarray, k: Optional[int] = None) -> AlgoResult:
        from sklearn.mixture import GaussianMixture

        assert k is not None
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        rng = np.random.default_rng(self.random_state)
        trajectory: list[Step] = []

        # 1. Build the same Laplacian-eigenvector basis LMM uses. We
        # piggyback on an Lmm instance so the basis path stays
        # apples-to-apples with LMM. The Lmm has its own k_hint so we
        # set it here.
        lmm = Lmm(
            n_eigvecs=self.n_eigvecs,
            n_neighbors=self.n_neighbors,
            affinity=self.affinity,
            row_normalize=self.row_normalize,
            nystrom=self.nystrom,
            n_landmarks=self.n_landmarks,
            nystrom_refine_iter=self.nystrom_refine_iter,
            random_state=self.random_state,
        )
        lmm._k_hint = k
        lmm.n_frequencies = lmm._resolve_n_eigvecs(k)
        _, phi, _ = lmm._build_basis(X, rng)
        trajectory.append(
            Step(
                step_idx=0,
                cost=0.0,
                accepted=True,
                action={"type": "laplacian_basis", "n_eigvecs": phi.shape[1]},
                state={"feature_dim": int(phi.shape[1])},
            )
        )

        # 2. Plain Gaussian mixture on the embedding.
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=self.covariance_type,
            max_iter=self.gmm_max_iter,
            n_init=self.gmm_n_init,
            reg_covar=1e-4,
            random_state=self.random_state,
        )
        labels = gmm.fit_predict(phi).astype(np.int64)
        trajectory.append(
            Step(
                step_idx=1,
                cost=float(-gmm.score(phi)),
                accepted=True,
                action={"type": "gmm_on_embedding", "n_clusters": int(k)},
                state={
                    "n_iter": int(gmm.n_iter_),
                    "log_likelihood": float(gmm.score(phi) * n),
                    "converged": bool(gmm.converged_),
                },
            )
        )

        return AlgoResult(
            labels=labels,
            extra={
                "feature_dim": int(phi.shape[1]),
                "gmm_n_iter": int(gmm.n_iter_),
                "gmm_converged": bool(gmm.converged_),
                "covariance_type": self.covariance_type,
                "log_likelihood": float(gmm.score(phi) * n),
            },
            trajectory=trajectory,
        )
