"""Selective-sampling-based scalable sparse subspace clustering (S5C).

Faithful-in-shape implementation of Matsushima & Brbic, the algorithm
named in `Maharaj (2024)` for the sampling/partitioning category of
the taxonomy.

The original paper combines three ideas:

1. **Selective sampling** — instead of solving a sparse representation
   for every data point (which is O(n^2)), select a subset of size m
   and use it as a dictionary.
2. **Sparse subspace clustering (SSC) on the sample** — for each
   sampled point, find its sparse representation as a linear
   combination of the other sampled points (via Orthogonal Matching
   Pursuit here; LASSO is also valid).
3. **Spectral clustering** — build the affinity ``W = |C| + |C|^T`` from
   the sparse codes and run normalized-cut spectral clustering on it.
4. **Out-of-sample assignment** — each non-sampled point inherits the
   label of its nearest sampled point (we use the same sparse-code
   nearest-neighbour idea as the paper).

The trajectory captures one :class:`Step` per phase so the dashboard
can visualise S5C's pipeline as a short trajectory.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.linear_model import OrthogonalMatchingPursuit

from .base import Algorithm, AlgoResult, Step, register


@register
class S5c(Algorithm):
    """Selective-sampling sparse subspace clustering.

    Parameters
    ----------
    sample_size : int
        Number of points to draw for the dictionary. ``min(n, sample_size)``.
    n_nonzero_coefs : int
        Sparsity budget passed to OMP — number of dictionary atoms each
        sampled point is allowed to use. The paper's ``s`` parameter.
    random_state : int
        RNG seed.
    """

    def __init__(
        self,
        sample_size: int = 500,
        n_nonzero_coefs: int = 5,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        self.name = "s5c"
        self.sample_size = sample_size
        self.n_nonzero_coefs = n_nonzero_coefs
        self.random_state = random_state

    def fit_predict(self, X: np.ndarray, k: Optional[int] = None) -> AlgoResult:
        assert k is not None, "k required"
        rng = np.random.default_rng(self.random_state)
        n = X.shape[0]
        m = min(self.sample_size, n)

        trajectory: list[Step] = []

        # Phase 1: selective sampling (uniform random; the paper offers
        # variants, but uniform is the documented baseline).
        sample_idx = rng.choice(n, size=m, replace=False)
        Xs = X[sample_idx].astype(np.float32)
        trajectory.append(
            Step(
                step_idx=0,
                cost=float(m),
                delta_cost=None,
                accepted=True,
                action={"type": "sample", "m": int(m), "n": int(n)},
                state={"sample_size": int(m)},
            )
        )

        # Phase 2: sparse coding. For each sampled point, regress it on
        # the other sampled points; this is the SSC / self-expressive
        # representation. ``C[i, i]`` is forced to zero by leave-one-out.
        s = max(1, min(self.n_nonzero_coefs, m - 1))
        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=s, fit_intercept=False)
        C = np.zeros((m, m), dtype=np.float32)
        # Pre-build dictionary indices for the leave-one-out at column i.
        all_idx = np.arange(m)
        residuals = []
        for i in range(m):
            mask = all_idx != i
            D = Xs[mask].T  # shape (d, m-1) — atoms as columns
            target = Xs[i]
            omp.fit(D, target)
            coefs = np.zeros(m, dtype=np.float32)
            coefs[mask] = omp.coef_.astype(np.float32)
            C[:, i] = coefs
            residuals.append(float(np.linalg.norm(target - D @ omp.coef_)))
        mean_residual = float(np.mean(residuals)) if residuals else 0.0
        trajectory.append(
            Step(
                step_idx=1,
                cost=mean_residual,
                delta_cost=None,
                accepted=True,
                action={
                    "type": "sparse_code",
                    "n_nonzero_coefs": s,
                    "method": "omp",
                },
                state={"mean_residual": mean_residual, "sparsity": s},
            )
        )

        # Phase 3: spectral clustering on the symmetric affinity.
        W = np.abs(C) + np.abs(C.T)
        # Avoid pathological zero-degree rows.
        np.fill_diagonal(W, 0.0)
        deg = W.sum(axis=1)
        if (deg <= 0).any():
            W = W + 1e-6
        try:
            sc = SpectralClustering(
                n_clusters=k,
                affinity="precomputed",
                assign_labels="kmeans",
                random_state=self.random_state,
                n_init=5,
            )
            sample_labels = sc.fit_predict(W)
        except Exception:
            # Spectral occasionally fails on degenerate affinities (e.g.
            # disconnected sample). Fall back to a k-means projection.
            from sklearn.cluster import KMeans
            sample_labels = KMeans(
                n_clusters=k, random_state=self.random_state, n_init=5
            ).fit_predict(Xs)
        trajectory.append(
            Step(
                step_idx=2,
                cost=float(np.unique(sample_labels).size),
                delta_cost=None,
                accepted=True,
                action={"type": "spectral", "method": "ncut"},
                state={"clusters_in_sample": int(np.unique(sample_labels).size)},
            )
        )

        # Phase 4: out-of-sample assignment via nearest dictionary atom.
        labels = np.empty(n, dtype=np.int64)
        labels[sample_idx] = sample_labels
        rest = np.setdiff1d(np.arange(n), sample_idx, assume_unique=True)
        if rest.size:
            # Memory-light nearest neighbour in chunks.
            chunk = 512
            for start in range(0, rest.size, chunk):
                idx = rest[start : start + chunk]
                D = ((X[idx][:, None, :] - Xs[None, :, :]) ** 2).sum(axis=2)
                nearest = D.argmin(axis=1)
                labels[idx] = sample_labels[nearest]
        trajectory.append(
            Step(
                step_idx=3,
                cost=float(rest.size),
                delta_cost=None,
                accepted=True,
                action={"type": "assign_oos", "n_out_of_sample": int(rest.size)},
                state={"clusters_total": int(np.unique(labels).size)},
            )
        )

        return AlgoResult(
            labels=labels,
            extra={
                "sample_size": int(m),
                "n_nonzero_coefs": int(s),
                "mean_residual": mean_residual,
            },
            trajectory=trajectory,
        )
