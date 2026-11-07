"""k-means clustering with an explicit EM loop so each iteration is captured.

When ``record_trajectory`` is False we delegate to :class:`sklearn.cluster.KMeans`
for speed. Otherwise we run k-means++ init manually and record the centroids,
assignment change, and inertia at every E/M pass.
"""

from __future__ import annotations

from typing import Any, Optional
import numpy as np
from sklearn.cluster import KMeans as SkKMeans

from ._kmeans_core import assign, kmeans_pp_init, partial_sums
from .base import Algorithm, AlgoResult, Step, register


# O(n k d) seeding; kept under the old name for callers that import it.
_kmeans_plus_plus_init = kmeans_pp_init


def _kmeans_em(
    X: np.ndarray,
    k: int,
    max_iter: int,
    tol: float,
    rng: np.random.Generator,
):
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=0)
    X = X - mu  # centred for numerically stable distances (see _kmeans_core.sq_dists)
    x_sq = np.einsum("ij,ij->i", X, X)
    centroids = kmeans_pp_init(X, k, rng)
    trajectory: list[Step] = []
    prev_labels = None
    prev_inertia: Optional[float] = None

    for step_idx in range(max_iter):
        labels, inertia = assign(X, centroids, x_sq)

        sums, counts = partial_sums(X, labels, k)
        new_centroids = centroids.copy()
        nz = counts > 0
        new_centroids[nz] = sums[nz] / counts[nz, None]

        changed = int((labels != prev_labels).sum()) if prev_labels is not None else X.shape[0]
        delta = None if prev_inertia is None else inertia - prev_inertia
        trajectory.append(
            Step(
                step_idx=step_idx,
                cost=inertia,
                delta_cost=delta,
                accepted=True,
                action={"type": "em", "n_reassigned": changed},
                state={"centroids": (new_centroids + mu).tolist()},
            )
        )

        shift = float(np.linalg.norm(new_centroids - centroids))
        centroids = new_centroids
        if prev_inertia is not None and abs(inertia - prev_inertia) < tol:
            break
        if shift < tol:
            break
        prev_labels = labels
        prev_inertia = inertia

    labels, inertia = assign(X, centroids, x_sq)
    return labels, centroids + mu, inertia, trajectory


@register
class Kmeans(Algorithm):
    """Standard k-means clustering.

    When ``record_trajectory`` is True (default), the EM loop is run in
    Python and every iteration is captured as a :class:`Step`. For pure
    benchmarking, set ``record_trajectory=False`` to get sklearn speed.
    """

    def __init__(
        self,
        max_iter: int = 300,
        n_init: int = 10,
        random_state: int = 42,
        tol: float = 1e-4,
        record_trajectory: bool = True,
        **kwargs: Any,
    ) -> None:
        self.name = "kmeans"
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state
        self.tol = tol
        self.record_trajectory = record_trajectory

    def fit_predict(self, X: np.ndarray, k: Optional[int] = None) -> AlgoResult:
        assert k is not None, "k (number of clusters) must be provided"
        if not self.record_trajectory:
            km = SkKMeans(
                n_clusters=k,
                max_iter=self.max_iter,
                n_init=self.n_init,
                random_state=self.random_state,
            )
            labels = km.fit_predict(X)
            return AlgoResult(labels=labels, extra={"inertia": float(km.inertia_)})

        rng = np.random.default_rng(self.random_state)
        best = None
        for init_idx in range(self.n_init):
            labels, centroids, inertia, trajectory = _kmeans_em(
                X, k, self.max_iter, self.tol, rng
            )
            if best is None or inertia < best[2]:
                best = (labels, centroids, inertia, trajectory, init_idx)

        labels, centroids, inertia, trajectory, init_idx = best
        return AlgoResult(
            labels=labels,
            extra={"inertia": inertia, "best_init": init_idx},
            trajectory=trajectory,
        )
