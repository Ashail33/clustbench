"""k-means clustering with an explicit EM loop so each iteration is captured.

When ``record_trajectory`` is False we delegate to :class:`sklearn.cluster.KMeans`
for speed. Otherwise we run k-means++ init manually and record the centroids,
assignment change, and inertia at every E/M pass.
"""

from __future__ import annotations

from typing import Any, Optional
import numpy as np
from sklearn.cluster import KMeans as SkKMeans

from .base import Algorithm, AlgoResult, Step, register


def _kmeans_plus_plus_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = X.shape[0]
    first = int(rng.integers(n))
    centers = [X[first]]
    for _ in range(1, k):
        d2 = np.min(
            np.sum((X[:, None, :] - np.stack(centers, axis=0)[None, :, :]) ** 2, axis=2),
            axis=1,
        )
        total = d2.sum()
        if total <= 0:
            idx = int(rng.integers(n))
        else:
            probs = d2 / total
            idx = int(rng.choice(n, p=probs))
        centers.append(X[idx])
    return np.stack(centers, axis=0)


def _kmeans_em(
    X: np.ndarray,
    k: int,
    max_iter: int,
    tol: float,
    rng: np.random.Generator,
):
    centroids = _kmeans_plus_plus_init(X, k, rng)
    trajectory: list[Step] = []
    prev_labels = None
    prev_inertia: Optional[float] = None

    for step_idx in range(max_iter):
        D = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        labels = D.argmin(axis=1)
        inertia = float(D[np.arange(X.shape[0]), labels].sum())

        new_centroids = centroids.copy()
        for j in range(k):
            members = X[labels == j]
            if len(members) > 0:
                new_centroids[j] = members.mean(axis=0)

        changed = int((labels != prev_labels).sum()) if prev_labels is not None else X.shape[0]
        delta = None if prev_inertia is None else inertia - prev_inertia
        trajectory.append(
            Step(
                step_idx=step_idx,
                cost=inertia,
                delta_cost=delta,
                accepted=True,
                action={"type": "em", "n_reassigned": changed},
                state={"centroids": new_centroids.tolist()},
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

    D = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    labels = D.argmin(axis=1)
    inertia = float(D[np.arange(X.shape[0]), labels].sum())
    return labels, centroids, inertia, trajectory


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
