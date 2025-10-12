"""k-means clustering algorithm wrapper."""

from __future__ import annotations

from typing import Optional, Any
import numpy as np
from sklearn.cluster import KMeans

from .base import Algorithm, AlgoResult, register


@register
class Kmeans(Algorithm):
    """Standard k-means clustering using scikit-learn."""

    def __init__(self, max_iter: int = 300, n_init: int = 10, random_state: int = 42, **kwargs: Any) -> None:
        self.name = "kmeans"
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state

    def fit_predict(self, X: np.ndarray, k: Optional[int] = None) -> AlgoResult:
        assert k is not None, "k (number of clusters) must be provided"
        km = KMeans(
            n_clusters=k,
            max_iter=self.max_iter,
            n_init=self.n_init,
            random_state=self.random_state,
        )
        labels = km.fit_predict(X)
        return AlgoResult(labels=labels, extra={"inertia": float(km.inertia_)})
