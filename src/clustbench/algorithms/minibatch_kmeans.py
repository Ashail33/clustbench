from __future__ import annotations

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from .base import Algorithm, AlgoResult, register

@register
class Minibatch_kmeans(Algorithm):
    def __init__(self, batch_size: int = 10000, max_iter: int = 100, random_state: int = 42, **kwargs):
        self.name = "minibatch_kmeans"
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.random_state = random_state

    def fit_predict(self, X: np.ndarray, k: int | None = None) -> AlgoResult:
        assert k is not None, "k required"
        model = MiniBatchKMeans(n_clusters=k, batch_size=self.batch_size, max_iter=self.max_iter, random_state=self.random_state)
        labels = model.fit_predict(X)
        return AlgoResult(labels=labels, extra={"inertia": float(model.inertia_)})
