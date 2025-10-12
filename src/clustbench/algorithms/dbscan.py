from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN

from .base import Algorithm, AlgoResult, register

@register
class Dbscan(Algorithm):
    def __init__(self, eps: float = 0.5, min_samples: int = 5, **kwargs):
        self.name = "dbscan"
        self.eps = eps
        self.min_samples = min_samples

    def fit_predict(self, X: np.ndarray, k: int | None = None) -> AlgoResult:
        model = DBSCAN(eps=self.eps, min_samples=self.min_samples, n_jobs=-1)
        labels = model.fit_predict(X)
        return AlgoResult(labels=labels, extra={})
