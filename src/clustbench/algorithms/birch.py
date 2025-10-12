from __future__ import annotations

import numpy as np
from sklearn.cluster import Birch

from .base import Algorithm, AlgoResult, register

@register
class Birch_algo(Algorithm):
    def __init__(self, threshold: float = 1.5, branching_factor: int = 50, **kwargs):
        self.name = "birch"
        self.threshold = threshold
        self.branching_factor = branching_factor

    def fit_predict(self, X: np.ndarray, k: int | None = None) -> AlgoResult:
        model = Birch(threshold=self.threshold, branching_factor=self.branching_factor, n_clusters=k)
        labels = model.fit_predict(X)
        return AlgoResult(labels=labels, extra={})
