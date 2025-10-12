from __future__ import annotations

import numpy as np
from .base import Algorithm, AlgoResult, register


def _pairwise_dist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)


def clarans_fit_predict(X: np.ndarray, k: int, numlocal: int = 3, maxneigh: int = 250, random_state: int = 42) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    n = X.shape[0]

    def assign_cost(meds_idx):
        meds = X[meds_idx]
        D = _pairwise_dist(X, meds)
        assign = D.argmin(axis=1)
        cost = D[np.arange(n), assign].sum()
        return assign, cost

    best_medoids = None
    best_cost = float("inf")
    for _ in range(numlocal):
        medoids = rng.choice(n, size=k, replace=False)
        assign, cost = assign_cost(medoids)
        improved = True
        neigh_cnt = 0
        while improved and neigh_cnt < maxneigh:
            improved = False
            o = rng.choice(medoids)
            non_medoids = np.setdiff1d(np.arange(n), medoids, assume_unique=True)
            p = rng.choice(non_medoids)
            new_medoids = medoids.copy()
            new_medoids[new_medoids == o] = p
            _, new_cost = assign_cost(new_medoids)
            neigh_cnt += 1
            if new_cost < cost:
                medoids = new_medoids
                cost = new_cost
                improved = True
                neigh_cnt = 0
        if cost < best_cost:
            best_cost = cost
            best_medoids = medoids

    labels, _ = assign_cost(best_medoids)
    return labels


@register
class Clarans(Algorithm):
    def __init__(self, numlocal: int = 3, maxneigh: int = 250, random_state: int = 42, **kwargs):
        self.name = "clarans"
        self.numlocal = numlocal
        self.maxneigh = maxneigh
        self.random_state = random_state

    def fit_predict(self, X: np.ndarray, k: int | None = None) -> AlgoResult:
        assert k is not None, "k required"
        labels = clarans_fit_predict(X, k, self.numlocal, self.maxneigh, self.random_state)
        return AlgoResult(labels=labels, extra={})
