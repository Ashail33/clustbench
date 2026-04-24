from __future__ import annotations

import numpy as np
from .base import Algorithm, AlgoResult, Step, register


def _pairwise_dist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)


def clarans_fit_predict(
    X: np.ndarray,
    k: int,
    numlocal: int = 3,
    maxneigh: int = 250,
    random_state: int = 42,
    record_trajectory: bool = True,
):
    """Run CLARANS and return ``(labels, trajectory)``.

    ``trajectory`` is a list of :class:`Step` objects describing every
    proposed swap: the state before the swap (current medoid set), the
    action (which medoid was swapped out for which non-medoid), the
    resulting cost and delta, and whether the swap was accepted. When
    ``record_trajectory`` is False, the list is empty — use for large
    problems where the trajectory would be too big to keep in memory.
    """
    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    trajectory: list[Step] = []

    def assign_cost(meds_idx):
        meds = X[meds_idx]
        D = _pairwise_dist(X, meds)
        assign = D.argmin(axis=1)
        cost = D[np.arange(n), assign].sum()
        return assign, float(cost)

    best_medoids = None
    best_cost = float("inf")
    step_idx = 0
    for local_idx in range(numlocal):
        medoids = rng.choice(n, size=k, replace=False)
        assign, cost = assign_cost(medoids)
        if record_trajectory:
            trajectory.append(
                Step(
                    step_idx=step_idx,
                    cost=cost,
                    delta_cost=None,
                    accepted=True,
                    action={"type": "init", "local": local_idx},
                    state={"medoids": medoids.tolist()},
                )
            )
            step_idx += 1
        improved = True
        neigh_cnt = 0
        while improved and neigh_cnt < maxneigh:
            improved = False
            o = int(rng.choice(medoids))
            non_medoids = np.setdiff1d(np.arange(n), medoids, assume_unique=True)
            p = int(rng.choice(non_medoids))
            new_medoids = medoids.copy()
            new_medoids[new_medoids == o] = p
            _, new_cost = assign_cost(new_medoids)
            neigh_cnt += 1
            delta = new_cost - cost
            accepted = new_cost < cost
            if record_trajectory:
                trajectory.append(
                    Step(
                        step_idx=step_idx,
                        cost=new_cost if accepted else cost,
                        delta_cost=delta,
                        accepted=accepted,
                        action={"type": "swap", "out": o, "in": p, "local": local_idx},
                        state={"medoids": (new_medoids if accepted else medoids).tolist()},
                    )
                )
                step_idx += 1
            if accepted:
                medoids = new_medoids
                cost = new_cost
                improved = True
                neigh_cnt = 0
        if cost < best_cost:
            best_cost = cost
            best_medoids = medoids

    labels, _ = assign_cost(best_medoids)
    return labels, trajectory


@register
class Clarans(Algorithm):
    def __init__(
        self,
        numlocal: int = 3,
        maxneigh: int = 250,
        random_state: int = 42,
        record_trajectory: bool = True,
        **kwargs,
    ):
        self.name = "clarans"
        self.numlocal = numlocal
        self.maxneigh = maxneigh
        self.random_state = random_state
        self.record_trajectory = record_trajectory

    def fit_predict(self, X: np.ndarray, k: int | None = None) -> AlgoResult:
        assert k is not None, "k required"
        labels, trajectory = clarans_fit_predict(
            X,
            k,
            self.numlocal,
            self.maxneigh,
            self.random_state,
            record_trajectory=self.record_trajectory,
        )
        return AlgoResult(labels=labels, extra={}, trajectory=trajectory)
