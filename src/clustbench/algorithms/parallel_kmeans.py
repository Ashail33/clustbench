"""Parallel k-means (Zhao 2009, MapReduce-style) — single-machine version.

The algorithm in `Maharaj (2024)` for the parallel/distributed-based
category is the Zhao et al. MapReduce variant of k-means. The shape of
the algorithm is:

    map:    each worker holds a chunk of X; given the broadcast centroids
            it emits per-cluster (sum, count) pairs for its chunk
    reduce: aggregate all workers' (sum, count) pairs into new centroids
    repeat: until centroids converge

That structure is what we instrument here. We swap MapReduce for a
``multiprocessing.Pool`` so it runs on a single machine, but keep the
two phases distinct so the trajectory can record one step per
map/reduce round and the per-worker partial sums end up in the action
payload (a tiny window into the distributed computation).
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Optional

import numpy as np

from ._kmeans_core import assign, kmeans_pp_init, partial_sums
from .base import Algorithm, AlgoResult, Step, register


def _map_chunk(args):
    chunk, centroids = args
    labels, inertia = assign(chunk, centroids)
    sums, counts = partial_sums(chunk, labels, centroids.shape[0])
    return labels, sums, counts, inertia


# O(n k d) seeding shared with ``kmeans``; kept under the old name for callers.
_kpp_init = kmeans_pp_init


@register
class Parallel_kmeans(Algorithm):
    """K-means with the MapReduce iteration shape.

    Parameters
    ----------
    n_workers : int
        Number of map workers. Defaults to the number of CPUs available.
        Falls back to in-process if ``n_workers <= 1`` (useful for tests).
    """

    def __init__(
        self,
        max_iter: int = 100,
        n_init: int = 3,
        n_workers: Optional[int] = None,
        random_state: int = 42,
        tol: float = 1e-4,
        record_trajectory: bool = True,
        **kwargs: Any,
    ) -> None:
        self.name = "parallel_kmeans"
        self.max_iter = max_iter
        self.n_init = n_init
        self.n_workers = n_workers if n_workers is not None else max(1, (os.cpu_count() or 2) // 2)
        self.random_state = random_state
        self.tol = tol
        self.record_trajectory = record_trajectory

    def _run_once(self, X: np.ndarray, k: int, rng: np.random.Generator):
        X = np.asarray(X, dtype=np.float64)
        mu = X.mean(axis=0)
        X = X - mu  # centred for numerically stable distances (see _kmeans_core.sq_dists)
        n = X.shape[0]
        centroids = kmeans_pp_init(X, k, rng)
        chunks = np.array_split(np.arange(n), max(1, self.n_workers))
        chunk_data = [X[idx] for idx in chunks]
        trajectory: list[Step] = []

        prev_inertia: Optional[float] = None
        labels_full = np.zeros(n, dtype=np.int64)
        executor = (
            ProcessPoolExecutor(max_workers=self.n_workers) if self.n_workers > 1 else None
        )
        try:
            for step_idx in range(self.max_iter):
                args_iter = [(c, centroids) for c in chunk_data]
                if executor is None:
                    results = [_map_chunk(a) for a in args_iter]
                else:
                    results = list(executor.map(_map_chunk, args_iter))

                # Reduce: aggregate sums/counts from each worker.
                total_sum = np.zeros_like(centroids)
                total_count = np.zeros(k, dtype=np.int64)
                inertia = 0.0
                for w, (lbl, s, c, iner) in enumerate(results):
                    total_sum += s
                    total_count += c
                    inertia += iner
                    labels_full[chunks[w]] = lbl

                new_centroids = centroids.copy()
                for j in range(k):
                    if total_count[j] > 0:
                        new_centroids[j] = total_sum[j] / total_count[j]

                if self.record_trajectory:
                    trajectory.append(
                        Step(
                            step_idx=step_idx,
                            cost=float(inertia),
                            delta_cost=None if prev_inertia is None else float(inertia - prev_inertia),
                            accepted=True,
                            action={
                                "type": "mapreduce",
                                "n_workers": self.n_workers,
                                "counts_per_cluster": total_count.tolist(),
                            },
                            state={"centroids": (new_centroids + mu).tolist()},
                        )
                    )

                shift = float(np.linalg.norm(new_centroids - centroids))
                centroids = new_centroids
                if prev_inertia is not None and abs(inertia - prev_inertia) < self.tol:
                    break
                if shift < self.tol:
                    break
                prev_inertia = inertia
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

        # Final assignment against the converged centroids, as in ``kmeans``;
        # the map-phase labels are one update behind.
        labels_full, inertia = assign(X, centroids)
        return labels_full, centroids + mu, float(inertia), trajectory

    def fit_predict(self, X: np.ndarray, k: Optional[int] = None) -> AlgoResult:
        assert k is not None, "k required"
        rng = np.random.default_rng(self.random_state)
        best = None
        for init_idx in range(self.n_init):
            labels, centroids, inertia, trajectory = self._run_once(X, k, rng)
            if best is None or inertia < best[2]:
                best = (labels, centroids, inertia, trajectory, init_idx)
        labels, centroids, inertia, trajectory, init_idx = best
        return AlgoResult(
            labels=labels,
            extra={
                "inertia": inertia,
                "n_workers": self.n_workers,
                "best_init": init_idx,
            },
            trajectory=trajectory,
        )
