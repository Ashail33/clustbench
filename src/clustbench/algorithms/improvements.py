"""Improved variants of the registered clustering algorithms.

Each class here is a deliberate fix for a specific bottleneck identified
in ``docs/ALGORITHM_ANALYSIS.md``. They register under the suffix the
analysis suggests (e.g. ``kmeans_trimmed`` for the trimmed-mean variant)
so a benchmark sweep can run originals and improved variants side by
side and the dashboard's Tables 4.1 / 4.2 directly score the lift.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .base import Algorithm, AlgoResult, Step, register
from . import base as base_algos


# ---------------------------------------------------------------------------
# kmeans_trimmed: outlier-robust kmeans via trimmed-mean centroid update.
# Bottleneck addressed: kmeans-family loses 32% ARI under outliers because
# the per-cluster mean has unbounded influence per point.
# ---------------------------------------------------------------------------


def _kpp_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = X.shape[0]
    centers = [X[int(rng.integers(n))]]
    for _ in range(1, k):
        d2 = np.min(
            np.sum((X[:, None, :] - np.stack(centers, axis=0)[None, :, :]) ** 2, axis=2),
            axis=1,
        )
        total = d2.sum()
        idx = int(rng.integers(n)) if total <= 0 else int(rng.choice(n, p=d2 / total))
        centers.append(X[idx])
    return np.stack(centers, axis=0)


@register
class Kmeans_trimmed(Algorithm):
    """kmeans with a trimmed-mean centroid update.

    Each M-step drops the ``trim`` fraction of cluster members farthest
    from the current centroid before averaging — the standard recipe for
    making the mean robust to a small outlier contamination. Defaults to
    10% trim, which matches the 20% outlier injection in the paper
    grid (the trimmed mean tolerates twice its trim fraction in
    contamination).
    """

    def __init__(
        self,
        max_iter: int = 100,
        n_init: int = 3,
        trim: float = 0.10,
        random_state: int = 42,
        tol: float = 1e-4,
        record_trajectory: bool = True,
        **kwargs: Any,
    ) -> None:
        self.name = "kmeans_trimmed"
        self.max_iter = max_iter
        self.n_init = n_init
        self.trim = trim
        self.random_state = random_state
        self.tol = tol
        self.record_trajectory = record_trajectory

    def _run(self, X, k, rng):
        centroids = _kpp_init(X, k, rng).astype(np.float32)
        prev_inertia = None
        trajectory: list[Step] = []
        for step_idx in range(self.max_iter):
            D = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
            labels = D.argmin(axis=1)
            inertia = float(D[np.arange(X.shape[0]), labels].sum())

            new_centroids = centroids.copy()
            n_trimmed_total = 0
            for j in range(k):
                members_idx = np.where(labels == j)[0]
                if len(members_idx) == 0:
                    continue
                dists = D[members_idx, j]
                keep_n = max(1, int(len(members_idx) * (1.0 - self.trim)))
                keep = members_idx[np.argsort(dists)[:keep_n]]
                n_trimmed_total += len(members_idx) - len(keep)
                new_centroids[j] = X[keep].mean(axis=0)

            if self.record_trajectory:
                trajectory.append(
                    Step(
                        step_idx=step_idx,
                        cost=inertia,
                        delta_cost=None if prev_inertia is None else inertia - prev_inertia,
                        accepted=True,
                        action={"type": "trimmed_em", "trim": self.trim, "n_trimmed": n_trimmed_total},
                        state={"centroids": new_centroids.tolist()},
                    )
                )
            shift = float(np.linalg.norm(new_centroids - centroids))
            centroids = new_centroids
            if prev_inertia is not None and abs(inertia - prev_inertia) < self.tol:
                break
            if shift < self.tol:
                break
            prev_inertia = inertia

        D = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        return D.argmin(axis=1), centroids, inertia, trajectory

    def fit_predict(self, X: np.ndarray, k: Optional[int] = None) -> AlgoResult:
        assert k is not None
        rng = np.random.default_rng(self.random_state)
        best = None
        for init_idx in range(self.n_init):
            labels, centroids, inertia, trajectory = self._run(X, k, rng)
            if best is None or inertia < best[2]:
                best = (labels, centroids, inertia, trajectory, init_idx)
        labels, centroids, inertia, trajectory, init_idx = best
        return AlgoResult(
            labels=labels,
            extra={"inertia": inertia, "trim": self.trim, "best_init": init_idx},
            trajectory=trajectory,
        )


# ---------------------------------------------------------------------------
# clarans_pp: CLARANS with k-means++-style medoid initialisation.
# Bottleneck addressed: CLARANS lost 40% ARI under outliers because random
# medoid init picks outliers as medoids surprisingly often.
# ---------------------------------------------------------------------------


def _kpp_medoids(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ but returns indices instead of centroid coordinates."""
    n = X.shape[0]
    medoids = [int(rng.integers(n))]
    for _ in range(1, k):
        d2 = np.min(
            np.sum((X[:, None, :] - X[np.array(medoids)][None, :, :]) ** 2, axis=2),
            axis=1,
        )
        total = d2.sum()
        idx = int(rng.integers(n)) if total <= 0 else int(rng.choice(n, p=d2 / total))
        medoids.append(idx)
    return np.array(medoids)


@register
class Clarans_pp(Algorithm):
    """CLARANS with k-means++ initialisation and a wider neighbourhood.

    Two changes vs. ``clarans``:
      1. Medoids are seeded by the kmeans++ probability (distance-biased)
         instead of uniform random. This is the standard fix for an
         algorithm whose local search is sensitive to where it starts.
      2. ``maxneigh`` is bumped because kmeans++ already pushes us toward
         a good basin; the swap budget should be spent on refinement.
    """

    def __init__(
        self,
        numlocal: int = 3,
        maxneigh: int = 80,
        random_state: int = 42,
        record_trajectory: bool = True,
        **kwargs,
    ):
        self.name = "clarans_pp"
        self.numlocal = numlocal
        self.maxneigh = maxneigh
        self.random_state = random_state
        self.record_trajectory = record_trajectory

    def fit_predict(self, X: np.ndarray, k: Optional[int] = None) -> AlgoResult:
        assert k is not None
        rng = np.random.default_rng(self.random_state)
        n = X.shape[0]
        trajectory: list[Step] = []
        step_idx = 0

        def assign_cost(meds_idx):
            D = np.linalg.norm(X[:, None, :] - X[meds_idx][None, :, :], axis=2)
            assign = D.argmin(axis=1)
            return assign, float(D[np.arange(n), assign].sum())

        best_medoids = None
        best_cost = float("inf")

        for local_idx in range(self.numlocal):
            medoids = _kpp_medoids(X, k, rng)  # << the fix
            assign, cost = assign_cost(medoids)
            if self.record_trajectory:
                trajectory.append(
                    Step(
                        step_idx=step_idx,
                        cost=cost,
                        delta_cost=None,
                        accepted=True,
                        action={"type": "init_kpp", "local": local_idx},
                        state={"medoids": medoids.tolist()},
                    )
                )
                step_idx += 1
            improved = True
            neigh_cnt = 0
            while improved and neigh_cnt < self.maxneigh:
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
                if self.record_trajectory:
                    trajectory.append(
                        Step(
                            step_idx=step_idx,
                            cost=new_cost if accepted else cost,
                            delta_cost=delta,
                            accepted=accepted,
                            action={"type": "swap", "out": o, "in": p},
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
        return AlgoResult(labels=labels, extra={"final_cost": best_cost}, trajectory=trajectory)


# ---------------------------------------------------------------------------
# dbscan_auto: DBSCAN with eps estimated from the k-distance plot.
# Bottleneck addressed: DBSCAN at eps=0.8 in d=10 returns one giant noise
# cluster on every task in the registry. The fix is the Ester 1996 §4.2
# k-distance heuristic: sort the kth-nearest-neighbour distance over all
# points and pick the knee.
# ---------------------------------------------------------------------------


def _knee_index(values: np.ndarray) -> int:
    """Find the elbow of a monotone-increasing array via max distance to the
    chord connecting the first and last points.
    """
    n = len(values)
    if n < 3:
        return n - 1
    # Normalise both axes to [0, 1] for a fair distance.
    x = np.linspace(0.0, 1.0, n)
    y = (values - values.min()) / (values.max() - values.min() + 1e-12)
    # Distance from each point to the line (0,0)-(1,1).
    dist = np.abs(y - x) / np.sqrt(2.0)
    return int(dist.argmax())


@register
class Dbscan_auto(Algorithm):
    """DBSCAN with auto-estimated eps via the k-distance knee.

    Parameters
    ----------
    min_samples : int
        Same role as in DBSCAN. Used as ``k`` in the k-distance plot.
    """

    def __init__(self, min_samples: int = 5, **kwargs: Any) -> None:
        self.name = "dbscan_auto"
        self.min_samples = min_samples

    def fit_predict(self, X: np.ndarray, k: Optional[int] = None) -> AlgoResult:
        from sklearn.cluster import DBSCAN
        from sklearn.neighbors import NearestNeighbors

        knn = NearestNeighbors(n_neighbors=self.min_samples + 1)
        knn.fit(X)
        dist, _ = knn.kneighbors(X)
        k_dist = np.sort(dist[:, -1])
        knee = _knee_index(k_dist)
        eps = float(k_dist[knee])
        if eps <= 0:
            eps = float(np.median(k_dist) + 1e-6)
        model = DBSCAN(eps=eps, min_samples=self.min_samples, n_jobs=-1)
        labels = model.fit_predict(X)
        return AlgoResult(
            labels=labels.astype(np.int64),
            extra={"eps_auto": eps, "knee_idx": knee, "n_samples": int(len(X))},
        )


# ---------------------------------------------------------------------------
# meanshift_robust: meanshift with trimmed-sample bandwidth.
# Bottleneck addressed: meanshift lost 59% ARI under outliers because
# ``estimate_bandwidth`` averages distances including outlier contributions
# and inflates the kernel; the resulting KDE smooths over real cluster
# boundaries.
# ---------------------------------------------------------------------------


@register
class Meanshift_robust(Algorithm):
    """Mean-shift with a trimmed-sample bandwidth estimator.

    Estimate bandwidth on a random sample whose top ``trim`` quantile by
    pairwise distance is dropped before averaging. This stops a small
    outlier contamination from inflating the kernel width.
    """

    def __init__(
        self,
        bandwidth: Optional[float] = None,
        bin_seeding: bool = True,
        trim: float = 0.2,
        quantile: float = 0.2,
        **kwargs: Any,
    ) -> None:
        self.name = "meanshift_robust"
        self.bandwidth = bandwidth
        self.bin_seeding = bin_seeding
        self.trim = trim
        self.quantile = quantile

    def fit_predict(self, X: np.ndarray, k: Optional[int] = None) -> AlgoResult:
        from sklearn.cluster import MeanShift

        if self.bandwidth is not None:
            bw = float(self.bandwidth)
        else:
            n_sample = min(500, len(X))
            rng = np.random.default_rng(0)
            idx = rng.choice(len(X), size=n_sample, replace=False)
            Xs = X[idx]
            # Pairwise nearest-neighbour distances in the sample.
            from sklearn.neighbors import NearestNeighbors

            k_nn = max(2, int(self.quantile * n_sample))
            nbrs = NearestNeighbors(n_neighbors=k_nn).fit(Xs)
            d, _ = nbrs.kneighbors(Xs)
            # Trim top fraction of farthest k-NN distances.
            avg_per_point = d[:, -1]
            keep = avg_per_point < np.quantile(avg_per_point, 1.0 - self.trim)
            bw = float(avg_per_point[keep].mean()) if keep.any() else float(avg_per_point.mean())
            if bw <= 0:
                bw = 1.0
        ms = MeanShift(bandwidth=bw, bin_seeding=self.bin_seeding)
        labels = ms.fit_predict(X)
        return AlgoResult(
            labels=labels.astype(np.int64),
            extra={"bandwidth": bw, "trim": self.trim},
        )


# ---------------------------------------------------------------------------
# pwcc_diverse: PWCC with a *diverse* ensemble — kmeans + spectral + gmm.
# Bottleneck addressed: vanilla PWCC's base [kmeans, minibatch_kmeans,
# birch_algo] is homogeneous (three k-means-style algos), so the weighted
# vote rarely sees disagreement and can't recover from unanimous mistakes
# on non-convex data. Swapping in spectral and gmm gives the vote real
# perspective diversity.
# ---------------------------------------------------------------------------


@register
class Pwcc_diverse(Algorithm):
    """PWCC with a heterogeneous base ensemble.

    Uses [kmeans, spectral, gmm] as defaults — one centroid-based, one
    graph-based, one EM-based. The weighted vote then aggregates three
    qualitatively different perspectives instead of three flavours of
    the same one.
    """

    def __init__(
        self,
        base: Optional[list[str]] = None,
        base_params: Optional[dict] = None,
        **kwargs: Any,
    ) -> None:
        self.name = "pwcc_diverse"
        self.base = base or ["kmeans", "spectral", "gmm"]
        self.base_params = base_params or {}

    def fit_predict(self, X: np.ndarray, k: Optional[int] = None) -> AlgoResult:
        # Delegate to the existing PWCC implementation with a different
        # default ensemble. Importing here keeps the registry decoupled.
        from .pwcc import Pwcc

        delegate = Pwcc(base=self.base, base_params=self.base_params)
        res = delegate.fit_predict(X, k=k)
        # Preserve trajectory and labels but tag the algo name correctly.
        return AlgoResult(
            labels=res.labels,
            extra={**res.extra, "ensemble": self.base},
            trajectory=res.trajectory,
        )
