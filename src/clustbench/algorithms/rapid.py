"""RAPID: Region-Adaptive Partitioning with Iterative Density.

A two-stage clustering pipeline that uses density-based clustering to
*partition the data into regions first*, then routes each region to the
algorithm best suited for it (per ``docs/ALGORITHM_ANALYSIS.md``: DBSCAN
isolates density-connected regions, k-means handles convex blobs,
spectral handles non-convex shapes).

Stage 1 (density-based region discovery):
    Estimate ``eps`` adaptively via the k-distance knee
    (same trick as :class:`Dbscan_auto`). Run DBSCAN with that eps.
    Each DBSCAN cluster becomes a region; noise points are held aside.

Stage 2 (per-region clustering):
    For each region, compute a small "shape signature" (convexity, size)
    and route:
      * convex + enough points  -> KMeans
      * non-convex              -> SpectralClustering
      * tiny                    -> single cluster id

Stage 3 (outlier reassignment):
    DBSCAN-noise points are reassigned to the cluster id of their
    nearest labelled neighbour, so the final array contains no ``-1``s.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .base import Algorithm, AlgoResult, Step, register


def _knee_index(values: np.ndarray) -> int:
    """Find the elbow of a monotone-increasing array via max distance to
    the chord connecting the first and last points.
    """
    n = len(values)
    if n < 3:
        return n - 1
    x = np.linspace(0.0, 1.0, n)
    y = (values - values.min()) / (values.max() - values.min() + 1e-12)
    dist = np.abs(y - x) / np.sqrt(2.0)
    return int(dist.argmax())


def _auto_eps(X: np.ndarray, min_samples: int) -> float:
    """Estimate ``eps`` from the k-distance knee (Ester 1996 §4.2)."""
    from sklearn.neighbors import NearestNeighbors

    n_neighbors = min(min_samples + 1, len(X))
    knn = NearestNeighbors(n_neighbors=n_neighbors)
    knn.fit(X)
    dist, _ = knn.kneighbors(X)
    k_dist = np.sort(dist[:, -1])
    knee = _knee_index(k_dist)
    eps = float(k_dist[knee])
    if eps <= 0:
        eps = float(np.median(k_dist) + 1e-6)
    return eps


@register
class Rapid(Algorithm):
    """Region-Adaptive Partitioning with Iterative Density.

    Parameters
    ----------
    min_samples : int
        ``min_samples`` for the Stage-1 DBSCAN (also the ``k`` in the
        k-distance knee estimate).
    random_state : int
        Seed forwarded to KMeans / SpectralClustering for reproducibility.
    """

    def __init__(
        self,
        min_samples: int = 5,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        self.name = "rapid"
        self.min_samples = min_samples
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Stage 1: density-based region discovery.
    # ------------------------------------------------------------------
    def _stage1_regions(self, X: np.ndarray) -> tuple[np.ndarray, float]:
        from sklearn.cluster import DBSCAN

        eps = _auto_eps(X, self.min_samples)
        dbscan = DBSCAN(eps=eps, min_samples=self.min_samples, n_jobs=-1)
        region_ids = dbscan.fit_predict(X).astype(np.int64)
        return region_ids, eps

    # ------------------------------------------------------------------
    # Stage 2: shape signature + per-region routing.
    # ------------------------------------------------------------------
    @staticmethod
    def _shape_signature(X_region: np.ndarray) -> tuple[float, int]:
        """Return ``(convexity, n_points)`` for a region.

        ``convexity`` = fraction of region points within 1 std of the
        region centroid (in Euclidean distance). High => convex.
        """
        n_points = int(X_region.shape[0])
        if n_points == 0:
            return 0.0, 0
        centroid = X_region.mean(axis=0)
        diffs = X_region - centroid
        d = np.linalg.norm(diffs, axis=1)
        # Std of the distance distribution; degenerate (all-identical) regions are perfectly convex.
        std = float(d.std())
        if std <= 1e-12:
            return 1.0, n_points
        convexity = float(np.mean(d <= std))
        return convexity, n_points

    def fit_predict(self, X: np.ndarray, k: Optional[int] = None) -> AlgoResult:
        assert k is not None, "k required"
        from sklearn.cluster import KMeans, SpectralClustering

        X = np.asarray(X)
        n = X.shape[0]
        trajectory: list[Step] = []
        step_idx = 0

        # ------------------------------------------------------------------
        # Stage 1: density-based region discovery.
        # ------------------------------------------------------------------
        region_ids, eps_auto = self._stage1_regions(X)
        unique_regions = [int(r) for r in np.unique(region_ids) if r != -1]
        n_regions = len(unique_regions)
        n_noise = int(np.sum(region_ids == -1))

        # Graceful degradation: 0 or 1 DBSCAN regions => one big uniform region.
        fallback = n_regions <= 1
        if fallback:
            # Treat every point as a single region; no noise.
            region_ids = np.zeros(n, dtype=np.int64)
            unique_regions = [0]
            n_regions = 1
            n_noise = 0

        trajectory.append(
            Step(
                step_idx=step_idx,
                cost=float(n_noise),
                delta_cost=None,
                accepted=True,
                action={"type": "stage1_density", "fallback": fallback},
                state={
                    "n_regions": n_regions,
                    "n_noise": n_noise,
                    "eps_auto": float(eps_auto),
                },
            )
        )
        step_idx += 1

        # ------------------------------------------------------------------
        # Stage 2: per-region routing & local clustering.
        # ------------------------------------------------------------------
        total_clustered = int(np.sum(region_ids != -1))
        labels = -np.ones(n, dtype=np.int64)
        next_cluster_id = 0
        region_routes: list[dict[str, Any]] = []

        for region_id in unique_regions:
            mask = region_ids == region_id
            idx = np.where(mask)[0]
            X_region = X[idx]
            convexity, n_points = self._shape_signature(X_region)

            # Per-region k allocation, proportional to region size.
            if total_clustered > 0:
                k_per_region = max(1, int(round(k * n_points / total_clustered)))
            else:
                k_per_region = 1
            # Local k can never exceed the number of points in the region.
            k_per_region = min(k_per_region, n_points) if n_points > 0 else 1

            method: str
            local_labels: np.ndarray
            if n_points < 5:
                method = "tiny"
                local_labels = np.zeros(n_points, dtype=np.int64)
            elif convexity > 0.5 and n_points >= 5 * k_per_region:
                method = "kmeans"
                if k_per_region <= 1:
                    local_labels = np.zeros(n_points, dtype=np.int64)
                else:
                    km = KMeans(
                        n_clusters=k_per_region,
                        init="k-means++",
                        n_init=10,
                        random_state=self.random_state,
                    )
                    local_labels = km.fit_predict(X_region).astype(np.int64)
            elif convexity <= 0.5:
                method = "spectral"
                if k_per_region <= 1:
                    local_labels = np.zeros(n_points, dtype=np.int64)
                else:
                    # Spectral requires k_per_region < n_points; clamp defensively.
                    k_eff = min(k_per_region, max(1, n_points - 1))
                    try:
                        sc = SpectralClustering(
                            n_clusters=k_eff,
                            affinity="nearest_neighbors",
                            assign_labels="kmeans",
                            random_state=self.random_state,
                            n_jobs=-1,
                        )
                        local_labels = sc.fit_predict(X_region).astype(np.int64)
                    except Exception:
                        # Spectral can fail on degenerate affinity graphs;
                        # fall back to kmeans rather than abort the run.
                        method = "kmeans_fallback"
                        if k_eff <= 1:
                            local_labels = np.zeros(n_points, dtype=np.int64)
                        else:
                            km = KMeans(
                                n_clusters=k_eff,
                                init="k-means++",
                                n_init=10,
                                random_state=self.random_state,
                            )
                            local_labels = km.fit_predict(X_region).astype(np.int64)
            else:
                # Convex but not enough points for the requested k => single cluster.
                method = "kmeans"
                local_labels = np.zeros(n_points, dtype=np.int64)

            n_local_clusters = int(local_labels.max()) + 1 if len(local_labels) else 0
            labels[idx] = local_labels + next_cluster_id

            route = {
                "region_id": int(region_id),
                "method": method,
                "n_points": int(n_points),
                "convexity": float(convexity),
                "k_per_region": int(k_per_region),
                "cluster_id_offset": int(next_cluster_id),
                "n_local_clusters": int(n_local_clusters),
            }
            region_routes.append(route)

            trajectory.append(
                Step(
                    step_idx=step_idx,
                    cost=float(n_points),
                    delta_cost=None,
                    accepted=True,
                    action={
                        "type": "route",
                        "method": method,
                        "region_id": int(region_id),
                        "n_points": int(n_points),
                        "convexity": float(convexity),
                    },
                    state={
                        "k_per_region": int(k_per_region),
                        "cluster_id_offset": int(next_cluster_id),
                        "n_local_clusters": int(n_local_clusters),
                    },
                )
            )
            step_idx += 1
            next_cluster_id += max(1, n_local_clusters)

        # ------------------------------------------------------------------
        # Stage 3: outlier reassignment via 1-NN to the labelled set.
        # ------------------------------------------------------------------
        noise_mask = labels == -1
        n_reassigned = int(noise_mask.sum())
        if n_reassigned > 0:
            labelled_mask = ~noise_mask
            if labelled_mask.any():
                from sklearn.neighbors import NearestNeighbors

                nbrs = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(X[labelled_mask])
                _, nn_idx = nbrs.kneighbors(X[noise_mask])
                labelled_labels = labels[labelled_mask]
                labels[noise_mask] = labelled_labels[nn_idx[:, 0]]
            else:
                # Pathological: no labelled points at all. Make everything one cluster.
                labels[:] = 0

        trajectory.append(
            Step(
                step_idx=step_idx,
                cost=float(n_reassigned),
                delta_cost=None,
                accepted=True,
                action={"type": "stage3_reassign_outliers", "n_reassigned": n_reassigned},
                state={"n_final_clusters": int(np.unique(labels).size)},
            )
        )

        return AlgoResult(
            labels=labels.astype(np.int64),
            extra={
                "eps_auto": float(eps_auto),
                "n_regions": int(n_regions),
                "n_noise": int(n_noise),
                "region_routes": region_routes,
            },
            trajectory=trajectory,
        )
