"""RAPID v2: Region-Adaptive Partitioning with Iterative Density,
hardened against outlier contamination of the stage-1 eps estimate.

This is a deliberate iteration on :class:`Rapid` (``rapid.py``). Empirically,
v1 was the strongest *synthesised* algorithm on the 16-task benchmark — the
only one to solve both moons and circles at the same time — but its outlier
delta was a regrettable -0.145: a handful of uniform outliers contaminated
the k-distance knee used to estimate ``eps``, which in turn made stage 1
either merge clusters into one giant region (eps too large) or shred them
into noise (eps too small), so stage 2 routed garbage.

v2 prepends a stage-0 robust outlier filter:

Stage 0 (LOF outlier hold-out, new):
    Run :class:`sklearn.neighbors.LocalOutlierFactor`. Set aside the points
    with the most negative LOF scores (top ``outlier_quantile`` fraction)
    as suspected outliers. Stage 1's k-distance knee is computed on the
    *cleaned* data only.

Stages 1-3 (region discovery / per-region routing / 1-NN noise reassign):
    Unchanged from v1, just applied to ``X_clean``.

Stage 4 (held-aside outlier reassignment, new):
    Each held-aside outlier inherits the cluster id of its nearest
    neighbour in ``X_clean``, so the final ``labels`` array matches the
    input length and contains no ``-1``s.
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
class Rapid_v2(Algorithm):
    """Region-Adaptive Partitioning with Iterative Density, v2.

    Parameters
    ----------
    min_samples : int
        ``min_samples`` for the Stage-1 DBSCAN (also the ``k`` in the
        k-distance knee estimate).
    outlier_quantile : float
        Fraction of points to hold aside as suspected outliers in stage 0.
        Defaults to ``0.10`` (top 10% LOF scores).
    lof_n_neighbors : int
        ``n_neighbors`` for :class:`LocalOutlierFactor`. Defaults to 20.
    random_state : int
        Seed forwarded to KMeans / SpectralClustering for reproducibility.
    """

    def __init__(
        self,
        min_samples: int = 5,
        outlier_quantile: float = 0.10,
        lof_n_neighbors: int = 20,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        self.name = "rapid_v2"
        self.min_samples = min_samples
        self.outlier_quantile = float(outlier_quantile)
        self.lof_n_neighbors = int(lof_n_neighbors)
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Stage 0: LOF-based robust outlier hold-out.
    # ------------------------------------------------------------------
    def _stage0_lof_filter(self, X: np.ndarray) -> tuple[np.ndarray, float]:
        """Return ``(outlier_mask, threshold)``.

        ``outlier_mask[i]`` is True if point ``i`` is in the top
        ``outlier_quantile`` of LOF outlier scores (most outlier-y).
        ``threshold`` is the cutoff on ``-negative_outlier_factor_``
        (higher = more outlier-y).

        Returns an all-False mask + NaN threshold when LOF can't run
        (n < lof_n_neighbors + 5) or when ``outlier_quantile <= 0``.
        """
        n = X.shape[0]
        # Need a couple of safety neighbours beyond lof_n_neighbors for LOF to be meaningful.
        if self.outlier_quantile <= 0 or n < self.lof_n_neighbors + 5 or n < 25:
            return np.zeros(n, dtype=bool), float("nan")

        from sklearn.neighbors import LocalOutlierFactor

        # contamination='auto' makes LOF return *scores* without committing to a label cut;
        # we apply our own quantile cutoff so the user can tune it independently.
        lof = LocalOutlierFactor(
            n_neighbors=min(self.lof_n_neighbors, n - 1),
            contamination="auto",
        )
        lof.fit(X)
        # negative_outlier_factor_ : the more negative, the more outlier-y.
        # Flip sign so "high = more outlier-y" => intuitive quantile cut.
        scores = -lof.negative_outlier_factor_
        threshold = float(np.quantile(scores, 1.0 - self.outlier_quantile))
        outlier_mask = scores >= threshold
        return outlier_mask, threshold

    # ------------------------------------------------------------------
    # Stage 1: density-based region discovery (verbatim from v1).
    # ------------------------------------------------------------------
    def _stage1_regions(self, X: np.ndarray) -> tuple[np.ndarray, float]:
        from sklearn.cluster import DBSCAN

        eps = _auto_eps(X, self.min_samples)
        dbscan = DBSCAN(eps=eps, min_samples=self.min_samples, n_jobs=-1)
        region_ids = dbscan.fit_predict(X).astype(np.int64)
        return region_ids, eps

    # ------------------------------------------------------------------
    # Stage 2: shape signature + per-region routing (verbatim from v1).
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
        std = float(d.std())
        if std <= 1e-12:
            return 1.0, n_points
        convexity = float(np.mean(d <= std))
        return convexity, n_points

    def fit_predict(self, X: np.ndarray, k: Optional[int] = None) -> AlgoResult:
        assert k is not None, "k required"
        from sklearn.cluster import KMeans, SpectralClustering

        X = np.asarray(X)
        n_total = X.shape[0]
        trajectory: list[Step] = []
        step_idx = 0

        # ------------------------------------------------------------------
        # Stage 0: LOF outlier hold-out.
        # ------------------------------------------------------------------
        outlier_mask_full, lof_threshold = self._stage0_lof_filter(X)
        n_outliers_removed = int(outlier_mask_full.sum())

        # Skip stage 0 entirely if LOF flagged < 1% (data wasn't noisy enough).
        if n_outliers_removed < max(1, int(0.01 * n_total)):
            outlier_mask_full = np.zeros(n_total, dtype=bool)
            n_outliers_removed = 0

        clean_idx = np.where(~outlier_mask_full)[0]
        outlier_idx = np.where(outlier_mask_full)[0]
        X_clean = X[clean_idx]
        X_outliers = X[outlier_idx]
        n_clean = X_clean.shape[0]

        trajectory.append(
            Step(
                step_idx=step_idx,
                cost=float(n_outliers_removed),
                delta_cost=None,
                accepted=True,
                action={"type": "stage0_lof_filter"},
                state={
                    "n_outliers_removed": int(n_outliers_removed),
                    "lof_threshold": float(lof_threshold) if np.isfinite(lof_threshold) else None,
                },
            )
        )
        step_idx += 1

        # ------------------------------------------------------------------
        # Stage 1: density-based region discovery (on X_clean).
        # ------------------------------------------------------------------
        region_ids, eps_auto = self._stage1_regions(X_clean)
        unique_regions = [int(r) for r in np.unique(region_ids) if r != -1]
        n_regions = len(unique_regions)
        n_noise = int(np.sum(region_ids == -1))

        # Graceful degradation: 0 or 1 DBSCAN regions => one big uniform region.
        fallback = n_regions <= 1
        if fallback:
            region_ids = np.zeros(n_clean, dtype=np.int64)
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
        # Stage 2: per-region routing & local clustering (verbatim from v1).
        # ------------------------------------------------------------------
        total_clustered = int(np.sum(region_ids != -1))
        labels_clean = -np.ones(n_clean, dtype=np.int64)
        next_cluster_id = 0
        region_routes: list[dict[str, Any]] = []

        for region_id in unique_regions:
            mask = region_ids == region_id
            idx = np.where(mask)[0]
            X_region = X_clean[idx]
            convexity, n_points = self._shape_signature(X_region)

            if total_clustered > 0:
                k_per_region = max(1, int(round(k * n_points / total_clustered)))
            else:
                k_per_region = 1
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
                method = "kmeans"
                local_labels = np.zeros(n_points, dtype=np.int64)

            n_local_clusters = int(local_labels.max()) + 1 if len(local_labels) else 0
            labels_clean[idx] = local_labels + next_cluster_id

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
        # Stage 3: stage-1 noise reassignment via 1-NN to labelled X_clean.
        # ------------------------------------------------------------------
        noise_mask = labels_clean == -1
        n_noise_reassigned = int(noise_mask.sum())
        if n_noise_reassigned > 0:
            labelled_mask = ~noise_mask
            if labelled_mask.any():
                from sklearn.neighbors import NearestNeighbors

                nbrs = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(X_clean[labelled_mask])
                _, nn_idx = nbrs.kneighbors(X_clean[noise_mask])
                labelled_labels = labels_clean[labelled_mask]
                labels_clean[noise_mask] = labelled_labels[nn_idx[:, 0]]
            else:
                labels_clean[:] = 0

        trajectory.append(
            Step(
                step_idx=step_idx,
                cost=float(n_noise_reassigned),
                delta_cost=None,
                accepted=True,
                action={"type": "stage3_reassign_noise", "n_reassigned": n_noise_reassigned},
                state={"n_clean_clusters": int(np.unique(labels_clean).size)},
            )
        )
        step_idx += 1

        # ------------------------------------------------------------------
        # Stage 4: assign held-aside outliers via 1-NN to X_clean.
        # ------------------------------------------------------------------
        labels = np.empty(n_total, dtype=np.int64)
        labels[clean_idx] = labels_clean
        n_outliers_reassigned = int(X_outliers.shape[0])
        if n_outliers_reassigned > 0:
            from sklearn.neighbors import NearestNeighbors

            nbrs = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(X_clean)
            _, nn_idx = nbrs.kneighbors(X_outliers)
            labels[outlier_idx] = labels_clean[nn_idx[:, 0]]

        trajectory.append(
            Step(
                step_idx=step_idx,
                cost=float(n_outliers_reassigned),
                delta_cost=None,
                accepted=True,
                action={"type": "stage4_reassign_outliers"},
                state={"n_outliers_reassigned": int(n_outliers_reassigned)},
            )
        )

        return AlgoResult(
            labels=labels.astype(np.int64),
            extra={
                "n_outliers_removed": int(n_outliers_removed),
                "lof_threshold": float(lof_threshold) if np.isfinite(lof_threshold) else None,
                "eps_auto": float(eps_auto),
                "n_regions": int(n_regions),
                "n_noise_reassigned": int(n_noise_reassigned),
                "region_routes": region_routes,
                "n_outliers_reassigned": int(n_outliers_reassigned),
            },
            trajectory=trajectory,
        )
