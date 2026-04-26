"""CHAMELEON-style graph clustering.

This is a scalable in-tree approximation of Karypis et al.'s CHAMELEON
shape: build an initial set of graph-respecting micro-clusters, then merge
them down to ``k`` clusters using relative closeness between partitions.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph

from .base import Algorithm, AlgoResult, Step, register


@register
class Chameleon(Algorithm):
    """Two-phase CHAMELEON-style clustering.

    Parameters
    ----------
    n_neighbors : int
        Number of neighbours in the sparse kNN graph.
    overcluster_factor : int
        Number of micro-clusters is approximately ``k * overcluster_factor``.
    max_partitions : int
        Upper bound on initial micro-clusters.
    """

    def __init__(
        self,
        n_neighbors: int = 15,
        overcluster_factor: int = 8,
        max_partitions: int = 120,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        self.name = "chameleon"
        self.n_neighbors = n_neighbors
        self.overcluster_factor = overcluster_factor
        self.max_partitions = max_partitions
        self.random_state = random_state

    def fit_predict(self, X: np.ndarray, k: Optional[int] = None) -> AlgoResult:
        assert k is not None, "k required"
        n = X.shape[0]
        n_neighbors = max(2, min(self.n_neighbors, n - 1))
        n_parts = max(k, min(self.max_partitions, k * self.overcluster_factor, n))
        trajectory: list[Step] = []

        graph = kneighbors_graph(
            X,
            n_neighbors=n_neighbors,
            mode="distance",
            include_self=False,
            n_jobs=-1,
        )
        graph = graph.maximum(graph.T)
        trajectory.append(
            Step(
                step_idx=0,
                cost=float(graph.nnz),
                action={"type": "knn_graph", "n_neighbors": int(n_neighbors)},
                state={"nodes": int(n), "edges": int(graph.nnz)},
            )
        )

        if n_parts == k:
            labels = MiniBatchKMeans(
                n_clusters=k,
                batch_size=min(2048, max(256, n)),
                random_state=self.random_state,
                n_init=3,
            ).fit_predict(X)
            return AlgoResult(
                labels=labels.astype(np.int64),
                extra={"n_neighbors": n_neighbors, "n_partitions": n_parts},
                trajectory=trajectory,
            )

        micro = MiniBatchKMeans(
            n_clusters=n_parts,
            batch_size=min(2048, max(256, n)),
            random_state=self.random_state,
            n_init=3,
        ).fit_predict(X)
        trajectory.append(
            Step(
                step_idx=1,
                cost=float(n_parts),
                action={"type": "partition", "method": "minibatch_kmeans"},
                state={"n_partitions": int(n_parts)},
            )
        )

        centroids = np.vstack([X[micro == i].mean(axis=0) for i in range(n_parts)])
        sizes = np.bincount(micro, minlength=n_parts).astype(float)
        dist = pairwise_distances(centroids)
        positive = dist[dist > 0]
        scale = float(np.median(positive)) if positive.size else 1.0
        closeness = np.exp(-dist / max(scale, 1e-12))

        graph_coo = graph.tocoo()
        inter = np.zeros((n_parts, n_parts), dtype=np.float32)
        for i, j, d in zip(graph_coo.row, graph_coo.col, graph_coo.data):
            a = micro[i]
            b = micro[j]
            if a != b:
                inter[a, b] += 1.0 / (1.0 + float(d))
        inter = np.maximum(inter, inter.T)
        score = closeness * np.log1p(inter)
        np.fill_diagonal(score, 0.0)
        dissimilarity = 1.0 / (1.0 + score)

        merger = AgglomerativeClustering(
            n_clusters=k,
            metric="precomputed",
            linkage="average",
        )
        merged = merger.fit_predict(dissimilarity)
        labels = merged[micro].astype(np.int64)
        trajectory.append(
            Step(
                step_idx=2,
                cost=float(k),
                action={"type": "merge", "method": "relative_closeness"},
                state={
                    "n_partitions": int(n_parts),
                    "target_clusters": int(k),
                    "median_centroid_distance": scale,
                },
            )
        )

        return AlgoResult(
            labels=labels,
            extra={
                "n_neighbors": int(n_neighbors),
                "n_partitions": int(n_parts),
                "overcluster_factor": int(self.overcluster_factor),
                "partition_size_min": int(sizes.min()),
                "partition_size_max": int(sizes.max()),
            },
            trajectory=trajectory,
        )
