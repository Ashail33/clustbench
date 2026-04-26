"""Standard sklearn clustering algorithms wrapped for clustbench.

Each class is a tiny adapter that registers itself in
``ALGO_REGISTRY``. None of these emit a trajectory yet — they're
batch / one-shot from sklearn's perspective. Add explicit instrumentation
(à la ``kmeans.py``) when one of these becomes interesting for the
state-action research.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .base import Algorithm, AlgoResult, register


@register
class Gmm(Algorithm):
    """Gaussian-mixture EM clustering."""

    def __init__(
        self,
        max_iter: int = 100,
        n_init: int = 1,
        covariance_type: str = "full",
        reg_covar: float = 1e-6,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        self.name = "gmm"
        self.max_iter = max_iter
        self.n_init = n_init
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar
        self.random_state = random_state

    def fit_predict(self, X: np.ndarray, k: Optional[int] = None) -> AlgoResult:
        from sklearn.mixture import GaussianMixture

        assert k is not None
        gm = GaussianMixture(
            n_components=k,
            covariance_type=self.covariance_type,
            reg_covar=self.reg_covar,
            max_iter=self.max_iter,
            n_init=self.n_init,
            random_state=self.random_state,
        )
        labels = gm.fit_predict(X)
        return AlgoResult(
            labels=labels.astype(np.int64),
            extra={"bic": float(gm.bic(X)), "converged": bool(gm.converged_)},
        )


@register
class Agglomerative(Algorithm):
    """Agglomerative (hierarchical) clustering, Ward linkage by default."""

    def __init__(self, linkage: str = "ward", **kwargs: Any) -> None:
        self.name = "agglomerative"
        self.linkage = linkage

    def fit_predict(self, X: np.ndarray, k: Optional[int] = None) -> AlgoResult:
        from sklearn.cluster import AgglomerativeClustering

        assert k is not None
        ag = AgglomerativeClustering(n_clusters=k, linkage=self.linkage)
        labels = ag.fit_predict(X)
        return AlgoResult(labels=labels.astype(np.int64), extra={"linkage": self.linkage})


@register
class Spectral(Algorithm):
    """Spectral clustering on a nearest-neighbours affinity."""

    def __init__(
        self,
        affinity: str = "nearest_neighbors",
        n_neighbors: int = 10,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        self.name = "spectral"
        self.affinity = affinity
        self.n_neighbors = n_neighbors
        self.random_state = random_state

    def fit_predict(self, X: np.ndarray, k: Optional[int] = None) -> AlgoResult:
        from sklearn.cluster import SpectralClustering

        assert k is not None
        sc = SpectralClustering(
            n_clusters=k,
            affinity=self.affinity,
            n_neighbors=self.n_neighbors,
            assign_labels="kmeans",
            random_state=self.random_state,
        )
        labels = sc.fit_predict(X)
        return AlgoResult(labels=labels.astype(np.int64), extra={})


@register
class Meanshift(Algorithm):
    """Mean-shift clustering. Bandwidth is estimated if not provided."""

    def __init__(self, bandwidth: Optional[float] = None, bin_seeding: bool = True, **kwargs: Any) -> None:
        self.name = "meanshift"
        self.bandwidth = bandwidth
        self.bin_seeding = bin_seeding

    def fit_predict(self, X: np.ndarray, k: Optional[int] = None) -> AlgoResult:
        from sklearn.cluster import MeanShift, estimate_bandwidth

        bw = self.bandwidth or estimate_bandwidth(X, quantile=0.2, n_samples=min(500, len(X)))
        if bw <= 0:
            bw = 1.0
        ms = MeanShift(bandwidth=bw, bin_seeding=self.bin_seeding)
        labels = ms.fit_predict(X)
        return AlgoResult(labels=labels.astype(np.int64), extra={"bandwidth": float(bw)})


@register
class Optics(Algorithm):
    """OPTICS density-based clustering. ``k`` is ignored."""

    def __init__(self, min_samples: int = 5, xi: float = 0.05, **kwargs: Any) -> None:
        self.name = "optics"
        self.min_samples = min_samples
        self.xi = xi

    def fit_predict(self, X: np.ndarray, k: Optional[int] = None) -> AlgoResult:
        from sklearn.cluster import OPTICS

        op = OPTICS(min_samples=self.min_samples, xi=self.xi, n_jobs=-1)
        labels = op.fit_predict(X)
        return AlgoResult(labels=labels.astype(np.int64), extra={})
