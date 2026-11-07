"""Equivalence and complexity checks for the k-means family.

Parallel k-means distributes the same Lloyd iteration, so from the same
seeding it must return the same partition as k-means. In Maharaj (2024) the
two disagreed (ARI ~1.0 vs ~0.25), which was an implementation artefact
reported as an algorithmic finding; this test would have caught it."""
from __future__ import annotations

import time

import numpy as np
import pytest
from sklearn.cluster import KMeans as SkKMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from clustbench.algorithms._kmeans_core import assign, kmeans_pp_init, partial_sums, sq_dists
from clustbench.algorithms.kmeans import Kmeans
from clustbench.algorithms.parallel_kmeans import Parallel_kmeans


@pytest.mark.parametrize("n_features,k", [(2, 3), (10, 5), (50, 10)])
def test_parallel_kmeans_matches_kmeans(n_features, k):
    X, _ = make_blobs(n_samples=2000, n_features=n_features, centers=k, random_state=1)
    a = Kmeans(max_iter=100, n_init=2, random_state=7).fit_predict(X, k).labels
    b = Parallel_kmeans(max_iter=100, n_init=2, n_workers=1, random_state=7).fit_predict(X, k).labels
    assert adjusted_rand_score(a, b) > 0.999


def test_parallel_kmeans_workers_do_not_change_result():
    X, _ = make_blobs(n_samples=3000, n_features=8, centers=6, random_state=3)
    one = Parallel_kmeans(n_init=1, n_workers=1, random_state=0).fit_predict(X, 6).labels
    two = Parallel_kmeans(n_init=1, n_workers=2, random_state=0).fit_predict(X, 6).labels
    assert adjusted_rand_score(one, two) > 0.999


def test_kmeans_matches_sklearn_quality():
    X, y = make_blobs(n_samples=3000, n_features=10, centers=8, cluster_std=1.0, random_state=5)
    ours = Kmeans(n_init=3, random_state=0).fit_predict(X, 8)
    sk = SkKMeans(n_clusters=8, n_init=3, random_state=0).fit(X)
    # Same objective: inertia within 1% of sklearn's on easy blobs.
    assert ours.extra["inertia"] <= sk.inertia_ * 1.01
    assert adjusted_rand_score(y, ours.labels) > 0.95


def test_core_helpers_match_brute_force():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 7))
    C = rng.normal(size=(6, 7))
    brute = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
    assert np.allclose(sq_dists(X, C), brute, atol=1e-9)
    labels, inertia = assign(X, C)
    assert np.array_equal(labels, brute.argmin(axis=1))
    assert np.isclose(inertia, brute.min(axis=1).sum())
    sums, counts = partial_sums(X, labels, 6)
    for j in range(6):
        assert counts[j] == (labels == j).sum()
        assert np.allclose(sums[j], X[labels == j].sum(axis=0))


def test_kpp_seeding_is_linear_in_k():
    """k-means++ must cost O(n k d): doubling k should roughly double the time,
    not quadruple it (the old O(n k^2 d) seeding grew ~4x)."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20000, 20))

    def t(k):
        best = np.inf
        for _ in range(3):
            t0 = time.perf_counter()
            kmeans_pp_init(X, k, np.random.default_rng(1))
            best = min(best, time.perf_counter() - t0)
        return best

    ratio = t(128) / t(32)  # 4x k
    assert ratio < 8.0, ratio  # linear ~4x; quadratic would be ~16x


@pytest.mark.parametrize("offset", [0.0, 1e9])
def test_large_offset_does_not_break_distances(offset):
    """Gram-expansion distances cancel when data sit far from the origin
    (e.g. timestamp features ~1e9); the k-means paths centre first."""
    X, y = make_blobs(n_samples=2000, n_features=5, centers=4, cluster_std=0.5, random_state=0)
    X = X + offset
    for algo in (Kmeans(n_init=2, random_state=0), Parallel_kmeans(n_init=2, n_workers=1, random_state=0)):
        res = algo.fit_predict(X, 4)
        assert adjusted_rand_score(y, res.labels) > 0.99
    C = kmeans_pp_init(X, 4, np.random.default_rng(0))
    assert np.all(np.abs(C - offset).max(axis=1) < 50)  # centres are real data points, shifted back
