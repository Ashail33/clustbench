"""Tests for the algorithm registry and individual clustering algorithms."""

from __future__ import annotations

import numpy as np
import pytest

from clustbench.algorithms import ALGO_REGISTRY, Algorithm, AlgoResult
from clustbench.algorithms.base import register


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def blobs_2d():
    """Small 2-D dataset with 3 tight clusters."""
    rng = np.random.default_rng(0)
    X = np.vstack([
        rng.normal([0, 0], 0.1, (50, 2)),
        rng.normal([5, 0], 0.1, (50, 2)),
        rng.normal([0, 5], 0.1, (50, 2)),
    ]).astype(np.float32)
    y = np.array([0] * 50 + [1] * 50 + [2] * 50)
    return X, y


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registry_has_expected_algorithms():
    expected = {"kmeans", "minibatch_kmeans", "dbscan", "birch_algo", "clarans", "consensus"}
    assert expected.issubset(ALGO_REGISTRY.keys()), (
        f"Missing algorithms: {expected - set(ALGO_REGISTRY.keys())}"
    )


def test_register_decorator():
    @register
    class _TestAlgo(Algorithm):
        name = "_testalgo"

        def fit_predict(self, X, k=None):
            return AlgoResult(labels=np.zeros(len(X), dtype=int), extra={})

    assert "_testalgo" in ALGO_REGISTRY
    # Clean up to avoid polluting other tests
    del ALGO_REGISTRY["_testalgo"]


# ---------------------------------------------------------------------------
# KMeans
# ---------------------------------------------------------------------------

def test_kmeans_basic(blobs_2d):
    X, y = blobs_2d
    from clustbench.algorithms.kmeans import Kmeans
    algo = Kmeans(max_iter=100, n_init=3, random_state=0)
    result = algo.fit_predict(X, k=3)
    assert isinstance(result, AlgoResult)
    assert result.labels.shape == (len(X),)
    assert len(set(result.labels)) == 3
    assert "inertia" in result.extra


def test_kmeans_requires_k(blobs_2d):
    X, _ = blobs_2d
    from clustbench.algorithms.kmeans import Kmeans
    with pytest.raises(AssertionError):
        Kmeans().fit_predict(X, k=None)


# ---------------------------------------------------------------------------
# MiniBatchKMeans
# ---------------------------------------------------------------------------

def test_minibatch_kmeans_basic(blobs_2d):
    X, _ = blobs_2d
    from clustbench.algorithms.minibatch_kmeans import Minibatch_kmeans
    algo = Minibatch_kmeans(batch_size=50, max_iter=50, random_state=0)
    result = algo.fit_predict(X, k=3)
    assert result.labels.shape == (len(X),)
    assert len(set(result.labels)) == 3


# ---------------------------------------------------------------------------
# DBSCAN
# ---------------------------------------------------------------------------

def test_dbscan_basic(blobs_2d):
    X, _ = blobs_2d
    from clustbench.algorithms.dbscan import Dbscan
    # With tight blobs and appropriate eps, DBSCAN should find 3 clusters
    algo = Dbscan(eps=0.5, min_samples=3)
    result = algo.fit_predict(X)
    assert result.labels.shape == (len(X),)
    n_clusters = len(set(result.labels) - {-1})
    assert n_clusters >= 1


# ---------------------------------------------------------------------------
# Birch
# ---------------------------------------------------------------------------

def test_birch_basic(blobs_2d):
    X, _ = blobs_2d
    from clustbench.algorithms.birch import Birch_algo
    algo = Birch_algo(threshold=0.5)
    result = algo.fit_predict(X, k=3)
    assert result.labels.shape == (len(X),)
    assert len(set(result.labels)) == 3


# ---------------------------------------------------------------------------
# CLARANS
# ---------------------------------------------------------------------------

def test_clarans_basic(blobs_2d):
    X, _ = blobs_2d
    from clustbench.algorithms.clarans import Clarans
    algo = Clarans(numlocal=1, maxneigh=20, random_state=0)
    result = algo.fit_predict(X, k=3)
    assert result.labels.shape == (len(X),)
    assert len(set(result.labels)) == 3


# ---------------------------------------------------------------------------
# Consensus
# ---------------------------------------------------------------------------

def test_consensus_basic(blobs_2d):
    X, _ = blobs_2d
    from clustbench.algorithms.consensus import Consensus
    algo = Consensus(base=["kmeans", "minibatch_kmeans", "birch_algo"])
    result = algo.fit_predict(X, k=3)
    assert result.labels.shape == (len(X),)
    assert len(set(result.labels)) == 3
    assert "bases" in result.extra


def test_consensus_missing_base_raises(blobs_2d):
    X, _ = blobs_2d
    from clustbench.algorithms.consensus import Consensus
    with pytest.raises(KeyError):
        Consensus(base=["nonexistent_algo"]).fit_predict(X, k=3)
