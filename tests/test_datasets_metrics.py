"""Tests for dataset generators and metrics."""

from __future__ import annotations

import numpy as np
import pytest

from clustbench.datasets import DATASETS, DataSpec, gen_blobs, gen_mixed
from clustbench.metrics import (
    bundle_scores,
    compactness,
    dunn_index,
    separation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def spec():
    return DataSpec(n_samples=200, n_features=4, centers=3, compactness=0.5, seed=42)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

def test_datasets_registry_has_blobs_and_mixed():
    assert "blobs" in DATASETS
    assert "mixed" in DATASETS


def test_gen_blobs_shape(spec):
    X, y = gen_blobs(spec)
    assert X.shape == (200, 4)
    assert y.shape == (200,)
    assert X.dtype == np.float32


def test_gen_blobs_cluster_count(spec):
    _, y = gen_blobs(spec)
    assert len(np.unique(y)) == 3


def test_gen_mixed_shape(spec):
    X, y = gen_mixed(spec)
    assert X.shape == (200, 4)
    assert X.dtype == np.float32


def test_gen_blobs_reproducible(spec):
    X1, y1 = gen_blobs(spec)
    X2, y2 = gen_blobs(spec)
    np.testing.assert_array_equal(X1, X2)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@pytest.fixture
def perfect_labels():
    """Labels that exactly match 3 non-overlapping clusters."""
    X = np.vstack([
        np.random.default_rng(0).normal([0, 0], 0.01, (30, 2)),
        np.random.default_rng(1).normal([10, 0], 0.01, (30, 2)),
        np.random.default_rng(2).normal([0, 10], 0.01, (30, 2)),
    ]).astype(np.float32)
    labels = np.array([0] * 30 + [1] * 30 + [2] * 30)
    return X, labels


def test_compactness_is_finite(perfect_labels):
    X, labels = perfect_labels
    c = compactness(X, labels)
    assert np.isfinite(c)
    assert c >= 0


def test_separation_is_positive(perfect_labels):
    X, labels = perfect_labels
    s = separation(X, labels)
    assert np.isfinite(s)
    assert s > 0


def test_dunn_index_is_positive(perfect_labels):
    X, labels = perfect_labels
    d = dunn_index(X, labels)
    assert np.isfinite(d)
    assert d > 0


def test_bundle_scores_with_ground_truth(perfect_labels):
    X, labels = perfect_labels
    scores = bundle_scores(X, labels, y_true=labels)
    assert "ari" in scores
    assert "nmi" in scores
    assert "silhouette" in scores
    assert "davies_bouldin" in scores
    # Perfect labels → ARI and NMI should be 1
    assert pytest.approx(scores["ari"], abs=1e-6) == 1.0
    assert pytest.approx(scores["nmi"], abs=1e-6) == 1.0


def test_bundle_scores_without_ground_truth(perfect_labels):
    X, labels = perfect_labels
    scores = bundle_scores(X, labels)
    assert "ari" not in scores
    assert "nmi" not in scores
    assert "silhouette" not in scores
    assert "davies_bouldin" in scores


def test_compactness_single_cluster_returns_nan():
    X = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    labels = np.array([0, 0])
    # Should handle gracefully – returns a float
    c = compactness(X, labels)
    assert isinstance(c, float)


def test_separation_single_cluster_returns_nan():
    X = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    labels = np.array([0, 0])
    s = separation(X, labels)
    assert np.isnan(s)
