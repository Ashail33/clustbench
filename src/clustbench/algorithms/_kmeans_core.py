"""Shared k-means primitives with the textbook complexity.

Earlier versions of ``kmeans`` and ``parallel_kmeans`` had two
implementation artefacts that showed up in the big-data sweep as if they
were properties of Lloyd's algorithm:

* k-means++ recomputed the distance from every point to *every* chosen
  centre at each seeding step, which is O(n k^2 d) instead of O(n k d).
  That alone produced the super-linear growth in ``k`` (exponents 1.5-1.9)
  reported for the Lloyd-type variants.
* the assignment step materialised an ``n x k x d`` difference tensor, so
  memory and time grew with ``n k d`` *elements* rather than with a
  BLAS matrix product.

These helpers keep the running minimum distance during seeding and use the
Gram expansion ``|x - c|^2 = |x|^2 - 2 x.c + |c|^2`` for distances. The
random draws are unchanged: the same ``rng`` produces the same sequence of
seeds as before (up to floating-point ties).
"""

from __future__ import annotations

import numpy as np


def sq_dists(X: np.ndarray, C: np.ndarray, x_sq: np.ndarray | None = None) -> np.ndarray:
    """Squared Euclidean distances between rows of ``X`` and ``C``, (n, k)."""
    if x_sq is None:
        x_sq = np.einsum("ij,ij->i", X, X)
    c_sq = np.einsum("ij,ij->i", C, C)
    D = x_sq[:, None] - 2.0 * (X @ C.T) + c_sq[None, :]
    np.maximum(D, 0.0, out=D)
    return D


def kmeans_pp_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ seeding in O(n k d) time and O(n) extra memory."""
    n = X.shape[0]
    centers = np.empty((k, X.shape[1]), dtype=np.float64)
    first = int(rng.integers(n))
    centers[0] = X[first]
    x_sq = np.einsum("ij,ij->i", X, X)
    d2 = sq_dists(X, centers[:1], x_sq)[:, 0]
    for j in range(1, k):
        total = d2.sum()
        if total <= 0:
            idx = int(rng.integers(n))
        else:
            idx = int(rng.choice(n, p=d2 / total))
        centers[j] = X[idx]
        np.minimum(d2, sq_dists(X, centers[j : j + 1], x_sq)[:, 0], out=d2)
    return centers


def assign(X: np.ndarray, C: np.ndarray, x_sq: np.ndarray | None = None):
    """Nearest-centre labels and inertia (sum of squared distances)."""
    D = sq_dists(X, C, x_sq)
    labels = D.argmin(axis=1)
    inertia = float(D[np.arange(X.shape[0]), labels].sum())
    return labels, inertia


def partial_sums(X: np.ndarray, labels: np.ndarray, k: int):
    """Per-cluster coordinate sums and counts, vectorised (no Python loop over k)."""
    from scipy.sparse import csr_matrix

    n = X.shape[0]
    counts = np.bincount(labels, minlength=k).astype(np.int64)
    onehot = csr_matrix((np.ones(n), (labels, np.arange(n))), shape=(k, n))
    sums = np.asarray(onehot @ X, dtype=np.float64)
    return sums, counts
