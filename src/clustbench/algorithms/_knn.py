"""k-nearest-neighbour graphs, exact or approximate.

Exact kNN (scikit-learn, brute force at moderate d) costs O(n^2 d) and is
what makes graph methods such as LMM and spectral clustering quadratic in n:
at n = 50k, d = 20 it is ~80% of LMM's fit time. ``method="nndescent"``
uses NN-descent (Dong, Charikar & Li, 2011; the ``pynndescent``
implementation), which builds an approximate graph in roughly
O(n^1.14 d) time. On MDCGen data it recovers 95-98% of the exact
neighbours at d = 20 (77% at d = 100), and a clustering graph only needs
neighbours from the same cluster, not the exact ones.

The exact search is used below ``min_n_approx`` points (10k) and at
``d <= 10``, where it is as fast or faster: scikit-learn then uses a k-d
tree (0.9 s vs 1.1 s for NN-descent at n = 40k, d = 5). pynndescent JIT-compiles on first use (~20 s);
:func:`warm` triggers that once per process so the cost does not land in a
timed run, and it is called on every ``nndescent`` request, so the
harness's untimed warm-up compiles it before any timed run is forked.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import scipy.sparse as sp

METHODS = ("exact", "nndescent")
_WARM = False


def warm() -> bool:
    """Compile pynndescent once per process; False if it is not installed."""
    global _WARM
    if _WARM:
        return True
    try:
        import numba

        # numba's default OpenMP layer aborts in a forked child ("fork() called
        # from a process already using GNU OpenMP"), and the harness forks each
        # isolated run from this warmed-up process. workqueue is fork-safe.
        if "NUMBA_THREADING_LAYER" not in os.environ:
            numba.config.THREADING_LAYER = "workqueue"
        import pynndescent
    except ImportError:
        return False
    # Large enough to reach every code path (tree init, descent); a 300-point
    # input leaves ~1 s of compilation for the first real call.
    X = np.random.default_rng(0).normal(size=(2000, 8)).astype(np.float32)
    pynndescent.NNDescent(X, n_neighbors=15, n_jobs=1, random_state=0).neighbor_graph
    _WARM = True
    return True


def knn_indices(X: np.ndarray, k: int, method: str = "exact", random_state: int = 0,
                min_n_approx: int = 10000) -> tuple[np.ndarray, np.ndarray]:
    """(n, k) neighbour indices and distances, excluding each point itself."""
    if method not in METHODS:
        raise ValueError(f"knn method must be one of {METHODS}")
    n = X.shape[0]
    k = max(1, min(int(k), n - 1))
    if method == "nndescent":
        if not warm():
            warnings.warn("pynndescent not installed; using exact kNN")
            method = "exact"
    if method == "nndescent" and n >= min_n_approx and X.shape[1] > 10:
        import pynndescent

        idx = pynndescent.NNDescent(np.ascontiguousarray(X, dtype=np.float32), n_neighbors=k + 1,
                                    n_jobs=1, random_state=random_state)
        I, D = idx.neighbor_graph
        I, D = _drop_self(I, D, k)
        return I, D.astype(np.float64)
    from sklearn.neighbors import NearestNeighbors

    D, I = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    return _drop_self(I, D, k)


def _drop_self(I: np.ndarray, D: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(I)
    is_self = I == np.arange(n)[:, None]
    # Rows without a self match (duplicates, or approximate search) drop the last column.
    no_self = ~is_self.any(axis=1)
    is_self[no_self, -1] = True
    keep = ~is_self
    return I[keep].reshape(n, k), D[keep].reshape(n, k)


def knn_graph(X: np.ndarray, k: int, method: str = "exact", mode: str = "connectivity",
              random_state: int = 0, min_n_approx: int = 10000) -> sp.csr_matrix:
    """Directed kNN graph, as :func:`sklearn.neighbors.kneighbors_graph` returns."""
    I, D = knn_indices(X, k, method, random_state, min_n_approx)
    n, kk = I.shape
    data = np.ones(n * kk) if mode == "connectivity" else D.ravel()
    return sp.csr_matrix((data, I.ravel(), np.arange(0, n * kk + 1, kk)), shape=(n, n))
