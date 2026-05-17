"""META-CLUSTERER: data-fingerprint dispatcher.

This algorithm operationalises the "What to try first" decision tree in
``docs/ALGORITHM_ANALYSIS.md``. It computes four cheap fingerprints of
``X`` and then dispatches to one already-registered base algorithm whose
inductive bias best matches the data:

    fingerprint              hint                 algorithm
    ---------------------    ------------------   ----------------------
    outlier_frac > 0.10      heavy-tailed noise   gmm
    eigengap < 0.05 & low D  non-convex manifold  spectral
    density_skew & clean     density-varying      dbscan_auto / birch_algo
    high eff_dim, clean      Euclidean, many dim  birch_algo
    default / mixed signal   safe consensus       pwcc_diverse / kmeans

The four fingerprints are each O(n) or O(n log n):

* ``eff_dim``   - number of PCA components capturing > 1% variance, up
  to ``min(n_features, 50)``.
* ``eigengap`` - normalised gap between the k-th and (k-1)-th smallest
  eigenvalues of the normalised graph Laplacian, computed on a kNN
  graph over a random sub-sample of at most 500 points.
* ``outlier_frac`` - fraction of points flagged ``-1`` by
  ``sklearn.neighbors.LocalOutlierFactor(n_neighbors=20)``.
* ``density_skew`` - std-of-kNN-distances / mean-of-kNN-distances; large
  means density varies across the dataset.

The trajectory records a single ``route`` step whose action lists the
chosen algorithm and the fingerprint dict, mirroring the consensus and
pwcc patterns.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from . import base as base_algos
from .base import Algorithm, AlgoResult, Step, register


def _eff_dim(X: np.ndarray) -> int:
    """Count PCA components capturing > 1% of variance each."""
    from sklearn.decomposition import PCA

    n_components = min(X.shape[0], X.shape[1], 50)
    if n_components < 1:
        return 1
    pca = PCA(n_components=n_components, svd_solver="auto", random_state=0)
    try:
        pca.fit(X)
    except Exception:
        return int(X.shape[1])
    return int(np.sum(pca.explained_variance_ratio_ > 0.01))


def _eigengap(X: np.ndarray, k: int, seed: int = 0) -> float:
    """Normalised gap ``(lambda[k] - lambda[k-1]) / lambda[k]`` of the
    normalised Laplacian on a small kNN graph. A small gap implies that
    the spectral embedding has no clean cut at ``k`` clusters - usually
    a sign of non-convex / manifold structure that spectral handles."""
    from sklearn.neighbors import kneighbors_graph
    from scipy.sparse.csgraph import laplacian
    from scipy.sparse.linalg import eigsh

    n = X.shape[0]
    m = min(n, 500)
    rng = np.random.default_rng(seed)
    if m < n:
        idx = rng.choice(n, size=m, replace=False)
        Xs = X[idx]
    else:
        Xs = X
    n_neighbors = min(10, max(2, m - 1))
    try:
        A = kneighbors_graph(
            Xs, n_neighbors=n_neighbors, mode="connectivity", include_self=False
        )
        # Symmetrise so the Laplacian is real-symmetric.
        A = 0.5 * (A + A.T)
        L = laplacian(A, normed=True)
        # ``eigsh`` needs k strictly less than the matrix dimension.
        n_eigs = min(max(k + 2, 3), m - 1)
        # ``which='SM'`` returns smallest-magnitude eigenvalues.
        vals = eigsh(L, k=n_eigs, which="SM", return_eigenvectors=False)
        vals = np.sort(np.real(vals))
    except Exception:
        return 1.0  # neutral / "no useful gap signal"
    if k < 1 or k >= len(vals):
        return 1.0
    lam_k = float(vals[k])
    lam_km1 = float(vals[k - 1])
    if lam_k <= 1e-12:
        return 1.0
    return (lam_k - lam_km1) / lam_k


def _outlier_frac(X: np.ndarray) -> float:
    """Fraction of points flagged by LocalOutlierFactor (n_neighbors=20)."""
    from sklearn.neighbors import LocalOutlierFactor

    n = X.shape[0]
    n_neighbors = min(20, max(2, n - 1))
    try:
        lof = LocalOutlierFactor(n_neighbors=n_neighbors)
        flags = lof.fit_predict(X)
    except Exception:
        return 0.0
    return float(np.mean(flags == -1))


def _density_skew(X: np.ndarray) -> float:
    """std / mean of mean-kNN distances. Large => density varies."""
    from sklearn.neighbors import NearestNeighbors

    n = X.shape[0]
    n_neighbors = min(10, max(2, n - 1))
    try:
        nn = NearestNeighbors(n_neighbors=n_neighbors)
        nn.fit(X)
        dists, _ = nn.kneighbors(X)
        # Drop the trivial self-distance (column 0) and take mean per row.
        per_point = dists[:, 1:].mean(axis=1)
    except Exception:
        return 0.0
    mu = float(per_point.mean())
    if mu <= 1e-12:
        return 0.0
    return float(per_point.std() / mu)


def _fingerprints(X: np.ndarray, k: int) -> Dict[str, float]:
    return {
        "eff_dim": int(_eff_dim(X)),
        "eigengap": float(_eigengap(X, k)),
        "outlier_frac": float(_outlier_frac(X)),
        "density_skew": float(_density_skew(X)),
    }


def _resolve(name: str, fallback: str) -> str:
    """Return ``name`` if registered, else ``fallback`` if registered, else
    ``kmeans`` as a last-resort baseline."""
    if name in base_algos.ALGO_REGISTRY:
        return name
    if fallback in base_algos.ALGO_REGISTRY:
        return fallback
    return "kmeans"


def _route(fp: Dict[str, float]) -> str:
    """Pick an algorithm key from the data fingerprints.

    Order matters: earlier conditions win. Each branch resolves to an
    available registered algorithm or a graceful fallback.
    """
    eff_dim = fp["eff_dim"]
    eigengap = fp["eigengap"]
    outlier_frac = fp["outlier_frac"]
    density_skew = fp["density_skew"]

    if outlier_frac > 0.10:
        return _resolve("gmm", "kmeans")
    if eigengap < 0.05 and eff_dim < 5:
        return _resolve("spectral", "kmeans")
    if density_skew > 0.5 and outlier_frac < 0.05:
        return _resolve("dbscan_auto", "birch_algo")
    if eff_dim >= 10 and outlier_frac < 0.05:
        return _resolve("birch_algo", "kmeans")
    return _resolve("pwcc_diverse", "kmeans")


@register
class Meta_clusterer(Algorithm):
    """Data-fingerprint dispatcher.

    Examines ``X``, picks a base algorithm, runs it, and returns its
    labels along with the fingerprints used to make the decision.

    Parameters
    ----------
    base_params : dict[str, dict] | None
        Per-base hyper-parameters keyed by the registered algorithm name,
        forwarded to the chosen algorithm's constructor. By default the
        spectral branch is invoked with ``n_neighbors=10`` (matching the
        affinity graph used by the eigengap fingerprint).
    seed : int
        Random seed for the eigengap sub-sample.
    """

    def __init__(
        self,
        base_params: dict | None = None,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        self.name = "meta_clusterer"
        self.base_params = base_params or {}
        # Spectral defaults that match the fingerprint graph.
        self.base_params.setdefault("spectral", {"n_neighbors": 10})
        self.seed = seed

    def fit_predict(self, X: np.ndarray, k: int | None = None) -> AlgoResult:
        assert k is not None, "k required"

        fp = _fingerprints(np.asarray(X), int(k))
        chosen = _route(fp)

        cls = base_algos.ALGO_REGISTRY.get(chosen)
        if cls is None:
            # Last-resort safety net: kmeans is always registered.
            chosen = "kmeans"
            cls = base_algos.ALGO_REGISTRY["kmeans"]
        params = dict(self.base_params.get(chosen, {}))
        res = cls(**params).fit_predict(X, k=k)

        trajectory = [
            Step(
                step_idx=0,
                cost=0.0,
                delta_cost=None,
                accepted=True,
                action={"type": "route", "to": chosen, "fingerprints": fp},
                state={"fingerprints": fp},
            )
        ]

        extra: Dict[str, Any] = {
            "chosen_algo": chosen,
            "fingerprints": fp,
        }
        # Merge the underlying algorithm's extra, namespacing collisions.
        for key, value in (res.extra or {}).items():
            if key in extra:
                extra[f"{chosen}.{key}"] = value
            else:
                extra[key] = value

        return AlgoResult(
            labels=np.asarray(res.labels, dtype=np.int64),
            extra=extra,
            trajectory=trajectory,
        )
