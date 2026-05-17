"""META-CLUSTERER v2: fingerprint dispatcher with non-convex detector and probe.

This is a deliberate iteration on :mod:`meta_clusterer` (v1). v1's
``eigengap < 0.05`` branch never fired on clean circles (a 2-circle
Laplacian has a *large* gap at index ``k`` because the structure is
clean) so the spectral route was effectively dead. v2 replaces the
spectral trigger with two more robust non-convexity diagnostics and
adds a cheap silhouette probe for low-confidence routing decisions.

The six fingerprints:

* ``eff_dim`` - number of PCA components with > 1% variance (same as v1).
* ``eigengap`` - kept from v1 for trajectory parity but no longer used
  for routing.
* ``outlier_frac`` - LOF outlier fraction (same as v1; the rule built
  on this is the one that worked).
* ``density_skew`` - std / mean of kNN distances (same as v1).
* ``convexity_ratio`` - fraction of points within 1 std of their kmeans
  centroid, averaged over clusters. Low => candidate clusters are not
  convex blobs => the true structure is non-convex.
* ``knn_modularity`` - Newman modularity of the kmeans partition on a
  10-NN graph. Low => kmeans cuts the graph poorly => a graph-based
  algorithm would do better.

Routing rules, first match wins:

1. ``outlier_frac > 0.10`` -> ``gmm`` (preserved from v1).
2. ``convexity_ratio < 0.45`` OR ``knn_modularity < 0.30`` -> ``spectral``
   with ``n_neighbors=10``.
3. ``eff_dim >= 10`` and ``outlier_frac < 0.05`` -> ``birch_algo``.
4. Default -> ``pwcc_diverse`` (else ``kmeans``).

A probe stage runs when the matching rule's confidence is low (e.g., the
convexity ratio is in the [0.40, 0.50] uncertainty band): fit three
candidate algorithms on a 20% subsample, score by silhouette, and
dispatch to the winner.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from . import base as base_algos
from .base import Algorithm, AlgoResult, Step, register


# -----------------------------------------------------------------------------
# Cheap fingerprints (eff_dim, eigengap, outlier_frac, density_skew are
# copied from v1 verbatim because they already do their job).
# -----------------------------------------------------------------------------

def _eff_dim(X: np.ndarray) -> int:
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
        A = 0.5 * (A + A.T)
        L = laplacian(A, normed=True)
        n_eigs = min(max(k + 2, 3), m - 1)
        vals = eigsh(L, k=n_eigs, which="SM", return_eigenvectors=False)
        vals = np.sort(np.real(vals))
    except Exception:
        return 1.0
    if k < 1 or k >= len(vals):
        return 1.0
    lam_k = float(vals[k])
    lam_km1 = float(vals[k - 1])
    if lam_k <= 1e-12:
        return 1.0
    return (lam_k - lam_km1) / lam_k


def _outlier_frac(X: np.ndarray) -> float:
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
    from sklearn.neighbors import NearestNeighbors

    n = X.shape[0]
    n_neighbors = min(10, max(2, n - 1))
    try:
        nn = NearestNeighbors(n_neighbors=n_neighbors)
        nn.fit(X)
        dists, _ = nn.kneighbors(X)
        per_point = dists[:, 1:].mean(axis=1)
    except Exception:
        return 0.0
    mu = float(per_point.mean())
    if mu <= 1e-12:
        return 0.0
    return float(per_point.std() / mu)


# -----------------------------------------------------------------------------
# New non-convex diagnostics
# -----------------------------------------------------------------------------

def _convexity_ratio(X: np.ndarray, k: int, seed: int = 0) -> Tuple[float, np.ndarray]:
    """Bounded ``[0, 1]`` score - high means the kmeans clusters look
    convex around their centroids, low means non-convex.

    The literal "fraction of points within 1.0 std of their cluster
    centroid" reading of the spec collapses to ~0.68-0.84 for *any*
    unimodal distance distribution (Chebyshev-style), so it can't
    distinguish a clean Gaussian blob from concentric circles split by
    kmeans. We instead use the **coefficient of variation** of centroid
    distances per cluster: ``CV = std(d) / mean(d)``.

    * Convex Gaussian blob: ``CV`` is small (chi-distribution is tightly
      peaked at its mode) -> ``CV ~ 0.28`` empirically.
    * Concentric circles cut laterally by kmeans: each half-cluster has
      a wide radial spread (some points near the centroid, others on
      the outer ring) -> ``CV ~ 0.48``.
    * Heavy-tailed cluster + outliers: ``CV ~ 0.74`` (caught upstream
      by the LOF rule, but we still want this to score low here).

    We map ``CV`` to a [0, 1] convexity score via ``clip(1.2 - 2*CV, 0, 1)``
    which puts blobs around ~0.65, circles around ~0.24, and outlier-
    heavy data near 0. The 0.45 routing threshold then cleanly catches
    circles while leaving blobs in the high-confidence convex band.

    Returns
    -------
    ratio : float
    kmeans_labels : np.ndarray
        Labels from the kmeans probe; reused by ``_knn_modularity`` so
        the kmeans fit isn't duplicated.
    """
    from sklearn.cluster import KMeans

    try:
        km = KMeans(n_clusters=k, n_init=4, random_state=seed)
        labels = km.fit_predict(X)
        centers = km.cluster_centers_
    except Exception:
        return 1.0, np.zeros(X.shape[0], dtype=np.int64)

    cvs: List[float] = []
    for j in range(k):
        mask = labels == j
        if not np.any(mask):
            continue
        d = np.linalg.norm(X[mask] - centers[j], axis=1)
        mu = float(d.mean())
        if mu <= 1e-12:
            cvs.append(0.0)
            continue
        cvs.append(float(d.std() / mu))
    if not cvs:
        return 1.0, labels.astype(np.int64)
    cv = float(np.mean(cvs))
    ratio = float(np.clip(1.2 - 2.0 * cv, 0.0, 1.0))
    return ratio, labels.astype(np.int64)


def _knn_modularity(
    X: np.ndarray, labels: np.ndarray, n_neighbors: int = 10
) -> float:
    """Newman (2006) modularity of ``labels`` on a symmetric 10-NN graph.

    For an undirected graph with adjacency ``A``, degree ``d``, edge total
    ``m = sum(A)/2`` and partition ``c``:

        Q = (1/2m) * sum_ij [ A_ij - d_i*d_j/(2m) ] * delta(c_i, c_j)

    High Q (close to 1) means the partition respects graph community
    structure; low Q (close to 0 or negative) means kmeans cut the graph
    arbitrarily - which is the failure mode for non-convex data, exactly
    the case where spectral wins.
    """
    from sklearn.neighbors import kneighbors_graph

    n = X.shape[0]
    nn = min(n_neighbors, max(2, n - 1))
    try:
        A = kneighbors_graph(X, n_neighbors=nn, mode="connectivity", include_self=False)
        A = (A + A.T)  # symmetric edge counts (1 or 2 per undirected edge)
        A.data = np.minimum(A.data, 1.0)  # collapse mutual edges to single edges
    except Exception:
        return 1.0

    A = A.tocsr()
    degrees = np.asarray(A.sum(axis=1)).ravel()
    two_m = float(degrees.sum())
    if two_m <= 1e-12:
        return 1.0

    # Sum A_ij over i, j in same community (twice each edge), minus null model.
    q = 0.0
    coo = A.tocoo()
    rows, cols, vals = coo.row, coo.col, coo.data
    same = labels[rows] == labels[cols]
    q_edges = float(vals[same].sum())

    # Null-model contribution: for each community c, (sum_{i in c} d_i)^2 / (2m).
    null = 0.0
    for c in np.unique(labels):
        d_c = float(degrees[labels == c].sum())
        null += d_c * d_c
    q = (q_edges / two_m) - (null / (two_m * two_m))
    return float(q)


# -----------------------------------------------------------------------------
# Fingerprint bundle, routing, probe
# -----------------------------------------------------------------------------

def _fingerprints(X: np.ndarray, k: int, seed: int = 0) -> Dict[str, float]:
    convexity, km_labels = _convexity_ratio(X, k, seed=seed)
    return {
        "eff_dim": int(_eff_dim(X)),
        "eigengap": float(_eigengap(X, k, seed=seed)),
        "outlier_frac": float(_outlier_frac(X)),
        "density_skew": float(_density_skew(X)),
        "convexity_ratio": float(convexity),
        "knn_modularity": float(_knn_modularity(X, km_labels)),
    }


def _resolve(name: str, fallback: str) -> str:
    if name in base_algos.ALGO_REGISTRY:
        return name
    if fallback in base_algos.ALGO_REGISTRY:
        return fallback
    return "kmeans"


def _route(fp: Dict[str, float]) -> Tuple[str, str, str, List[str]]:
    """Return ``(rule_name, chosen_algo, confidence, alternatives)``.

    ``alternatives`` lists two backup algorithms used by the probe when
    confidence is "low".
    """
    eff_dim = fp["eff_dim"]
    outlier_frac = fp["outlier_frac"]
    convexity = fp["convexity_ratio"]
    modularity = fp["knn_modularity"]

    if outlier_frac > 0.10:
        # Outlier rule from v1; high-confidence (it worked empirically).
        return "outlier_gmm", _resolve("gmm", "kmeans"), "high", ["kmeans", "pwcc_diverse"]

    if convexity < 0.45 or modularity < 0.30:
        # Non-convex detector. Low confidence when the convexity ratio
        # sits in the [0.40, 0.50] uncertainty band.
        conf = "low" if 0.40 <= convexity <= 0.50 else "high"
        return (
            "nonconvex_spectral",
            _resolve("spectral", "kmeans"),
            conf,
            [_resolve("pwcc_diverse", "kmeans"), "kmeans"],
        )

    if eff_dim >= 10 and outlier_frac < 0.05:
        return "highdim_birch", _resolve("birch_algo", "kmeans"), "high", ["kmeans", "pwcc_diverse"]

    return "default", _resolve("pwcc_diverse", "kmeans"), "high", ["kmeans", _resolve("birch_algo", "kmeans")]


def _params_for(algo: str) -> Dict[str, Any]:
    """Per-algorithm hyper-parameters baked in by v2."""
    if algo == "spectral":
        return {"n_neighbors": 10}
    return {}


def _run_algo(algo: str, X: np.ndarray, k: int) -> AlgoResult:
    cls = base_algos.ALGO_REGISTRY.get(algo)
    if cls is None:
        cls = base_algos.ALGO_REGISTRY["kmeans"]
    return cls(**_params_for(algo)).fit_predict(X, k=k)


def _silhouette_probe(
    candidates: List[str], X: np.ndarray, k: int, seed: int = 0
) -> Tuple[str, List[float]]:
    """Fit each candidate on a 20% subsample, score by silhouette.

    Returns the winning algorithm and the silhouette score list (same
    order as ``candidates``). Candidates that fail or produce < 2
    distinct clusters get a silhouette of ``-inf`` so they lose.
    """
    from sklearn.metrics import silhouette_score

    n = X.shape[0]
    rng = np.random.default_rng(seed)
    sub_size = max(50, int(0.2 * n))
    sub_size = min(sub_size, n)
    idx = rng.choice(n, size=sub_size, replace=False)
    Xs = X[idx]

    scores: List[float] = []
    for algo in candidates:
        try:
            res = _run_algo(algo, Xs, k)
            labels = np.asarray(res.labels)
            uniq = np.unique(labels[labels >= 0])
            if len(uniq) < 2:
                scores.append(float("-inf"))
                continue
            scores.append(float(silhouette_score(Xs, labels)))
        except Exception:
            scores.append(float("-inf"))

    best_idx = int(np.argmax(scores)) if scores else 0
    return candidates[best_idx], scores


@register
class Meta_clusterer_v2(Algorithm):
    """Data-fingerprint dispatcher v2 with non-convex detector and probe.

    Parameters
    ----------
    seed : int
        Random seed for sampling and probe subsampling.
    """

    def __init__(self, seed: int = 0, **kwargs: Any) -> None:
        self.name = "meta_clusterer_v2"
        self.seed = int(seed)

    def fit_predict(self, X: np.ndarray, k: int | None = None) -> AlgoResult:
        assert k is not None, "k required"
        X = np.asarray(X)

        # Step 0: fingerprints.
        fp = _fingerprints(X, int(k), seed=self.seed)

        # Step 1: routing rule.
        rule, routed, confidence, alternatives = _route(fp)

        trajectory: List[Step] = [
            Step(
                step_idx=0,
                cost=0.0,
                delta_cost=None,
                accepted=True,
                action={"type": "compute_fingerprints"},
                state={"fingerprints": dict(fp)},
            ),
            Step(
                step_idx=1,
                cost=0.0,
                delta_cost=None,
                accepted=True,
                action={
                    "type": "rule_match",
                    "rule": rule,
                    "to": routed,
                    "confidence": confidence,
                },
                state={},
            ),
        ]

        # Step 2 (optional): probe.
        probe_winner: str | None = None
        final_algo = routed
        if confidence == "low":
            candidates = [routed] + [a for a in alternatives if a != routed]
            # Deduplicate while preserving order.
            seen: set = set()
            candidates = [c for c in candidates if not (c in seen or seen.add(c))]
            winner, sils = _silhouette_probe(candidates, X, int(k), seed=self.seed)
            probe_winner = winner
            final_algo = winner
            trajectory.append(
                Step(
                    step_idx=2,
                    cost=0.0,
                    delta_cost=None,
                    accepted=True,
                    action={
                        "type": "probe",
                        "candidates": list(candidates),
                        "silhouettes": [float(s) for s in sils],
                        "winner": winner,
                    },
                    state={},
                )
            )

        # Dispatch.
        try:
            result = _run_algo(final_algo, X, int(k))
        except Exception:
            final_algo = "kmeans"
            result = _run_algo("kmeans", X, int(k))

        labels = np.asarray(result.labels, dtype=np.int64)
        n_clusters_found = int(len(np.unique(labels[labels >= 0])))

        trajectory.append(
            Step(
                step_idx=len(trajectory),
                cost=0.0,
                delta_cost=None,
                accepted=True,
                action={"type": "dispatched"},
                state={"final_algo": final_algo, "n_clusters_found": n_clusters_found},
            )
        )

        extra: Dict[str, Any] = {
            "fingerprints": dict(fp),
            "routed_algo": routed,
            "probe_winner": probe_winner,
            "final_algo": final_algo,
            "rule": rule,
            "confidence": confidence,
        }
        for key, value in (result.extra or {}).items():
            if key in extra:
                extra[f"{final_algo}.{key}"] = value
            else:
                extra[key] = value

        return AlgoResult(labels=labels, extra=extra, trajectory=trajectory)
