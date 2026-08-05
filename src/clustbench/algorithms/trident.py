"""Trident: three cheap probes plus specialist escalation, no external router.

Winners in runs/paper_demo_r14/results.csv cluster into three families that
win almost every non-router dataset outright: centroid (parallel_kmeans,
clarans_pp on PowerLawStudentT / heavy_tailed_mixture), mixture (gmm, lmm,
isomap_bgmm on iris / wine / extreme_outliers), and graph/spectral
(louvain_knn, aura, aura_v3 on moons / graph_sbm / VariableDensityBridges).
Trident runs one cheap probe per family, gates on silhouette + pairwise
agreement, and escalates only when disagreement plus a heavy-tail or
density-variation signature signals a router blind spot.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

from . import base as base_algos
from .base import Algorithm, AlgoResult, Step, register


# ---------------------------------------------------------------------------
# Cheap data signatures (heavy tails, density variation)
# ---------------------------------------------------------------------------

def _heavy_tail_score(X: np.ndarray) -> float:
    """Robust heavy-tail indicator: mean per-feature MAD / std ratio inverted.

    On a Gaussian, MAD * 1.4826 approx equals std, so 1 - MAD/std is near 0.
    Heavy-tailed columns push std much higher than MAD, so the score climbs
    toward 1. Kurtosis proper is noisier on small n; MAD/std is bounded and
    cheap.
    """
    if X.shape[0] < 4:
        return 0.0
    mad = np.median(np.abs(X - np.median(X, axis=0, keepdims=True)), axis=0)
    std = X.std(axis=0)
    ratio = (1.4826 * mad) / (std + 1e-12)
    ratio = np.clip(ratio, 0.0, 1.0)
    return float(np.mean(1.0 - ratio))


def _density_variation_score(X: np.ndarray, k_nn: int = 10) -> float:
    """Coefficient of variation of local density (kNN radius).

    Regions with dense clusters and sparse bridges (VariableDensityBridges,
    hierarchical_nested) have wildly different kNN radii; the CV of the
    kth-neighbour distance is a good proxy.
    """
    n = X.shape[0]
    if n <= k_nn + 1:
        return 0.0
    knn = NearestNeighbors(n_neighbors=k_nn + 1).fit(X)
    d, _ = knn.kneighbors(X)
    rk = d[:, -1]
    mu = float(rk.mean())
    if mu <= 0.0:
        return 0.0
    return float(rk.std() / mu)


# ---------------------------------------------------------------------------
# Fast probes (small budgets)
# ---------------------------------------------------------------------------

def _probe_kmeans(X: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Centroid probe: MiniBatchKMeans for scale, KMeans for small n."""
    n = X.shape[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if n > 2000:
            km = MiniBatchKMeans(
                n_clusters=k,
                n_init=2,
                max_iter=40,
                batch_size=min(256, n // 4),
                random_state=seed,
            )
        else:
            km = KMeans(n_clusters=k, n_init=2, max_iter=60, random_state=seed)
        return km.fit_predict(X).astype(np.int64)


def _probe_gmm(X: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Mixture probe: diagonal GMM (cheap yet handles anisotropy)."""
    n, d = X.shape
    cov = "diag" if (d >= n // 4 or n < 5 * k) else "full"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm = GaussianMixture(
            n_components=k,
            covariance_type=cov,
            reg_covar=1e-4,
            max_iter=40,
            n_init=1,
            random_state=seed,
        )
        gm.fit(X)
        return gm.predict(X).astype(np.int64)


def _probe_graph(X: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Graph probe: delegate to louvain_knn (already in registry).

    louvain_knn is cheap on n<~2000 (kNN + numpy Louvain) and reconciles
    naturally with target k via its merge/split heuristic.
    """
    cls = base_algos.ALGO_REGISTRY["louvain_knn"]
    res = cls(random_state=seed).fit_predict(X, k=k)
    return res.labels.astype(np.int64)


# ---------------------------------------------------------------------------
# Gating helpers
# ---------------------------------------------------------------------------

def _safe_silhouette(X: np.ndarray, labels: np.ndarray, seed: int) -> float:
    """Silhouette with sample_size cap for scale; nan-safe."""
    uniq = np.unique(labels)
    if uniq.size < 2 or uniq.size >= X.shape[0]:
        return -1.0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            n = X.shape[0]
            sz = min(n, 800)
            return float(
                silhouette_score(X, labels, sample_size=sz, random_state=seed)
            )
    except Exception:
        return -1.0


def _pairwise_ari(partitions: List[np.ndarray]) -> np.ndarray:
    """Symmetric ARI matrix among the given partitions."""
    m = len(partitions)
    M = np.eye(m, dtype=np.float64)
    for i in range(m):
        for j in range(i + 1, m):
            M[i, j] = M[j, i] = float(
                adjusted_rand_score(partitions[i], partitions[j])
            )
    return M


def _propagate_1nn(
    X_sub: np.ndarray, labels_sub: np.ndarray, X_full: np.ndarray
) -> np.ndarray:
    """1-NN label propagation from a probe subsample back to the full X."""
    knn = NearestNeighbors(n_neighbors=1).fit(X_sub)
    _, idx = knn.kneighbors(X_full)
    return labels_sub[idx[:, 0]].astype(np.int64)


# ---------------------------------------------------------------------------
# Main algorithm
# ---------------------------------------------------------------------------

@register
class Trident(Algorithm):
    """Three-probe pipeline with silhouette-gated specialist escalation.

    Cheap probes: kmeans (centroid), diag GMM (mixture), louvain_knn (graph).
    Escalates only if probes disagree and a router-blind failure mode is
    detected (heavy tails -> isomap_bgmm; density variation -> aura_v3).
    """

    def __init__(
        self,
        agreement_threshold: float = 0.85,
        silhouette_confidence: float = 0.25,
        heavy_tail_threshold: float = 0.35,
        density_variation_threshold: float = 0.65,
        probe_subsample: int = 3000,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        self.name = "trident"
        self.agreement_threshold = float(agreement_threshold)
        self.silhouette_confidence = float(silhouette_confidence)
        self.heavy_tail_threshold = float(heavy_tail_threshold)
        self.density_variation_threshold = float(density_variation_threshold)
        self.probe_subsample = int(probe_subsample)
        self.random_state = int(random_state)

    # ------------------------------------------------------------------
    # Subsampling backbone (scaling)
    # ------------------------------------------------------------------

    def _maybe_subsample(
        self, X: np.ndarray, rng: np.random.Generator
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        n = X.shape[0]
        if n <= self.probe_subsample:
            return X, None
        idx = rng.choice(n, size=self.probe_subsample, replace=False)
        idx.sort()
        return X[idx], idx

    # ------------------------------------------------------------------
    # Escalation dispatcher
    # ------------------------------------------------------------------

    def _run_specialist(
        self, name: str, X: np.ndarray, k: int
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        cls = base_algos.ALGO_REGISTRY[name]
        res = cls(random_state=self.random_state).fit_predict(X, k=k)
        return res.labels.astype(np.int64), dict(res.extra or {})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_predict(self, X: np.ndarray, k: Optional[int] = None) -> AlgoResult:
        assert k is not None, "Trident requires k."
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("Trident expects X with shape (n, d).")
        n = X.shape[0]
        rng = np.random.default_rng(self.random_state)
        trajectory: List[Step] = []

        # ---- 1. Probes on (optionally subsampled) X ----
        X_probe, sub_idx = self._maybe_subsample(X, rng)
        probe_names = ["kmeans", "gmm_diag", "louvain_knn"]
        probes: List[np.ndarray] = []
        probe_sils: List[float] = []
        for i, name in enumerate(probe_names):
            try:
                if name == "kmeans":
                    lab = _probe_kmeans(X_probe, k, self.random_state + i)
                elif name == "gmm_diag":
                    lab = _probe_gmm(X_probe, k, self.random_state + i)
                else:
                    lab = _probe_graph(X_probe, k, self.random_state + i)
            except Exception:
                lab = np.zeros(X_probe.shape[0], dtype=np.int64)
            probes.append(lab)
            probe_sils.append(_safe_silhouette(X_probe, lab, self.random_state))

        trajectory.append(
            Step(
                step_idx=0,
                cost=float(-max(probe_sils)) if probe_sils else 0.0,
                action={
                    "type": "run_probes",
                    "probes": list(probe_names),
                    "subsampled": bool(sub_idx is not None),
                    "n_probe": int(X_probe.shape[0]),
                },
                state={
                    "silhouettes": [float(s) for s in probe_sils],
                    "sizes": [
                        int(np.unique(p).size) for p in probes
                    ],
                },
            )
        )

        # ---- 2. Gate: pairwise ARI + best silhouette ----
        ari_mat = _pairwise_ari(probes)
        max_pair_ari = float(np.max(ari_mat - np.eye(len(probes))))
        best_probe_i = int(np.argmax(probe_sils))
        best_sil = float(probe_sils[best_probe_i])

        # Cheap failure-mode signatures
        heavy_tail = _heavy_tail_score(X_probe)
        density_var = _density_variation_score(
            X_probe, k_nn=min(10, X_probe.shape[0] - 1)
        )
        # Convex-vs-graph dissent: kmeans/gmm agree, louvain disagrees.
        # This is the classic non-convex signature; the fix for the
        # router-invisible "silhouette lies on rings" regime.
        km_gm, km_gr, gm_gr = float(ari_mat[0, 1]), float(ari_mat[0, 2]), float(ari_mat[1, 2])
        graph_dissent = km_gm >= 0.6 and km_gr <= 0.4 and gm_gr <= 0.4
        off = ari_mat[np.triu_indices(len(probes), k=1)]
        min_pair_ari = float(off.min()) if off.size else 0.0

        # Decision logic: baked-in selection, no external router.
        if graph_dissent:
            # centroid/mixture concur but louvain dissents -> non-convex.
            chose = "aura_v3"
            path = "escalate_nonconvex"
        elif min_pair_ari >= self.agreement_threshold and best_sil >= self.silhouette_confidence:
            # Every probe concurs AND leader is well-separated -> trust it.
            chose = probe_names[best_probe_i]
            path = "consensus_trust"
        elif heavy_tail >= self.heavy_tail_threshold:
            chose = "isomap_bgmm"
            path = "escalate_heavy_tail"
        elif density_var >= self.density_variation_threshold:
            chose = "aura_v3"
            path = "escalate_density_variation"
        elif max_pair_ari >= self.agreement_threshold and best_sil >= self.silhouette_confidence:
            # Two of three concur strongly; leader is well-separated.
            chose = probe_names[best_probe_i]
            path = "majority_trust"
        elif best_sil >= self.silhouette_confidence:
            chose = probe_names[best_probe_i]
            path = "silhouette_fallback"
        else:
            # No probe looks confident and no clear failure signature:
            # prefer the graph probe when features look non-convex-ish
            # (density_var moderate), else the mixture probe.
            if density_var >= 0.4:
                chose = "louvain_knn"
                path = "graph_fallback"
            else:
                chose = "gmm_diag"
                path = "mixture_fallback"

        trajectory.append(
            Step(
                step_idx=1,
                cost=float(-best_sil),
                action={
                    "type": "gate",
                    "chose": chose,
                    "path": path,
                    "max_pair_ari": max_pair_ari,
                    "best_silhouette": best_sil,
                    "heavy_tail_score": heavy_tail,
                    "density_variation_score": density_var,
                },
                state={
                    "ari_matrix": ari_mat.tolist(),
                    "best_probe": probe_names[best_probe_i],
                },
            )
        )

        # ---- 3. Materialise the chosen partition ----
        specialist_extra: Dict[str, Any] = {}
        if chose in probe_names:
            probe_i = probe_names.index(chose)
            labels_probe = probes[probe_i]
            final_cost = float(-probe_sils[probe_i])
            source = f"probe:{chose}"
        else:
            # Escalate. Run the specialist on the probe subsample too,
            # then propagate. Specialists are heavier so this keeps cost
            # bounded even on large n.
            labels_probe, specialist_extra = self._run_specialist(
                chose, X_probe, int(k)
            )
            final_cost = float(-_safe_silhouette(X_probe, labels_probe, self.random_state))
            source = f"specialist:{chose}"

        if sub_idx is not None:
            labels_full = _propagate_1nn(X_probe, labels_probe, X)
        else:
            labels_full = labels_probe

        trajectory.append(
            Step(
                step_idx=2,
                cost=final_cost,
                delta_cost=final_cost - float(-best_sil),
                action={
                    "type": "materialise",
                    "source": source,
                    "propagated": bool(sub_idx is not None),
                },
                state={
                    "n_final_clusters": int(np.unique(labels_full).size),
                    "specialist_keys": sorted(specialist_extra.keys()),
                },
            )
        )

        extra: Dict[str, Any] = {
            "chose": chose,
            "path": path,
            "max_pair_ari": max_pair_ari,
            "best_silhouette": best_sil,
            "probe_silhouettes": [float(s) for s in probe_sils],
            "heavy_tail_score": float(heavy_tail),
            "density_variation_score": float(density_var),
            "subsampled": bool(sub_idx is not None),
            "n_probe": int(X_probe.shape[0]),
            "probes": list(probe_names),
            "ari_matrix": ari_mat.tolist(),
            "specialist_extra": specialist_extra,
        }
        return AlgoResult(labels=labels_full, extra=extra, trajectory=trajectory)
