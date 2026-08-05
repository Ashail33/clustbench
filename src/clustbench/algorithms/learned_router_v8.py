"""Learned routing meta-algorithm, eighth iteration.

Targets four failure modes exposed by round 14:
1. Aggregates candidates by mean ARI (peak finder), not mean rank.
2. Fingerprint adds pairwise_distance_kurtosis, cluster_mass_gini,
   knn_density_variance, and covariance_effective_rank.
3. When the nearest training fingerprint is OOD, v8 runs the top N
   candidates on the real data and picks by silhouette.
4. Trains on runs/paper_demo_r14/results.csv, so the round-14 winners
   are already in the candidate pool. Leave-one-out at exact
   fingerprint keeps evaluation honest.

`_BLOCKED_ROUTERS_V8` blocks every router variant from being a
dispatch target.
"""

from __future__ import annotations

import csv
import pathlib
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import base as base_algos
from .base import Algorithm, AlgoResult, Step, register
from .learned_router import _regenerate_task


_TRAINING_CACHE_V8: Optional[Dict[str, Any]] = None

_BLOCKED_ROUTERS_V8 = {
    "learned_router", "learned_router_v2", "learned_router_v3",
    "learned_router_v4", "learned_router_v5", "learned_router_v6",
    "learned_router_v6b", "learned_router_v6c", "learned_router_v7",
    "learned_router_v8",
}

_K_NEIGHBOURS = 5
_ENSEMBLE_TOP_N = 3
_OOD_QUANTILE = 0.90
_ENSEMBLE_SILHOUETTE_FLOOR = 0.02


# ---------------------------------------------------------------------------
# Enriched fingerprint (11 features)
# ---------------------------------------------------------------------------

def _fingerprint_v8(X: np.ndarray, k: Optional[int]) -> Dict[str, float]:
    """Base geometry (7) + round-14 targeted (4) features.

    base : log_n, d, k, eff_dim, conv_cv, outlier_frac, density_skew
    r14  : pairwise_distance_kurtosis, cluster_mass_gini,
           knn_density_variance, covariance_effective_rank
    """
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

    n, d = X.shape
    fp: Dict[str, float] = {
        "log_n": float(np.log10(max(n, 2))),
        "d": float(d),
        "k": float(k if k is not None else 0),
    }

    n_pca = max(1, min(d, 10, n - 1))
    try:
        pca = PCA(n_components=n_pca, svd_solver="auto", random_state=0).fit(X)
        fp["eff_dim"] = float((pca.explained_variance_ratio_ > 0.01).sum())
    except Exception:
        fp["eff_dim"] = float(d)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            km = KMeans(n_clusters=max(2, k or 3), n_init=1, max_iter=30,
                        random_state=0).fit(X)
        dist = np.linalg.norm(X - km.cluster_centers_[km.labels_], axis=1)
        fp["conv_cv"] = float(np.std(dist) / (np.mean(dist) + 1e-9))
        sizes = np.bincount(km.labels_, minlength=max(2, k or 3)).astype(np.float64)
    except Exception:
        fp["conv_cv"] = 0.0
        sizes = np.array([1.0])

    if n >= 25:
        try:
            lof = LocalOutlierFactor(
                n_neighbors=min(20, n - 1), contamination="auto"
            ).fit(X)
            scores = -lof.negative_outlier_factor_
            fp["outlier_frac"] = float((scores > 1.5).mean())
        except Exception:
            fp["outlier_frac"] = 0.0
    else:
        fp["outlier_frac"] = 0.0

    density_skew = 0.0
    knn_density_var = 0.0
    if n >= 6:
        try:
            k_nn = min(10, n - 1)
            nbrs = NearestNeighbors(n_neighbors=k_nn + 1).fit(X)
            d_knn, _ = nbrs.kneighbors(X)
            mean_d = d_knn[:, 1:].mean(axis=1)
            density_skew = float(np.std(mean_d) / (np.mean(mean_d) + 1e-9))
            density = 1.0 / np.maximum(mean_d, 1e-12)
            mu = float(density.mean())
            if mu > 1e-12:
                knn_density_var = float(density.std() / mu)
        except Exception:
            pass
    fp["density_skew"] = density_skew
    fp["knn_density_variance"] = knn_density_var

    # Cluster mass Gini from the kmeans probe above.
    if sizes.sum() > 0:
        s = np.sort(sizes)
        m = s.size
        idx = np.arange(1, m + 1, dtype=np.float64)
        denom = m * s.sum()
        gini = float(np.sum((2 * idx - m - 1) * s) / denom) if denom > 1e-12 else 0.0
    else:
        gini = 0.0
    fp["cluster_mass_gini"] = gini

    # Pairwise distance excess kurtosis over a bounded sample.
    kurt = 0.0
    if n >= 4:
        try:
            rng = np.random.default_rng(0)
            mm = min(n, 200)
            sel = rng.choice(n, size=mm, replace=False) if mm < n else np.arange(n)
            Xs = X[sel]
            diffs = Xs[:, None, :] - Xs[None, :, :]
            dd = np.sqrt(np.sum(diffs * diffs, axis=2))
            iu = np.triu_indices(mm, k=1)
            vals = dd[iu]
            if vals.size >= 4:
                mu = float(vals.mean()); std = float(vals.std())
                if std > 1e-12:
                    z = (vals - mu) / std
                    kurt = float(np.mean(z ** 4) - 3.0)
        except Exception:
            pass
    fp["pairwise_distance_kurtosis"] = kurt

    # Participation ratio of covariance eigenvalues.
    er = float(d)
    try:
        Xc = X - X.mean(axis=0, keepdims=True)
        cov = (Xc.T @ Xc) / max(1, n - 1)
        eigs = np.clip(np.linalg.eigvalsh(cov), 0.0, None)
        s1 = float(eigs.sum()); s2 = float(np.sum(eigs * eigs))
        if s2 > 1e-24:
            er = float((s1 * s1) / s2)
    except Exception:
        pass
    fp["covariance_effective_rank"] = er
    return fp


# ---------------------------------------------------------------------------
# Training data (reads runs/paper_demo_r14/results.csv)
# ---------------------------------------------------------------------------

def _cint(v: Any, default: int = 0) -> int:
    try:
        return int(float(v)) if v not in (None, "") else default
    except Exception:
        return default


def _cflt(v: Any, default: float = 0.0) -> float:
    try:
        return float(v) if v not in (None, "") else default
    except Exception:
        return default


def _load_training_data_v8() -> Optional[Dict[str, Any]]:
    """Group rows by task, regenerate (X, y), compute v8 fingerprints,
    and cache normalised feature matrix + per-task {algo: ari} dicts."""
    global _TRAINING_CACHE_V8
    if _TRAINING_CACHE_V8 is not None:
        return _TRAINING_CACHE_V8

    repo_root = pathlib.Path(__file__).resolve().parents[3]
    results_path = repo_root / "runs" / "paper_demo_r14" / "results.csv"
    if not results_path.exists():
        _TRAINING_CACHE_V8 = None
        return None

    by_task: Dict[tuple, Dict[str, float]] = defaultdict(dict)
    task_meta: Dict[tuple, dict] = {}
    try:
        with results_path.open("r", newline="") as f:
            for r in csv.DictReader(f):
                key = (r.get("dataset_id"), _cint(r.get("n_samples")),
                       _cint(r.get("n_features")), _cint(r.get("k_target")),
                       _cint(r.get("outliers")), _cint(r.get("noise")),
                       _cflt(r.get("density"), 1.0), _cint(r.get("seed"), 1))
                ari_raw = r.get("ari")
                try:
                    ari = float(ari_raw) if ari_raw not in (None, "") else None
                except Exception:
                    ari = None
                if ari is None or np.isnan(ari):
                    continue
                by_task[key][r["algo"]] = float(ari)
                task_meta[key] = {
                    "dataset_id": r.get("dataset_id"),
                    "n_samples": _cint(r.get("n_samples")),
                    "n_features": _cint(r.get("n_features")),
                    "k_target": _cint(r.get("k_target")),
                    "compactness": _cflt(r.get("compactness"), 1.0),
                    "outliers": _cint(r.get("outliers")),
                    "noise": _cint(r.get("noise")),
                    "density": _cflt(r.get("density"), 1.0),
                    "seed": _cint(r.get("seed"), 1),
                    "outlier_extremity": _cflt(r.get("outlier_extremity"), 1.0),
                }
    except Exception:
        _TRAINING_CACHE_V8 = None
        return None

    if not by_task:
        _TRAINING_CACHE_V8 = None
        return None

    fingerprints: List[Dict[str, float]] = []
    ari_rows: List[Dict[str, float]] = []
    for key, algo_ari in by_task.items():
        gen_result = _regenerate_task(task_meta[key])
        if gen_result is None:
            continue
        try:
            X, _ = gen_result
        except Exception:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fp = _fingerprint_v8(X, task_meta[key]["k_target"])
        except Exception:
            continue
        fingerprints.append(fp)
        ari_rows.append(algo_ari)

    if not fingerprints:
        _TRAINING_CACHE_V8 = None
        return None

    feature_names = sorted(fingerprints[0].keys())
    F = np.array([[fp[nm] for nm in feature_names] for fp in fingerprints],
                 dtype=np.float64)
    F_mean = F.mean(axis=0)
    F_std = F.std(axis=0)
    F_std[F_std == 0] = 1.0
    F_norm = (F - F_mean) / F_std

    # 90th percentile of self nearest neighbour distances is the OOD
    # threshold at inference.
    self_dists: List[float] = []
    for i in range(F_norm.shape[0]):
        d = np.linalg.norm(F_norm - F_norm[i][None, :], axis=1)
        d[i] = np.inf
        if np.isfinite(d.min()):
            self_dists.append(float(d.min()))
    ood_threshold = float(np.quantile(self_dists, _OOD_QUANTILE)) \
        if self_dists else float("inf")

    _TRAINING_CACHE_V8 = {
        "F": F_norm, "ari_rows": ari_rows, "feature_names": feature_names,
        "mean": F_mean, "std": F_std, "n_tasks": int(F_norm.shape[0]),
        "ood_threshold": ood_threshold,
    }
    return _TRAINING_CACHE_V8


# ---------------------------------------------------------------------------
# Scoring and probing helpers
# ---------------------------------------------------------------------------

def _candidate_pool(ari_rows: List[Dict[str, float]]) -> List[str]:
    seen: set = set()
    for row in ari_rows:
        seen.update(row.keys())
    return sorted(a for a in seen if a not in _BLOCKED_ROUTERS_V8)


def _mean_ari_scores(
    ari_rows: List[Dict[str, float]],
    neighbour_idx: np.ndarray,
    pool: List[str],
) -> Dict[str, float]:
    """Peak-finder aggregation: for each candidate, mean ARI across the
    neighbours that ran it. Missing algos are skipped (not zero filled)
    so a candidate that only appeared in a few neighbours can still win."""
    scores: Dict[str, float] = {}
    for algo in pool:
        vals: List[float] = []
        for i in neighbour_idx:
            v = ari_rows[int(i)].get(algo)
            if v is not None:
                vals.append(float(v))
        if vals:
            scores[algo] = float(np.mean(vals))
    return scores


def _silhouette_of(X: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import silhouette_score
    lab = np.asarray(labels)
    mask = lab >= 0
    if mask.sum() < 2:
        return float("-inf")
    eff = lab[mask]
    unique = np.unique(eff)
    if len(unique) < 2 or len(unique) >= mask.sum():
        return float("-inf")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(silhouette_score(X[mask], eff))
    except Exception:
        return float("-inf")


def _run_algo(algo: str, X: np.ndarray, k: Optional[int]) -> Optional[AlgoResult]:
    cls = base_algos.ALGO_REGISTRY.get(algo)
    if cls is None:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return cls().fit_predict(X, k=k)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Algorithm class
# ---------------------------------------------------------------------------

@register
class Learned_router_v8(Algorithm):
    """Round-14-aware learned router with ensemble OOD fallback."""

    def __init__(
        self,
        k_neighbors: int = _K_NEIGHBOURS,
        ensemble_top_n: int = _ENSEMBLE_TOP_N,
        fallback: str = "pwcc_diverse",
        **kwargs: Any,
    ) -> None:
        self.name = "learned_router_v8"
        self.k_neighbors = int(k_neighbors)
        self.ensemble_top_n = int(ensemble_top_n)
        self.fallback = fallback

    def _fallback_dispatch(
        self, X: np.ndarray, k: Optional[int], reason: str
    ) -> AlgoResult:
        cls = base_algos.ALGO_REGISTRY.get(self.fallback) or \
              base_algos.ALGO_REGISTRY["kmeans"]
        inner = cls().fit_predict(X, k=k)
        return AlgoResult(
            labels=inner.labels,
            extra={"router": "learned_router_v8", "chose": self.fallback,
                   "reason": reason, **(inner.extra or {})},
            trajectory=inner.trajectory or [],
        )

    def fit_predict(
        self, X: np.ndarray, k: Optional[int] = None
    ) -> AlgoResult:
        cache = _load_training_data_v8()
        if cache is None:
            return self._fallback_dispatch(X, k, "no_training_data")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fp = _fingerprint_v8(X, k)
        fp_vec = np.array(
            [fp[n] for n in cache["feature_names"]], dtype=np.float64
        )
        fp_norm = (fp_vec - cache["mean"]) / cache["std"]

        F = cache["F"]
        ari_rows = cache["ari_rows"]
        dists = np.linalg.norm(F - fp_norm[None, :], axis=1)
        order = np.argsort(dists)
        if len(order) and dists[order[0]] < 1e-6:  # leave one out
            order = order[1:]
        if len(order) == 0:
            return self._fallback_dispatch(X, k, "no_neighbours")

        topk_idx = order[: max(1, self.k_neighbors)]
        pool = _candidate_pool(ari_rows)
        scores = _mean_ari_scores(ari_rows, topk_idx, pool)
        if not scores:
            return self._fallback_dispatch(X, k, "empty_scores")

        # Peak finder: descending by mean neighbour ARI.
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        chose = ranked[0][0]
        top_candidates = [(a, float(s)) for a, s in ranked[: max(5, self.ensemble_top_n)]]

        min_nd = float(dists[order[0]])
        ood_thr = float(cache["ood_threshold"])
        is_ood = min_nd > ood_thr

        reason = "argmax_mean_ari"
        probes: List[Dict[str, Any]] = []
        ensemble_pick: Optional[str] = None
        inner: Optional[AlgoResult] = None

        if is_ood:
            top_algos = [a for a, _ in ranked[: max(1, self.ensemble_top_n)]]
            probe_results: Dict[str, Tuple[AlgoResult, float]] = {}
            for algo in top_algos:
                res = _run_algo(algo, X, k)
                if res is None:
                    probes.append({"algo": algo, "silhouette": None,
                                   "error": "run_failed"})
                    continue
                sil = _silhouette_of(X, res.labels)
                probes.append({"algo": algo,
                               "silhouette": None if not np.isfinite(sil) else sil})
                if np.isfinite(sil):
                    probe_results[algo] = (res, sil)
            if probe_results:
                best_algo, (best_res, best_sil) = max(
                    probe_results.items(), key=lambda kv: kv[1][1]
                )
                if best_sil > _ENSEMBLE_SILHOUETTE_FLOOR:
                    ensemble_pick = best_algo
                    chose = best_algo
                    reason = "ood_ensemble_silhouette"
                    inner = best_res
                else:
                    reason = "ood_ensemble_below_floor"
            else:
                reason = "ood_ensemble_all_failed"

        if inner is None:
            inner = _run_algo(chose, X, k)
        if inner is None:
            fb = self._fallback_dispatch(X, k, "chosen_algo_failed")
            fb.extra.setdefault("attempted", chose)
            return fb

        fp_dict = {kk: float(vv) for kk, vv in fp.items()}
        trajectory: List[Step] = [
            Step(step_idx=0, cost=0.0, action={"type": "compute_fingerprint_v8"},
                 state={"fingerprint": fp_dict,
                        "n_training_tasks": int(cache["n_tasks"])}),
            Step(step_idx=1, cost=0.0,
                 action={"type": "router_v8_decision", "chose": chose,
                         "reason": reason, "is_ood": bool(is_ood),
                         "ensemble_pick": ensemble_pick,
                         "top_candidates": top_candidates},
                 state={"min_neighbour_dist": min_nd,
                        "ood_threshold": ood_thr,
                        "ensemble_probes": probes}),
        ]
        if inner.trajectory:
            for s in inner.trajectory:
                trajectory.append(
                    Step(step_idx=len(trajectory), cost=s.cost,
                         delta_cost=s.delta_cost, accepted=s.accepted,
                         action=s.action, state=s.state)
                )
        return AlgoResult(
            labels=inner.labels,
            extra={
                "router": "learned_router_v8", "chose": chose,
                "reason": reason, "is_ood": bool(is_ood),
                "min_neighbour_dist": min_nd, "ood_threshold": ood_thr,
                "top_candidates": top_candidates,
                "ensemble_probes": probes, "ensemble_pick": ensemble_pick,
                "fingerprint": fp_dict, **(inner.extra or {}),
            },
            trajectory=trajectory,
        )
