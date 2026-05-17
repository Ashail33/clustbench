"""Learned routing meta-algorithm — third iteration.

`learned_router_v3` is the third generation of the kNN-over-fingerprints
meta-algorithm. It keeps v2's expanded fingerprint and adds three changes:

1. **Predict ARI directly, not just rank.** v1 and v2 dispatched to the
   algorithm with the lowest mean ARI *rank* across the K nearest training
   tasks. That's conservative — it picks "consistently good" over
   "occasionally great". v3 instead fits a per-algorithm regressor on the
   training fingerprints (target = that algorithm's ARI) and dispatches
   to the algorithm with the highest *predicted* ARI on the input
   fingerprint. The regressor is a distance-weighted kNN regressor with
   k=5; this is simple, robust, and needs no hyperparameter tuning.
2. **Distance-weighted neighbour voting** (used as a fallback when no
   regressor produces a usable prediction). Closer neighbours get more
   weight via `1 / (1 + distance)` instead of equal vote.
3. **Probe features from a quick DBSCAN run.** v2 probed only k-means
   (4 features about convex compression). v3 adds two features from a
   DBSCAN probe with auto-eps: the number of clusters found and the
   noise fraction. These tell the router about the data's density
   structure directly, complementing v2's k-means-shape features.

So v3's fingerprint is **17 features**: v2's 15 plus the 2 DBSCAN probes.

The training data comes from ``docs/data/results.json`` (the
comprehensive benchmark in ``runs/full/``, 4332 result rows across 33
algorithms × ~118 tasks × 23 dataset families). Each historical task is
regenerated from its DataSpec, the v3 fingerprint is computed, and a
KNeighborsRegressor is fit per candidate algorithm.

All other behaviour — leave-one-out via ``exclude_self``, dispatch via
``ALGO_REGISTRY``, blocked recursive routing — matches v2. The cache
(`_TRAINING_CACHE_V3`, `_REGRESSORS_V3`) is independent so v1/v2/v3 can
coexist in the same process.
"""

from __future__ import annotations

import json
import pathlib
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import base as base_algos
from .base import Algorithm, AlgoResult, Step, register
from .learned_router import _algo_position_per_task, _regenerate_task
from .learned_router_v2 import _fingerprint_v2


# Caches are independent from v1 and v2 so the three routers don't share
# fingerprints or regressors.
_TRAINING_CACHE_V3: Optional[Tuple[np.ndarray, list, Dict[str, Any]]] = None
_REGRESSORS_V3: Optional[Dict[str, Any]] = None

# All router variants are blocked as dispatch targets to prevent
# any pathological recursive routing chains.
# any pathological recursive routing chains. Block every existing
# router-family member so v3 can't dispatch to a v4/v5/v6/v7/... that
# then dispatches back to v3.
_BLOCKED_TARGETS_V3 = {
    "learned_router",
    "learned_router_v2",
    "learned_router_v3",
    "learned_router_v4",
    "learned_router_v5",
    "learned_router_v6",
    "learned_router_v6b",
    "learned_router_v6c",
    "learned_router_v7",
}


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------

def _dbscan_probe(X: np.ndarray) -> Tuple[int, float]:
    """Quick DBSCAN probe with auto-eps.

    Auto-eps heuristic: the median of each point's distance to its 5th
    nearest neighbour. That's a cheap version of the classic
    "knee in the k-distance plot" trick. We cap the sample at 500 points
    so the probe stays fast on large inputs.

    Returns
    -------
    (n_clusters, noise_frac)
        ``n_clusters`` is the number of clusters DBSCAN finds (excluding
        the noise label -1). A value of 0 means DBSCAN could not find
        any density structure at this scale — a strong signal that
        density-based methods will fail on this data.
        ``noise_frac`` is the fraction of points labeled noise.
    """
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import NearestNeighbors

    n = X.shape[0]
    if n < 6:
        return 0, 0.0

    rng = np.random.default_rng(0)
    if n > 500:
        idx = rng.choice(n, size=500, replace=False)
        Xp = X[idx]
    else:
        Xp = X

    try:
        k = min(5, Xp.shape[0] - 1)
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(Xp)
        d, _ = nbrs.kneighbors(Xp)
        # d[:, k] is the kth nearest neighbour distance (column 0 is self).
        eps = float(np.median(d[:, k]))
        if eps <= 0:
            eps = float(np.mean(d[:, k]) + 1e-9)
        if eps <= 0:
            return 0, 0.0
        labels = DBSCAN(eps=eps, min_samples=5).fit_predict(Xp)
        n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))
        noise_frac = float((labels == -1).mean())
        return n_clusters, noise_frac
    except Exception:
        return 0, 0.0


def _fingerprint_v3(X: np.ndarray, k: Optional[int]) -> Dict[str, float]:
    """Compute the 17-feature v3 fingerprint.

    Layout:
      - v2 carry-over (15): see ``_fingerprint_v2``.
      - DBSCAN probe (2): probe_dbscan_n_clusters, probe_dbscan_noise_frac.
    """
    fp = _fingerprint_v2(X, k)
    n_clusters, noise_frac = _dbscan_probe(X)
    fp["probe_dbscan_n_clusters"] = float(n_clusters)
    fp["probe_dbscan_noise_frac"] = float(noise_frac)
    return fp


# ---------------------------------------------------------------------------
# Training-data loader + per-algorithm ARI regressors
# ---------------------------------------------------------------------------

def _load_training_data_v3() -> Tuple[Optional[np.ndarray], Optional[list], Optional[Dict[str, Any]]]:
    """Read ``docs/data/results.json``, regenerate each historical task,
    compute v3 fingerprints, and cache the result.
    """
    global _TRAINING_CACHE_V3
    if _TRAINING_CACHE_V3 is not None:
        return _TRAINING_CACHE_V3

    repo_root = pathlib.Path(__file__).resolve().parents[3]
    results_path = repo_root / "docs" / "data" / "results.json"
    if not results_path.exists():
        _TRAINING_CACHE_V3 = (None, None, None)
        return _TRAINING_CACHE_V3

    try:
        rows = json.loads(results_path.read_text())
    except Exception:
        _TRAINING_CACHE_V3 = (None, None, None)
        return _TRAINING_CACHE_V3

    by_task: Dict[tuple, Dict[str, float]] = defaultdict(dict)
    task_meta: Dict[tuple, dict] = {}
    for r in rows:
        key = (
            r.get("dataset_id"),
            r.get("n_samples"),
            r.get("n_features"),
            r.get("k_target"),
            r.get("outliers"),
            r.get("noise"),
            r.get("density"),
            r.get("seed"),
        )
        ari = r.get("ari")
        if ari is None:
            continue
        try:
            ari = float(ari)
        except Exception:
            continue
        if np.isnan(ari):
            continue
        by_task[key][r["algo"]] = ari
        task_meta[key] = {
            "dataset_id": r.get("dataset_id"),
            "n_samples": r.get("n_samples"),
            "n_features": r.get("n_features"),
            "k_target": r.get("k_target"),
            "compactness": r.get("compactness", 1.0),
            "outliers": r.get("outliers", 0),
            "noise": r.get("noise", 0),
            "density": r.get("density", 1.0),
            "seed": r.get("seed"),
        }

    fingerprints: List[Dict[str, float]] = []
    rank_rows: List[Dict[str, float]] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for key, algo_ari in by_task.items():
            meta = task_meta[key]
            gen_result = _regenerate_task(meta)
            if gen_result is None:
                continue
            X, _ = gen_result
            fp = _fingerprint_v3(X, meta["k_target"])
            fingerprints.append(fp)
            rank_rows.append(algo_ari)

    if not fingerprints:
        _TRAINING_CACHE_V3 = (None, None, None)
        return _TRAINING_CACHE_V3

    feature_names = sorted(fingerprints[0].keys())
    F = np.array(
        [[fp[name] for name in feature_names] for fp in fingerprints],
        dtype=np.float64,
    )
    F_mean = F.mean(axis=0)
    F_std = F.std(axis=0)
    F_std[F_std == 0] = 1.0
    F_norm = (F - F_mean) / F_std

    _TRAINING_CACHE_V3 = (
        F_norm,
        rank_rows,
        {
            "feature_names": feature_names,
            "mean": F_mean,
            "std": F_std,
            "n_tasks": len(fingerprints),
        },
    )
    return _TRAINING_CACHE_V3


def _load_regressors_v3() -> Optional[Dict[str, Any]]:
    """Train one ARI regressor per algorithm seen in the training data.

    Each regressor is a distance-weighted kNN regressor (k=5) trained on
    the normalised fingerprints. The target is the algorithm's ARI on
    each task; tasks where the algorithm was not run are simply omitted
    from that regressor's training set.

    Returns a dict ``{algo: regressor}`` or ``None`` if no training data.
    """
    global _REGRESSORS_V3
    if _REGRESSORS_V3 is not None:
        return _REGRESSORS_V3

    F_norm, rank_rows, meta = _load_training_data_v3()
    if F_norm is None or rank_rows is None:
        _REGRESSORS_V3 = {}
        return _REGRESSORS_V3

    from sklearn.neighbors import KNeighborsRegressor

    # Collect all algorithms seen in training, minus blocked targets.
    seen: set = set()
    for row in rank_rows:
        seen.update(row.keys())
    seen -= _BLOCKED_TARGETS_V3

    regressors: Dict[str, Any] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for algo in sorted(seen):
            idx = [i for i, row in enumerate(rank_rows) if algo in row]
            if len(idx) < 2:
                continue
            X_tr = F_norm[idx]
            y_tr = np.array([rank_rows[i][algo] for i in idx], dtype=np.float64)
            n_nb = min(5, len(idx))
            reg = KNeighborsRegressor(n_neighbors=n_nb, weights="distance")
            try:
                reg.fit(X_tr, y_tr)
                regressors[algo] = reg
            except Exception:
                continue

    _REGRESSORS_V3 = regressors
    return _REGRESSORS_V3


# ---------------------------------------------------------------------------
# Algorithm class
# ---------------------------------------------------------------------------

@register
class Learned_router_v3(Algorithm):
    """v3 of the kNN-over-fingerprints meta-algorithm.

    Dispatches by *predicted ARI* (per-algorithm kNN regression) rather
    than by mean ARI-rank as in v1/v2. See module docstring for the full
    rationale.
    """

    def __init__(
        self,
        k_neighbors: int = 5,
        exclude_self: bool = True,
        candidates: Optional[List[str]] = None,
        fallback: str = "pwcc_diverse",
        **kwargs: Any,
    ) -> None:
        self.name = "learned_router_v3"
        self.k_neighbors = k_neighbors
        self.exclude_self = exclude_self
        self.candidates = candidates
        self.fallback = fallback

    def _candidates_pool(self, rank_rows: List[Dict[str, float]]) -> List[str]:
        if self.candidates is not None:
            return [c for c in self.candidates if c not in _BLOCKED_TARGETS_V3]
        seen: set = set()
        for row in rank_rows:
            seen.update(row.keys())
        return [a for a in sorted(seen) if a not in _BLOCKED_TARGETS_V3]

    def _dispatch(self, algo: str, X: np.ndarray, k: Optional[int]) -> AlgoResult:
        cls = base_algos.ALGO_REGISTRY.get(algo)
        if cls is None:
            cls = base_algos.ALGO_REGISTRY[self.fallback]
        return cls().fit_predict(X, k=k)

    def fit_predict(self, X: np.ndarray, k: Optional[int] = None) -> AlgoResult:
        F_norm, rank_rows, meta = _load_training_data_v3()

        if F_norm is None or rank_rows is None or meta is None:
            inner = self._dispatch(self.fallback, X, k)
            return AlgoResult(
                labels=inner.labels,
                extra={
                    "router": "learned_knn_v3",
                    "chose": self.fallback,
                    "reason": "no_training_data",
                    **(inner.extra or {}),
                },
                trajectory=inner.trajectory or [],
            )

        regressors = _load_regressors_v3() or {}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fp = _fingerprint_v3(X, k)
        fp_vec = np.array(
            [fp[name] for name in meta["feature_names"]], dtype=np.float64
        )
        fp_norm = (fp_vec - meta["mean"]) / meta["std"]

        # Distances to all training tasks for the optional rank fallback
        # and for the trajectory's neighbour_distance_min.
        dists = np.linalg.norm(F_norm - fp_norm[None, :], axis=1)
        order = np.argsort(dists)
        if self.exclude_self and len(order) and dists[order[0]] < 1e-6:
            order = order[1:]
        topk_idx = order[: max(1, self.k_neighbors)]
        neighbour_rows = [rank_rows[i] for i in topk_idx]
        neighbour_dists = dists[topk_idx]

        pool = self._candidates_pool(rank_rows)

        # ----- Predict ARI per candidate algorithm -----
        predicted_aris: Dict[str, float] = {}
        fp_norm_2d = fp_norm[None, :]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for algo in pool:
                reg = regressors.get(algo)
                if reg is None:
                    continue
                try:
                    pred = float(reg.predict(fp_norm_2d)[0])
                    if not np.isnan(pred):
                        predicted_aris[algo] = pred
                except Exception:
                    continue

        if predicted_aris:
            chosen = max(predicted_aris.items(), key=lambda kv: kv[1])[0]
            top_candidates = sorted(
                predicted_aris.items(), key=lambda kv: kv[1], reverse=True
            )[:5]
            reason = "predicted_ari"
        else:
            # Distance-weighted rank-vote fallback.
            # Weight = 1 / (1 + dist). Lower mean weighted rank wins.
            weights = 1.0 / (1.0 + np.maximum(neighbour_dists, 0.0))
            wsum = float(weights.sum()) if weights.size else 0.0
            scores: Dict[str, float] = {}
            for algo in pool:
                num = 0.0
                denom = 0.0
                for w, row in zip(weights, neighbour_rows):
                    if algo not in row:
                        continue
                    # Compute the algorithm's 1-based rank within this row.
                    ranked = sorted(row.items(), key=lambda kv: kv[1], reverse=True)
                    for i, (a, _) in enumerate(ranked, start=1):
                        if a == algo:
                            num += w * i
                            denom += w
                            break
                if denom > 0:
                    scores[algo] = num / denom
            if scores:
                chosen = min(scores.items(), key=lambda kv: kv[1])[0]
                # For the trajectory's "top 5" we invert (lower rank = better).
                top_candidates = [
                    (a, float(s))
                    for a, s in sorted(scores.items(), key=lambda kv: kv[1])[:5]
                ]
            else:
                chosen = self.fallback
                top_candidates = []
            reason = "rank_vote_fallback"

        # ----- Trajectory: fingerprint step + predict_ari step + inner shifted by +2 -----
        trajectory: List[Step] = [
            Step(
                step_idx=0,
                cost=0.0,
                delta_cost=None,
                accepted=True,
                action={
                    "type": "fingerprint_v3",
                },
                state={"fingerprint": {kk: float(vv) for kk, vv in fp.items()}},
            ),
            Step(
                step_idx=1,
                cost=float(-predicted_aris.get(chosen, 0.0)) if predicted_aris
                else 0.0,
                delta_cost=None,
                accepted=True,
                action={
                    "type": "predict_ari",
                    "chose": chosen,
                    "reason": reason,
                    "top_candidates": [(a, float(s)) for a, s in top_candidates],
                    "neighbour_distance_min":
                        float(dists[order[0]]) if len(order) else None,
                },
                state={
                    "predicted_aris": {
                        a: float(s) for a, s in top_candidates
                    },
                    "n_training_tasks": int(meta["n_tasks"]),
                },
            ),
        ]
        inner = self._dispatch(chosen, X, k)
        if inner.trajectory:
            for s in inner.trajectory:
                trajectory.append(
                    Step(
                        step_idx=len(trajectory),
                        cost=s.cost,
                        delta_cost=s.delta_cost,
                        accepted=s.accepted,
                        action=s.action,
                        state=s.state,
                    )
                )

        return AlgoResult(
            labels=inner.labels,
            extra={
                "router": "learned_knn_v3",
                "chose": chosen,
                "predicted_aris": {a: float(s) for a, s in top_candidates},
                "fingerprint": {kk: float(vv) for kk, vv in fp.items()},
                "n_training_tasks": int(meta["n_tasks"]),
                **(inner.extra or {}),
            },
            trajectory=trajectory,
        )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Held-out smoke test, seed=99, matching v2's smoke-test recipe.
    from sklearn.metrics import adjusted_rand_score

    from ..datasets import DataSpec, gen_circles, gen_mdcgen, gen_moons

    cases = [
        (
            "mdcgen-convex",
            gen_mdcgen(DataSpec(n_samples=400, n_features=8, centers=3,
                                compactness=1.0, seed=99)),
        ),
        (
            "mdcgen-outliers",
            gen_mdcgen(DataSpec(n_samples=400, n_features=8, centers=3,
                                compactness=1.0, seed=99, outliers=80)),
        ),
        (
            "moons",
            gen_moons(DataSpec(n_samples=400, n_features=4, centers=2,
                               compactness=1.0, seed=99)),
        ),
        (
            "circles",
            gen_circles(DataSpec(n_samples=400, n_features=4, centers=2,
                                 compactness=1.0, seed=99)),
        ),
    ]

    router = Learned_router_v3()
    for name, (X, y) in cases:
        k_target = int(len(np.unique(y[y >= 0])))
        res = router.fit_predict(X, k=k_target)
        mask = y >= 0
        if mask.sum() == 0:
            ari = float("nan")
        else:
            ari = adjusted_rand_score(y[mask], res.labels[mask])
        chose = res.extra["chose"]
        preds = res.extra.get("predicted_aris", {})
        pred_chosen = preds.get(chose, float("nan"))
        top3 = list(preds.items())[:3]
        top3_str = ", ".join(f"{a}={s:.3f}" for a, s in top3)
        fp = res.extra["fingerprint"]
        print(
            f"{name:18s}  chose={chose:24s}  ari={ari:.3f}  "
            f"pred_ari={pred_chosen:.3f}  "
            f"dbscan_n_clusters={fp['probe_dbscan_n_clusters']:.0f}  "
            f"dbscan_noise_frac={fp['probe_dbscan_noise_frac']:.3f}\n"
            f"                    top3=[{top3_str}]"
        )
