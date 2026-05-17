"""Learned routing meta-algorithm — fourth iteration of the synthesis chain.

The v1/v2/v3 generations of `aura`, `meta_clusterer`, and `rapid`
encoded routing rules by hand:
- v1 used direct heuristics (eigengap, convexity ratio, ...).
- v2 fixed v1's bottlenecks with more robust heuristics.
- v3 dispatched between v1 and v2 based on a cheap data signature.

`learned_router` replaces the hand-coded dispatch with a k-NN classifier
over data fingerprints, trained on past benchmark results stored in
`docs/data/results.json`. At inference time the router

1. computes a fingerprint of the input data,
2. finds the K nearest training tasks (by Euclidean distance in
   normalized fingerprint space),
3. for each candidate algorithm, computes its mean rank across those
   neighbouring tasks (where "rank" is its ARI position within each
   task's leaderboard),
4. dispatches to the candidate with the lowest mean rank.

**Data leakage.** When the same dataset (matched by exact fingerprint)
appears in both training and inference, `exclude_self=True` drops it
— a strict leave-one-out. That doesn't eliminate near-neighbour leakage
(two tasks that differ only in seed have very similar fingerprints);
the learned_router still benefits from having seen the dataset family
during training. The honest story is: this router shows the upper bound
of what a learned dispatch policy could do, given full coverage of the
benchmark's data regimes. To eliminate near-neighbour leakage in
follow-up evaluation, hold out an entire (dataset_id, parameter slice)
from training and evaluate only on that slice.
"""

from __future__ import annotations

import json
import pathlib
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import base as base_algos
from .base import Algorithm, AlgoResult, Step, register


# Module-level cache so the training data is built once per process.
_TRAINING_CACHE: Optional[Tuple[np.ndarray, list, Dict[str, Any]]] = None

# Algorithms the router is *not* allowed to dispatch to. learned_router
# itself is excluded to prevent infinite recursion; algorithms that have
# a strict superset (e.g. v1 when v3 exists for the same family) are
# allowed because the kNN may rank them higher on specific regimes.
_BLOCKED_TARGETS = {"learned_router"}


def _fingerprint(X: np.ndarray, k: Optional[int]) -> Dict[str, float]:
    """Compute a small, cheap, fixed-size fingerprint of (X, k).

    Designed to capture the regime axes that the analysis doc identifies
    as the main differentiators: scale, dimensionality, cluster count,
    convexity, outlier presence, and density variation.
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

    # PCA-based effective dimensionality.
    n_pca = max(1, min(d, 10, n - 1))
    try:
        pca = PCA(n_components=n_pca, svd_solver="auto", random_state=0).fit(X)
        fp["eff_dim"] = float((pca.explained_variance_ratio_ > 0.01).sum())
    except Exception:
        fp["eff_dim"] = float(d)

    # Convexity via coefficient of variation of intra-cluster distances.
    try:
        km = KMeans(
            n_clusters=max(2, k or 3),
            n_init=3,
            max_iter=50,
            random_state=0,
        ).fit(X)
        dist = np.linalg.norm(X - km.cluster_centers_[km.labels_], axis=1)
        fp["conv_cv"] = float(np.std(dist) / (np.mean(dist) + 1e-9))
    except Exception:
        fp["conv_cv"] = 0.0

    # Outlier fraction via LOF score.
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

    # k-NN density skew — high when clusters have very different densities.
    try:
        nbrs = NearestNeighbors(n_neighbors=min(11, n)).fit(X)
        d_knn, _ = nbrs.kneighbors(X)
        knn_mean = d_knn[:, 1:].mean(axis=1)
        fp["density_skew"] = float(np.std(knn_mean) / (np.mean(knn_mean) + 1e-9))
    except Exception:
        fp["density_skew"] = 0.0

    return fp


def _regenerate_task(meta: dict):
    """Regenerate the (X, y) pair for one historical task using DataSpec."""
    from ..datasets import DATASETS, DataSpec

    gen = DATASETS.get(meta["dataset_id"])
    if gen is None:
        return None
    try:
        spec = DataSpec(
            n_samples=int(meta["n_samples"]),
            n_features=int(meta["n_features"]),
            centers=int(meta["k_target"] or 2),
            compactness=float(meta.get("compactness", 1.0) or 1.0),
            seed=int(meta["seed"]),
            outliers=int(meta.get("outliers", 0) or 0),
            noise=int(meta.get("noise", 0) or 0),
            density=float(meta.get("density", 1.0) or 1.0),
        )
        return gen(spec)
    except Exception:
        return None


def _load_training_data() -> Tuple[Optional[np.ndarray], Optional[list], Optional[Dict[str, Any]]]:
    """Read docs/data/results.json, regenerate each historical task,
    compute fingerprints, and cache (F_norm, per-task ARI dicts, meta)."""
    global _TRAINING_CACHE
    if _TRAINING_CACHE is not None:
        return _TRAINING_CACHE

    repo_root = pathlib.Path(__file__).resolve().parents[3]
    results_path = repo_root / "docs" / "data" / "results.json"
    if not results_path.exists():
        _TRAINING_CACHE = (None, None, None)
        return _TRAINING_CACHE

    try:
        rows = json.loads(results_path.read_text())
    except Exception:
        _TRAINING_CACHE = (None, None, None)
        return _TRAINING_CACHE

    # Group rows by exact task identity.
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
    for key, algo_ari in by_task.items():
        meta = task_meta[key]
        gen_result = _regenerate_task(meta)
        if gen_result is None:
            continue
        X, _ = gen_result
        fp = _fingerprint(X, meta["k_target"])
        fingerprints.append(fp)
        rank_rows.append(algo_ari)

    if not fingerprints:
        _TRAINING_CACHE = (None, None, None)
        return _TRAINING_CACHE

    feature_names = sorted(fingerprints[0].keys())
    F = np.array([[fp[name] for name in feature_names] for fp in fingerprints], dtype=np.float64)
    F_mean = F.mean(axis=0)
    F_std = F.std(axis=0)
    F_std[F_std == 0] = 1.0
    F_norm = (F - F_mean) / F_std

    _TRAINING_CACHE = (
        F_norm,
        rank_rows,
        {
            "feature_names": feature_names,
            "mean": F_mean,
            "std": F_std,
            "n_tasks": len(fingerprints),
        },
    )
    return _TRAINING_CACHE


def _algo_position_per_task(rank_rows: List[Dict[str, float]], algo: str) -> List[int]:
    """For each task, return the 1-based position of ``algo`` in the
    descending-ARI leaderboard. Returns -1 if the algo wasn't run on
    that task."""
    out = []
    for row in rank_rows:
        if algo not in row:
            continue
        order = sorted(row.items(), key=lambda kv: kv[1], reverse=True)
        pos = next(i for i, (a, _) in enumerate(order, start=1) if a == algo)
        out.append(pos)
    return out


@register
class Learned_router(Algorithm):
    """kNN-over-fingerprints meta-algorithm.

    Parameters
    ----------
    k_neighbors : int
        Number of nearest training tasks to vote with.
    exclude_self : bool
        Drop the nearest training task if its fingerprint exactly matches
        the inference fingerprint (leave-one-out for benchmark evaluation
        on the same data the router was trained on).
    candidates : list[str] | None
        Restrict the dispatch to this subset of registered algorithms.
        Defaults to every algorithm except `learned_router` itself.
    fallback : str
        Algorithm to dispatch to if training data is unavailable.
    """

    def __init__(
        self,
        k_neighbors: int = 5,
        exclude_self: bool = True,
        candidates: Optional[List[str]] = None,
        fallback: str = "pwcc_diverse",
        **kwargs: Any,
    ) -> None:
        self.name = "learned_router"
        self.k_neighbors = k_neighbors
        self.exclude_self = exclude_self
        self.candidates = candidates
        self.fallback = fallback

    def _candidates_pool(self, rank_rows: List[Dict[str, float]]) -> List[str]:
        if self.candidates is not None:
            return [c for c in self.candidates if c not in _BLOCKED_TARGETS]
        seen: set = set()
        for row in rank_rows:
            seen.update(row.keys())
        return [a for a in sorted(seen) if a not in _BLOCKED_TARGETS]

    def _dispatch(self, algo: str, X: np.ndarray, k: Optional[int]) -> AlgoResult:
        cls = base_algos.ALGO_REGISTRY.get(algo)
        if cls is None:
            cls = base_algos.ALGO_REGISTRY[self.fallback]
        return cls().fit_predict(X, k=k)

    def fit_predict(self, X: np.ndarray, k: Optional[int] = None) -> AlgoResult:
        F_norm, rank_rows, meta = _load_training_data()

        if F_norm is None or rank_rows is None or meta is None:
            inner = self._dispatch(self.fallback, X, k)
            return AlgoResult(
                labels=inner.labels,
                extra={
                    "router": "learned_knn",
                    "chose": self.fallback,
                    "reason": "no_training_data",
                    **(inner.extra or {}),
                },
                trajectory=inner.trajectory or [],
            )

        fp = _fingerprint(X, k)
        fp_vec = np.array([fp[name] for name in meta["feature_names"]], dtype=np.float64)
        fp_norm = (fp_vec - meta["mean"]) / meta["std"]

        # k-NN in normalised fingerprint space.
        dists = np.linalg.norm(F_norm - fp_norm[None, :], axis=1)
        order = np.argsort(dists)
        if self.exclude_self and len(order) and dists[order[0]] < 1e-6:
            order = order[1:]
        topk_idx = order[: max(1, self.k_neighbors)]
        neighbour_rows = [rank_rows[i] for i in topk_idx]

        # Aggregate: each candidate's mean rank across the neighbours.
        pool = self._candidates_pool(rank_rows)
        scores: Dict[str, float] = {}
        for algo in pool:
            positions = _algo_position_per_task(neighbour_rows, algo)
            if positions:
                scores[algo] = float(np.mean(positions))

        if not scores:
            chosen = self.fallback
        else:
            chosen = min(scores.items(), key=lambda kv: kv[1])[0]

        top_candidates = sorted(scores.items(), key=lambda kv: kv[1])[:5]

        # Build a trajectory that records the routing decision and then
        # appends the inner algorithm's trajectory shifted by +1.
        trajectory: List[Step] = [
            Step(
                step_idx=0,
                cost=float(scores.get(chosen, 0.0)),
                delta_cost=None,
                accepted=True,
                action={
                    "type": "learned_route",
                    "chose": chosen,
                    "k_neighbors": self.k_neighbors,
                    "top_candidates": [(a, float(s)) for a, s in top_candidates],
                    "neighbour_distance_min": float(dists[order[0]]) if len(order) else None,
                },
                state={
                    "fingerprint": {kk: float(vv) for kk, vv in fp.items()},
                    "n_training_tasks": int(meta["n_tasks"]),
                },
            )
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
                "router": "learned_knn",
                "chose": chosen,
                "top_candidates": [(a, float(s)) for a, s in top_candidates],
                "fingerprint": {kk: float(vv) for kk, vv in fp.items()},
                **(inner.extra or {}),
            },
            trajectory=trajectory,
        )
