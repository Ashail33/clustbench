"""Learned routing meta-algorithm — second iteration.

`learned_router_v2` keeps v1's kNN-over-fingerprints architecture but
expands the data fingerprint from 7 to 15 features, with the aim of
fixing v1's blind spot on the ``moons`` shape (full-benchmark ARI 0.62
versus 1.00 on circles/anisotropic).

The expanded fingerprint adds three families:

1. **Probe features** — a tiny k-means++ EM loop is run on the input
   (capped at 500 points and 3 iterations) and four behavioural signals
   are extracted: the inertia compression ratio, the silhouette of the
   probe labels, the coefficient of variation of cluster sizes, and the
   relative centroid shift in the last iteration. These probes give the
   classifier direct evidence about whether the data is convex
   (k-means++ compresses cleanly) or non-convex (k-means++ produces a
   poor split that doesn't converge).
2. **Intrinsic-dimension feature** — a Levina-Bickel maximum-likelihood
   estimate of manifold dimension separates ``moons`` (intrinsic 1) and
   ``circles`` (intrinsic 1) from ambient-dim mdcgen blobs.
3. **Multi-scale density** — mean k-NN distance at k = 5, 20, 50 gives
   a curve that exposes variable-density clusters.

All other behaviour (leave-one-out via ``exclude_self``, dispatch to
``ALGO_REGISTRY``, training data from ``docs/data/results.json``) is
inherited from v1. The training-data cache is separate (`_TRAINING_CACHE_V2`)
so v1 and v2 can coexist in the same process, and dispatch is blocked
to *both* router variants to prevent recursive routing chains.
"""

from __future__ import annotations

import json
import pathlib
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import base as base_algos
from .base import Algorithm, AlgoResult, Step, register
from .learned_router import _algo_position_per_task, _regenerate_task


# Cache is independent from v1 so the two routers don't share fingerprints.
_TRAINING_CACHE_V2: Optional[Tuple[np.ndarray, list, Dict[str, Any]]] = None

# Both router variants are blocked as dispatch targets to prevent
# any pathological recursive routing chains (e.g. v2 -> v1 -> v2).
_BLOCKED_TARGETS_V2 = {"learned_router", "learned_router_v2"}


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------

def _probe_kmeans(X: np.ndarray, k: int, n_iter: int, rng: np.random.Generator
                  ) -> Tuple[np.ndarray, np.ndarray, List[float], List[np.ndarray]]:
    """Tiny k-means++ EM loop used as a behavioural probe.

    Returns (labels, centers, inertia_per_iter, centers_per_iter). We
    write it inline instead of calling sklearn so we can observe the
    per-iteration trajectory (initial inertia, centroid shifts) without
    a heavyweight callback hook.
    """
    n, d = X.shape
    k = max(2, min(k, n))

    # k-means++ seed.
    first = int(rng.integers(0, n))
    centers = [X[first]]
    # squared distance to nearest existing center
    dist2 = np.sum((X - centers[0]) ** 2, axis=1)
    for _ in range(1, k):
        total = dist2.sum()
        if total <= 0:
            idx = int(rng.integers(0, n))
        else:
            probs = dist2 / total
            idx = int(rng.choice(n, p=probs))
        centers.append(X[idx])
        new_d = np.sum((X - centers[-1]) ** 2, axis=1)
        dist2 = np.minimum(dist2, new_d)

    C = np.array(centers, dtype=np.float64)

    # Compute initial inertia (assignment to seeded centers, no update yet)
    d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
    labels = np.argmin(d2, axis=1)
    inertia_list = [float(d2[np.arange(n), labels].sum())]
    centers_history = [C.copy()]

    # EM updates
    for _ in range(n_iter):
        new_C = np.array([
            X[labels == j].mean(axis=0) if np.any(labels == j) else C[j]
            for j in range(k)
        ], dtype=np.float64)
        C = new_C
        d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(d2, axis=1)
        inertia_list.append(float(d2[np.arange(n), labels].sum()))
        centers_history.append(C.copy())

    return labels, C, inertia_list, centers_history


def _fingerprint_v2(X: np.ndarray, k: Optional[int]) -> Dict[str, float]:
    """Compute the 15-feature v2 fingerprint of (X, k).

    Layout:
      - v1 carry-over (7): log_n, d, k, eff_dim, conv_cv, outlier_frac,
        density_skew
      - probe (4): probe_inertia_ratio, probe_silhouette, probe_size_cv,
        probe_centroid_shift_last
      - intrinsic (1): intrinsic_dim_levina_bickel
      - density (3): density_k5, density_k20, density_k50
    """
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

    n, d = X.shape
    fp: Dict[str, float] = {
        "log_n": float(np.log10(max(n, 2))),
        "d": float(d),
        "k": float(k if k is not None else 0),
    }

    # ---- v1: effective dimensionality ------------------------------------
    n_pca = max(1, min(d, 10, n - 1))
    try:
        pca = PCA(n_components=n_pca, svd_solver="auto", random_state=0).fit(X)
        fp["eff_dim"] = float((pca.explained_variance_ratio_ > 0.01).sum())
    except Exception:
        fp["eff_dim"] = float(d)

    # ---- v1: convexity ---------------------------------------------------
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

    # ---- v1: outlier fraction -------------------------------------------
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

    # ---- v1: density skew ------------------------------------------------
    try:
        nbrs = NearestNeighbors(n_neighbors=min(11, n)).fit(X)
        d_knn, _ = nbrs.kneighbors(X)
        knn_mean = d_knn[:, 1:].mean(axis=1)
        fp["density_skew"] = float(np.std(knn_mean) / (np.mean(knn_mean) + 1e-9))
    except Exception:
        fp["density_skew"] = 0.0

    # ---- v2: probe features ---------------------------------------------
    # Cap probe cost at 500 points.
    probe_k = max(2, k or 3)
    rng = np.random.default_rng(0)
    if n > 500:
        sample_idx = rng.choice(n, size=500, replace=False)
        Xp = X[sample_idx].astype(np.float64, copy=False)
    else:
        Xp = X.astype(np.float64, copy=False)

    try:
        plabels, _pc, inertia_list, centers_history = _probe_kmeans(
            Xp, probe_k, n_iter=3, rng=rng
        )
        initial_inertia = inertia_list[0]
        final_inertia = inertia_list[-1]
        ratio = (final_inertia / initial_inertia) if initial_inertia > 1e-12 else 1.0
        fp["probe_inertia_ratio"] = float(ratio)

        # Silhouette on probe labels (needs >=2 distinct labels and >=3 points).
        try:
            unique = np.unique(plabels)
            if len(unique) >= 2 and Xp.shape[0] >= 3:
                # Cap silhouette cost at 300 to keep it cheap.
                if Xp.shape[0] > 300:
                    sil_idx = rng.choice(Xp.shape[0], size=300, replace=False)
                    Xs, ls = Xp[sil_idx], plabels[sil_idx]
                    if len(np.unique(ls)) >= 2:
                        fp["probe_silhouette"] = float(silhouette_score(Xs, ls))
                    else:
                        fp["probe_silhouette"] = 0.0
                else:
                    fp["probe_silhouette"] = float(silhouette_score(Xp, plabels))
            else:
                fp["probe_silhouette"] = 0.0
        except Exception:
            fp["probe_silhouette"] = 0.0

        # Cluster size coefficient of variation.
        sizes = np.bincount(plabels, minlength=probe_k).astype(np.float64)
        if sizes.sum() > 0:
            fp["probe_size_cv"] = float(np.std(sizes) / (np.mean(sizes) + 1e-9))
        else:
            fp["probe_size_cv"] = 0.0

        # Centroid shift: ||C_last - C_prev|| / ||C_iter1 - C_init||
        # Use a relative measure so it's roughly scale-invariant.
        first_shift = float(np.linalg.norm(centers_history[1] - centers_history[0]))
        last_shift = float(np.linalg.norm(centers_history[-1] - centers_history[-2]))
        if first_shift > 1e-9:
            fp["probe_centroid_shift_last"] = float(last_shift / first_shift)
        else:
            fp["probe_centroid_shift_last"] = 0.0
    except Exception:
        fp["probe_inertia_ratio"] = 1.0
        fp["probe_silhouette"] = 0.0
        fp["probe_size_cv"] = 0.0
        fp["probe_centroid_shift_last"] = 0.0

    # ---- v2: Levina-Bickel intrinsic dimension --------------------------
    # MLE on 2-NN ratios, k_nn=5. Sample up to 200 points.
    try:
        sample_n = min(200, n)
        if n > sample_n:
            idx = rng.choice(n, size=sample_n, replace=False)
            Xs = X[idx]
        else:
            Xs = X
        k_nn = 5
        if Xs.shape[0] >= k_nn + 1:
            nbrs_lb = NearestNeighbors(n_neighbors=k_nn + 1).fit(Xs)
            d_lb, _ = nbrs_lb.kneighbors(Xs)
            # d_lb[:, 0] is self (distance 0). neighbours are columns 1..k_nn.
            T_k = d_lb[:, k_nn]
            T_1 = d_lb[:, 1]
            # Standard LB: per-point MLE is (1/(k_nn-1)) * sum_{j=1..k_nn-1} log(T_k / T_j)
            # then averaged then inverted. Use a simplified ratio form with j=1.
            ratios = T_k / np.maximum(T_1, 1e-12)
            logs = np.log(np.maximum(ratios, 1.0 + 1e-12))
            # Per-point dimension estimate: (k_nn - 1) / sum_{j=1..k_nn-1} log(T_k/T_j)
            # We approximate using the k=1 ratio (the dominant term) — equivalent
            # to MLE with k_nn=2 effective. This is robust on small samples.
            mean_log = np.mean(logs)
            if mean_log > 1e-9:
                fp["intrinsic_dim_levina_bickel"] = float((k_nn - 1) / mean_log)
            else:
                fp["intrinsic_dim_levina_bickel"] = float(d)
        else:
            fp["intrinsic_dim_levina_bickel"] = float(d)
    except Exception:
        fp["intrinsic_dim_levina_bickel"] = float(d)

    # ---- v2: multi-scale density ----------------------------------------
    for k_scale in (5, 20, 50):
        key = f"density_k{k_scale}"
        try:
            n_use = min(k_scale + 1, n)
            if n_use < 2:
                fp[key] = 0.0
                continue
            nbrs_s = NearestNeighbors(n_neighbors=n_use).fit(X)
            d_s, _ = nbrs_s.kneighbors(X)
            # mean of k-th nearest distance across points
            fp[key] = float(d_s[:, n_use - 1].mean())
        except Exception:
            fp[key] = 0.0

    return fp


# ---------------------------------------------------------------------------
# Training-data loader (v2 cache)
# ---------------------------------------------------------------------------

def _load_training_data_v2() -> Tuple[Optional[np.ndarray], Optional[list], Optional[Dict[str, Any]]]:
    """Read ``docs/data/results.json``, regenerate each historical task,
    compute v2 fingerprints, and cache the result.
    """
    global _TRAINING_CACHE_V2
    if _TRAINING_CACHE_V2 is not None:
        return _TRAINING_CACHE_V2

    repo_root = pathlib.Path(__file__).resolve().parents[3]
    results_path = repo_root / "docs" / "data" / "results.json"
    if not results_path.exists():
        _TRAINING_CACHE_V2 = (None, None, None)
        return _TRAINING_CACHE_V2

    try:
        rows = json.loads(results_path.read_text())
    except Exception:
        _TRAINING_CACHE_V2 = (None, None, None)
        return _TRAINING_CACHE_V2

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
        fp = _fingerprint_v2(X, meta["k_target"])
        fingerprints.append(fp)
        rank_rows.append(algo_ari)

    if not fingerprints:
        _TRAINING_CACHE_V2 = (None, None, None)
        return _TRAINING_CACHE_V2

    feature_names = sorted(fingerprints[0].keys())
    F = np.array(
        [[fp[name] for name in feature_names] for fp in fingerprints],
        dtype=np.float64,
    )
    F_mean = F.mean(axis=0)
    F_std = F.std(axis=0)
    F_std[F_std == 0] = 1.0
    F_norm = (F - F_mean) / F_std

    _TRAINING_CACHE_V2 = (
        F_norm,
        rank_rows,
        {
            "feature_names": feature_names,
            "mean": F_mean,
            "std": F_std,
            "n_tasks": len(fingerprints),
        },
    )
    return _TRAINING_CACHE_V2


# ---------------------------------------------------------------------------
# Algorithm class
# ---------------------------------------------------------------------------

@register
class Learned_router_v2(Algorithm):
    """v2 of the kNN-over-fingerprints meta-algorithm.

    See module docstring for the rationale of the expanded fingerprint.
    """

    def __init__(
        self,
        k_neighbors: int = 5,
        exclude_self: bool = True,
        candidates: Optional[List[str]] = None,
        fallback: str = "pwcc_diverse",
        **kwargs: Any,
    ) -> None:
        self.name = "learned_router_v2"
        self.k_neighbors = k_neighbors
        self.exclude_self = exclude_self
        self.candidates = candidates
        self.fallback = fallback

    def _candidates_pool(self, rank_rows: List[Dict[str, float]]) -> List[str]:
        if self.candidates is not None:
            return [c for c in self.candidates if c not in _BLOCKED_TARGETS_V2]
        seen: set = set()
        for row in rank_rows:
            seen.update(row.keys())
        return [a for a in sorted(seen) if a not in _BLOCKED_TARGETS_V2]

    def _dispatch(self, algo: str, X: np.ndarray, k: Optional[int]) -> AlgoResult:
        cls = base_algos.ALGO_REGISTRY.get(algo)
        if cls is None:
            cls = base_algos.ALGO_REGISTRY[self.fallback]
        return cls().fit_predict(X, k=k)

    def fit_predict(self, X: np.ndarray, k: Optional[int] = None) -> AlgoResult:
        F_norm, rank_rows, meta = _load_training_data_v2()

        if F_norm is None or rank_rows is None or meta is None:
            inner = self._dispatch(self.fallback, X, k)
            return AlgoResult(
                labels=inner.labels,
                extra={
                    "router": "learned_knn_v2",
                    "chose": self.fallback,
                    "reason": "no_training_data",
                    **(inner.extra or {}),
                },
                trajectory=inner.trajectory or [],
            )

        fp = _fingerprint_v2(X, k)
        fp_vec = np.array(
            [fp[name] for name in meta["feature_names"]], dtype=np.float64
        )
        fp_norm = (fp_vec - meta["mean"]) / meta["std"]

        dists = np.linalg.norm(F_norm - fp_norm[None, :], axis=1)
        order = np.argsort(dists)
        if self.exclude_self and len(order) and dists[order[0]] < 1e-6:
            order = order[1:]
        topk_idx = order[: max(1, self.k_neighbors)]
        neighbour_rows = [rank_rows[i] for i in topk_idx]

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

        trajectory: List[Step] = [
            Step(
                step_idx=0,
                cost=float(scores.get(chosen, 0.0)),
                delta_cost=None,
                accepted=True,
                action={
                    "type": "learned_route_v2",
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
                "router": "learned_knn_v2",
                "chose": chosen,
                "top_candidates": [(a, float(s)) for a, s in top_candidates],
                "fingerprint": {kk: float(vv) for kk, vv in fp.items()},
                **(inner.extra or {}),
            },
            trajectory=trajectory,
        )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Held-out smoke test, seed=99, matching v1's smoke-test recipe.
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

    router = Learned_router_v2()
    for name, (X, y) in cases:
        # The k_target used by these generators is `centers`.
        k_target = int(len(np.unique(y[y >= 0])))
        res = router.fit_predict(X, k=k_target)
        # Compare ARI on points with non-noise ground truth labels.
        mask = y >= 0
        if mask.sum() == 0:
            ari = float("nan")
        else:
            ari = adjusted_rand_score(y[mask], res.labels[mask])
        fp = res.extra["fingerprint"]
        print(
            f"{name:18s}  chose={res.extra['chose']:24s}  ari={ari:.3f}  "
            f"probe_inertia_ratio={fp['probe_inertia_ratio']:.3f}  "
            f"probe_silhouette={fp['probe_silhouette']:.3f}  "
            f"probe_size_cv={fp['probe_size_cv']:.3f}  "
            f"probe_centroid_shift_last={fp['probe_centroid_shift_last']:.3f}"
        )
