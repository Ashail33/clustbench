"""Learned routing meta-algorithm — fourth iteration.

`learned_router_v4` is the fourth generation of the learned-dispatch
meta-algorithm. It inherits v3's architecture (per-algorithm regressors
over a normalised fingerprint, argmax dispatch) and introduces three
focused improvements:

1. **Gradient-boosted per-algorithm regressors.** v3 used a distance-
   weighted ``KNeighborsRegressor(k=5)`` per candidate algorithm. kNN
   regression is robust but it only interpolates; it cannot capture
   non-linear *interactions* between fingerprint features. v4 swaps in
   ``GradientBoostingRegressor(n_estimators=40, max_depth=3,
   learning_rate=0.1, random_state=0)`` for every algorithm. The
   training-set is small (~118 unique tasks per algorithm) so we keep
   ``n_estimators`` modest to avoid memorisation; depth 3 lets the
   model use feature interactions without overfitting.

2. **Three dataset-meta features.** The comprehensive sweep now carries
   ``dataset_source``, ``dataset_domain`` and ``outlier_extremity`` on
   every row, so v4 enriches the v3 fingerprint with three derived
   features computed from the input task's identifiers when they're
   known (defaulting to 0 otherwise):

       is_real_data       1.0 if ``dataset_id`` is in REAL_METADATA
                          (source ∈ {sklearn, openml}), else 0.0.
       is_image_domain    1.0 if ``dataset_id`` ∈ {digits, olivetti_faces}.
       log_outlier_extremity  log(outlier_extremity + 1).

   The fingerprint becomes **20 features** (v3's 17 plus these 3).
   The dataset_source/domain columns in results.json happen to be all
   ``None`` at this snapshot, so we derive ``is_real_data`` /
   ``is_image_domain`` purely from ``dataset_id`` via the
   :data:`clustbench.datasets_real.REAL_METADATA` table — same shape as
   what the spec describes once the columns get populated.

3. **Ensemble with v3 as the conservative anchor.** v4 internally
   instantiates ``Learned_router_v3`` (lazily, on first call) and gets
   v3's predicted-ARI vector for every candidate. If v4's argmax pick
   disagrees with v3 by more than 0.05 predicted ARI — i.e. v3 would
   rate v4's choice meaningfully worse than its own — we *fall back to
   v3's pick*. This is a guardrail: v3 already lands rank-1 on the
   benchmark, so we only let v4 override on cases where the two models
   roughly agree. Mild disagreements (≤0.05) leave v4's pick intact so
   v4 still has room to find new wins on the freshly added
   compactness=1.25 / extreme-outliers regimes.

Dispatch blocking is narrower than in v3: only router variants
(``learned_router``, ``learned_router_v2``, ``learned_router_v3``,
``learned_router_v4``) are blocked to prevent recursive routing chains.
``mutant_kmeans_meta`` — a regular registry algorithm shipped alongside
v4 — is *not* blocked and v4 can dispatch to it freely.

Caches (``_TRAINING_CACHE_V4``, ``_REGRESSORS_V4``) are independent so
all four router versions can coexist in the same process.
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
from .learned_router import _regenerate_task
from .learned_router_v3 import _fingerprint_v3


# Caches are independent from v1/v2/v3 so the four routers don't share
# fingerprints or regressors.
_TRAINING_CACHE_V4: Optional[Tuple[np.ndarray, list, Dict[str, Any]]] = None
_REGRESSORS_V4: Optional[Dict[str, Any]] = None

# Only block router variants to prevent recursive routing chains.
# ``mutant_kmeans_meta`` is a regular registry algorithm and is a valid
# dispatch target for v4.
_BLOCKED_TARGETS_V4 = {
    "learned_router",
    "learned_router_v2",
    "learned_router_v3",
    "learned_router_v4",
}

# Datasets considered "image-domain" for the is_image_domain feature.
_IMAGE_DOMAIN_DATASETS = {"digits", "olivetti_faces"}


# ---------------------------------------------------------------------------
# Dataset-meta feature derivation
# ---------------------------------------------------------------------------

def _dataset_meta_features(
    dataset_id: Optional[str],
    outlier_extremity: Optional[float],
) -> Dict[str, float]:
    """Compute the three dataset-meta features.

    All three default to 0.0 when their inputs are unknown. The spec
    instructs us to look up REAL_METADATA for ``is_real_data`` so that
    the feature reflects "this is a real-world dataset" regardless of
    whether the ``dataset_source`` column has been populated yet.
    """
    try:
        from ..datasets_real import REAL_METADATA
    except Exception:
        REAL_METADATA = {}

    is_real = 1.0 if (dataset_id is not None and dataset_id in REAL_METADATA) else 0.0
    is_image = 1.0 if (dataset_id in _IMAGE_DOMAIN_DATASETS) else 0.0
    try:
        ext = float(outlier_extremity) if outlier_extremity is not None else 0.0
    except Exception:
        ext = 0.0
    log_ext = float(np.log(ext + 1.0))
    return {
        "is_real_data": float(is_real),
        "is_image_domain": float(is_image),
        "log_outlier_extremity": log_ext,
    }


def _fingerprint_v4(
    X: np.ndarray,
    k: Optional[int],
    *,
    dataset_id: Optional[str] = None,
    outlier_extremity: Optional[float] = None,
) -> Dict[str, float]:
    """Compute the 20-feature v4 fingerprint.

    Layout:
      - v3 carry-over (17): see ``_fingerprint_v3`` (= v2's 15 + DBSCAN 2).
      - dataset-meta (3): is_real_data, is_image_domain,
        log_outlier_extremity.

    At training time the meta keys are taken from the historical row;
    at inference time callers can pass ``dataset_id`` and
    ``outlier_extremity`` to enable the meta features, otherwise they
    default to 0.
    """
    fp = _fingerprint_v3(X, k)
    fp.update(_dataset_meta_features(dataset_id, outlier_extremity))
    return fp


# ---------------------------------------------------------------------------
# Training-data loader
# ---------------------------------------------------------------------------

def _regenerate_task_v4(meta: dict):
    """Regenerate the (X, y) pair for a historical task.

    Identical to v1's ``_regenerate_task`` (the v4 extra fields are not
    part of DataSpec — they only participate as fingerprint inputs).
    Wrapped under a v4-specific name to satisfy the spec's naming
    convention.
    """
    return _regenerate_task(meta)


def _load_training_data_v4() -> Tuple[
    Optional[np.ndarray], Optional[list], Optional[Dict[str, Any]]
]:
    """Read ``docs/data/results.json``, regenerate each historical task,
    compute v4 fingerprints, and cache the result.
    """
    global _TRAINING_CACHE_V4
    if _TRAINING_CACHE_V4 is not None:
        return _TRAINING_CACHE_V4

    repo_root = pathlib.Path(__file__).resolve().parents[3]
    results_path = repo_root / "docs" / "data" / "results.json"
    if not results_path.exists():
        _TRAINING_CACHE_V4 = (None, None, None)
        return _TRAINING_CACHE_V4

    try:
        rows = json.loads(results_path.read_text())
    except Exception:
        _TRAINING_CACHE_V4 = (None, None, None)
        return _TRAINING_CACHE_V4

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
            r.get("compactness"),
            r.get("outlier_extremity"),
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
            "dataset_source": r.get("dataset_source"),
            "dataset_domain": r.get("dataset_domain"),
            "outlier_extremity": r.get("outlier_extremity"),
        }

    fingerprints: List[Dict[str, float]] = []
    rank_rows: List[Dict[str, float]] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for key, algo_ari in by_task.items():
            meta = task_meta[key]
            gen_result = _regenerate_task_v4(meta)
            if gen_result is None:
                continue
            X, _ = gen_result
            fp = _fingerprint_v4(
                X,
                meta["k_target"],
                dataset_id=meta.get("dataset_id"),
                outlier_extremity=meta.get("outlier_extremity"),
            )
            fingerprints.append(fp)
            rank_rows.append(algo_ari)

    if not fingerprints:
        _TRAINING_CACHE_V4 = (None, None, None)
        return _TRAINING_CACHE_V4

    feature_names = sorted(fingerprints[0].keys())
    F = np.array(
        [[fp[name] for name in feature_names] for fp in fingerprints],
        dtype=np.float64,
    )
    F_mean = F.mean(axis=0)
    F_std = F.std(axis=0)
    F_std[F_std == 0] = 1.0
    F_norm = (F - F_mean) / F_std

    _TRAINING_CACHE_V4 = (
        F_norm,
        rank_rows,
        {
            "feature_names": feature_names,
            "mean": F_mean,
            "std": F_std,
            "n_tasks": len(fingerprints),
        },
    )
    return _TRAINING_CACHE_V4


def _load_regressors_v4() -> Optional[Dict[str, Any]]:
    """Train one gradient-boosted ARI regressor per candidate algorithm.

    Each regressor sees only the tasks where its target algorithm was
    actually run. ``GradientBoostingRegressor`` is used with modest
    capacity to suit the small training set (~118 tasks/algo).
    """
    global _REGRESSORS_V4
    if _REGRESSORS_V4 is not None:
        return _REGRESSORS_V4

    F_norm, rank_rows, meta = _load_training_data_v4()
    if F_norm is None or rank_rows is None:
        _REGRESSORS_V4 = {}
        return _REGRESSORS_V4

    from sklearn.ensemble import GradientBoostingRegressor

    seen: set = set()
    for row in rank_rows:
        seen.update(row.keys())
    seen -= _BLOCKED_TARGETS_V4

    regressors: Dict[str, Any] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for algo in sorted(seen):
            idx = [i for i, row in enumerate(rank_rows) if algo in row]
            if len(idx) < 2:
                continue
            X_tr = F_norm[idx]
            y_tr = np.array(
                [rank_rows[i][algo] for i in idx], dtype=np.float64
            )
            reg = GradientBoostingRegressor(
                n_estimators=40,
                max_depth=3,
                learning_rate=0.1,
                random_state=0,
            )
            try:
                reg.fit(X_tr, y_tr)
                regressors[algo] = reg
            except Exception:
                continue

    _REGRESSORS_V4 = regressors
    return _REGRESSORS_V4


# ---------------------------------------------------------------------------
# Algorithm class
# ---------------------------------------------------------------------------

@register
class Learned_router_v4(Algorithm):
    """v4 of the learned-dispatch meta-algorithm.

    See module docstring for the rationale: gradient-boosted per-algo
    regressors over a 20-feature fingerprint, with a v3 ensemble
    guardrail.
    """

    def __init__(
        self,
        exclude_self: bool = True,
        candidates: Optional[List[str]] = None,
        fallback: str = "pwcc_diverse",
        v3_disagreement_threshold: float = 0.05,
        **kwargs: Any,
    ) -> None:
        self.name = "learned_router_v4"
        self.exclude_self = exclude_self
        self.candidates = candidates
        self.fallback = fallback
        self.v3_disagreement_threshold = v3_disagreement_threshold
        # Lazily instantiated on first call to avoid double-loading the
        # training data at module import time.
        self._v3: Optional[Algorithm] = None

    # ----- helpers -------------------------------------------------------

    def _candidates_pool(self, rank_rows: List[Dict[str, float]]) -> List[str]:
        if self.candidates is not None:
            return [c for c in self.candidates if c not in _BLOCKED_TARGETS_V4]
        seen: set = set()
        for row in rank_rows:
            seen.update(row.keys())
        return [a for a in sorted(seen) if a not in _BLOCKED_TARGETS_V4]

    def _dispatch(
        self, algo: str, X: np.ndarray, k: Optional[int]
    ) -> AlgoResult:
        cls = base_algos.ALGO_REGISTRY.get(algo)
        if cls is None:
            cls = base_algos.ALGO_REGISTRY[self.fallback]
        return cls().fit_predict(X, k=k)

    def _v3_predictions(
        self, X: np.ndarray, k: Optional[int], pool: List[str]
    ) -> Dict[str, float]:
        """Return v3's predicted-ARI vector for ``pool`` algorithms.

        Uses v3's normalisation + regressors directly (without
        re-dispatching) so we don't pay for v3's inner clustering call
        when all we need is its forecast.
        """
        from .learned_router_v3 import (
            _fingerprint_v3 as _fp3,
            _load_regressors_v3,
            _load_training_data_v3,
        )

        F_norm, rank_rows, meta = _load_training_data_v3()
        if F_norm is None or meta is None:
            return {}
        regs = _load_regressors_v3() or {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fp = _fp3(X, k)
        fp_vec = np.array(
            [fp[name] for name in meta["feature_names"]], dtype=np.float64
        )
        fp_norm = (fp_vec - meta["mean"]) / meta["std"]
        fp_norm_2d = fp_norm[None, :]
        preds: Dict[str, float] = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for algo in pool:
                reg = regs.get(algo)
                if reg is None:
                    continue
                try:
                    val = float(reg.predict(fp_norm_2d)[0])
                    if not np.isnan(val):
                        preds[algo] = val
                except Exception:
                    continue
        return preds

    # ----- main entry point ---------------------------------------------

    def fit_predict(
        self,
        X: np.ndarray,
        k: Optional[int] = None,
        *,
        dataset_id: Optional[str] = None,
        outlier_extremity: Optional[float] = None,
    ) -> AlgoResult:
        F_norm, rank_rows, meta = _load_training_data_v4()

        if F_norm is None or rank_rows is None or meta is None:
            inner = self._dispatch(self.fallback, X, k)
            return AlgoResult(
                labels=inner.labels,
                extra={
                    "router": "learned_gbr_v4",
                    "chose": self.fallback,
                    "reason": "no_training_data",
                    **(inner.extra or {}),
                },
                trajectory=inner.trajectory or [],
            )

        regressors = _load_regressors_v4() or {}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fp = _fingerprint_v4(
                X,
                k,
                dataset_id=dataset_id,
                outlier_extremity=outlier_extremity,
            )
        fp_vec = np.array(
            [fp[name] for name in meta["feature_names"]], dtype=np.float64
        )
        fp_norm = (fp_vec - meta["mean"]) / meta["std"]

        # Nearest training distance is reported in the trajectory.
        dists = np.linalg.norm(F_norm - fp_norm[None, :], axis=1)
        order = np.argsort(dists)
        if self.exclude_self and len(order) and dists[order[0]] < 1e-6:
            order = order[1:]
        nearest_dist = float(dists[order[0]]) if len(order) else None

        pool = self._candidates_pool(rank_rows)

        # ----- v4 predicted ARI per candidate ----------------------------
        v4_preds: Dict[str, float] = {}
        fp_norm_2d = fp_norm[None, :]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for algo in pool:
                reg = regressors.get(algo)
                if reg is None:
                    continue
                try:
                    val = float(reg.predict(fp_norm_2d)[0])
                    if not np.isnan(val):
                        v4_preds[algo] = val
                except Exception:
                    continue

        # ----- v3 predicted ARI (ensemble companion) ---------------------
        v3_preds = self._v3_predictions(X, k, pool)

        # ----- Argmax pick under v4, then ensemble guardrail -------------
        if v4_preds:
            v4_pick = max(v4_preds.items(), key=lambda kv: kv[1])[0]
            reason = "v4_predicted_ari"
        else:
            v4_pick = self.fallback
            reason = "no_v4_predictions"

        v3_pick = None
        if v3_preds:
            v3_pick = max(v3_preds.items(), key=lambda kv: kv[1])[0]

        # Disagreement = how much v3 down-rates v4's pick vs v3's pick.
        # If v3 has no prediction for v4's pick we treat it as "no
        # disagreement information" and keep v4's choice.
        fell_back_to_v3 = False
        chosen = v4_pick
        if v3_pick is not None and v4_pick in v3_preds:
            gap = float(v3_preds[v3_pick] - v3_preds[v4_pick])
            if gap > self.v3_disagreement_threshold and v3_pick != v4_pick:
                chosen = v3_pick
                reason = "v3_fallback_disagreement"
                fell_back_to_v3 = True

        # ----- Ensemble (averaged) predictions for telemetry ------------
        ensemble: Dict[str, float] = {}
        for algo in set(v4_preds) | set(v3_preds):
            vals = []
            if algo in v4_preds:
                vals.append(v4_preds[algo])
            if algo in v3_preds:
                vals.append(v3_preds[algo])
            if vals:
                ensemble[algo] = float(np.mean(vals))

        top_v4 = sorted(v4_preds.items(), key=lambda kv: kv[1], reverse=True)[:5]
        top_v3 = sorted(v3_preds.items(), key=lambda kv: kv[1], reverse=True)[:5]
        top_ens = sorted(ensemble.items(), key=lambda kv: kv[1], reverse=True)[:5]

        # ----- Trajectory -------------------------------------------------
        trajectory: List[Step] = [
            Step(
                step_idx=0,
                cost=0.0,
                delta_cost=None,
                accepted=True,
                action={"type": "fingerprint_v4"},
                state={
                    "fingerprint": {kk: float(vv) for kk, vv in fp.items()},
                    "n_training_tasks": int(meta["n_tasks"]),
                },
            ),
            Step(
                step_idx=1,
                cost=float(-v4_preds.get(v4_pick, 0.0)) if v4_preds else 0.0,
                delta_cost=None,
                accepted=True,
                action={
                    "type": "predict_v4",
                    "v4_pick": v4_pick,
                    "v4_pred_ari": float(v4_preds.get(v4_pick, float("nan"))),
                    "top_candidates_v4": [(a, float(s)) for a, s in top_v4],
                    "neighbour_distance_min": nearest_dist,
                },
                state={
                    "predicted_aris_v4": {a: float(s) for a, s in top_v4},
                },
            ),
        ]

        if fell_back_to_v3:
            trajectory.append(
                Step(
                    step_idx=len(trajectory),
                    cost=float(-v3_preds.get(v3_pick, 0.0)),
                    delta_cost=None,
                    accepted=True,
                    action={
                        "type": "v3_consult",
                        "v3_pick": v3_pick,
                        "v3_pred_ari_v3_pick": float(v3_preds[v3_pick]),
                        "v3_pred_ari_v4_pick":
                            float(v3_preds.get(v4_pick, float("nan"))),
                        "disagreement_gap": float(
                            v3_preds[v3_pick] - v3_preds.get(v4_pick, 0.0)
                        ),
                        "threshold": float(self.v3_disagreement_threshold),
                        "chose": chosen,
                        "top_candidates_v3":
                            [(a, float(s)) for a, s in top_v3],
                    },
                    state={
                        "predicted_aris_v3":
                            {a: float(s) for a, s in top_v3},
                        "ensemble_predicted_aris":
                            {a: float(s) for a, s in top_ens},
                    },
                )
            )

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
                "router": "learned_gbr_v4",
                "chose": chosen,
                "reason": reason,
                "v4_pick": v4_pick,
                "v3_pick": v3_pick,
                "fell_back_to_v3": bool(fell_back_to_v3),
                "v4_predicted_aris": {a: float(s) for a, s in top_v4},
                "v3_predicted_aris": {a: float(s) for a, s in top_v3},
                "ensemble_predicted_aris":
                    {a: float(s) for a, s in top_ens},
                "fingerprint": {kk: float(vv) for kk, vv in fp.items()},
                "n_training_tasks": int(meta["n_tasks"]),
                "v3_disagreement_threshold":
                    float(self.v3_disagreement_threshold),
                **(inner.extra or {}),
            },
            trajectory=trajectory,
        )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Held-out smoke test, seed=99, matching v3's smoke-test recipe.
    from sklearn.metrics import adjusted_rand_score

    from ..datasets import DataSpec, gen_circles, gen_mdcgen, gen_moons
    from .learned_router_v3 import Learned_router_v3

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

    router_v4 = Learned_router_v4()
    router_v3 = Learned_router_v3()

    for name, (X, y) in cases:
        k_target = int(len(np.unique(y[y >= 0])))
        res = router_v4.fit_predict(X, k=k_target)
        mask = y >= 0
        if mask.sum() == 0:
            ari = float("nan")
        else:
            ari = adjusted_rand_score(y[mask], res.labels[mask])

        chose = res.extra["chose"]
        v4_pick = res.extra["v4_pick"]
        v3_pick = res.extra["v3_pick"]
        fell_back = res.extra["fell_back_to_v3"]
        v4_preds = res.extra.get("v4_predicted_aris", {})
        v3_preds = res.extra.get("v3_predicted_aris", {})
        v4_pred_for_pick = v4_preds.get(chose, float("nan"))
        v3_pred_for_pick = v3_preds.get(chose, float("nan"))

        # What did v3 (standalone) pick on the same input?
        res_v3 = router_v3.fit_predict(X, k=k_target)
        v3_standalone_pick = res_v3.extra["chose"]
        v3_standalone_preds = res_v3.extra.get("predicted_aris", {})
        v3_standalone_pred = v3_standalone_preds.get(
            v3_standalone_pick, float("nan")
        )

        print(
            f"{name:18s}  v4_chose={chose:24s}  v3_chose={v3_standalone_pick:24s}  "
            f"fell_back_to_v3={fell_back}\n"
            f"                    v4_pred_for_chose={v4_pred_for_pick:.3f}  "
            f"v3_pred_for_v3_pick={v3_standalone_pred:.3f}  "
            f"actual_ari={ari:.3f}\n"
            f"                    v4_argmax={v4_pick}  v3_inner_pick={v3_pick}  "
            f"v3_pred_for_v4_pick="
            f"{v3_preds.get(v4_pick, float('nan')):.3f}"
        )
