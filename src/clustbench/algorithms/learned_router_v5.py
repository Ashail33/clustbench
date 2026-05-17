"""Learned routing meta-algorithm — fifth iteration (stacker).

`learned_router_v5` is a *stacker* over its two best predecessors. It
keeps both v3's predicted-ARI kNN router (current rank 1 of 36, mean ARI
0.851) and v4's gradient-boosted router (rank 2, mean ARI 0.836) and
learns a per-task gating function that decides which of the two should
own the dispatch on this input.

The intuition is straightforward. v4 lost the bake-off to v3 overall
(-0.015 mean ARI), but the loss is not uniform: v4 wins on some tasks
where its 20-feature fingerprint and GBR-style nonlinear interactions
catch a regime that v3's kNN smoothing misses. v3 wins everywhere else.
A scalar threshold won't separate those two regimes, but the v4
fingerprint (which already encodes shape, density, dataset-meta and
k-means probe features) plausibly does. So we train a small classifier
on the historical benchmark whose target is "which router would have
picked the higher-ARI algorithm on this task?"; at inference we just
follow the prediction.

Training procedure
------------------
1. Read ``docs/data/results.json`` and group rows by task identity
   (dataset_id, n_samples, n_features, k_target, outliers, noise,
   density, seed). For each unique task, regenerate the (X, y) pair
   from its DataSpec and compute the v4 fingerprint (20 features).
2. For each task, simulate what v3 and v4 would *pick* — but only by
   running their internal prediction step, NOT their full ``fit_predict``
   (we never want to re-run clustering during training). v3's pick is
   the argmax over its per-algo kNN-ARI regressors; v4's pick is the
   argmax over its per-algo GBR-ARI regressors, with the same
   ``v3_disagreement_threshold=0.05`` guardrail v4 ships with so the
   simulation matches the actual deployed behaviour.
3. Look the picked algorithm's recorded ARI up in the task's row and
   compute ``v4_better = 1 if ARI(v4_pick) > ARI(v3_pick) else 0``.
4. Deduplicate by fingerprint hash (some tasks have identical regenerated
   fingerprints — e.g., real-data tasks where the same X is reused with
   different seeds), using the median of v3/v4 ARI within each
   fingerprint group before computing ``v4_better``.
5. Train ``RandomForestClassifier(n_estimators=50, max_depth=4,
   random_state=0)`` on (fingerprint, v4_better).

If the training data contains no ``v4_better=1`` rows the classifier
would only ever see one class; v5 then degenerates to "always dispatch
to v3" and emits a warning.

Inference
---------
At ``fit_predict`` time we compute the input's 20-feature fingerprint,
call ``predict_proba`` on the classifier, take the class with ``proba >
0.5`` (defaulting to v3 on ties), instantiate the chosen router lazily
and forward its result. The returned ``extra`` is decorated with the
classifier's choice and confidence, and the trajectory prepends two new
records (``fingerprint`` + ``stacker_decision``) before the inner
router's own trajectory.

Dispatch blocking
-----------------
By construction v5 only ever forwards to v3 or v4, but
``_BLOCKED_ROUTERS_V5`` still includes every router variant
(``learned_router``, ``learned_router_v2``, ``learned_router_v3``,
``learned_router_v4``, ``learned_router_v5``) as defence-in-depth in
case the inner router's own decision tries to recurse.

Caches
------
Module-level caches ``_TRAINING_CACHE_V5`` and ``_CLASSIFIER_V5`` mean
the meta training set is built once per process. The two inner routers
(``Learned_router_v3`` and ``Learned_router_v4``) are instantiated
lazily on the first ``fit_predict`` call and cached as instance
attributes so we don't pay v3/v4's training-data load cost at module
import.
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
from .learned_router_v4 import _fingerprint_v4


# Module-level caches: built once per process.
_TRAINING_CACHE_V5: Optional[Dict[str, Any]] = None
_CLASSIFIER_V5: Optional[Any] = None

# All router variants are blocked as dispatch targets. v5 only forwards
# to v3 / v4 by construction, but this is defence-in-depth.
_BLOCKED_ROUTERS_V5 = {
    "learned_router",
    "learned_router_v2",
    "learned_router_v3",
    "learned_router_v4",
    "learned_router_v5",
}


# ---------------------------------------------------------------------------
# Simulated picks from v3 / v4 internal predictors
# ---------------------------------------------------------------------------

def _simulate_v3_pick(
    fp_v3_norm: np.ndarray,
    pool: List[str],
    v3_regressors: Dict[str, Any],
    fallback: str,
) -> Tuple[str, Dict[str, float]]:
    """Replay v3's prediction step on a (already normalised) v3 fingerprint.

    Returns ``(picked_algo, predicted_aris)``. Mirrors the argmax-over-
    predicted-ARI logic in ``Learned_router_v3.fit_predict``. Skips the
    rank-vote fallback because at training time we always have
    regressors available.
    """
    fp_2d = fp_v3_norm[None, :]
    preds: Dict[str, float] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for algo in pool:
            reg = v3_regressors.get(algo)
            if reg is None:
                continue
            try:
                val = float(reg.predict(fp_2d)[0])
                if not np.isnan(val):
                    preds[algo] = val
            except Exception:
                continue
    if not preds:
        return fallback, preds
    pick = max(preds.items(), key=lambda kv: kv[1])[0]
    return pick, preds


def _simulate_v4_pick(
    fp_v4_norm: np.ndarray,
    fp_v3_norm: np.ndarray,
    pool: List[str],
    v4_regressors: Dict[str, Any],
    v3_regressors: Dict[str, Any],
    disagreement_threshold: float,
    fallback: str,
) -> Tuple[str, Dict[str, float], Dict[str, float], bool]:
    """Replay v4's prediction step (including v3 ensemble guardrail).

    Returns ``(picked_algo, v4_preds, v3_preds, fell_back_to_v3)``.
    Mirrors the argmax-then-guardrail logic in
    ``Learned_router_v4.fit_predict``.
    """
    fp4_2d = fp_v4_norm[None, :]
    v4_preds: Dict[str, float] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for algo in pool:
            reg = v4_regressors.get(algo)
            if reg is None:
                continue
            try:
                val = float(reg.predict(fp4_2d)[0])
                if not np.isnan(val):
                    v4_preds[algo] = val
            except Exception:
                continue

    _, v3_preds = _simulate_v3_pick(fp_v3_norm, pool, v3_regressors, fallback)

    if v4_preds:
        v4_pick = max(v4_preds.items(), key=lambda kv: kv[1])[0]
    else:
        v4_pick = fallback

    chosen = v4_pick
    fell_back = False
    if v3_preds:
        v3_pick = max(v3_preds.items(), key=lambda kv: kv[1])[0]
        if v4_pick in v3_preds:
            gap = float(v3_preds[v3_pick] - v3_preds[v4_pick])
            if gap > disagreement_threshold and v3_pick != v4_pick:
                chosen = v3_pick
                fell_back = True
    return chosen, v4_preds, v3_preds, fell_back


# ---------------------------------------------------------------------------
# Build the meta training set + classifier
# ---------------------------------------------------------------------------

def _load_training_data_v5() -> Optional[Dict[str, Any]]:
    """Build the v5 meta training set.

    For each historical task we simulate v3's pick and v4's pick using
    their cached regressors (no clustering is run), look the picks'
    actual recorded ARIs up in the row, and emit a labelled example
    ``(fingerprint, v4_better)`` for the classifier.

    Returns a dict with keys:
        F           : (N_unique, 20) normalised feature matrix
        y           : (N_unique,) binary labels (1 if v4 picked better)
        feature_names, mean, std, n_total, n_unique, n_v4_better,
        n_v3_only_fallback (count of tasks where v4 == v3 — they
        contribute ``v4_better = 0`` since picks are identical).

    Returns ``None`` if results.json is missing or empty.
    """
    global _TRAINING_CACHE_V5
    if _TRAINING_CACHE_V5 is not None:
        return _TRAINING_CACHE_V5

    repo_root = pathlib.Path(__file__).resolve().parents[3]
    results_path = repo_root / "docs" / "data" / "results.json"
    if not results_path.exists():
        _TRAINING_CACHE_V5 = None
        return None

    try:
        rows = json.loads(results_path.read_text())
    except Exception:
        _TRAINING_CACHE_V5 = None
        return None

    # Group by task identity (same key v3 uses).
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
            "outlier_extremity": r.get("outlier_extremity"),
        }

    if not by_task:
        _TRAINING_CACHE_V5 = None
        return None

    # We need v3's training cache (for v3 normalisation + regressors) and
    # v4's training cache (for v4 normalisation + regressors). Loading
    # them here primes the module-level caches in those routers, but
    # that's fine — they would have been loaded on the first inference
    # call anyway, and loading them now keeps training self-contained.
    from .learned_router_v3 import (
        _fingerprint_v3,
        _load_regressors_v3,
        _load_training_data_v3,
    )
    from .learned_router_v4 import _load_regressors_v4, _load_training_data_v4

    F3_norm, _, meta3 = _load_training_data_v3()
    F4_norm, _, meta4 = _load_training_data_v4()
    if meta3 is None or meta4 is None:
        _TRAINING_CACHE_V5 = None
        return None

    v3_regs = _load_regressors_v3() or {}
    v4_regs = _load_regressors_v4() or {}

    # Candidate pool is the union of algorithms seen in the historical
    # rows, minus the blocked router variants. Mirrors how v3/v4 build
    # their pools internally.
    seen: set = set()
    for algos in by_task.values():
        seen.update(algos.keys())
    pool = sorted(seen - _BLOCKED_ROUTERS_V5)
    fallback = "pwcc_diverse"

    # Walk each task: simulate picks, look up actual ARIs, compute label.
    feature_names_v4 = meta4["feature_names"]
    mean4 = meta4["mean"]
    std4 = meta4["std"]
    feature_names_v3 = meta3["feature_names"]
    mean3 = meta3["mean"]
    std3 = meta3["std"]

    fps: List[np.ndarray] = []      # raw v4 fingerprints (unnormalised)
    v3_ari: List[float] = []
    v4_ari: List[float] = []
    n_total = 0
    n_excluded = 0
    n_v3_eq_v4 = 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for key, algo_ari in by_task.items():
            meta = task_meta[key]
            gen_result = _regenerate_task(meta)
            if gen_result is None:
                n_excluded += 1
                continue
            try:
                X, _ = gen_result
            except Exception:
                n_excluded += 1
                continue

            try:
                fp = _fingerprint_v4(
                    X,
                    meta["k_target"],
                    dataset_id=meta.get("dataset_id"),
                    outlier_extremity=meta.get("outlier_extremity"),
                )
                fp_v3 = _fingerprint_v3(X, meta["k_target"])
            except Exception:
                n_excluded += 1
                continue

            try:
                fp_v4_vec = np.array(
                    [fp[name] for name in feature_names_v4], dtype=np.float64
                )
                fp_v4_norm = (fp_v4_vec - mean4) / std4
                fp_v3_vec = np.array(
                    [fp_v3[name] for name in feature_names_v3], dtype=np.float64
                )
                fp_v3_norm = (fp_v3_vec - mean3) / std3
            except Exception:
                n_excluded += 1
                continue

            try:
                v3_pick, _ = _simulate_v3_pick(
                    fp_v3_norm, pool, v3_regs, fallback
                )
                v4_pick, _, _, _ = _simulate_v4_pick(
                    fp_v4_norm, fp_v3_norm, pool, v4_regs, v3_regs,
                    disagreement_threshold=0.05, fallback=fallback,
                )
            except Exception:
                warnings.warn(
                    f"learned_router_v5: simulator failed on task "
                    f"{key}, excluding"
                )
                n_excluded += 1
                continue

            # Look up the picks' actual ARIs in the row. If a pick isn't
            # in the row (e.g., the algorithm wasn't run on this task)
            # we treat its ARI as NaN -> exclude the task.
            v3_pick_ari = algo_ari.get(v3_pick)
            v4_pick_ari = algo_ari.get(v4_pick)
            if v3_pick_ari is None or v4_pick_ari is None:
                n_excluded += 1
                continue

            n_total += 1
            if v3_pick == v4_pick:
                n_v3_eq_v4 += 1
            fps.append(fp_v4_vec)
            v3_ari.append(float(v3_pick_ari))
            v4_ari.append(float(v4_pick_ari))

    if not fps:
        _TRAINING_CACHE_V5 = None
        return None

    # Deduplicate by fingerprint hash; aggregate by median ARI.
    fps_arr = np.array(fps, dtype=np.float64)
    v3_ari_arr = np.array(v3_ari, dtype=np.float64)
    v4_ari_arr = np.array(v4_ari, dtype=np.float64)

    groups: Dict[bytes, List[int]] = defaultdict(list)
    for i, row in enumerate(fps_arr):
        # Round to a stable hash key (small float noise from regenerate
        # could otherwise split identical fingerprints).
        groups[row.round(8).tobytes()].append(i)

    F_unique: List[np.ndarray] = []
    y_unique: List[int] = []
    for _, idxs in groups.items():
        F_unique.append(fps_arr[idxs[0]])  # all rows in the group are equal
        v3_med = float(np.median(v3_ari_arr[idxs]))
        v4_med = float(np.median(v4_ari_arr[idxs]))
        y_unique.append(1 if v4_med > v3_med else 0)

    F = np.vstack(F_unique)
    F_norm = (F - mean4) / std4   # reuse v4's normalisation for consistency
    y = np.array(y_unique, dtype=np.int64)

    _TRAINING_CACHE_V5 = {
        "F": F_norm,
        "y": y,
        "feature_names": feature_names_v4,
        "mean": mean4,
        "std": std4,
        "n_total": int(n_total),
        "n_unique": int(F.shape[0]),
        "n_v4_better": int(y.sum()),
        "n_v3_eq_v4": int(n_v3_eq_v4),
        "n_excluded": int(n_excluded),
    }
    return _TRAINING_CACHE_V5


def _load_classifier_v5() -> Optional[Any]:
    """Train the stacker's RandomForest classifier.

    Returns the fitted classifier, or ``None`` if there is only one
    class in the training labels (in which case callers should fall
    back to always dispatching to v3).
    """
    global _CLASSIFIER_V5
    if _CLASSIFIER_V5 is not None:
        return _CLASSIFIER_V5

    cache = _load_training_data_v5()
    if cache is None:
        return None

    F = cache["F"]
    y = cache["y"]
    if len(set(y.tolist())) < 2:
        # Only one class observed; cannot train a discriminative stacker.
        warnings.warn(
            "learned_router_v5: training labels contain only one class "
            f"(n_v4_better={int(y.sum())}/{len(y)}). "
            "Stacker will degenerate to always-v3."
        )
        return None

    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(
        n_estimators=50,
        max_depth=4,
        random_state=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(F, y)
    _CLASSIFIER_V5 = clf
    return _CLASSIFIER_V5


# ---------------------------------------------------------------------------
# Algorithm class
# ---------------------------------------------------------------------------

@register
class Learned_router_v5(Algorithm):
    """Stacker over v3 (kNN-ARI) and v4 (GBR-ARI) routers.

    At inference, computes the v4 20-feature fingerprint, predicts
    ``use_v4`` via a RandomForestClassifier trained on historical wins,
    and forwards ``fit_predict`` to the chosen inner router. Carries
    the stacker's confidence in ``extra``.
    """

    def __init__(
        self,
        fallback: str = "pwcc_diverse",
        **kwargs: Any,
    ) -> None:
        self.name = "learned_router_v5"
        self.fallback = fallback
        # Inner routers are instantiated lazily.
        self._inner_v3: Optional[Algorithm] = None
        self._inner_v4: Optional[Algorithm] = None

    def _get_inner(self, which: str) -> Algorithm:
        # Imported lazily so module init doesn't double-load v3/v4
        # training data.
        if which == "v3":
            if self._inner_v3 is None:
                from .learned_router_v3 import Learned_router_v3
                self._inner_v3 = Learned_router_v3()
            return self._inner_v3
        if which == "v4":
            if self._inner_v4 is None:
                from .learned_router_v4 import Learned_router_v4
                self._inner_v4 = Learned_router_v4()
            return self._inner_v4
        raise ValueError(f"unknown inner router: {which}")

    def _fallback_dispatch(
        self, X: np.ndarray, k: Optional[int]
    ) -> AlgoResult:
        """Used when no v5 training data is available."""
        cls = base_algos.ALGO_REGISTRY.get(self.fallback)
        if cls is None:
            # Last-ditch: try v3 even though we have no training data.
            return self._get_inner("v3").fit_predict(X, k=k)
        return cls().fit_predict(X, k=k)

    def fit_predict(
        self, X: np.ndarray, k: Optional[int] = None
    ) -> AlgoResult:
        cache = _load_training_data_v5()

        if cache is None:
            inner = self._fallback_dispatch(X, k)
            return AlgoResult(
                labels=inner.labels,
                extra={
                    "router": "learned_stacker_v5",
                    "chose_router": None,
                    "reason": "no_training_data",
                    **(inner.extra or {}),
                },
                trajectory=inner.trajectory or [],
            )

        # Compute the v4 fingerprint (20 features). v5 doesn't accept
        # dataset_id/outlier_extremity in its public API (it's a
        # registered algo and the registry calls fit_predict(X, k)
        # only), so the dataset-meta features default to 0 — same
        # convention v4 uses for unknown inputs.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fp = _fingerprint_v4(X, k)
        fp_vec = np.array(
            [fp[name] for name in cache["feature_names"]],
            dtype=np.float64,
        )
        fp_norm = (fp_vec - cache["mean"]) / cache["std"]

        clf = _load_classifier_v5()

        # Stacker decision -------------------------------------------------
        if clf is None:
            chose_router = "v3"
            proba_v4 = 0.0
            reason = "classifier_unavailable_default_v3"
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                proba = clf.predict_proba(fp_norm[None, :])[0]
            # classes_ ordering: 0 = use_v3, 1 = use_v4.
            classes = clf.classes_.tolist()
            if 1 in classes:
                proba_v4 = float(proba[classes.index(1)])
            else:
                proba_v4 = 0.0
            chose_router = "v4" if proba_v4 > 0.5 else "v3"
            reason = "stacker_prediction"

        # Trajectory: fingerprint + stacker_decision, then inner shifted +2.
        trajectory: List[Step] = [
            Step(
                step_idx=0,
                cost=0.0,
                delta_cost=None,
                accepted=True,
                action={"type": "compute_fingerprint"},
                state={
                    "fingerprint": {kk: float(vv) for kk, vv in fp.items()},
                    "n_features": int(len(fp)),
                },
            ),
            Step(
                step_idx=1,
                cost=0.0,
                delta_cost=None,
                accepted=True,
                action={
                    "type": "stacker_decision",
                    "chose_router": chose_router,
                    "reason": reason,
                },
                state={
                    "v5_classifier_confidence": float(proba_v4),
                    "chose_router": chose_router,
                    "n_meta_training_tasks": int(cache["n_unique"]),
                    "n_v4_better": int(cache["n_v4_better"]),
                },
            ),
        ]

        # Forward dispatch -------------------------------------------------
        inner = self._get_inner(chose_router).fit_predict(X, k=k)

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
                "router": "learned_stacker_v5",
                "chose_router": chose_router,
                "v5_classifier_confidence": float(proba_v4),
                "reason": reason,
                "n_meta_training_tasks": int(cache["n_unique"]),
                "n_v4_better": int(cache["n_v4_better"]),
                "inner_extra": dict(inner.extra or {}),
                **(inner.extra or {}),
            },
            trajectory=trajectory,
        )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
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

    cache = _load_training_data_v5()
    if cache is not None:
        print(
            f"meta training set: n_unique={cache['n_unique']}  "
            f"n_v4_better={cache['n_v4_better']}/{cache['n_unique']}  "
            f"n_v3_eq_v4={cache['n_v3_eq_v4']}  "
            f"n_total={cache['n_total']}  "
            f"n_excluded={cache['n_excluded']}"
        )
    else:
        print("meta training set: unavailable")

    router = Learned_router_v5()
    for name, (X, y) in cases:
        k_target = int(len(np.unique(y[y >= 0])))
        res = router.fit_predict(X, k=k_target)
        mask = y >= 0
        if mask.sum() == 0:
            ari = float("nan")
        else:
            ari = adjusted_rand_score(y[mask], res.labels[mask])
        chose_router = res.extra.get("chose_router")
        conf = res.extra.get("v5_classifier_confidence", float("nan"))
        # The inner router's own "chose" is the algorithm it dispatched
        # to (e.g., "pwcc_diverse"). v3 puts it under "chose"; v4 too.
        inner_extra = res.extra.get("inner_extra", {})
        inner_chose = inner_extra.get("chose")
        print(
            f"{name:18s}  v5_chose_router={chose_router}  "
            f"proba_v4={conf:.3f}  "
            f"inner_chose={inner_chose}  "
            f"ari={ari:.3f}"
        )
