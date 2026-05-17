"""Learned routing meta-algorithm — seventh iteration (v3/v6 stacker).

`learned_router_v7` is a *stacker* over `learned_router_v3` and
`learned_router_v6`. It picks one of the two routers on a per-task basis
using a learned classifier over the 20-feature v4 fingerprint (17 v3
features + 3 dataset-meta features).

Why this and not v5
-------------------
v5 stacked over {v3, v4} and degenerated to always-v3 because v4 never
strictly beat v3 on any benchmark task — same fingerprint, same
similarity-to-history paradigm, no complementarity to exploit. v6 is
architecturally different: it asks "what's actually working on this
data right now?" by subsampling and probing, not "what worked on data
that looks like this in history?". On the existing 52-task benchmark
v6 strictly wins on exactly one task (``inverse_pca`` with seed=3,
where v3 gets ARI 0.00 and v6 gets 0.38) — small, but nonzero, so
unlike {v3, v4} the {v3, v6} pair has at least one labelled win the
classifier can learn to recognise.

Training procedure
------------------
1. Read ``docs/data/results.json`` and group rows by task identity
   (dataset_id, n_samples, n_features, k_target, outliers, noise,
   density, seed). For each task, regenerate (X, y) from its DataSpec
   and compute the 20-feature v4 fingerprint.
2. Both ``learned_router_v3`` and ``learned_router_v6`` ran on the
   benchmark — their per-task ARIs are stored in the results rows
   themselves. We don't simulate; we just look up:
       v3_ari = row[task]['learned_router_v3'].ari
       v6_ari = row[task]['learned_router_v6'].ari
   Tasks missing either router are skipped.
3. Label: ``label = 1 if v6_ari > v3_ari + 0.02 else 0``. The 0.02
   margin matches the spec and screens out noise-level ties.
4. Train ``RandomForestClassifier(n_estimators=50, max_depth=4,
   random_state=0)`` on (fingerprint, label).

If the training labels contain only one class (likely v3-everywhere)
the classifier can't discriminate and v7 falls back to always-v3.

Inference
---------
Compute the input's 20-feature fingerprint, predict ``use_v6_proba``,
dispatch to v6 if ``> 0.5`` else v3. Trajectory prepends two records
(``compute_fingerprint`` + ``stacker_decision``) before the inner
router's trajectory, shifted by +2.

Edge cases
----------
- v6 not in registry: fall back to v3.
- ``docs/data/results.json`` missing or empty: fall back to v3.

Dispatch blocking
-----------------
``_BLOCKED_ROUTERS_V7`` blocks every router variant (v1-v7 and v6b)
defence-in-depth in case an inner router's own decision tries to
recurse.

Caches
------
Module-level caches ``_TRAINING_CACHE_V7`` and ``_CLASSIFIER_V7`` mean
the meta training set is built once per process. The two inner routers
(``Learned_router_v3`` and ``Learned_router_v6``) are instantiated
lazily on the first ``fit_predict`` call.
"""

from __future__ import annotations

import json
import pathlib
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np

from . import base as base_algos
from .base import Algorithm, AlgoResult, Step, register
from .learned_router import _regenerate_task
from .learned_router_v4 import _fingerprint_v4


# Module-level caches: built once per process.
_TRAINING_CACHE_V7: Optional[Dict[str, Any]] = None
_CLASSIFIER_V7: Optional[Any] = None

# All router variants are blocked as dispatch targets. v7 only forwards
# to v3 / v6 by construction, but this is defence-in-depth.
_BLOCKED_ROUTERS_V7 = {
    "learned_router",
    "learned_router_v2",
    "learned_router_v3",
    "learned_router_v4",
    "learned_router_v5",
    "learned_router_v6",
    "learned_router_v6b",
    "learned_router_v7",
}

# Margin above which v6 must beat v3 to be labelled the better choice.
_V6_WIN_MARGIN = 0.02


# ---------------------------------------------------------------------------
# Build the meta training set + classifier
# ---------------------------------------------------------------------------

def _load_training_data_v7() -> Optional[Dict[str, Any]]:
    """Build the v7 meta training set.

    For each historical task, look up the actual ARIs of
    ``learned_router_v3`` and ``learned_router_v6`` in the results row,
    regenerate (X, y), compute the 20-feature v4 fingerprint, and emit
    a labelled example ``(fingerprint, v6_better)``.

    Returns a dict with keys:
        F           : (N_unique, 20) normalised feature matrix
        y           : (N_unique,) binary labels (1 if v6 strictly better)
        feature_names, mean, std, n_total, n_unique, n_v6_better,
        n_excluded.

    Returns ``None`` if results.json is missing/empty or no usable rows.
    """
    global _TRAINING_CACHE_V7
    if _TRAINING_CACHE_V7 is not None:
        return _TRAINING_CACHE_V7

    repo_root = pathlib.Path(__file__).resolve().parents[3]
    results_path = repo_root / "docs" / "data" / "results.json"
    if not results_path.exists():
        _TRAINING_CACHE_V7 = None
        return None

    try:
        rows = json.loads(results_path.read_text())
    except Exception:
        _TRAINING_CACHE_V7 = None
        return None

    # Group by task identity (same key v5 uses).
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
        _TRAINING_CACHE_V7 = None
        return None

    # We piggy-back on v4's normalisation parameters so the input
    # fingerprint at inference time is normalised consistently with
    # what the classifier was trained on.
    from .learned_router_v4 import _load_training_data_v4

    _, _, meta4 = _load_training_data_v4()
    if meta4 is None:
        _TRAINING_CACHE_V7 = None
        return None

    feature_names = meta4["feature_names"]
    mean4 = meta4["mean"]
    std4 = meta4["std"]

    fps: List[np.ndarray] = []      # raw v4 fingerprints (unnormalised)
    v3_ari_list: List[float] = []
    v6_ari_list: List[float] = []
    n_total = 0
    n_excluded = 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for key, algo_ari in by_task.items():
            v3_ari = algo_ari.get("learned_router_v3")
            v6_ari = algo_ari.get("learned_router_v6")
            if v3_ari is None or v6_ari is None:
                n_excluded += 1
                continue

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
            except Exception:
                n_excluded += 1
                continue

            try:
                fp_vec = np.array(
                    [fp[name] for name in feature_names], dtype=np.float64
                )
            except Exception:
                n_excluded += 1
                continue

            n_total += 1
            fps.append(fp_vec)
            v3_ari_list.append(float(v3_ari))
            v6_ari_list.append(float(v6_ari))

    if not fps:
        _TRAINING_CACHE_V7 = None
        return None

    # Deduplicate by fingerprint hash; aggregate by median ARI. Mirrors
    # v5's approach: identical regenerated X's (real-data tasks reused
    # across seeds) shouldn't get repeated examples.
    fps_arr = np.array(fps, dtype=np.float64)
    v3_arr = np.array(v3_ari_list, dtype=np.float64)
    v6_arr = np.array(v6_ari_list, dtype=np.float64)

    groups: Dict[bytes, List[int]] = defaultdict(list)
    for i, row in enumerate(fps_arr):
        groups[row.round(8).tobytes()].append(i)

    F_unique: List[np.ndarray] = []
    y_unique: List[int] = []
    for _, idxs in groups.items():
        F_unique.append(fps_arr[idxs[0]])  # all rows in the group are equal
        v3_med = float(np.median(v3_arr[idxs]))
        v6_med = float(np.median(v6_arr[idxs]))
        y_unique.append(1 if v6_med > v3_med + _V6_WIN_MARGIN else 0)

    F = np.vstack(F_unique)
    F_norm = (F - mean4) / std4
    y = np.array(y_unique, dtype=np.int64)

    _TRAINING_CACHE_V7 = {
        "F": F_norm,
        "y": y,
        "feature_names": feature_names,
        "mean": mean4,
        "std": std4,
        "n_total": int(n_total),
        "n_unique": int(F.shape[0]),
        "n_v6_better": int(y.sum()),
        "n_excluded": int(n_excluded),
    }
    return _TRAINING_CACHE_V7


def _load_classifier_v7() -> Optional[Any]:
    """Train the stacker's RandomForest classifier.

    Returns the fitted classifier, or ``None`` if there is only one
    class in the training labels (in which case callers fall back to
    always dispatching to v3).
    """
    global _CLASSIFIER_V7
    if _CLASSIFIER_V7 is not None:
        return _CLASSIFIER_V7

    cache = _load_training_data_v7()
    if cache is None:
        return None

    F = cache["F"]
    y = cache["y"]
    if len(set(y.tolist())) < 2:
        warnings.warn(
            "learned_router_v7: training labels contain only one class "
            f"(n_v6_better={int(y.sum())}/{len(y)}). "
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
    _CLASSIFIER_V7 = clf
    return _CLASSIFIER_V7


# ---------------------------------------------------------------------------
# Algorithm class
# ---------------------------------------------------------------------------

@register
class Learned_router_v7(Algorithm):
    """Stacker over v3 (fingerprint-kNN) and v6 (subsample-probe) routers.

    At inference, computes the 20-feature v4 fingerprint, predicts
    ``use_v6`` via a RandomForestClassifier trained on historical wins,
    and forwards ``fit_predict`` to the chosen inner router. Carries
    the stacker's confidence in ``extra``.
    """

    def __init__(
        self,
        fallback: str = "pwcc_diverse",
        **kwargs: Any,
    ) -> None:
        self.name = "learned_router_v7"
        self.fallback = fallback
        # Inner routers are instantiated lazily.
        self._inner_v3: Optional[Algorithm] = None
        self._inner_v6: Optional[Algorithm] = None

    def _get_inner(self, which: str) -> Algorithm:
        if which == "v3":
            if self._inner_v3 is None:
                from .learned_router_v3 import Learned_router_v3
                self._inner_v3 = Learned_router_v3()
            return self._inner_v3
        if which == "v6":
            if self._inner_v6 is None:
                from .learned_router_v6 import Learned_router_v6
                self._inner_v6 = Learned_router_v6()
            return self._inner_v6
        raise ValueError(f"unknown inner router: {which}")

    def _fallback_dispatch(
        self, X: np.ndarray, k: Optional[int]
    ) -> AlgoResult:
        """Used when no v7 training data is available."""
        cls = base_algos.ALGO_REGISTRY.get(self.fallback)
        if cls is None:
            return self._get_inner("v3").fit_predict(X, k=k)
        return cls().fit_predict(X, k=k)

    def fit_predict(
        self, X: np.ndarray, k: Optional[int] = None
    ) -> AlgoResult:
        cache = _load_training_data_v7()

        if cache is None:
            inner = self._fallback_dispatch(X, k)
            return AlgoResult(
                labels=inner.labels,
                extra={
                    "router": "learned_stacker_v7",
                    "chose_router": None,
                    "reason": "no_training_data",
                    **(inner.extra or {}),
                },
                trajectory=inner.trajectory or [],
            )

        # Defensive guard: if v6 isn't in the registry for some reason,
        # always dispatch to v3.
        v6_available = "learned_router_v6" in base_algos.ALGO_REGISTRY

        # Compute the v4 fingerprint (20 features). At inference time
        # the dataset-meta features default to 0 since the registry
        # only passes (X, k) — same convention v4/v5 use.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fp = _fingerprint_v4(X, k)
        fp_vec = np.array(
            [fp[name] for name in cache["feature_names"]],
            dtype=np.float64,
        )
        fp_norm = (fp_vec - cache["mean"]) / cache["std"]

        clf = _load_classifier_v7()

        # Stacker decision -------------------------------------------------
        if clf is None:
            chose_router = "v3"
            use_v6_proba = 0.0
            reason = "classifier_unavailable_default_v3"
        elif not v6_available:
            chose_router = "v3"
            use_v6_proba = 0.0
            reason = "v6_not_in_registry_default_v3"
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                proba = clf.predict_proba(fp_norm[None, :])[0]
            # classes_ ordering: 0 = use_v3, 1 = use_v6.
            classes = clf.classes_.tolist()
            if 1 in classes:
                use_v6_proba = float(proba[classes.index(1)])
            else:
                use_v6_proba = 0.0
            chose_router = "v6" if use_v6_proba > 0.5 else "v3"
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
                    "use_v6_proba": float(use_v6_proba),
                    "chose_router": chose_router,
                    "n_meta_training_tasks": int(cache["n_unique"]),
                    "n_v6_better": int(cache["n_v6_better"]),
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
                "router": "learned_stacker_v7",
                "chose_router": chose_router,
                "use_v6_proba": float(use_v6_proba),
                "reason": reason,
                "n_meta_training_tasks": int(cache["n_unique"]),
                "n_v6_better": int(cache["n_v6_better"]),
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

    cache = _load_training_data_v7()
    if cache is not None:
        print(
            f"meta training set: n_unique={cache['n_unique']}  "
            f"n_v6_better={cache['n_v6_better']}/{cache['n_unique']}  "
            f"n_total={cache['n_total']}  "
            f"n_excluded={cache['n_excluded']}"
        )
    else:
        print("meta training set: unavailable")

    router = Learned_router_v7()
    for name, (X, y) in cases:
        k_target = int(len(np.unique(y[y >= 0])))
        res = router.fit_predict(X, k=k_target)
        mask = y >= 0
        if mask.sum() == 0:
            ari = float("nan")
        else:
            ari = adjusted_rand_score(y[mask], res.labels[mask])
        chose_router = res.extra.get("chose_router")
        proba = res.extra.get("use_v6_proba", float("nan"))
        inner_extra = res.extra.get("inner_extra", {})
        inner_chose = inner_extra.get("chose")
        print(
            f"{name:18s}  v7_chose_router={chose_router}  "
            f"use_v6_proba={proba:.3f}  "
            f"inner_chose={inner_chose}  "
            f"ari={ari:.3f}"
        )
