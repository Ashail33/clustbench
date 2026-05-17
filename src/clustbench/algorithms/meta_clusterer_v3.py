"""META-CLUSTERER v3: run v1 and v2 routing in parallel, probe on disagreement.

This is the third iteration of META-CLUSTERER. Where v1 hand-codes four
fingerprint rules and v2 swaps in a convexity / modularity non-convex
detector, v3 deliberately **runs both routers** and only commits to a
single base algorithm after reconciling them:

* If v1 and v2 choose the *same* base algorithm, route once - no probe.
* If they *disagree*, fit each candidate on a 20% random subsample,
  compute ``sklearn.metrics.silhouette_score`` on the subsample
  predictions, and dispatch the full dataset to whichever candidate
  wins.

The motivation: v1 is conservative (its eigengap rule never fires for
clean circles, so circles fall through to the convex default branch);
v2 is sharper on circles via the convexity ratio but still has its own
borderline cases. The two routers' agreement set is a strong signal -
when they agree, we commit; when they disagree, an evidence-based tie
break (silhouette on a cheap subsample) picks the better of the two.

Trajectory steps emitted, in order:

1. ``fingerprint_v1`` - v1's chosen algorithm and its fingerprint dict.
2. ``fingerprint_v2`` - v2's chosen algorithm and its fingerprint dict.
3. ``probe_or_agree`` - either ``{"type": "agree", "chose": ...}`` (when
   no probe is needed) or ``{"type": "probe", "v1_choice": ...,
   "v2_choice": ..., "v1_sil": ..., "v2_sil": ..., "winner": ...}``.
4. ... the chosen base algorithm's own trajectory steps, with
   ``step_idx`` shifted to continue past the routing steps.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from . import base as base_algos
from .base import Algorithm, AlgoResult, Step, register

# Reuse the routing helpers from v1 and v2. These are private module-level
# functions today; importing them keeps v3 in lock-step with whatever
# fingerprint / routing logic v1 and v2 currently use without duplicating
# code or risking drift.
from .meta_clusterer import _fingerprints as _v1_fingerprints
from .meta_clusterer import _route as _v1_route
from .meta_clusterer_v2 import _fingerprints as _v2_fingerprints
from .meta_clusterer_v2 import _route as _v2_route
from .meta_clusterer_v2 import _params_for as _v2_params_for


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _params_for(algo: str) -> Dict[str, Any]:
    """Per-algorithm hyper-parameters for v3 dispatch.

    Reuses v2's per-algo defaults (notably ``spectral.n_neighbors=10`` to
    match the affinity graph used by the fingerprints). Falls back to no
    overrides for everything else.
    """
    return dict(_v2_params_for(algo))


def _run_algo(algo: str, X: np.ndarray, k: int) -> AlgoResult:
    cls = base_algos.ALGO_REGISTRY.get(algo)
    if cls is None:
        cls = base_algos.ALGO_REGISTRY["kmeans"]
        algo = "kmeans"
    return cls(**_params_for(algo)).fit_predict(X, k=k)


def _silhouette_of(X: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette score, with the standard fallback for degenerate cases.

    sklearn requires at least 2 distinct labels and < n labels. If the
    candidate collapsed to a single cluster (or labels are all noise),
    return ``-1`` so the other candidate wins the probe.
    """
    from sklearn.metrics import silhouette_score

    labels = np.asarray(labels)
    valid = labels[labels >= 0]
    uniq = np.unique(valid)
    if len(uniq) < 2 or len(uniq) >= len(labels):
        return -1.0
    try:
        return float(silhouette_score(X, labels))
    except Exception:
        return -1.0


def _probe(
    v1_algo: str, v2_algo: str, X: np.ndarray, k: int, seed: int
) -> Tuple[str, float, float]:
    """Fit ``v1_algo`` and ``v2_algo`` on a 20%-or-500-point subsample.

    Returns ``(winner, v1_sil, v2_sil)``. Whichever silhouette is higher
    on the subsample wins; if both are -1 (both degenerate), v2 wins by
    convention (v2 is the more recent / generally stronger router).
    """
    n = X.shape[0]
    sub_size = min(int(round(0.2 * n)), 500)
    sub_size = max(sub_size, 2)
    sub_size = min(sub_size, n)
    rng = np.random.default_rng(seed)
    if sub_size < n:
        idx = rng.choice(n, size=sub_size, replace=False)
        Xs = X[idx]
    else:
        Xs = X

    try:
        r1 = _run_algo(v1_algo, Xs, k)
        v1_sil = _silhouette_of(Xs, np.asarray(r1.labels))
    except Exception:
        v1_sil = -1.0
    try:
        r2 = _run_algo(v2_algo, Xs, k)
        v2_sil = _silhouette_of(Xs, np.asarray(r2.labels))
    except Exception:
        v2_sil = -1.0

    if v2_sil >= v1_sil:
        winner = v2_algo
    else:
        winner = v1_algo
    return winner, float(v1_sil), float(v2_sil)


# -----------------------------------------------------------------------------
# Public algorithm
# -----------------------------------------------------------------------------

@register
class Meta_clusterer_v3(Algorithm):
    """Parallel-routing meta-clusterer.

    Computes both v1's and v2's routing decisions on the same data. If
    they agree, dispatches once to that algorithm; if they disagree,
    fits both candidates on a 20% random subsample and dispatches the
    full data to whichever yielded a higher silhouette.

    Parameters
    ----------
    seed : int
        Random seed for fingerprint sub-sampling and the probe sub-sample.
    """

    def __init__(self, seed: int = 0, **kwargs: Any) -> None:
        self.name = "meta_clusterer_v3"
        self.seed = int(seed)

    def fit_predict(self, X: np.ndarray, k: int | None = None) -> AlgoResult:
        assert k is not None, "k required"
        X = np.asarray(X)
        k = int(k)

        # Step 0: v1 fingerprints + routing.
        fp_v1 = _v1_fingerprints(X, k)
        v1_choice = _v1_route(fp_v1)

        # Step 1: v2 fingerprints + routing. v2's _route returns a tuple
        # (rule_name, chosen_algo, confidence, alternatives).
        fp_v2 = _v2_fingerprints(X, k, seed=self.seed)
        _v2_rule, v2_choice, _v2_conf, _v2_alts = _v2_route(fp_v2)

        trajectory: List[Step] = [
            Step(
                step_idx=0,
                cost=0.0,
                delta_cost=None,
                accepted=True,
                action={"type": "fingerprint_v1", "chose": v1_choice},
                state={"v1_choice": v1_choice, "v1_fingerprint": dict(fp_v1)},
            ),
            Step(
                step_idx=1,
                cost=0.0,
                delta_cost=None,
                accepted=True,
                action={"type": "fingerprint_v2", "chose": v2_choice},
                state={"v2_choice": v2_choice, "v2_fingerprint": dict(fp_v2)},
            ),
        ]

        # Step 2: agree or probe.
        probed = False
        v1_sil: float | None = None
        v2_sil: float | None = None
        if v1_choice == v2_choice:
            chose = v1_choice
            trajectory.append(
                Step(
                    step_idx=2,
                    cost=0.0,
                    delta_cost=None,
                    accepted=True,
                    action={"type": "agree", "chose": chose},
                    state={"chose": chose},
                )
            )
        else:
            probed = True
            winner, v1_sil, v2_sil = _probe(
                v1_choice, v2_choice, X, k, seed=self.seed
            )
            chose = winner
            trajectory.append(
                Step(
                    step_idx=2,
                    cost=0.0,
                    delta_cost=None,
                    accepted=True,
                    action={
                        "type": "probe",
                        "v1_choice": v1_choice,
                        "v2_choice": v2_choice,
                        "v1_sil": float(v1_sil),
                        "v2_sil": float(v2_sil),
                        "winner": winner,
                    },
                    state={"chose": chose},
                )
            )

        # Dispatch on the full data. Fall back to kmeans if the chosen
        # algorithm explodes for any reason.
        try:
            result = _run_algo(chose, X, k)
        except Exception:
            chose = "kmeans"
            result = _run_algo("kmeans", X, k)

        # Forward the chosen algo's trajectory with shifted step_idx.
        base_offset = len(trajectory)
        for sub_step in (result.trajectory or []):
            trajectory.append(
                Step(
                    step_idx=base_offset + int(sub_step.step_idx),
                    cost=float(sub_step.cost),
                    delta_cost=(
                        None
                        if sub_step.delta_cost is None
                        else float(sub_step.delta_cost)
                    ),
                    accepted=bool(sub_step.accepted),
                    action=dict(sub_step.action or {}),
                    state=dict(sub_step.state or {}),
                )
            )

        extra: Dict[str, Any] = {
            "v1_choice": v1_choice,
            "v2_choice": v2_choice,
            "chose": chose,
            "probed": probed,
            "v1_fingerprint": dict(fp_v1),
            "v2_fingerprint": dict(fp_v2),
        }
        if probed:
            extra["v1_sil"] = float(v1_sil) if v1_sil is not None else None
            extra["v2_sil"] = float(v2_sil) if v2_sil is not None else None
        # Merge underlying algorithm's extras under a namespaced key on
        # collision so we don't shadow v3's bookkeeping.
        for key, value in (result.extra or {}).items():
            if key in extra:
                extra[f"{chose}.{key}"] = value
            else:
                extra[key] = value

        labels = np.asarray(result.labels, dtype=np.int64)
        return AlgoResult(labels=labels, extra=extra, trajectory=trajectory)
