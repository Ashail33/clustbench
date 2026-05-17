"""Learned routing meta-algorithm — v6c (multi-metric probe + cost-weighted).

`learned_router_v6c` is a refinement of `learned_router_v6b` that **adds
computational complexity to the multi-metric probe aggregate**.

Why v6b underperformed (smoke evidence)
---------------------------------------
On moons, v6b's ``kneighbor_purity`` correctly favours spectral, but
spectral loses on the other three metrics (silhouette / calinski /
db_inv). Because v6b averages the four metrics *unweighted*, the
compact-partition algorithms (kmeans / agglomerative) win the aggregate
even though the kn_purity signal is right.

v6c's fix
---------
Two changes, both confined to the metric step:

1. **Add a 5th metric ``cost_inv``.** During the probe, wrap each
   candidate's ``fit_predict`` in ``time.perf_counter()`` and store the
   elapsed wall time. Then::

       cost_inv[algo] = 1 / (1 + wall_time[algo])

   This yields ~1.0 for instant algos, ~0.5 for a 1-second probe, ~0.1
   for a 10-second probe. ``cost_inv`` is min-max normalised across the
   five candidates just like the other metrics.

2. **Switch from unweighted to weighted aggregate.** A constructor
   parameter ``metric_weights`` (default below) sets the per-metric
   weight; the aggregate is the weighted mean of the *available*
   (non-NaN) normalised metrics with their weights renormalised::

       metric_weights = {
           "silhouette":         0.20,
           "calinski_harabasz":  0.15,
           "davies_bouldin_inv": 0.15,
           "kneighbor_purity":   0.30,  # boosted — handles non-convex
           "cost_inv":           0.20,
       }

   The ``kneighbor_purity`` weight is boosted to 0.30 because it is the
   single best signal for non-convex data; ``cost_inv`` enters at 0.20
   as a tie-breaker that prefers cheap algorithms when quality is
   roughly equal.

Dispatch + fallback follow the v6c spec: re-run the winning algorithm
on the full X; fall back to v3 if the chosen aggregate < 0.10 or every
metric is NaN for every candidate. Unlike v6b, the optional
"v3-hint" pre-emptive fallback is **off by default** (the spec only
mentions the two conditions above). Pass ``use_v3_hint=True`` to opt
in.

Recursion is prevented by ``_BLOCKED_ROUTERS_V6C``.
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import base as base_algos
from .base import Algorithm, AlgoResult, Step, register


# All router variants are blocked as dispatch targets to prevent
# recursive routing chains.
_BLOCKED_ROUTERS_V6C = {
    "learned_router",
    "learned_router_v2",
    "learned_router_v3",
    "learned_router_v4",
    "learned_router_v5",
    "learned_router_v6",
    "learned_router_v6b",
    "learned_router_v6c",
}

# Candidate algorithms probed at inference time. Order is fixed so the
# trajectory and extra dictionaries are reproducible.
_PROBE_CANDIDATES: List[str] = [
    "kmeans",
    "spectral",
    "gmm",
    "agglomerative",
    "dbscan_auto",
]

# Aggregate threshold below which the probe is considered indecisive
# and we fall back to v3.
_AGGREGATE_DECISIVE_THRESHOLD = 0.10

# If v3 predicts ARI < this for v6c's pick, prefer v3's pick instead.
_V3_HINT_MIN_PREDICTED_ARI = 0.30

# Default weighting for the 5 probe metrics. Must sum to 1.0.
_DEFAULT_METRIC_WEIGHTS: Dict[str, float] = {
    "silhouette":         0.20,
    "calinski_harabasz":  0.15,
    "davies_bouldin_inv": 0.15,
    "kneighbor_purity":   0.30,
    "cost_inv":           0.20,
}

_METRIC_NAMES = (
    "silhouette",
    "calinski_harabasz",
    "davies_bouldin_inv",
    "kneighbor_purity",
    "cost_inv",
)


# ---------------------------------------------------------------------------
# Metric helpers (silhouette / CH / DB-inv / kneighbor_purity unchanged from v6b)
# ---------------------------------------------------------------------------

def _effective_labels(labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Drop DBSCAN-style noise (-1) and return ``(mask, eff_labels)``."""
    labels = np.asarray(labels)
    mask = labels >= 0
    return mask, labels[mask]


def _silhouette(Xs: np.ndarray, labels: np.ndarray) -> float:
    """Sklearn silhouette on non-noise points. NaN if undefined."""
    from sklearn.metrics import silhouette_score

    mask, eff = _effective_labels(labels)
    if mask.sum() < 2:
        return float("nan")
    unique = np.unique(eff)
    if len(unique) < 2 or len(unique) >= mask.sum():
        return float("nan")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(silhouette_score(Xs[mask], eff))
    except Exception:
        return float("nan")


def _calinski_harabasz(Xs: np.ndarray, labels: np.ndarray) -> float:
    """Sklearn Calinski-Harabasz on non-noise points. NaN if undefined."""
    from sklearn.metrics import calinski_harabasz_score

    mask, eff = _effective_labels(labels)
    if mask.sum() < 3:
        return float("nan")
    unique = np.unique(eff)
    if len(unique) < 2 or len(unique) >= mask.sum():
        return float("nan")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(calinski_harabasz_score(Xs[mask], eff))
    except Exception:
        return float("nan")


def _davies_bouldin_inv(Xs: np.ndarray, labels: np.ndarray) -> float:
    """Davies-Bouldin inverted via ``1 / (1 + db)``. NaN if undefined."""
    from sklearn.metrics import davies_bouldin_score

    mask, eff = _effective_labels(labels)
    if mask.sum() < 2:
        return float("nan")
    unique = np.unique(eff)
    if len(unique) < 2 or len(unique) >= mask.sum():
        return float("nan")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            db = float(davies_bouldin_score(Xs[mask], eff))
        if np.isnan(db) or np.isinf(db):
            return float("nan")
        return 1.0 / (1.0 + db)
    except Exception:
        return float("nan")


def _kneighbor_purity(Xs: np.ndarray, labels: np.ndarray) -> float:
    """Connectivity-aware purity on a small kNN graph over the subsample.

    For each point, compute the fraction of its kNN that share its
    cluster label, then average over all (non-noise) points. ``k`` is
    ``max(3, min(10, m // 20))``.

    This rewards algorithms that respect local connectivity. Spectral
    on moons preserves the manifold structure and scores high; k-means
    cuts straight across the moons and scores low.
    """
    from sklearn.neighbors import NearestNeighbors

    labels = np.asarray(labels)
    m = int(Xs.shape[0])
    if m < 4:
        return float("nan")

    mask, eff = _effective_labels(labels)
    if mask.sum() < 2:
        return float("nan")
    unique = np.unique(eff)
    if len(unique) < 2:
        return float("nan")

    k_nbr = max(3, min(10, m // 20))
    n_neighbors = min(k_nbr + 1, m)
    if n_neighbors < 2:
        return float("nan")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(Xs)
            _, idx = nbrs.kneighbors(Xs)
    except Exception:
        return float("nan")

    neighbour_idx = idx[:, 1:]
    point_labels = labels[:, None]
    neighbour_labels = labels[neighbour_idx]
    matches = (neighbour_labels == point_labels).astype(np.float64)
    point_purity = matches.mean(axis=1)

    if mask.sum() == 0:
        return float("nan")
    return float(point_purity[mask].mean())


def _cost_inv_from_wall_time(wall_time: float) -> float:
    """Convert a probe wall time (seconds) into a higher-is-better score.

    ``1 / (1 + wall_time)``: 1.0 for instant probes, ~0.5 at 1s, ~0.1 at
    10s. We never return NaN here — every successful probe has a wall
    time. (For failed probes we hand back NaN at the call site.)
    """
    if not np.isfinite(wall_time) or wall_time < 0:
        return float("nan")
    return float(1.0 / (1.0 + float(wall_time)))


def _compute_quality_metrics(
    Xs: np.ndarray, labels: np.ndarray
) -> Dict[str, float]:
    """Return the four quality metrics for one probe (cost_inv is added
    by the caller because it depends on the wall time, not the labels)."""
    return {
        "silhouette": _silhouette(Xs, labels),
        "calinski_harabasz": _calinski_harabasz(Xs, labels),
        "davies_bouldin_inv": _davies_bouldin_inv(Xs, labels),
        "kneighbor_purity": _kneighbor_purity(Xs, labels),
    }


# ---------------------------------------------------------------------------
# Probe runner
# ---------------------------------------------------------------------------

def _run_probe(
    algo_name: str, Xs: np.ndarray, k: Optional[int]
) -> Dict[str, Any]:
    """Run one probe and return ``{metrics, wall_time, error}``.

    Always returns a dict (never raises). ``metrics`` is the five-metric
    dict (cost_inv included), with NaN for any metric that couldn't be
    computed; ``error`` is the exception's class name or ``None``.
    """
    cls = base_algos.ALGO_REGISTRY.get(algo_name)
    nan_metrics = {
        "silhouette": float("nan"),
        "calinski_harabasz": float("nan"),
        "davies_bouldin_inv": float("nan"),
        "kneighbor_purity": float("nan"),
        "cost_inv": float("nan"),
    }
    if cls is None:
        return {
            "metrics": nan_metrics,
            "wall_time": 0.0,
            "error": "algo_not_registered",
        }

    t0 = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = cls().fit_predict(Xs, k=k)
        wall_time = float(time.perf_counter() - t0)
        labels = np.asarray(result.labels)
        metrics = _compute_quality_metrics(Xs, labels)
        metrics["cost_inv"] = _cost_inv_from_wall_time(wall_time)
        return {
            "metrics": metrics,
            "wall_time": wall_time,
            "error": None,
        }
    except Exception as e:  # noqa: BLE001 — whole point is to swallow
        wall_time = float(time.perf_counter() - t0)
        # Failed probe: NaN cost_inv (no usable timing signal for a
        # failure mode — we don't want to reward an algo for crashing
        # fast).
        return {
            "metrics": nan_metrics,
            "wall_time": wall_time,
            "error": type(e).__name__,
        }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate_scores(
    per_algo_metrics: Dict[str, Dict[str, float]],
    metric_weights: Dict[str, float],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    """Min-max normalise each metric column, then take the weighted mean
    per candidate.

    Parameters
    ----------
    per_algo_metrics
        ``{algo: {metric_name: raw_value}}`` — must include all five
        metric names listed in ``_METRIC_NAMES``.
    metric_weights
        ``{metric_name: weight}`` summing to 1.0 over the five known
        metrics. NaN metrics drop out for a given candidate and the
        remaining weights are renormalised for that candidate.

    Returns
    -------
    normalised
        ``{algo: {metric_name: normalised_value_or_nan}}``.
    aggregate
        ``{algo: weighted_mean_of_available_metrics}``. Algos whose
        five metrics are all NaN get ``-inf``.
    """
    algos = list(per_algo_metrics.keys())
    normalised: Dict[str, Dict[str, float]] = {a: {} for a in algos}

    for metric in _METRIC_NAMES:
        col = np.array(
            [per_algo_metrics[a].get(metric, float("nan")) for a in algos],
            dtype=np.float64,
        )
        finite_mask = np.isfinite(col)
        if not finite_mask.any():
            for a in algos:
                normalised[a][metric] = float("nan")
            continue
        finite = col[finite_mask]
        c_min = float(finite.min())
        c_max = float(finite.max())
        if c_max - c_min < 1e-12:
            for a, v in zip(algos, col):
                normalised[a][metric] = 1.0 if np.isfinite(v) else float("nan")
            continue
        for a, v in zip(algos, col):
            if np.isfinite(v):
                normalised[a][metric] = float((v - c_min) / (c_max - c_min))
            else:
                normalised[a][metric] = float("nan")

    aggregate: Dict[str, float] = {}
    for a in algos:
        num = 0.0
        denom = 0.0
        for metric in _METRIC_NAMES:
            v = normalised[a].get(metric, float("nan"))
            w = float(metric_weights.get(metric, 0.0))
            if np.isfinite(v) and w > 0:
                num += w * v
                denom += w
        if denom <= 0:
            aggregate[a] = float("-inf")
        else:
            aggregate[a] = float(num / denom)

    return normalised, aggregate


# ---------------------------------------------------------------------------
# Algorithm class
# ---------------------------------------------------------------------------

@register
class Learned_router_v6c(Algorithm):
    """Subsample-and-probe router with a 5-metric weighted aggregate.

    Identical to v6b except: (1) wall time per probe becomes a 5th
    metric ``cost_inv``; (2) the aggregate is a weighted mean instead
    of an unweighted mean. See module docstring for the rationale.
    """

    def __init__(
        self,
        random_state: int = 42,
        fallback: str = "learned_router_v3",
        use_v3_hint: bool = False,
        metric_weights: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ) -> None:
        self.name = "learned_router_v6c"
        self.random_state = random_state
        self.fallback = fallback
        self.use_v3_hint = bool(use_v3_hint)
        if metric_weights is None:
            self.metric_weights: Dict[str, float] = dict(_DEFAULT_METRIC_WEIGHTS)
        else:
            self.metric_weights = {
                m: float(metric_weights.get(m, 0.0)) for m in _METRIC_NAMES
            }
        # Inner v3 router is instantiated lazily so importing this
        # module doesn't trigger v3's training-data load.
        self._inner_v3: Optional[Algorithm] = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_v3(self) -> Algorithm:
        if self._inner_v3 is None:
            from .learned_router_v3 import Learned_router_v3
            self._inner_v3 = Learned_router_v3()
        return self._inner_v3

    def _subsample_size(self, n: int) -> int:
        """``m = min(200, max(50, int(0.3 * n)))`` — same as v6/v6b."""
        return int(min(200, max(50, int(0.3 * n))))

    def _build_probe_metrics_extra(
        self,
        per_algo_metrics: Dict[str, Dict[str, float]],
        probe_times: Dict[str, float],
        normalised: Dict[str, Dict[str, float]],
        aggregate: Dict[str, float],
    ) -> Dict[str, Dict[str, float]]:
        """Pack raw metrics + cost_inv + wall_time + aggregate into the
        per-algo dict used in ``extra["probe_metrics"]``."""
        out: Dict[str, Dict[str, float]] = {}
        for a, raw in per_algo_metrics.items():
            out[a] = {
                "silhouette": float(raw.get("silhouette", float("nan"))),
                "calinski": float(raw.get("calinski_harabasz", float("nan"))),
                "db_inv": float(raw.get("davies_bouldin_inv", float("nan"))),
                "kn_purity": float(raw.get("kneighbor_purity", float("nan"))),
                "cost_inv": float(raw.get("cost_inv", float("nan"))),
                "wall_time": float(probe_times.get(a, float("nan"))),
                "silhouette_norm":
                    float(normalised[a].get("silhouette", float("nan"))),
                "calinski_norm":
                    float(normalised[a].get("calinski_harabasz", float("nan"))),
                "db_inv_norm":
                    float(normalised[a].get("davies_bouldin_inv", float("nan"))),
                "kn_purity_norm":
                    float(normalised[a].get("kneighbor_purity", float("nan"))),
                "cost_inv_norm":
                    float(normalised[a].get("cost_inv", float("nan"))),
                "aggregate": float(aggregate.get(a, float("-inf"))),
            }
        return out

    def _fallback_to_v3(
        self,
        X: np.ndarray,
        k: Optional[int],
        reason: str,
        prefix_steps: List[Step],
        probe_metrics: Dict[str, Dict[str, float]],
        subsample_size: int,
    ) -> AlgoResult:
        """Run v3 and stitch its trajectory after the v6c prefix."""
        v3 = self._get_v3()
        inner = v3.fit_predict(X, k=k)

        decision = Step(
            step_idx=len(prefix_steps),
            cost=0.0,
            delta_cost=None,
            accepted=True,
            action={
                "type": "aggregate_decision",
                "chose": "learned_router_v3",
                "fallback_used": True,
                "reason": reason,
            },
            state={
                "probe_metrics": probe_metrics,
                "metric_weights": dict(self.metric_weights),
                "fallback_used": True,
            },
        )
        trajectory: List[Step] = list(prefix_steps) + [decision]
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

        chose_inner = (inner.extra or {}).get("chose")
        return AlgoResult(
            labels=inner.labels,
            extra={
                "router": "subsample_probe_v6c",
                "chose": chose_inner if chose_inner else "learned_router_v3",
                "probe_metrics": probe_metrics,
                "metric_weights": dict(self.metric_weights),
                "subsample_size": int(subsample_size),
                "fallback_used": True,
                "fallback_reason": reason,
                "inner_router": "learned_router_v3",
                **(inner.extra or {}),
            },
            trajectory=trajectory,
        )

    def _v3_predicted_ari_for(
        self, algo: str, X: np.ndarray, k: Optional[int]
    ) -> Optional[float]:
        """Query v3's per-algorithm regressor for the predicted ARI of
        ``algo`` on this input. Returns ``None`` on any failure."""
        try:
            from .learned_router_v3 import (
                _fingerprint_v3,
                _load_regressors_v3,
                _load_training_data_v3,
            )

            F_norm, _rows, meta = _load_training_data_v3()
            if F_norm is None or meta is None:
                return None
            regressors = _load_regressors_v3() or {}
            reg = regressors.get(algo)
            if reg is None:
                return None
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fp = _fingerprint_v3(X, k)
            fp_vec = np.array(
                [fp[name] for name in meta["feature_names"]],
                dtype=np.float64,
            )
            fp_norm = (fp_vec - meta["mean"]) / meta["std"]
            pred = float(reg.predict(fp_norm[None, :])[0])
            if np.isnan(pred):
                return None
            return pred
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Main entrypoint
    # ------------------------------------------------------------------

    def fit_predict(
        self, X: np.ndarray, k: Optional[int] = None
    ) -> AlgoResult:
        X = np.asarray(X)
        n = int(X.shape[0])

        # ----- Tiny-input guardrail -----------------------------------
        if n < 50:
            setup = Step(
                step_idx=0,
                cost=0.0,
                delta_cost=None,
                accepted=True,
                action={
                    "type": "probe_setup",
                    "candidates": list(_PROBE_CANDIDATES),
                    "reason": "n_too_small_for_probe",
                },
                state={
                    "n": n,
                    "subsample_size": 0,
                    "candidates": list(_PROBE_CANDIDATES),
                },
            )
            return self._fallback_to_v3(
                X, k,
                reason="n<50_subsample_too_small",
                prefix_steps=[setup],
                probe_metrics={},
                subsample_size=0,
            )

        # ----- Subsample ----------------------------------------------
        m = self._subsample_size(n)
        rng = np.random.default_rng(self.random_state)
        if m >= n:
            sub_idx = np.arange(n)
        else:
            sub_idx = rng.choice(n, size=m, replace=False)
        Xs = X[sub_idx]

        # ----- Setup step ---------------------------------------------
        setup_step = Step(
            step_idx=0,
            cost=0.0,
            delta_cost=None,
            accepted=True,
            action={
                "type": "probe_setup",
                "candidates": list(_PROBE_CANDIDATES),
            },
            state={
                "n": n,
                "subsample_size": int(m),
                "candidates": list(_PROBE_CANDIDATES),
                "metric_weights": dict(self.metric_weights),
                "k": k,
            },
        )
        steps: List[Step] = [setup_step]

        # ----- Run probes ---------------------------------------------
        per_algo_metrics: Dict[str, Dict[str, float]] = {}
        probe_times: Dict[str, float] = {}
        probe_errors: Dict[str, Optional[str]] = {}
        for algo in _PROBE_CANDIDATES:
            res = _run_probe(algo, Xs, k)
            metrics = res["metrics"]
            per_algo_metrics[algo] = metrics
            probe_times[algo] = float(res["wall_time"])
            probe_errors[algo] = res["error"]
            steps.append(
                Step(
                    step_idx=len(steps),
                    cost=0.0,
                    delta_cost=None,
                    accepted=res["error"] is None,
                    action={
                        "type": f"probe_{algo}",
                        "algo": algo,
                        "error": res["error"],
                    },
                    state={
                        "silhouette": float(metrics["silhouette"]),
                        "calinski_harabasz": float(metrics["calinski_harabasz"]),
                        "davies_bouldin_inv": float(metrics["davies_bouldin_inv"]),
                        "kneighbor_purity": float(metrics["kneighbor_purity"]),
                        "cost_inv": float(metrics["cost_inv"]),
                        "wall_time": float(res["wall_time"]),
                    },
                )
            )

        # ----- Aggregate ----------------------------------------------
        normalised, aggregate = _aggregate_scores(
            per_algo_metrics, self.metric_weights
        )
        probe_metrics_extra = self._build_probe_metrics_extra(
            per_algo_metrics, probe_times, normalised, aggregate
        )

        # If every candidate is all-NaN, we have no signal.
        all_neg_inf = all(v == float("-inf") for v in aggregate.values())
        if all_neg_inf:
            return self._fallback_to_v3(
                X, k,
                reason="all_metrics_nan_for_all_candidates",
                prefix_steps=steps,
                probe_metrics=probe_metrics_extra,
                subsample_size=m,
            )

        chosen, chosen_score = max(aggregate.items(), key=lambda kv: kv[1])

        # Low-confidence guardrail.
        if chosen_score < _AGGREGATE_DECISIVE_THRESHOLD:
            return self._fallback_to_v3(
                X, k,
                reason=(
                    f"best_aggregate_{chosen_score:.3f}"
                    f"_below_threshold_{_AGGREGATE_DECISIVE_THRESHOLD}"
                ),
                prefix_steps=steps,
                probe_metrics=probe_metrics_extra,
                subsample_size=m,
            )

        # Optional v3-hint fallback: if v3 thinks the chosen algo will
        # fail badly on this input, use v3's pick instead.
        v3_hint_used = False
        v3_predicted_ari: Optional[float] = None
        if self.use_v3_hint:
            v3_predicted_ari = self._v3_predicted_ari_for(chosen, X, k)
            if (
                v3_predicted_ari is not None
                and v3_predicted_ari < _V3_HINT_MIN_PREDICTED_ARI
            ):
                v3_hint_used = True
                return self._fallback_to_v3(
                    X, k,
                    reason=(
                        f"v3_hint_predicted_ari_{v3_predicted_ari:.3f}"
                        f"_for_{chosen}_below_{_V3_HINT_MIN_PREDICTED_ARI}"
                    ),
                    prefix_steps=steps,
                    probe_metrics=probe_metrics_extra,
                    subsample_size=m,
                )

        # ----- Dispatch decision step ---------------------------------
        steps.append(
            Step(
                step_idx=len(steps),
                cost=float(-chosen_score),
                delta_cost=None,
                accepted=True,
                action={
                    "type": "aggregate_decision",
                    "chose": chosen,
                    "fallback_used": False,
                    "v3_predicted_ari": v3_predicted_ari,
                    "v3_hint_used": v3_hint_used,
                },
                state={
                    "probe_metrics": probe_metrics_extra,
                    "metric_weights": dict(self.metric_weights),
                    "aggregate_scores": {
                        a: float(s) for a, s in aggregate.items()
                    },
                    "chosen_aggregate": float(chosen_score),
                    "fallback_used": False,
                },
            )
        )

        # Block recursive routers — should never happen because chosen
        # is constrained to _PROBE_CANDIDATES, but defence-in-depth.
        if chosen in _BLOCKED_ROUTERS_V6C:
            return self._fallback_to_v3(
                X, k,
                reason=f"chosen_{chosen}_is_blocked_router",
                prefix_steps=steps[:-1],
                probe_metrics=probe_metrics_extra,
                subsample_size=m,
            )

        # ----- Re-run winner on full X --------------------------------
        cls = base_algos.ALGO_REGISTRY.get(chosen)
        if cls is None:
            return self._fallback_to_v3(
                X, k,
                reason=f"chosen_algo_{chosen}_not_in_registry",
                prefix_steps=steps[:-1],
                probe_metrics=probe_metrics_extra,
                subsample_size=m,
            )

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                inner = cls().fit_predict(X, k=k)
        except Exception as e:  # noqa: BLE001
            return self._fallback_to_v3(
                X, k,
                reason=f"chosen_algo_failed_on_full_X_{type(e).__name__}",
                prefix_steps=steps[:-1],
                probe_metrics=probe_metrics_extra,
                subsample_size=m,
            )

        # ----- Stitch inner trajectory --------------------------------
        if inner.trajectory:
            for s in inner.trajectory:
                steps.append(
                    Step(
                        step_idx=len(steps),
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
                "router": "subsample_probe_v6c",
                "chose": chosen,
                "probe_metrics": probe_metrics_extra,
                "metric_weights": dict(self.metric_weights),
                "probe_wall_times": probe_times,
                "probe_errors": {a: e for a, e in probe_errors.items()},
                "subsample_size": int(m),
                "fallback_used": False,
                "chosen_aggregate": float(chosen_score),
                "v3_predicted_ari": v3_predicted_ari,
                "v3_hint_used": v3_hint_used,
                **(inner.extra or {}),
            },
            trajectory=steps,
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

    router = Learned_router_v6c()
    for name, (X, y) in cases:
        k_target = int(len(np.unique(y[y >= 0])))
        res = router.fit_predict(X, k=k_target)
        mask = y >= 0
        if mask.sum() == 0:
            ari = float("nan")
        else:
            ari = adjusted_rand_score(y[mask], res.labels[mask])
        chose = res.extra["chose"]
        pm = res.extra.get("probe_metrics", {})
        sub_size = res.extra.get("subsample_size", 0)
        fallback = res.extra.get("fallback_used", False)
        wt_str = "  ".join(
            f"{a}={pm.get(a, {}).get('wall_time', float('nan')):.4f}s"
            for a in _PROBE_CANDIDATES
        )
        agg_str = "  ".join(
            f"{a}={pm.get(a, {}).get('aggregate', float('nan')):+.3f}"
            for a in _PROBE_CANDIDATES
        )
        print(
            f"{name:18s}  sub={sub_size:3d}  fallback={fallback!s:5s}  "
            f"chose={chose:24s}  ari={ari:.3f}\n"
            f"                    wall_times: {wt_str}\n"
            f"                    aggregates: {agg_str}"
        )
        if name == "moons":
            km = pm.get("kmeans", {})
            sp = pm.get("spectral", {})
            print(
                f"                    moons-detail: "
                f"kmeans kn_purity={km.get('kn_purity', float('nan')):+.3f}  "
                f"cost_inv={km.get('cost_inv', float('nan')):+.3f}  "
                f"aggregate={km.get('aggregate', float('nan')):+.3f}  | "
                f"spectral kn_purity={sp.get('kn_purity', float('nan')):+.3f}  "
                f"cost_inv={sp.get('cost_inv', float('nan')):+.3f}  "
                f"aggregate={sp.get('aggregate', float('nan')):+.3f}"
            )
