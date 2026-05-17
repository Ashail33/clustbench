"""Learned routing meta-algorithm — v6b (multi-metric subsample-and-probe).

`learned_router_v6b` keeps v6's subsample-and-probe architecture but
replaces v6's **single silhouette score** with a **multi-metric
aggregate** that is robust to non-convex cluster shapes.

Why v6 underperformed
---------------------
v6 ran 5 candidates on a subsample, scored each by silhouette, and
dispatched to the silhouette-argmax. Silhouette is biased toward compact
(convex) partitions: on moons / circles, k-means wins the silhouette
probe even though spectral is the correct answer. Empirically v6's
full-benchmark rank was 28 (mean ARI 0.633) vs v3's rank 1 (0.855).

v6b's fix
---------
For each probe, compute four internal-validity metrics on the subsample
partition:

    1. ``silhouette``         — sklearn.metrics.silhouette_score
    2. ``calinski_harabasz``  — sklearn.metrics.calinski_harabasz_score
    3. ``davies_bouldin``     — inverted as ``1 / (1 + db)`` so higher = better
    4. ``kneighbor_purity``   — custom connectivity-aware purity

``kneighbor_purity`` is the critical addition: on a small kNN graph
built on the subsample, score each point by the fraction of its
neighbours that share its cluster label, then average. Spectral on
moons preserves local connectivity and scores high here; k-means on
moons cuts across the moon and scores low. This is the signal silhouette
lacks for non-convex data.

Each metric is min-max scaled across the 5 candidate algorithms to
[0, 1] (NaN-aware: NaN values are skipped, fully-NaN columns drop out).
The unweighted mean across the four normalised metrics is the
aggregate. The candidate with the highest aggregate wins.

Dispatch
--------
1. Re-run the winning algorithm on the full X (same as v6).
2. Fall back to v3 if the winning aggregate < 0.10 (low-confidence) or
   if every metric is NaN for every candidate.
3. **Optional v3-hint fallback.** If v3's predicted ARI for the v6b
   pick is < 0.30, prefer v3's pick instead — the probe is confident
   the chosen algo will fail.

Recursion is prevented by ``_BLOCKED_ROUTERS_V6B``.
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
_BLOCKED_ROUTERS_V6B = {
    "learned_router",
    "learned_router_v2",
    "learned_router_v3",
    "learned_router_v4",
    "learned_router_v5",
    "learned_router_v6",
    "learned_router_v6b",
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

# If v3 predicts ARI < this for v6b's pick, prefer v3's pick instead.
_V3_HINT_MIN_PREDICTED_ARI = 0.30


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _effective_labels(labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Drop DBSCAN-style noise (-1) and return ``(mask, eff_labels)``.

    The mask indexes into the original subsample. ``eff_labels`` is the
    labels at those positions.
    """
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
    """Davies-Bouldin inverted via ``1 / (1 + db)``. NaN if undefined.

    DB is bounded below by 0 (lower = better). The inversion turns it
    into a higher-is-better score in [0, 1].
    """
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
    cluster label, then average over all points. ``k`` is set by
    ``max(3, min(10, m // 20))`` where ``m`` is the subsample size, so
    it scales gently with sample size while staying small.

    This rewards algorithms that respect local connectivity. Spectral
    on moons preserves the manifold structure and scores high; k-means
    cuts straight across the moons and scores low.

    Noise points (label -1, DBSCAN convention) are excluded from the
    averaging pool but are still allowed to appear as a neighbour of an
    in-cluster point (where they always count as a label mismatch).
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
        # All non-noise points share one label — purity is trivially 1
        # but conveys no discrimination signal between candidates that
        # collapsed to a single cluster. NaN is the honest answer.
        return float("nan")

    k_nbr = max(3, min(10, m // 20))
    # We need k+1 because the point itself is its own first neighbour.
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

    # idx[:, 0] is the point itself — drop it.
    neighbour_idx = idx[:, 1:]
    point_labels = labels[:, None]
    neighbour_labels = labels[neighbour_idx]
    matches = (neighbour_labels == point_labels).astype(np.float64)
    point_purity = matches.mean(axis=1)

    # Average only over non-noise points so that DBSCAN runs which
    # label most points as noise aren't artificially rewarded by the
    # "noise vs noise" matches.
    if mask.sum() == 0:
        return float("nan")
    return float(point_purity[mask].mean())


def _compute_all_metrics(
    Xs: np.ndarray, labels: np.ndarray
) -> Dict[str, float]:
    """Return all four metrics for one probe. Each metric is wrapped
    in its own try/except inside the helper, so this just dispatches.
    """
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

    Always returns a dict (never raises). ``metrics`` is the four-metric
    dict, with NaN for any metric that couldn't be computed; ``error``
    is the exception's class name or ``None`` on success.
    """
    cls = base_algos.ALGO_REGISTRY.get(algo_name)
    nan_metrics = {
        "silhouette": float("nan"),
        "calinski_harabasz": float("nan"),
        "davies_bouldin_inv": float("nan"),
        "kneighbor_purity": float("nan"),
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
        labels = np.asarray(result.labels)
        metrics = _compute_all_metrics(Xs, labels)
        return {
            "metrics": metrics,
            "wall_time": float(time.perf_counter() - t0),
            "error": None,
        }
    except Exception as e:  # noqa: BLE001 — whole point is to swallow
        return {
            "metrics": nan_metrics,
            "wall_time": float(time.perf_counter() - t0),
            "error": type(e).__name__,
        }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

_METRIC_NAMES = (
    "silhouette",
    "calinski_harabasz",
    "davies_bouldin_inv",
    "kneighbor_purity",
)


def _aggregate_scores(
    per_algo_metrics: Dict[str, Dict[str, float]],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    """Min-max normalise each metric column across candidates, then
    take the unweighted mean per candidate.

    Parameters
    ----------
    per_algo_metrics
        ``{algo: {metric_name: raw_value}}``.

    Returns
    -------
    normalised
        ``{algo: {metric_name: normalised_value_or_nan}}``.
    aggregate
        ``{algo: mean_of_available_metrics}``. Algos whose four metrics
        are all NaN get ``-inf``.
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
            # Metric drops out entirely.
            for a in algos:
                normalised[a][metric] = float("nan")
            continue
        finite = col[finite_mask]
        c_min = float(finite.min())
        c_max = float(finite.max())
        if c_max - c_min < 1e-12:
            # All finite values equal — every candidate scores 1.0 on
            # this metric (no discrimination), NaN stays NaN.
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
        vals = [v for v in normalised[a].values() if np.isfinite(v)]
        if not vals:
            aggregate[a] = float("-inf")
        else:
            aggregate[a] = float(np.mean(vals))

    return normalised, aggregate


# ---------------------------------------------------------------------------
# Algorithm class
# ---------------------------------------------------------------------------

@register
class Learned_router_v6b(Algorithm):
    """Subsample-and-probe router with a multi-metric aggregate.

    See module docstring for the full rationale. Drop-in replacement
    for ``learned_router_v6`` whose only difference is the score used
    to pick a probe winner.
    """

    def __init__(
        self,
        random_state: int = 42,
        fallback: str = "learned_router_v3",
        use_v3_hint: bool = True,
        **kwargs: Any,
    ) -> None:
        self.name = "learned_router_v6b"
        self.random_state = random_state
        self.fallback = fallback
        self.use_v3_hint = bool(use_v3_hint)
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
        """``m = min(200, max(50, int(0.3 * n)))`` — same as v6."""
        return int(min(200, max(50, int(0.3 * n))))

    def _build_probe_metrics_extra(
        self,
        per_algo_metrics: Dict[str, Dict[str, float]],
        normalised: Dict[str, Dict[str, float]],
        aggregate: Dict[str, float],
    ) -> Dict[str, Dict[str, float]]:
        """Pack the four raw metrics + aggregate into the per-algo dict
        used in ``extra["probe_metrics"]``. Renamed to match the spec
        keys (``calinski``, ``db_inv``, ``kn_purity``).
        """
        out: Dict[str, Dict[str, float]] = {}
        for a, raw in per_algo_metrics.items():
            out[a] = {
                "silhouette": float(raw.get("silhouette", float("nan"))),
                "calinski": float(raw.get("calinski_harabasz", float("nan"))),
                "db_inv": float(raw.get("davies_bouldin_inv", float("nan"))),
                "kn_purity": float(raw.get("kneighbor_purity", float("nan"))),
                "silhouette_norm":
                    float(normalised[a].get("silhouette", float("nan"))),
                "calinski_norm":
                    float(normalised[a].get("calinski_harabasz", float("nan"))),
                "db_inv_norm":
                    float(normalised[a].get("davies_bouldin_inv", float("nan"))),
                "kn_purity_norm":
                    float(normalised[a].get("kneighbor_purity", float("nan"))),
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
        """Run v3 and stitch its trajectory after the v6b prefix."""
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
                "router": "subsample_probe_v6b",
                "chose": chose_inner if chose_inner else "learned_router_v3",
                "probe_metrics": probe_metrics,
                "subsample_size": int(subsample_size),
                "fallback_used": True,
                "fallback_reason": reason,
                "inner_router": "learned_router_v3",
                **(inner.extra or {}),
            },
            trajectory=trajectory,
        )

    def _v3_predicted_ari_for(self, algo: str, X: np.ndarray, k: Optional[int]) -> Optional[float]:
        """Query v3's per-algorithm regressor for the predicted ARI of
        ``algo`` on this input. Returns ``None`` if v3's training data
        or regressors are unavailable. Best-effort and silent.
        """
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
                        "wall_time": float(res["wall_time"]),
                    },
                )
            )

        # ----- Aggregate ----------------------------------------------
        normalised, aggregate = _aggregate_scores(per_algo_metrics)
        probe_metrics_extra = self._build_probe_metrics_extra(
            per_algo_metrics, normalised, aggregate
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
                    "aggregate_scores": {a: float(s) for a, s in aggregate.items()},
                    "chosen_aggregate": float(chosen_score),
                    "fallback_used": False,
                },
            )
        )

        # ----- Re-run winner on full X --------------------------------
        cls = base_algos.ALGO_REGISTRY.get(chosen)
        if cls is None:
            # Defence in depth — chosen comes from _PROBE_CANDIDATES.
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
                "router": "subsample_probe_v6b",
                "chose": chosen,
                "probe_metrics": probe_metrics_extra,
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

    router = Learned_router_v6b()
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
        agg_str = "  ".join(
            f"{a}={pm.get(a, {}).get('aggregate', float('nan')):+.3f}"
            for a in _PROBE_CANDIDATES
        )
        print(
            f"{name:18s}  sub={sub_size:3d}  fallback={fallback!s:5s}  "
            f"chose={chose:24s}  ari={ari:.3f}\n"
            f"                    aggregates: {agg_str}"
        )
        if name == "moons":
            km = pm.get("kmeans", {})
            sp = pm.get("spectral", {})
            print(
                f"                    moons-detail: "
                f"kmeans kn_purity={km.get('kn_purity', float('nan')):+.3f}  "
                f"aggregate={km.get('aggregate', float('nan')):+.3f}  | "
                f"spectral kn_purity={sp.get('kn_purity', float('nan')):+.3f}  "
                f"aggregate={sp.get('aggregate', float('nan')):+.3f}"
            )
