"""Learned routing meta-algorithm — sixth iteration (subsample-and-probe).

`learned_router_v6` is the **first router in the family whose dispatch
decision is grounded in actually running candidate algorithms on the
input data**, not in fingerprint similarity to historical tasks. v1-v5
all asked the same question — "which algorithm did best on tasks that
*look* like this one?" — and answered it with kNN / GBR / stacker models
over a static feature vector. v5's bake-off proved that question is
saturated: no same-shape-different-model router beats v3 per-task on the
current benchmark.

v6 asks a qualitatively different question: **which algorithm is
working on this specific data right now?** It picks the answer by
subsampling, running 5 candidate algorithms on the subsample, and
dispatching to whichever showed the best silhouette on the probe.

Design
------
1. **Sample.** Take ``m = min(200, max(50, int(0.3 * n)))`` points from
   the input with a fixed RNG seed. If ``n < 50`` fall back to v3 —
   silhouette on fewer than ~50 points is too noisy to trust.

2. **Probe.** Run 5 candidates on the subsample, each wrapped in
   try/except so a single failure can't kill the router:

       kmeans, spectral, gmm, agglomerative, dbscan_auto

   All are called with the requested ``k`` (``dbscan_auto`` ignores it
   but its silhouette is still meaningful, so we keep it in the
   comparison). For each probe we record the silhouette score on the
   subsample labels and the wall time; failed probes get silhouette
   ``-1``.

3. **Dispatch.** Direct silhouette-argmax: pick the candidate with the
   highest probe silhouette. Two guardrails:
     - If all probes return undefined / negative silhouette, fall back
       to v3.
     - If the winner's silhouette is below ``0.05`` (effectively no
       structure) the probe wasn't decisive — fall back to v3.

4. **Final dispatch.** Re-run the winning algorithm on the *full*
   ``X`` (not the subsample). Wrap in try/except; on failure, fall back
   to v3.

The ``use_regressor`` constructor flag is reserved for the planned
probe-augmented regression variant (combine probe silhouettes with the
v4 fingerprint into a 25-feature vector, per-algorithm kNN regressors).
For this first version it must remain ``False`` — the path is not yet
implemented.

Recursion is prevented by ``_BLOCKED_ROUTERS_V6``, which blocks every
router variant including v6 itself. The non-router candidates
(``kmeans``, ``spectral``, ``gmm``, ``agglomerative``, ``dbscan_auto``)
are obviously not blocked.
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Dict, List, Optional

import numpy as np

from . import base as base_algos
from .base import Algorithm, AlgoResult, Step, register


# All router variants are blocked as dispatch targets to prevent
# recursive routing chains.
_BLOCKED_ROUTERS_V6 = {
    "learned_router",
    "learned_router_v2",
    "learned_router_v3",
    "learned_router_v4",
    "learned_router_v5",
    "learned_router_v6",
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

# Silhouette threshold below which the probe is considered indecisive
# and we fall back to v3.
_SILHOUETTE_DECISIVE_THRESHOLD = 0.05


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------

def _probe_silhouette(
    Xs: np.ndarray, labels: np.ndarray
) -> float:
    """Compute silhouette on probe labels, returning -1 if undefined.

    Silhouette is undefined when there is only one effective cluster
    (after dropping any DBSCAN-style noise label -1). In that case we
    return -1 so the candidate loses the argmax.
    """
    from sklearn.metrics import silhouette_score

    # Drop noise labels (DBSCAN convention) before measuring silhouette.
    mask = labels >= 0
    if mask.sum() < 2:
        return -1.0
    eff_labels = labels[mask]
    unique = np.unique(eff_labels)
    if len(unique) < 2:
        return -1.0
    if len(unique) >= mask.sum():
        # Every point its own cluster — silhouette is undefined.
        return -1.0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(silhouette_score(Xs[mask], eff_labels))
    except Exception:
        return -1.0


def _run_probe(
    algo_name: str, Xs: np.ndarray, k: Optional[int]
) -> Dict[str, Any]:
    """Run one probe and return ``{silhouette, wall_time, error}``.

    Always returns a dict (never raises). ``silhouette`` is -1 on
    failure or when undefined; ``error`` is the exception's class name
    or ``None`` on success.
    """
    cls = base_algos.ALGO_REGISTRY.get(algo_name)
    if cls is None:
        return {
            "silhouette": -1.0,
            "wall_time": 0.0,
            "error": "algo_not_registered",
        }

    t0 = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = cls().fit_predict(Xs, k=k)
        labels = np.asarray(result.labels)
        sil = _probe_silhouette(Xs, labels)
        return {
            "silhouette": float(sil),
            "wall_time": float(time.perf_counter() - t0),
            "error": None,
        }
    except Exception as e:  # noqa: BLE001 — the whole point is to swallow
        return {
            "silhouette": -1.0,
            "wall_time": float(time.perf_counter() - t0),
            "error": type(e).__name__,
        }


# ---------------------------------------------------------------------------
# Algorithm class
# ---------------------------------------------------------------------------

@register
class Learned_router_v6(Algorithm):
    """Subsample-and-probe meta-algorithm.

    On every call, subsample the data, run 5 candidate algorithms on the
    subsample, and dispatch to whichever showed the best silhouette.
    Falls back to ``learned_router_v3`` when the subsample is too small
    or the probe is indecisive. See module docstring for the full
    rationale.
    """

    def __init__(
        self,
        random_state: int = 42,
        use_regressor: bool = False,
        fallback: str = "learned_router_v3",
        **kwargs: Any,
    ) -> None:
        self.name = "learned_router_v6"
        self.random_state = random_state
        # Reserved for the planned probe + fingerprint regression path.
        # Not yet implemented; v6's first version must use direct
        # silhouette-argmax only.
        self.use_regressor = bool(use_regressor)
        self.fallback = fallback
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
        """``m = min(200, max(50, int(0.3 * n)))`` per spec."""
        return int(min(200, max(50, int(0.3 * n))))

    def _fallback_to_v3(
        self,
        X: np.ndarray,
        k: Optional[int],
        reason: str,
        prefix_steps: List[Step],
        probe_silhouettes: Dict[str, float],
        subsample_size: int,
    ) -> AlgoResult:
        """Run v3 and stitch its trajectory after the v6 prefix."""
        v3 = self._get_v3()
        inner = v3.fit_predict(X, k=k)

        # Decision step recording the fallback.
        decision = Step(
            step_idx=len(prefix_steps),
            cost=0.0,
            delta_cost=None,
            accepted=True,
            action={
                "type": "dispatch_decision",
                "chose": "learned_router_v3",
                "fallback_used": True,
                "reason": reason,
            },
            state={
                "silhouette": None,
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
                "router": "subsample_probe_v6",
                "chose": chose_inner if chose_inner else "learned_router_v3",
                "probe_silhouettes": probe_silhouettes,
                "subsample_size": int(subsample_size),
                "fallback_used": True,
                "fallback_reason": reason,
                "inner_router": "learned_router_v3",
                **(inner.extra or {}),
            },
            trajectory=trajectory,
        )

    # ------------------------------------------------------------------
    # Main entrypoint
    # ------------------------------------------------------------------

    def fit_predict(
        self, X: np.ndarray, k: Optional[int] = None
    ) -> AlgoResult:
        X = np.asarray(X)
        n = int(X.shape[0])

        # Probe-augmented regression path is reserved for a future
        # iteration. Spec says default False; if someone constructs
        # the router with True, surface a warning and ignore the flag
        # rather than silently misbehaving.
        if self.use_regressor:
            warnings.warn(
                "learned_router_v6: use_regressor=True is reserved for a "
                "future iteration; falling back to direct silhouette-argmax."
            )

        # ----- Tiny-input guardrail -----------------------------------
        if n < 50:
            # Build a single probe_setup step so the trajectory still
            # records why we never probed.
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
                probe_silhouettes={},
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
        probe_silhouettes: Dict[str, float] = {}
        probe_times: Dict[str, float] = {}
        probe_errors: Dict[str, Optional[str]] = {}
        for algo in _PROBE_CANDIDATES:
            res = _run_probe(algo, Xs, k)
            sil = float(res["silhouette"])
            probe_silhouettes[algo] = sil
            probe_times[algo] = float(res["wall_time"])
            probe_errors[algo] = res["error"]
            steps.append(
                Step(
                    step_idx=len(steps),
                    cost=float(-sil) if sil > -1.0 else 0.0,
                    delta_cost=None,
                    accepted=res["error"] is None,
                    action={
                        "type": f"probe_{algo}",
                        "algo": algo,
                        "error": res["error"],
                    },
                    state={
                        "silhouette": sil,
                        "wall_time": float(res["wall_time"]),
                    },
                )
            )

        # ----- Decide --------------------------------------------------
        # Argmax over silhouettes. ``-1`` flags either failure or an
        # undefined silhouette (collapsed labels), both treated as
        # losing candidates.
        usable = {a: s for a, s in probe_silhouettes.items() if s > -1.0}

        if not usable:
            return self._fallback_to_v3(
                X, k,
                reason="all_probes_failed_or_undefined",
                prefix_steps=steps,
                probe_silhouettes=probe_silhouettes,
                subsample_size=m,
            )

        chosen, chosen_sil = max(usable.items(), key=lambda kv: kv[1])

        if chosen_sil < _SILHOUETTE_DECISIVE_THRESHOLD:
            return self._fallback_to_v3(
                X, k,
                reason=(
                    f"best_silhouette_{chosen_sil:.3f}"
                    f"_below_threshold_{_SILHOUETTE_DECISIVE_THRESHOLD}"
                ),
                prefix_steps=steps,
                probe_silhouettes=probe_silhouettes,
                subsample_size=m,
            )

        # ----- Dispatch decision step ---------------------------------
        steps.append(
            Step(
                step_idx=len(steps),
                cost=float(-chosen_sil),
                delta_cost=None,
                accepted=True,
                action={
                    "type": "dispatch_decision",
                    "chose": chosen,
                    "fallback_used": False,
                },
                state={
                    "silhouette": float(chosen_sil),
                    "fallback_used": False,
                },
            )
        )

        # ----- Re-run winner on full X --------------------------------
        cls = base_algos.ALGO_REGISTRY.get(chosen)
        if cls is None:
            # Should not happen (chosen comes from _PROBE_CANDIDATES)
            # but defence in depth.
            return self._fallback_to_v3(
                X, k,
                reason=f"chosen_algo_{chosen}_not_in_registry",
                prefix_steps=steps[:-1],  # drop the now-misleading decision
                probe_silhouettes=probe_silhouettes,
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
                probe_silhouettes=probe_silhouettes,
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
                "router": "subsample_probe_v6",
                "chose": chosen,
                "probe_silhouettes": probe_silhouettes,
                "probe_wall_times": probe_times,
                "probe_errors": {a: e for a, e in probe_errors.items()},
                "subsample_size": int(m),
                "fallback_used": False,
                "chosen_probe_silhouette": float(chosen_sil),
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

    router = Learned_router_v6()
    for name, (X, y) in cases:
        k_target = int(len(np.unique(y[y >= 0])))
        res = router.fit_predict(X, k=k_target)
        mask = y >= 0
        if mask.sum() == 0:
            ari = float("nan")
        else:
            ari = adjusted_rand_score(y[mask], res.labels[mask])
        chose = res.extra["chose"]
        sils = res.extra.get("probe_silhouettes", {})
        sub_size = res.extra.get("subsample_size", 0)
        fallback = res.extra.get("fallback_used", False)
        sil_str = "  ".join(f"{a}={sils.get(a, float('nan')):+.3f}"
                            for a in _PROBE_CANDIDATES)
        print(
            f"{name:18s}  sub={sub_size:3d}  fallback={fallback!s:5s}  "
            f"chose={chose:24s}  ari={ari:.3f}\n"
            f"                    silhouettes: {sil_str}"
        )
