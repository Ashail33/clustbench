"""RAPID v3: conditional outlier prefilter router over v1 and v2.

The v1/v2 trade-off (per ``docs/ALGORITHM_ANALYSIS.md``):

* :class:`Rapid` (v1) — two-stage region-adaptive pipeline with **no
  outlier prefilter**. Best base quality on the synthesised-algo
  registry (ARI rank 5/28; the only one solving moons (0.85) and
  circles (0.84) jointly). But fragile under outlier contamination:
  -0.145 ARI delta when 100 uniform outliers are injected.

* :class:`Rapid_v2` — same pipeline with an *unconditional* stage-0
  LOF outlier hold-out (top-10% LOF scores). Closes the outlier delta
  to -0.050 (+95 ARI-points on the targeted dimension), but regresses
  base quality whenever cluster density approximates outlier density,
  because LOF starts deleting real cluster points. Net overall:
  -0.069 ARI, rank 5 → 13.

v3's thesis: keep the outlier defence up front but **make it
conditional**. Use a single cheap LOF pass to estimate whether the
data is outlier-heavy *before* committing to a stage-0 removal. If
yes, run the v2 pipeline; if no, run the v1 pipeline. v3 is then
*automatically* v1 on clean data and *automatically* v2 on
outlier-heavy data — recovering v1's base quality on clean inputs and
v2's outlier robustness on dirty inputs, without paying either's
penalty on the wrong regime.

Stage -1 (router, new):
    Compute LOF scores once on ``X``. Estimate
    ``outlier_frac = (lof_scores > 1.5).mean()`` (1.5 is the
    conventional "likely outlier" cutoff on the LOF score). If
    ``outlier_frac > outlier_frac_threshold`` (default 0.05), delegate
    ``fit_predict`` to :class:`Rapid_v2`; else delegate to
    :class:`Rapid`. The router's decision is recorded as the first
    trajectory step and the inner algorithm's trajectory is appended
    with ``step_idx`` shifted by +1.

Edge cases:
    * ``n < 25``: LOF can't be trusted, default to v1.
    * LOF raises: default to v1, log the error in
      ``extra["lof_error"]``.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .base import Algorithm, AlgoResult, Step, register
from .rapid import Rapid
from .rapid_v2 import Rapid_v2

# Conventional "likely outlier" cutoff on LOF score (>1 = denser than
# its neighbours, >1.5 is empirically what LOF reference treatments
# call out as the threshold worth taking seriously).
_LOF_OUTLIER_SCORE_THRESHOLD = 1.5


@register
class Rapid_v3(Algorithm):
    """Region-Adaptive Partitioning with Iterative Density, v3 (router).

    Parameters
    ----------
    outlier_frac_threshold : float
        Fraction of points with LOF score > 1.5 above which v3 dispatches
        to :class:`Rapid_v2` (with its stage-0 LOF hold-out). At or
        below this fraction, v3 dispatches to :class:`Rapid` (no
        prefilter). Defaults to ``0.05``.
    min_samples : int
        ``min_samples`` for the downstream RAPID stage-1 DBSCAN.
    random_state : int
        Seed forwarded to the inner algorithm for reproducibility.
    """

    def __init__(
        self,
        outlier_frac_threshold: float = 0.05,
        min_samples: int = 5,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        self.name = "rapid_v3"
        self.outlier_frac_threshold = float(outlier_frac_threshold)
        self.min_samples = int(min_samples)
        self.random_state = random_state
        # Stash any extra kwargs to forward downstream (e.g. a sweep
        # might pass ``outlier_quantile`` for v2's stage 0).
        self._extra_kwargs = kwargs

    # ------------------------------------------------------------------
    # Stage -1: cheap one-shot LOF dispatch.
    # ------------------------------------------------------------------
    @staticmethod
    def _estimate_outlier_frac(X: np.ndarray) -> tuple[float, Optional[str]]:
        """Return ``(outlier_frac, error)``.

        ``outlier_frac`` is the fraction of points whose LOF score
        exceeds :data:`_LOF_OUTLIER_SCORE_THRESHOLD` (1.5). Returns
        ``(0.0, error_msg)`` if LOF can't run (n<25) or raises — that
        is the "default to v1" signal.
        """
        n = X.shape[0]
        if n < 25:
            return 0.0, "n<25"
        try:
            from sklearn.neighbors import LocalOutlierFactor

            lof = LocalOutlierFactor(
                n_neighbors=min(20, max(2, n - 1)),
                contamination="auto",
            )
            lof.fit(X)
            # negative_outlier_factor_: more negative = more outlier-y.
            # The conventional "LOF score" people threshold at 1.5 is
            # the unflipped magnitude, i.e. -negative_outlier_factor_.
            scores = -lof.negative_outlier_factor_
            frac = float((scores > _LOF_OUTLIER_SCORE_THRESHOLD).mean())
            return frac, None
        except Exception as exc:  # pragma: no cover - defensive
            return 0.0, f"{type(exc).__name__}: {exc}"

    def fit_predict(self, X: np.ndarray, k: Optional[int] = None) -> AlgoResult:
        assert k is not None, "k required"
        X = np.asarray(X)
        n = X.shape[0]

        outlier_frac, lof_error = self._estimate_outlier_frac(X)

        # If LOF failed we default to v1 regardless of the threshold.
        if lof_error is not None:
            chose = "v1"
        elif outlier_frac > self.outlier_frac_threshold:
            chose = "v2"
        else:
            chose = "v1"

        # Build the router step. This is always step 0 of the merged
        # trajectory; inner-algo steps are shifted by +1.
        router_step = Step(
            step_idx=0,
            cost=float(outlier_frac),
            delta_cost=None,
            accepted=True,
            action={
                "type": "router",
                "outlier_frac": float(outlier_frac),
                "lof_threshold": _LOF_OUTLIER_SCORE_THRESHOLD,
                "chose": chose,
            },
            state={
                "n": int(n),
                "outlier_frac_threshold": float(self.outlier_frac_threshold),
                "lof_error": lof_error,
            },
        )

        # Instantiate and dispatch. Forward shared knobs explicitly so
        # the inner algo behaves identically to a direct call.
        inner: Algorithm
        if chose == "v2":
            inner = Rapid_v2(
                min_samples=self.min_samples,
                random_state=self.random_state,
                **self._extra_kwargs,
            )
        else:
            inner = Rapid(
                min_samples=self.min_samples,
                random_state=self.random_state,
                **self._extra_kwargs,
            )

        inner_result = inner.fit_predict(X, k=k)

        # Merge trajectories: router first, then inner steps shifted +1.
        merged_trajectory: list[Step] = [router_step]
        if inner_result.trajectory:
            for s in inner_result.trajectory:
                merged_trajectory.append(
                    Step(
                        step_idx=s.step_idx + 1,
                        cost=s.cost,
                        delta_cost=s.delta_cost,
                        accepted=s.accepted,
                        action=s.action,
                        state=s.state,
                    )
                )

        extra = {
            "chose": chose,
            "outlier_frac": float(outlier_frac),
            "outlier_frac_threshold": float(self.outlier_frac_threshold),
            "lof_threshold": _LOF_OUTLIER_SCORE_THRESHOLD,
            "n": int(n),
            "lof_error": lof_error,
            "inner": inner_result.extra,
        }

        return AlgoResult(
            labels=inner_result.labels.astype(np.int64),
            extra=extra,
            trajectory=merged_trajectory,
        )
