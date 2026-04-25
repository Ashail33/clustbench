"""Purity-Weighted Consensus Clustering (PWCC), per Alguliyev et al.

Run several base clustering algorithms, compute a *purity-like* weight
for each base partition (higher when the partition is internally
consistent with the consensus candidate), then build a final partition
by weighted majority vote over the aligned base labels.

The paper formulates this as

    pi*  =  argmax_pi   sum_i  w_i * Xi(pi, pi_i)

where ``Xi`` is a partition-similarity utility. We approximate that
greedy maximization with one Hungarian re-alignment per base partition
and a weighted vote per data point — the standard practical recipe
that is several orders of magnitude faster than searching the full
space of partitions.

The trajectory captures one :class:`Step` per base partition (the
state is the per-cluster size distribution, the action is "run base
algo X", the cost is the negative weight contribution) and a final
"consensus" step that records the assembled partition.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from . import base as base_algos
from .base import Algorithm, AlgoResult, Step, register


def _align(reference: np.ndarray, other: np.ndarray) -> np.ndarray:
    u_ref = np.unique(reference[reference != -1])
    u_other = np.unique(other[other != -1])
    if len(u_ref) == 0 or len(u_other) == 0:
        return other
    C = np.zeros((u_ref.size, u_other.size), dtype=np.int64)
    for i, r in enumerate(u_ref):
        for j, l in enumerate(u_other):
            C[i, j] = int(np.sum((reference == r) & (other == l)))
    row_ind, col_ind = linear_sum_assignment(-C)
    mapping = {int(u_other[j]): int(u_ref[i]) for i, j in zip(row_ind, col_ind)}
    out = other.copy()
    for j in u_other:
        out[other == j] = mapping.get(int(j), int(j))
    return out


def _purity_weight(reference: np.ndarray, other: np.ndarray) -> float:
    """Fraction of points in ``other`` whose dominant ``reference`` label
    matches their ``other`` label after alignment. In [0, 1]."""
    valid = reference != -1
    if not valid.any():
        return 0.0
    matched = int(np.sum(reference[valid] == other[valid]))
    return matched / int(valid.sum())


@register
class Pwcc(Algorithm):
    """Purity-Weighted Consensus Clustering.

    Parameters
    ----------
    base : list[str]
        Algorithm-registry keys to use as base partitions. Defaults to
        a sensible mix that the paper's design implies.
    base_params : dict[str, dict]
        Per-base hyper-parameters keyed by the same names as ``base``.
    """

    def __init__(
        self,
        base: list[str] | None = None,
        base_params: dict | None = None,
        **kwargs: Any,
    ) -> None:
        self.name = "pwcc"
        self.base = base or ["kmeans", "minibatch_kmeans", "birch_algo"]
        self.base_params = base_params or {}

    def fit_predict(self, X: np.ndarray, k: int | None = None) -> AlgoResult:
        assert k is not None, "k required"
        n = X.shape[0]
        partitions: list[np.ndarray] = []
        weights: list[float] = []
        trajectory: list[Step] = []

        # First pass: run each base, record per-base step.
        for step_idx, name in enumerate(self.base):
            cls = base_algos.ALGO_REGISTRY[name]
            res = cls(**self.base_params.get(name, {})).fit_predict(X, k=k)
            partitions.append(res.labels)
            sizes = Counter(int(v) for v in res.labels if v != -1)
            trajectory.append(
                Step(
                    step_idx=step_idx,
                    cost=0.0,  # filled in below once weights are known
                    delta_cost=None,
                    accepted=True,
                    action={"type": "run_base", "algo": name},
                    state={"sizes": dict(sizes)},
                )
            )

        # Align all partitions to the first one, then weight each by purity
        # against the (aligned) majority vote — equivalent to the weighted
        # utility maximisation described by Alguliyev et al.
        ref = partitions[0]
        aligned = [ref] + [_align(ref, p) for p in partitions[1:]]
        stacked = np.stack(aligned, axis=1)

        # Initial unweighted majority vote (drop -1 unless every base
        # called it noise).
        first_pass = np.empty(n, dtype=np.int64)
        for i in range(n):
            cnt = Counter(int(v) for v in stacked[i])
            if -1 in cnt and len(cnt) > 1:
                cnt.pop(-1, None)
            first_pass[i] = cnt.most_common(1)[0][0]

        for s, p in enumerate(aligned):
            w = _purity_weight(first_pass, p)
            weights.append(w)
            trajectory[s].cost = -float(w)
        weights_arr = np.array(weights, dtype=np.float32)
        if weights_arr.sum() > 0:
            weights_arr = weights_arr / weights_arr.sum()

        # Final weighted vote.
        final = np.empty(n, dtype=np.int64)
        for i in range(n):
            tally: dict[int, float] = {}
            for j, lbl in enumerate(stacked[i]):
                lbl = int(lbl)
                if lbl == -1 and any(int(v) != -1 for v in stacked[i]):
                    continue
                tally[lbl] = tally.get(lbl, 0.0) + float(weights_arr[j])
            final[i] = max(tally.items(), key=lambda kv: kv[1])[0]

        # Final consensus step.
        agreement = float(np.mean(final == first_pass))
        trajectory.append(
            Step(
                step_idx=len(self.base),
                cost=-agreement,
                delta_cost=None,
                accepted=True,
                action={
                    "type": "weighted_vote",
                    "weights": [float(w) for w in weights_arr],
                    "bases": list(self.base),
                },
                state={"agreement_with_unweighted": agreement},
            )
        )

        return AlgoResult(
            labels=final,
            extra={
                "bases": list(self.base),
                "weights": [float(w) for w in weights_arr],
                "agreement_with_unweighted": agreement,
            },
            trajectory=trajectory,
        )
