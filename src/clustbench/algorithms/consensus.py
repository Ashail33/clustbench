"""Consensus clustering algorithm – majority vote over an ensemble of base algorithms."""

from __future__ import annotations

import numpy as np
from collections import Counter
from scipy.optimize import linear_sum_assignment

from .base import Algorithm, AlgoResult, register
from . import base as _registry


def _align_labels(ref: np.ndarray, lab: np.ndarray) -> np.ndarray:
    """Re-label *lab* so its cluster IDs best match those in *ref* (Hungarian algorithm)."""
    u_ref = np.unique(ref[ref != -1])
    u_lab = np.unique(lab[lab != -1])
    if len(u_ref) == 0 or len(u_lab) == 0:
        return lab
    C = np.zeros((u_ref.size, u_lab.size), dtype=int)
    for i, r in enumerate(u_ref):
        for j, l in enumerate(u_lab):
            C[i, j] = np.sum((ref == r) & (lab == l))
    row_ind, col_ind = linear_sum_assignment(-C)
    mapping = {u_lab[j]: u_ref[i] for i, j in zip(row_ind, col_ind)}
    out = lab.copy()
    for j in u_lab:
        out[lab == j] = mapping.get(j, j)
    return out


@register
class Consensus(Algorithm):
    """Majority-vote ensemble over a list of base clustering algorithms.

    Parameters
    ----------
    base:
        List of algorithm names (lowercase) that exist in :data:`ALGO_REGISTRY`.
    base_params:
        Optional dict mapping algorithm name → constructor kwargs.
    """

    def __init__(self, base: list[str], base_params: dict | None = None, **kwargs):
        self.name = "consensus"
        self.base = base
        self.base_params = base_params or {}

    def fit_predict(self, X: np.ndarray, k: int | None = None) -> AlgoResult:
        label_list = []
        extras = {}
        for b in self.base:
            cls = _registry.ALGO_REGISTRY[b]
            res = cls(**self.base_params.get(b, {})).fit_predict(X, k=k)
            label_list.append(res.labels)
            extras[b] = res.extra
        ref = label_list[0]
        aligned = [ref] + [_align_labels(ref, L) for L in label_list[1:]]
        stacked = np.stack(aligned, axis=1)
        final = np.empty(stacked.shape[0], dtype=int)
        for i in range(stacked.shape[0]):
            cnt = Counter(stacked[i].tolist())
            if -1 in cnt and len(cnt) > 1:
                cnt.pop(-1, None)
            final[i] = cnt.most_common(1)[0][0]
        return AlgoResult(labels=final, extra={"bases": list(extras.keys())})
