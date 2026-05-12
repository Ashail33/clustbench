"""MWC: Mann-Whitney rank-test consensus clustering.

Ensemble clustering that combines several base partitions via a
hypothesis-test-driven merge rule. Each candidate merge between two
clusters is gated by a Mann-Whitney U test on the co-association
matrix:

    between = M[i, j] for i ∈ A, j ∈ B
    within  = M[i, j] for i, j ∈ A, i ≠ j  (plus the same for B)

    H0: between ≥ within  (the two clusters look like one)
    H1: between <  within  (the two clusters are genuinely separate)

Reject H0 at level ``alpha`` → keep the clusters separate.
Fail to reject  → merge them.

The procedure:
  1. Run every base algorithm; build the (n × n) co-association matrix
     ``M[i, j] = (1/K) * Σ_k [y_k[i] == y_k[j]]``.
  2. Get an over-clustered initial partition by running agglomerative
     average-linkage on ``1 - M`` to ``max(2k, k+5)`` clusters
     (oversampling so the test has merge candidates).
  3. Repeat: pick the pair of current clusters with the highest mean
     co-association (most "merge-like"); run the Mann-Whitney test; if
     the test does *not* reject H0, merge them; otherwise the pair is
     genuinely separate, try the next-best pair. Stop when no merge
     passes, or when the cluster count drops to ``k`` (when given).

Conceptually:
  - Classic consensus (CSPA, our ``consensus`` algorithm) takes a
    majority vote, treating every pair the same.
  - PWCC weights pairs by path probability through the co-association
    graph.
  - MWC adds a *significance test* per merge: clusters that look
    separate to most base algorithms stay separate even if a few
    algorithms disagree, which is exactly the noise-robustness
    consensus methods are supposed to give you.

Where it shines: ensembles where the base algorithms disagree (so plain
majority voting is unstable). Where it doesn't: ensembles where every
base algorithm already agrees (the test always says "merge", you might
as well use plain consensus).
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import scipy.stats
from sklearn.cluster import AgglomerativeClustering

from .base import ALGO_REGISTRY, Algorithm, AlgoResult, Step, register


@register
class Mwc(Algorithm):
    """Mann-Whitney rank-test consensus clustering.

    Parameters
    ----------
    base : list[str]
        Names of base algorithms in :data:`ALGO_REGISTRY` to ensemble.
    base_params : dict[str, dict] | None
        Per-algorithm parameter overrides, e.g. ``{"lmm": {"n_neighbors": 15}}``.
    alpha : float
        Mann-Whitney significance level. ``p < alpha`` means the pair is
        significantly separate → keep them split. ``p >= alpha`` →
        merge.
    max_initial_clusters : int | None
        Over-clustering target for the initial agglomerative step. If
        ``None``, uses ``max(2*k, k+5)`` when ``k`` is provided, else
        ``20``. Higher values give the merge test more candidates;
        smaller values run faster.
    max_pair_sample : int
        Cap on how many pairs are sampled when computing the within /
        between distributions for the Mann-Whitney test. Keeps each
        merge decision O(max_pair_sample) instead of
        O(|cluster|^2).
    random_state : int
        RNG seed for the pair subsampling.
    """

    def __init__(
        self,
        base: list[str],
        base_params: Optional[dict] = None,
        alpha: float = 0.05,
        max_initial_clusters: Optional[int] = None,
        max_pair_sample: int = 200,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        self.name = "mwc"
        self.base = list(base)
        self.base_params = base_params or {}
        self.alpha = float(alpha)
        self.max_initial_clusters = max_initial_clusters
        self.max_pair_sample = int(max_pair_sample)
        self.random_state = int(random_state)

    @staticmethod
    def _co_association(labels_list: list[np.ndarray]) -> np.ndarray:
        """Compute the (n × n) co-association matrix from a list of partitions."""
        K = len(labels_list)
        n = labels_list[0].shape[0]
        M = np.zeros((n, n), dtype=np.float32)
        for L in labels_list:
            same = (L[:, None] == L[None, :]).astype(np.float32)
            M += same
        M /= K
        np.fill_diagonal(M, 0.0)
        return M

    def _sample_pairs(
        self,
        pts_a: np.ndarray,
        pts_b: np.ndarray | None,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample point-pair indices, capped at ``max_pair_sample``.

        If ``pts_b is None``, samples within ``pts_a`` (excluding the
        diagonal); otherwise samples cross pairs between ``pts_a`` and
        ``pts_b``.
        """
        if pts_b is None:
            na = pts_a.shape[0]
            if na < 2:
                return np.empty(0, dtype=int), np.empty(0, dtype=int)
            n_sample = min(self.max_pair_sample, na * (na - 1) // 2)
            i = rng.integers(0, na, size=n_sample)
            j = rng.integers(0, na, size=n_sample)
            mask = i != j
            return pts_a[i[mask]], pts_a[j[mask]]
        na, nb = pts_a.shape[0], pts_b.shape[0]
        if na == 0 or nb == 0:
            return np.empty(0, dtype=int), np.empty(0, dtype=int)
        n_sample = min(self.max_pair_sample, na * nb)
        i = rng.integers(0, na, size=n_sample)
        j = rng.integers(0, nb, size=n_sample)
        return pts_a[i], pts_b[j]

    def fit_predict(self, X: np.ndarray, k: Optional[int] = None) -> AlgoResult:
        n = X.shape[0]
        rng = np.random.default_rng(self.random_state)
        trajectory: list[Step] = []

        # 1. Run each base algorithm. ``k`` is forwarded; algorithms that
        # don't take k (DBSCAN etc.) will produce their own count.
        labels_list: list[np.ndarray] = []
        for b in self.base:
            cls = ALGO_REGISTRY[b]
            res = cls(**self.base_params.get(b, {})).fit_predict(X, k=k)
            labels_list.append(res.labels)
        trajectory.append(
            Step(
                step_idx=0,
                cost=0.0,
                accepted=True,
                action={"type": "run_base", "n_base": len(labels_list)},
                state={"bases": list(self.base)},
            )
        )

        # 2. Co-association matrix.
        M = self._co_association(labels_list)

        # 3. Over-clustered initial partition via agglomerative on (1 - M).
        if self.max_initial_clusters is not None:
            m_init = int(self.max_initial_clusters)
        elif k is not None:
            m_init = min(max(2 * k, k + 5), n)
        else:
            m_init = min(20, n)
        m_init = max(2, min(m_init, n - 1))
        try:
            init_labels = AgglomerativeClustering(
                n_clusters=m_init, metric="precomputed", linkage="average"
            ).fit_predict(1.0 - M.astype(np.float64))
        except Exception:
            # Tiny / degenerate inputs — fall back to the first base partition.
            init_labels = labels_list[0].copy()
        labels = init_labels.astype(np.int64)
        trajectory.append(
            Step(
                step_idx=1,
                cost=float(m_init),
                accepted=True,
                action={"type": "init_overcluster", "m_init": int(m_init)},
                state={"n_clusters_init": int(np.unique(labels).size)},
            )
        )

        # 4. Iterative Mann-Whitney-gated merging.
        em_iter = 0
        while True:
            unique = np.unique(labels)
            if len(unique) <= 2:
                break
            if k is not None and len(unique) <= k:
                break
            pts_by = {int(c): np.where(labels == c)[0] for c in unique}

            # Rank candidate merges by mean cross-cluster co-association.
            candidates: list[tuple[float, int, int]] = []
            for i_idx, c1 in enumerate(unique):
                for c2 in unique[i_idx + 1 :]:
                    pa, pb = self._sample_pairs(pts_by[int(c1)], pts_by[int(c2)], rng)
                    if pa.size == 0:
                        continue
                    mean_between = float(M[pa, pb].mean())
                    candidates.append((mean_between, int(c1), int(c2)))
            if not candidates:
                break
            candidates.sort(reverse=True)

            # Try the best-supported merges first.
            merged = False
            for mean_between, c1, c2 in candidates:
                pa, pb = pts_by[c1], pts_by[c2]
                ia, ja = self._sample_pairs(pa, None, rng)
                ib, jb = self._sample_pairs(pb, None, rng)
                ix, jx = self._sample_pairs(pa, pb, rng)
                within = np.concatenate([M[ia, ja], M[ib, jb]])
                between = M[ix, jx]
                if len(within) < 5 or len(between) < 5:
                    p = 1.0  # not enough data → merge if forced, else keep
                else:
                    _, p = scipy.stats.mannwhitneyu(
                        between, within, alternative="less"
                    )
                accept_merge = p >= self.alpha
                # When ``k`` is required and we'd otherwise stop above it,
                # force the highest-similarity merge even if the test would
                # have rejected.
                must_merge = (k is not None) and (len(unique) > k)
                if accept_merge or (must_merge and not merged):
                    labels[labels == c2] = c1
                    trajectory.append(
                        Step(
                            step_idx=2 + em_iter,
                            cost=float(p),
                            accepted=True,
                            action={
                                "type": "merge",
                                "c1": int(c1),
                                "c2": int(c2),
                                "forced": bool(must_merge and not accept_merge),
                            },
                            state={
                                "p_value": float(p),
                                "mean_between": float(mean_between),
                                "n_clusters_after": int(np.unique(labels).size),
                            },
                        )
                    )
                    merged = True
                    em_iter += 1
                    break
            if not merged:
                break

        # 5. Renumber labels to contiguous 0..K-1.
        _, labels = np.unique(labels, return_inverse=True)
        return AlgoResult(
            labels=labels.astype(np.int64),
            extra={
                "n_base": len(labels_list),
                "n_clusters_found": int(np.unique(labels).size),
                "co_association_mean": float(M.mean()),
                "bases": list(self.base),
            },
            trajectory=trajectory,
        )
