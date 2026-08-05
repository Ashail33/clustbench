"""Streaming coreset k-means (BICO-style).

Consumes X in batches. Maintains a set of leaf Clustering Features
CF=(n, LS, SS) -- like BIRCH but with an absolute radius bound ``tau``
calibrated from the first batch (10th percentile of pairwise distances).
Each point walks the (flat) leaf set greedily: if it fits into some
leaf's tau-ball, it is absorbed (n+=1, LS+=x, SS+=x*x); otherwise a
new leaf is opened. When the leaf count exceeds a budget M = 200*k
(capped at 2000) we run merge-and-reduce (Har-Peled & Mazumdar): repeatedly
merge the two CFs whose union has the smallest radius until back under
budget. At query time we extract weighted coreset points c_j = LS_j/n_j
with weight n_j and run weighted k-means++ (sklearn KMeans with
sample_weight); the full X is assigned by nearest coreset centroid.

Refs: Fichtenberger et al., 'BICO: BIRCH meets Coresets for k-means',
ESA 2013; Har-Peled & Mazumdar, STOC 2004.
"""

from __future__ import annotations

from typing import Any, Optional
import numpy as np
from scipy.spatial.distance import pdist, cdist
from sklearn.cluster import KMeans

from .base import Algorithm, AlgoResult, Step, register


def _radius(n: float, LS: np.ndarray, SS: np.ndarray) -> float:
    """CF radius: sqrt(sum_d Var_d) = sqrt(sum_d (SS_d/n - (LS_d/n)^2))."""
    if n <= 1:
        return 0.0
    c = LS / n
    var = np.maximum(SS / n - c * c, 0.0).sum()
    return float(np.sqrt(var))


def _weighted_coreset_inertia(centers: np.ndarray, weights: np.ndarray, k: int, seed: int) -> float:
    if len(centers) < k:
        return 0.0
    km = KMeans(n_clusters=k, n_init=3, random_state=seed).fit(centers, sample_weight=weights)
    return float(km.inertia_)


@register
class Bico_stream(Algorithm):
    """Streaming BICO-style coreset construction + weighted k-means query."""

    def __init__(
        self,
        batch_size: int = 4096,
        budget_mult: int = 200,
        max_budget: int = 2000,
        tau: Optional[float] = None,
        tau_percentile: float = 10.0,
        n_init: int = 10,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        self.name = "bico_stream"
        self.batch_size = batch_size
        self.budget_mult = budget_mult
        self.max_budget = max_budget
        self.tau = tau
        self.tau_percentile = tau_percentile
        self.n_init = n_init
        self.random_state = random_state

    # ------------------------------------------------------------------
    def _calibrate_tau(self, X0: np.ndarray) -> float:
        if self.tau is not None:
            return float(self.tau)
        rng = np.random.default_rng(self.random_state)
        m = min(200, X0.shape[0])
        if X0.shape[0] > m:
            idx = rng.choice(X0.shape[0], size=m, replace=False)
            sample = X0[idx]
        else:
            sample = X0
        d = pdist(sample)
        d = d[d > 0]
        if d.size == 0:
            return 1.0
        return float(np.percentile(d, self.tau_percentile))

    # ------------------------------------------------------------------
    def _absorb_or_open(
        self,
        x: np.ndarray,
        leaves_n: list,
        leaves_LS: list,
        leaves_SS: list,
        tau: float,
    ) -> str:
        """Route one point. Returns 'absorbed' or 'opened'."""
        if leaves_n:
            LS_arr = np.stack(leaves_LS, axis=0)
            n_arr = np.asarray(leaves_n, dtype=np.float64)[:, None]
            centers = LS_arr / n_arr
            dist = np.linalg.norm(centers - x, axis=1)
            j = int(dist.argmin())
            nn = leaves_n[j] + 1
            LS_new = leaves_LS[j] + x
            SS_new = leaves_SS[j] + x * x
            if _radius(nn, LS_new, SS_new) <= tau:
                leaves_n[j] = nn
                leaves_LS[j] = LS_new
                leaves_SS[j] = SS_new
                return "absorbed"
        leaves_n.append(1.0)
        leaves_LS.append(x.copy())
        leaves_SS.append(x * x)
        return "opened"

    # ------------------------------------------------------------------
    def _merge_reduce(
        self,
        leaves_n: list,
        leaves_LS: list,
        leaves_SS: list,
        budget: int,
    ) -> int:
        """Iteratively merge closest-center pairs until len <= budget."""
        n_merged = 0
        while len(leaves_n) > budget:
            LS_arr = np.stack(leaves_LS, axis=0)
            n_arr = np.asarray(leaves_n, dtype=np.float64)[:, None]
            centers = LS_arr / n_arr
            D = cdist(centers, centers)
            np.fill_diagonal(D, np.inf)
            flat = int(D.argmin())
            M = len(leaves_n)
            i, j = flat // M, flat % M
            if i > j:
                i, j = j, i
            leaves_n[i] = leaves_n[i] + leaves_n[j]
            leaves_LS[i] = leaves_LS[i] + leaves_LS[j]
            leaves_SS[i] = leaves_SS[i] + leaves_SS[j]
            del leaves_n[j]
            del leaves_LS[j]
            del leaves_SS[j]
            n_merged += 1
        return n_merged

    # ------------------------------------------------------------------
    def fit_predict(self, X: np.ndarray, k: Optional[int] = None) -> AlgoResult:
        assert k is not None, "k required"
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape
        B = max(1, int(self.batch_size))
        budget = min(int(self.max_budget), int(self.budget_mult) * k)

        tau = self._calibrate_tau(X[: min(B, n)])

        leaves_n: list = []
        leaves_LS: list = []
        leaves_SS: list = []
        counts = {"absorbed": 0, "opened": 0, "merged": 0}
        trajectory: list[Step] = []

        n_batches = int(np.ceil(n / B))
        for b_idx in range(n_batches):
            batch = X[b_idx * B : (b_idx + 1) * B]
            for x in batch:
                verdict = self._absorb_or_open(x, leaves_n, leaves_LS, leaves_SS, tau)
                counts[verdict] += 1
            n_merged_this = self._merge_reduce(leaves_n, leaves_LS, leaves_SS, budget)
            counts["merged"] += n_merged_this

            # Record 2 trajectory points: after first batch and after last batch.
            if b_idx == 0 or b_idx == n_batches - 1:
                centers = np.stack(leaves_LS, axis=0) / np.asarray(leaves_n)[:, None]
                weights = np.asarray(leaves_n, dtype=np.float64)
                cost = _weighted_coreset_inertia(centers, weights, k, self.random_state)
                trajectory.append(
                    Step(
                        step_idx=len(trajectory),
                        cost=cost,
                        action={
                            "type": "stream_batch",
                            "batch_idx": b_idx,
                            **{k_: int(v) for k_, v in counts.items()},
                        },
                        state={
                            "n_leaves": int(len(leaves_n)),
                            "tau": float(tau),
                        },
                    )
                )

        # ---------- final query: weighted k-means++ on coreset ----------
        centers = np.stack(leaves_LS, axis=0) / np.asarray(leaves_n)[:, None]
        weights = np.asarray(leaves_n, dtype=np.float64)

        if len(leaves_n) < k:
            km = KMeans(n_clusters=k, n_init=self.n_init, random_state=self.random_state).fit(X)
            labels = km.labels_.astype(int)
            trajectory.append(
                Step(
                    step_idx=len(trajectory),
                    cost=float(km.inertia_),
                    action={"type": "fallback_full_kmeans"},
                    state={"n_leaves": int(len(leaves_n)), "tau": float(tau)},
                )
            )
            return AlgoResult(
                labels=labels,
                extra={
                    "inertia": float(km.inertia_),
                    "n_leaves": int(len(leaves_n)),
                    "tau": float(tau),
                    "fallback": True,
                    "note": "Coreset too small; fell back to full k-means.",
                },
                trajectory=trajectory,
            )

        km = KMeans(n_clusters=k, n_init=self.n_init, random_state=self.random_state).fit(
            centers, sample_weight=weights
        )
        final_centers = km.cluster_centers_
        D = cdist(X, final_centers)
        labels = D.argmin(axis=1).astype(int)
        inertia = float(D[np.arange(n), labels].sum())

        trajectory.append(
            Step(
                step_idx=len(trajectory),
                cost=inertia,
                action={"type": "weighted_kmeans_query"},
                state={
                    "n_leaves": int(len(leaves_n)),
                    "tau": float(tau),
                    "centroids": final_centers.tolist(),
                },
            )
        )

        return AlgoResult(
            labels=labels,
            extra={
                "inertia": inertia,
                "coreset_inertia": float(km.inertia_),
                "n_leaves": int(len(leaves_n)),
                "tau": float(tau),
                "budget": int(budget),
                "batch_size": int(B),
                "counts": {k_: int(v) for k_, v in counts.items()},
                "note": "Result is order-dependent under batch reordering (streaming).",
            },
            trajectory=trajectory,
        )
