"""o-means: mini-batch k-means with moving-average centroids and a two-level (cell) assignment.

CPU/NumPy re-implementation of the method described at owlgebra.ai/blog/o-means (built on e-means). The published
code is a CUDA contrib package; this version follows the described algorithm so it can run in the harness:

* **e-means update.** Each centroid keeps a running sum ``S_j`` and count ``N_j``; after a batch with sums ``s_j``
  and counts ``n_j``, ``S_j <- (1 - a) S_j + a s_j`` and ``N_j <- (1 - a) N_j + a n_j``, centroid ``S_j / N_j``. The
  step size ``a`` follows a cosine schedule from ``alpha_max`` down to ~0 over the run. Every centroid decays every
  batch, hit or not, so the step size has to shrink with the batch: the default is
  ``alpha_max = min(0.5, 0.5 b / k)`` for batch size ``b``. What matters is the number of updates per epoch: the
  default batch ``n / 200`` (clipped to 256-4,096) was best or near-best in all three checks in docs/data_omeans.
* **Revival.** Every ``revive_every`` batches, centroids whose count has fallen below ``dead_frac`` (1%) of the
  mean are replaced by splitting the busiest ones (half of the donor's sums move to the child, which is nudged along
  the donor's spread). Splitting busy centroids keeps refining dense regions, so reviving generously helps: at
  k = 4,096 a 1% threshold beats 0.1% (+0.63% vs +1.90% error after 3 epochs) and beats reviving only centroids that
  received no points for a long window (+3.10%; ``revive="hits"``).
* **Two-level assignment.** With ``k`` above ``two_level_min_k``, centroids are grouped into cells of ~128 around
  ``nlist`` cell centres; a point is compared with the cell centres, then only with centroids in its ``nprobe``
  nearest cells. The cell index is refreshed every ``revive_every`` batches in the first epoch (centroids re-bucketed
  to fixed cell centres) and rebuilt every half epoch after; for each split, the child is also checked whenever its
  donor wins (the "split pointer").
* **Closing mean.** For ``epochs >= 2`` each centroid becomes the plain mean of the points assigned to it in the last
  epoch. Returned labels come from an exact assignment to the final centroids.

The batch-size and step-size power laws of e-means are not published in the post, so ``batch_size``,
``alpha_max`` and ``dead_frac`` are explicit parameters; the defaults come from the sweeps in docs/data_omeans.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base import Algorithm, AlgoResult, register


def _sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.maximum((A * A).sum(1)[:, None] - 2 * A @ B.T + (B * B).sum(1)[None], 0)


def exact_assign(X: np.ndarray, C: np.ndarray, chunk: int = 2048) -> np.ndarray:
    out = np.empty(len(X), dtype=np.int64)
    for s in range(0, len(X), chunk):
        out[s:s + chunk] = _sqdist(X[s:s + chunk], C).argmin(1)
    return out


class _CellIndex:
    """Centroids bucketed into cells around ``nlist`` cell centres (an IVF index over the centroids)."""

    def __init__(self, C: np.ndarray, nlist: int, nprobe: int, rng: np.random.Generator, iters: int = 5):
        self.nlist, self.nprobe = nlist, min(nprobe, nlist)
        self.centres = C[rng.choice(len(C), nlist, replace=False)].copy()
        for _ in range(iters):                                      # a few Lloyd steps over the centroids
            a = _sqdist(C, self.centres).argmin(1)
            for c in range(nlist):
                if (a == c).any():
                    self.centres[c] = C[a == c].mean(0)
        self.rebucket(C)

    def rebucket(self, C: np.ndarray) -> None:
        """Re-assign centroids to the (fixed) cell centres."""
        self.cell_of = _sqdist(C, self.centres).argmin(1)
        order = np.argsort(self.cell_of, kind="stable")
        counts = np.bincount(self.cell_of, minlength=self.nlist)
        self.members = np.split(order, np.cumsum(counts)[:-1])

    def assign(self, X: np.ndarray, C: np.ndarray, children: dict) -> tuple[np.ndarray, int]:
        """Approximate nearest centroid; returns labels and the number of distance computations.

        Cell-major, as in the GPU kernel: each cell's centroids form one dense block, multiplied against just the
        points that probe that cell, so memory stays O(points x cell size)."""
        dc = _sqdist(X, self.centres)
        if self.nprobe < self.nlist:
            probe = np.argpartition(dc, self.nprobe - 1, axis=1)[:, :self.nprobe]
        else:
            probe = np.broadcast_to(np.arange(self.nlist), (len(X), self.nlist))
        pts = np.repeat(np.arange(len(X)), probe.shape[1]); cells = probe.ravel()
        order = np.argsort(cells, kind="stable"); pts, cells = pts[order], cells[order]
        bounds = np.searchsorted(cells, np.arange(self.nlist + 1))
        best = np.full(len(X), np.inf); lab = np.zeros(len(X), dtype=np.int64)
        ncomp = len(X) * self.nlist
        for c in range(self.nlist):
            m = self.members[c]; p = pts[bounds[c]:bounds[c + 1]]
            if len(m) == 0 or len(p) == 0:
                continue
            D = _sqdist(X[p], C[m]); j = D.argmin(1); dmin = D[np.arange(len(p)), j]
            better = dmin < best[p]; best[p[better]] = dmin[better]; lab[p[better]] = m[j[better]]
            ncomp += len(p) * len(m)
        if children:                                                    # split pointers: check the donor's children
            for donor, kids in children.items():
                sel = np.where(lab == donor)[0]
                if len(sel):
                    cand = np.r_[donor, np.array(kids)]
                    lab[sel] = cand[_sqdist(X[sel], C[cand]).argmin(1)]
                    ncomp += len(sel) * len(cand)
        return lab, ncomp


def omeans(X: np.ndarray, k: int, epochs: float = 3.0, batch_size: Optional[int] = None, alpha_max: Optional[float] = None,
           two_level_min_k: int = 256, cell_size: int = 128, nprobe: Optional[int] = None, revive_every: int = 5,
           random_state: int = 0, two_level: bool = True, revive: str = "count", dead_frac: float = 1e-2, trace: Optional[list] = None, trace_every: int = 1) -> dict:
    """Fit o-means; returns centroids, exact labels, the k-means objective and distance-computation counts.

    With ``trace`` (a list), every ``trace_every`` batches ``(passes so far, objective of the current centroids)`` is
    appended, where passes = distance computations / (n k); the objective pass is not counted.
    """
    rng = np.random.default_rng(random_state)
    X = np.ascontiguousarray(X, dtype=np.float64)
    n, d = X.shape
    k = int(min(k, n))
    b = int(batch_size or min(n, max(256, min(4096, n // 200))))     # ~200 updates per epoch
    nb_epoch = max(1, int(np.ceil(n / b)))
    alpha_max = float(alpha_max if alpha_max is not None else min(0.5, 0.5 * b / k))
    window = max(2 * revive_every, int(np.ceil(4 * k / b)))            # batches without a hit before 'dead'
    last_hit = np.zeros(k, dtype=np.int64)
    total = max(1, int(round(epochs * nb_epoch)))
    C = X[rng.choice(n, k, replace=False)].copy()
    S, N = C.copy(), np.ones(k)
    use_cells = two_level and k >= two_level_min_k
    idx = None
    if use_cells:
        nlist = int(16 * np.ceil(np.ceil(k / cell_size) / 16))
        nprobe = nprobe or (24 if k > 16384 else (16 if d < 512 else 12))
        idx = _CellIndex(C, min(nlist, k), nprobe, rng)
    children: dict = {}
    ncomp, last_sum, last_cnt = 0, np.zeros((k, d)), np.zeros(k)
    perm = rng.permutation(n)
    for t in range(total):
        pos = (t % nb_epoch) * b
        if pos == 0 and t > 0:
            perm = rng.permutation(n)
        rows = perm[pos:pos + b]
        xb = X[rows]
        if use_cells:
            lab, nc = idx.assign(xb, C, children)
        else:
            lab, nc = exact_assign(xb, C), len(xb) * k
        ncomp += nc
        a = alpha_max * 0.5 * (1 + np.cos(np.pi * t / total)) + 1e-4      # cosine schedule
        cnt = np.bincount(lab, minlength=k).astype(np.float64)
        last_hit[cnt > 0] = t
        sums = np.zeros((k, d)); np.add.at(sums, lab, xb)
        S = (1 - a) * S + a * sums
        N = (1 - a) * N + a * cnt
        C = S / np.maximum(N, 1e-12)[:, None]
        if trace is not None and (t % trace_every == 0 or t == total - 1):
            trace.append((ncomp / (n * k), float(((X - C[exact_assign(X, C)]) ** 2).sum(1).mean())))
        if t >= total - nb_epoch:                                         # last epoch: collect plain means
            last_sum += sums; last_cnt += cnt
        if (t + 1) % revive_every == 0:
            dead = np.where(t - last_hit > window)[0] if revive == "hits" else np.where(N < dead_frac * N.mean())[0]
            if len(dead):
                donors = np.argsort(-N)[:len(dead)]
                spread = np.sqrt(np.maximum(((xb - C[lab]) ** 2).mean(0), 1e-12))
                for j, dn in zip(dead, donors):
                    S[j] = S[dn] / 2; N[j] = N[dn] / 2; S[dn] /= 2; N[dn] /= 2
                    C[dn] = S[dn] / N[dn]
                    C[j] = C[dn] + 0.01 * spread * rng.standard_normal(d); S[j] = C[j] * N[j]
                    children.setdefault(int(dn), []).append(int(j))
                    last_hit[j] = last_hit[dn] = t
            if use_cells and t < nb_epoch:
                idx.rebucket(C); children = {}
        if use_cells and t >= nb_epoch and (t - nb_epoch) % max(1, nb_epoch // 2) == 0:
            idx = _CellIndex(C, idx.nlist, idx.nprobe, rng); children = {}
    if epochs >= 2:
        keep = last_cnt > 0
        C[keep] = last_sum[keep] / last_cnt[keep][:, None]
    labels = exact_assign(X, C)
    obj = float(((X - C[labels]) ** 2).sum(1).mean())
    return {"centroids": C, "labels": labels, "objective": obj, "distance_computations": ncomp,
            "exact_pass_equivalent": ncomp / (n * k), "batches": total, "two_level": use_cells}


@register
class O_means(Algorithm):
    """o-means as a clustering algorithm (k = number of clusters)."""

    def __init__(self, epochs: float = 3.0, batch_size: Optional[int] = None, alpha_max: Optional[float] = None,
                 n_init: int = 3, random_state: int = 0, **kwargs):
        self.name = "o_means"
        self.epochs, self.batch_size, self.alpha_max = epochs, batch_size, alpha_max
        self.n_init, self.random_state = n_init, random_state

    def fit_predict(self, X: np.ndarray, k: Optional[int] = None) -> AlgoResult:
        assert k is not None, "k required"
        best = None
        for r in range(self.n_init):                                      # keep the lowest-objective restart
            res = omeans(X, k, self.epochs, self.batch_size, self.alpha_max, random_state=self.random_state + r)
            if best is None or res["objective"] < best["objective"]:
                best = res
        return AlgoResult(labels=best["labels"], extra={"inertia": best["objective"] * len(X),
                                                        "exact_pass_equivalent": best["exact_pass_equivalent"]})
