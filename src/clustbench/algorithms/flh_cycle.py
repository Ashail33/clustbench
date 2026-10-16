"""FLH-cycle: co-EM that cycles between Fourier, Laplace and Hilbert views.

Instead of concatenating the three transform blocks of
:class:`~clustbench.algorithms.flhmm.Flhmm` into one feature space, this
keeps one mixture per view, ``log p_k^v(x) = a_k^v . phi_v(x) - log Z_k^v``
(self-normalised over the data, as in FMM), and shares only the soft
assignments between views. One *step* is

1. M-step for view ``v``: one damped Newton step on ``a^v`` (and the mixing
   weights) given the current responsibilities;
2. E-step from view ``v`` alone: responsibilities are recomputed from that
   view's log-likelihood, then handed to the next view.

Steps cycle Fourier -> Laplace -> Hilbert -> Fourier ... until the hard
labels stop changing for a whole cycle (or ``n_cycles`` is reached). This is
co-EM (Nigam & Ghani, 2000), applied to clustering as in Bickel & Scheffer
(2004), with transform features as the views. Each view can pull the shared
partition towards what it sees and the next view either agrees or pulls back.

Two knobs soften the hand-off: ``handoff`` < 1 blends each view's new
responsibilities with the previous ones, and ``estep="product"`` computes
responsibilities from all views' likelihoods while still updating one view's
parameters per step (block-coordinate EM on the product-of-experts model).

The final labels use ``final="consensus"`` (sum of all views'
log-likelihoods, i.e. a product of experts) or ``final="last"`` (the last
view's E-step). Initialisation is k-means on the concatenated features. Every
step is recorded in the trajectory (action = which view, cost = negative
consensus log-likelihood).
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
from scipy.special import logsumexp

from .base import AlgoResult, Algorithm, Step, register
from .flhmm import _BLOCKS, Flhmm


class _View:
    """One self-normalised exponential-family mixture over a fixed basis."""

    def __init__(self, phi: np.ndarray, k: int, l2: float, step: float) -> None:
        self.phi = phi
        self.k = k
        self.l2 = l2
        self.step = step
        self.alpha = np.zeros((k, phi.shape[1]))

    def log_dens(self) -> np.ndarray:
        """(n, k) matrix of log p_k(x_i) (self-normalised over the n points)."""
        E = self.phi @ self.alpha.T
        return E - logsumexp(E, axis=0, keepdims=True)

    def m_step(self, gamma: np.ndarray) -> None:
        Nk = gamma.sum(axis=0) + 1e-12
        E = self.phi @ self.alpha.T
        Q = np.exp(E - logsumexp(E, axis=0, keepdims=True))  # (n, k) model weights
        m = self.phi.shape[1]
        for j in range(self.k):
            target = gamma[:, j] @ self.phi / Nk[j]
            q = Q[:, j]
            mean_q = q @ self.phi
            cov = (self.phi * q[:, None]).T @ self.phi - np.outer(mean_q, mean_q)
            g = target - mean_q - self.l2 * self.alpha[j]
            try:
                delta = np.linalg.solve(cov + self.l2 * np.eye(m), g)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(cov + self.l2 * np.eye(m), g, rcond=None)[0]
            self.alpha[j] += self.step * delta


def _posterior(log_dens: np.ndarray, log_pi: np.ndarray) -> np.ndarray:
    L = log_dens + log_pi[None, :]
    return np.exp(L - logsumexp(L, axis=1, keepdims=True))


@register
class Flh_cycle(Algorithm):
    """Co-EM cycling between Fourier, Laplace and Hilbert transform views."""

    def __init__(
        self,
        blocks: Sequence[str] = _BLOCKS,
        n_cycles: int = 15,
        final: str = "consensus",
        estep: str = "view",
        handoff: float = 1.0,
        l2: float = 1e-3,
        newton_step: float = 0.5,
        random_state: int = 42,
        **feature_kwargs: Any,
    ) -> None:
        self.name = "flh_cycle"
        if final not in ("consensus", "last"):
            raise ValueError("final must be 'consensus' or 'last'")
        self.blocks = tuple(b for b in _BLOCKS if b in blocks)
        if len(self.blocks) < 1 or set(blocks) - set(_BLOCKS):
            raise ValueError(f"blocks must be a non-empty subset of {_BLOCKS}")
        if estep not in ("view", "product"):
            raise ValueError("estep must be 'view' or 'product'")
        self.n_cycles = int(n_cycles)
        self.final = final
        self.estep = estep
        self.handoff = float(handoff)
        self.l2 = float(l2)
        self.newton_step = float(newton_step)
        self.random_state = int(random_state)
        self.feature_kwargs = feature_kwargs

    def fit_predict(self, X: np.ndarray, k: int | None = None) -> AlgoResult:
        from sklearn.cluster import KMeans

        assert k is not None, "k must be given"
        rng = np.random.default_rng(self.random_state)
        feats = Flhmm(blocks=self.blocks, random_state=self.random_state,
                      **self.feature_kwargs).block_features(X, rng)
        n = len(X)
        views: Dict[str, _View] = {b: _View(feats[b], k, self.l2, self.newton_step) for b in self.blocks}

        km = KMeans(n_clusters=k, n_init=10, random_state=self.random_state).fit(
            np.concatenate([feats[b] for b in self.blocks], axis=1))
        gamma = np.full((n, k), 1e-3)
        gamma[np.arange(n), km.labels_] = 1.0
        gamma /= gamma.sum(axis=1, keepdims=True)

        traj: List[Step] = []
        labels = gamma.argmax(axis=1)
        prev_cost = None
        cycles_run = 0
        for c in range(self.n_cycles):
            start_labels = labels.copy()
            for b in self.blocks:
                log_pi = np.log(gamma.mean(axis=0) + 1e-12)
                views[b].m_step(gamma)
                if self.estep == "view":
                    new = _posterior(views[b].log_dens(), log_pi)
                else:  # all views' likelihoods; only view b's parameters moved
                    new = _posterior(sum(v.log_dens() for v in views.values()), log_pi)
                gamma = self.handoff * new + (1.0 - self.handoff) * gamma
                labels = gamma.argmax(axis=1)
                consensus = sum(v.log_dens() for v in views.values()) + log_pi[None, :]
                cost = float(-logsumexp(consensus, axis=1).sum())
                traj.append(Step(step_idx=len(traj), cost=cost,
                                 delta_cost=None if prev_cost is None else cost - prev_cost,
                                 action={"type": "view_em_step", "view": b, "cycle": c},
                                 state={"cluster_sizes": np.bincount(labels, minlength=k).tolist()}))
                prev_cost = cost
            cycles_run = c + 1
            if c > 0 and np.array_equal(labels, start_labels):
                break

        if self.final == "consensus":
            log_pi = np.log(gamma.mean(axis=0) + 1e-12)
            labels = (sum(v.log_dens() for v in views.values()) + log_pi[None, :]).argmax(axis=1)
        agree = {b: float(np.mean(views[b].log_dens().argmax(axis=1) == labels)) for b in self.blocks}
        return AlgoResult(labels=labels.astype(np.int64),
                          extra={"blocks": list(self.blocks), "cycles": cycles_run, "final": self.final,
                                 "estep": self.estep, "handoff": self.handoff,
                                 "view_agreement": agree},
                          trajectory=traj)
