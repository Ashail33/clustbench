"""FMM: exponential-family Fourier-basis mixture model.

A mixture model in the spirit of GMM, but every component is a
**learned Fourier series in log-density** rather than a Gaussian. The
generative story is

    p(x) = sum_k pi_k * p_k(x; alpha_k),
    p_k(x; alpha_k) ∝ exp(alpha_k . phi(x)),
    phi(x) = [cos(omega_1.x), sin(omega_1.x), ..., cos(omega_M.x), sin(omega_M.x)],

so each component is, quite literally, "a list of sinusoidal waveforms
added up" — exponentiated to get a non-negative density. The
frequencies ``omega_m`` are sampled once (random Fourier features
style) and shared across clusters; only the per-cluster amplitudes
``alpha_k`` are learned.

The intractable log-partition is sidestepped by self-normalising over
the data:

    log p_k(x_i) = alpha_k . phi(x_i) - log sum_j exp(alpha_k . phi(x_j)).

This turns ``p_k`` into a discrete distribution over the training
points (no base measure or Monte-Carlo integration needed) and makes
the per-cluster M-step a concave logistic-regression-style problem.

EM
--
* **E-step** — soft responsibilities ``gamma_ik``.
* **M-step** — for each cluster, take a few gradient-ascent steps to
  moment-match

      (1/N_k) sum_i gamma_ik phi(x_i)  ~=  sum_j q_jk phi(x_j),
      q_jk = softmax_j(alpha_k . phi(x_j)).

  An L2 penalty on ``alpha_k`` keeps the optimisation well-posed when
  the Fourier basis is high-dimensional.

Initialisation
--------------
k-means provides a warm start. Each cluster's ``alpha_k`` is then set
to point ``phi`` toward that cluster's mean (cheap analytic init that
keeps the first E-step well-behaved).

Trajectory
----------
One :class:`Step` per phase: ``fourier_basis``, ``kmeans_init``, and
one ``em_step`` per EM iteration carrying the running log-likelihood.
"""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
from scipy.special import logsumexp
from sklearn.cluster import KMeans

from .base import Algorithm, AlgoResult, Step, register


@register
class Fmm(Algorithm):
    """Exponential-family Fourier mixture model.

    Parameters
    ----------
    n_frequencies : int
        Number of random Fourier frequencies. The sufficient-statistic
        vector has length ``2 * n_frequencies`` (one cos + one sin per
        frequency).
    freq_scale : float or "auto"
        Std of the random Gaussian frequencies. ``"auto"`` uses
        ``1 / median(pairwise distance)`` (the standard RFF heuristic).
    max_iter : int
        Outer EM iterations.
    n_inner_iter : int
        Gradient-ascent steps per M-step.
    learning_rate : float
        Step size for the M-step gradient updates.
    l2 : float
        Ridge regulariser on ``alpha_k``.
    tol : float
        Stop when the relative change in log-likelihood is below this.
    random_state : int
        RNG seed.
    """

    def __init__(
        self,
        n_frequencies: int = 64,
        freq_scale: Union[float, str] = "auto",
        max_iter: int = 40,
        n_inner_iter: int = 5,
        learning_rate: float = 0.5,
        l2: float = 1e-2,
        tol: float = 1e-4,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        self.name = "fmm"
        self.n_frequencies = n_frequencies
        self.freq_scale = freq_scale
        self.max_iter = max_iter
        self.n_inner_iter = n_inner_iter
        self.learning_rate = learning_rate
        self.l2 = l2
        self.tol = tol
        self.random_state = random_state

    @staticmethod
    def _features(X: np.ndarray, omega: np.ndarray) -> np.ndarray:
        proj = X @ omega.T
        return np.concatenate([np.cos(proj), np.sin(proj)], axis=1)

    @staticmethod
    def _median_pairwise(X: np.ndarray, rng: np.random.Generator, n_sub: int = 500) -> float:
        n = X.shape[0]
        if n <= 1:
            return 1.0
        idx = rng.choice(n, size=min(n, n_sub), replace=False)
        Xs = X[idx]
        diff = Xs[:, None, :] - Xs[None, :, :]
        d = np.sqrt((diff * diff).sum(-1))
        iu = np.triu_indices(Xs.shape[0], k=1)
        if iu[0].size == 0:
            return 1.0
        med = float(np.median(d[iu]))
        return max(med, 1e-12)

    def fit_predict(self, X: np.ndarray, k: Optional[int] = None) -> AlgoResult:
        assert k is not None, "k (number of clusters) must be provided"
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape
        rng = np.random.default_rng(self.random_state)
        eps = 1e-12
        trajectory: list[Step] = []

        # --- 1. Sample the Fourier basis.
        if isinstance(self.freq_scale, str) and self.freq_scale == "auto":
            med = self._median_pairwise(X, rng)
            freq_scale = 1.0 / med
        else:
            freq_scale = float(self.freq_scale)
        omega = rng.normal(size=(self.n_frequencies, d)) * freq_scale
        phi = self._features(X, omega)  # (n, 2M)
        M2 = phi.shape[1]
        trajectory.append(
            Step(
                step_idx=0,
                cost=0.0,
                accepted=True,
                action={
                    "type": "fourier_basis",
                    "n_frequencies": int(self.n_frequencies),
                    "freq_scale": float(freq_scale),
                },
                state={"feature_dim": int(M2)},
            )
        )

        # --- 2. k-means warm start for alpha_k.
        km = KMeans(
            n_clusters=k, n_init=5, random_state=self.random_state, max_iter=100
        ).fit(X)
        init_labels = km.labels_
        alpha = np.zeros((k, M2))
        phi_var = phi.var(axis=0) + eps
        global_mean = phi.mean(axis=0)
        for j in range(k):
            members = phi[init_labels == j]
            if members.size:
                alpha[j] = (members.mean(axis=0) - global_mean) / phi_var
        pi = np.ones(k) / k
        trajectory.append(
            Step(
                step_idx=1,
                cost=float(km.inertia_),
                accepted=True,
                action={"type": "kmeans_init", "n_clusters": int(k)},
                state={"init_inertia": float(km.inertia_)},
            )
        )

        # --- 3. EM loop with self-normalised log-partition.
        prev_ll: Optional[float] = None
        gamma = np.zeros((n, k))
        em_iter = 0
        for em_iter in range(self.max_iter):
            # E-step.
            logits = phi @ alpha.T                          # (n, k)
            log_Zk = logsumexp(logits, axis=0)              # (k,)
            log_pkx = logits - log_Zk[None, :]              # (n, k)
            log_post = log_pkx + np.log(pi + eps)[None, :]
            log_marg = logsumexp(log_post, axis=1, keepdims=True)
            log_post = log_post - log_marg
            gamma = np.exp(log_post)
            ll = float(log_marg.sum())

            # M-step: pi from responsibilities, alpha by gradient ascent.
            Nk = gamma.sum(axis=0) + eps
            pi = Nk / n
            data_moments = (gamma.T @ phi) / Nk[:, None]    # (k, 2M)
            for _ in range(self.n_inner_iter):
                logits = phi @ alpha.T
                log_q = logits - logsumexp(logits, axis=0, keepdims=True)
                q = np.exp(log_q)                           # (n, k); cols sum to 1
                model_moments = q.T @ phi                   # (k, 2M)
                grad = data_moments - model_moments - self.l2 * alpha
                alpha = alpha + self.learning_rate * grad

            delta = None if prev_ll is None else ll - prev_ll
            trajectory.append(
                Step(
                    step_idx=2 + em_iter,
                    cost=-ll,  # neg log-lik; lower is better by convention.
                    delta_cost=None if delta is None else -delta,
                    accepted=True,
                    action={"type": "em_step", "em_iter": int(em_iter)},
                    state={
                        "log_likelihood": ll,
                        "pi": [float(v) for v in pi],
                        "alpha_norm": float(np.linalg.norm(alpha)),
                    },
                )
            )
            if (
                prev_ll is not None
                and em_iter > 2
                and abs(ll - prev_ll) < self.tol * max(abs(ll), 1.0)
            ):
                prev_ll = ll
                break
            prev_ll = ll

        labels = gamma.argmax(axis=1).astype(np.int64)
        return AlgoResult(
            labels=labels,
            extra={
                "log_likelihood": float(prev_ll if prev_ll is not None else float("nan")),
                "n_em_iter": int(em_iter + 1),
                "n_frequencies": int(self.n_frequencies),
                "feature_dim": int(M2),
                "freq_scale": float(freq_scale),
                "alpha_norm": float(np.linalg.norm(alpha)),
            },
            trajectory=trajectory,
        )
