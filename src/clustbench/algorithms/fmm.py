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
* **M-step** — for each cluster, one damped **Newton** step solves

      (Cov_q[phi] + l2 * I) * Δalpha_k = data_moments_k - model_moments_k - l2 * alpha_k,

  where ``data_moments_k = (1/N_k) Σ_i gamma_ik phi(x_i)`` and
  ``model_moments_k = Σ_j q_jk phi(x_j)`` with ``q_jk`` the softmax of
  ``alpha_k . phi(x_j)`` over the dataset. The Hessian is a covariance
  matrix (PSD) so EM converges in a handful of outer iterations.

Multi-scale frequencies
-----------------------
The frequencies ``omega_m`` are sampled from a *mixture* of Gaussian
scales (``n_scales`` bandwidths centred on ``1 / median pairwise
distance``). One model therefore resolves both fine and coarse
structure — a single bandwidth either over- or under-smooths.

BIC
---
Self-normalised log-likelihood is well-defined over the training set,
so the standard BIC

    BIC = -2 * log_likelihood + p * log(n),
    p   = k * (2M) + (k - 1)       # alpha and pi

is reported in ``extra``. When ``k_search`` is provided and ``k`` is
``None``, the algorithm runs once per ``k`` in the range and returns
the model with the lowest BIC.

Trajectory
----------
One :class:`Step` per phase: ``fourier_basis``, ``kmeans_init``, and
one ``newton_step`` per EM iteration carrying the running
log-likelihood. When ``k_search`` triggers, the per-``k`` traces are
discarded and a single ``k_search`` step records the BIC profile.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import logsumexp
from sklearn.cluster import KMeans

from .base import Algorithm, AlgoResult, Step, register


@register
class Fmm(Algorithm):
    """Exponential-family Fourier mixture model with multi-scale RFF basis,
    Newton M-step, and built-in BIC.

    Parameters
    ----------
    n_frequencies : int
        Total number of random Fourier frequencies (split evenly across
        the ``n_scales`` bandwidths). The sufficient-statistic vector
        has length ``2 * n_frequencies``.
    n_scales : int
        Number of bandwidths in the multi-scale RFF mixture. The
        bandwidth grid is ``base * 2^l`` for ``l = 0, 1, ..., n_scales-1``,
        where ``base = 1 / median(pairwise distance)`` when
        ``freq_scale="auto"`` (otherwise ``base = freq_scale``).
    freq_scale : float or "auto"
        Base bandwidth. ``"auto"`` uses the RFF heuristic.
    max_iter : int
        Outer EM iterations.
    n_inner_iter : int
        Newton steps inside each M-step. ``1`` is the textbook EM ratio
        but the per-cluster objective is concave, so 2-3 inner steps
        produce a near-exact M-step and tighter EM monotonicity.
    l2 : float
        Ridge regulariser on ``alpha_k``. Also keeps the Newton Hessian
        positive-definite even when ``Cov_q[phi]`` is rank-deficient.
    damping : float
        Maximum Newton step size in (0, 1]. The actual step is chosen
        by backtracking so the marginal log-likelihood never decreases.
    tol : float
        Stop when the relative change in log-likelihood is below this.
    k_search : tuple[int, int] | None
        ``(k_lo, k_hi)`` inclusive. When ``k`` is ``None`` at call time,
        fits FMM for every ``k`` in the range and returns the
        minimum-BIC model.
    random_state : int
        RNG seed.
    """

    def __init__(
        self,
        n_frequencies: int = 64,
        n_scales: int = 3,
        freq_scale: Union[float, str] = "auto",
        max_iter: int = 40,
        n_inner_iter: int = 1,
        l2: float = 1e-4,
        damping: float = 0.2,
        tol: float = 1e-4,
        k_search: Optional[Tuple[int, int]] = None,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        self.name = "fmm"
        self.n_frequencies = n_frequencies
        self.n_scales = max(1, n_scales)
        self.freq_scale = freq_scale
        self.max_iter = max_iter
        self.n_inner_iter = max(1, n_inner_iter)
        self.l2 = l2
        self.damping = damping
        self.tol = tol
        self.k_search = k_search
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

    def _sample_omega(
        self, X: np.ndarray, rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Multi-scale Gaussian frequency sampler.

        Returns ``(omega, scales_per_freq)``.
        """
        d = X.shape[1]
        if isinstance(self.freq_scale, str) and self.freq_scale == "auto":
            base = 1.0 / self._median_pairwise(X, rng)
        else:
            base = float(self.freq_scale)

        # Geometric bandwidth grid centred on ``base`` so the median scale
        # equals ``base`` regardless of ``n_scales``.
        l = self.n_scales
        offsets = np.arange(l) - (l - 1) / 2.0
        scales = base * (2.0 ** offsets)

        # Split frequencies as evenly as possible across scales.
        per_scale = np.full(l, self.n_frequencies // l, dtype=int)
        per_scale[: self.n_frequencies % l] += 1

        omega_blocks = []
        scales_per_freq = []
        for s, m_s in zip(scales, per_scale):
            if m_s == 0:
                continue
            omega_blocks.append(rng.normal(size=(m_s, d)) * s)
            scales_per_freq.append(np.full(m_s, s))
        omega = np.concatenate(omega_blocks, axis=0)
        scales_per_freq = np.concatenate(scales_per_freq, axis=0)
        return omega, scales_per_freq

    def _newton_directions(
        self,
        alpha: np.ndarray,        # (k, M2)
        phi: np.ndarray,          # (n, M2)
        data_moments: np.ndarray, # (k, M2)
    ) -> np.ndarray:
        """Compute the per-cluster Newton ascent direction. Returns ``(k, M2)``."""
        n, m2 = phi.shape
        k = alpha.shape[0]
        logits = phi @ alpha.T                                    # (n, k)
        log_q = logits - logsumexp(logits, axis=0, keepdims=True)
        q = np.exp(log_q)                                         # cols sum to 1
        step = np.zeros_like(alpha)
        eye = self.l2 * np.eye(m2)
        for j in range(k):
            qj = q[:, j]
            phi_w = phi * qj[:, None]
            m_j = phi_w.sum(axis=0)
            cov_j = phi_w.T @ phi - np.outer(m_j, m_j)
            H = cov_j + eye
            g = data_moments[j] - m_j - self.l2 * alpha[j]
            try:
                c, low = cho_factor(H, lower=True)
                step[j] = cho_solve((c, low), g)
            except np.linalg.LinAlgError:
                step[j] = np.linalg.lstsq(H, g, rcond=None)[0]
        return step

    def _fit_one(
        self,
        X: np.ndarray,
        k: int,
        phi: np.ndarray,
        omega: np.ndarray,
        rng: np.random.Generator,
    ) -> Tuple[AlgoResult, float]:
        n, d = X.shape
        m2 = phi.shape[1]
        eps = 1e-12
        trajectory: list[Step] = []

        trajectory.append(
            Step(
                step_idx=0,
                cost=0.0,
                accepted=True,
                action={
                    "type": "fourier_basis",
                    "n_frequencies": int(self.n_frequencies),
                    "n_scales": int(self.n_scales),
                },
                state={"feature_dim": int(m2)},
            )
        )

        # --- k-means warm start for alpha_k.
        # Set alpha_k to point ``phi`` toward cluster k's mean, but keep
        # the magnitude small so the softmax stays in its responsive
        # regime — a saturated softmax (||alpha|| >> 1) has near-zero
        # gradient and traps Newton at the very first iteration.
        km = KMeans(
            n_clusters=k, n_init=5, random_state=self.random_state, max_iter=100
        ).fit(X)
        init_labels = km.labels_
        alpha = np.zeros((k, m2))
        global_mean = phi.mean(axis=0)
        # Standard-deviation-based scaling keeps each feature contribution
        # O(1), so ||alpha . phi(x)|| stays around unit magnitude.
        phi_std = phi.std(axis=0) + eps
        init_strength = 1.0 / np.sqrt(m2)
        for j in range(k):
            members = phi[init_labels == j]
            if members.size:
                alpha[j] = init_strength * (members.mean(axis=0) - global_mean) / phi_std
        # Small nudge breaks any residual saddle in the Q-function.
        alpha = alpha + rng.normal(scale=1e-3, size=alpha.shape)
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

        def _ll(alpha_eval: np.ndarray, pi_eval: np.ndarray) -> Tuple[float, np.ndarray]:
            logits = phi @ alpha_eval.T
            log_Zk = logsumexp(logits, axis=0)
            log_pkx = logits - log_Zk[None, :]
            log_post = log_pkx + np.log(pi_eval + eps)[None, :]
            log_marg = logsumexp(log_post, axis=1, keepdims=True)
            log_post = log_post - log_marg
            return float(log_marg.sum()), np.exp(log_post)

        # --- EM loop with self-normalised log-partition and Newton M-step.
        ll, gamma = _ll(alpha, pi)
        em_iter = 0
        for em_iter in range(self.max_iter):
            # M-step. Closed-form pi; Newton direction for alpha with
            # backtracking line search so EM stays monotone.
            Nk = gamma.sum(axis=0) + eps
            new_pi = Nk / n
            data_moments = (gamma.T @ phi) / Nk[:, None]
            cur_alpha = alpha
            cur_ll = ll
            cur_gamma = gamma
            last_step_size = 0.0
            for _ in range(self.n_inner_iter):
                direction = self._newton_directions(cur_alpha, phi, data_moments)
                step_size = self.damping
                improved = False
                for _bt in range(6):  # at most 6 halvings
                    cand_alpha = cur_alpha + step_size * direction
                    cand_ll, cand_gamma = _ll(cand_alpha, new_pi)
                    if cand_ll >= cur_ll - 1e-9:
                        cur_alpha = cand_alpha
                        cur_ll = cand_ll
                        cur_gamma = cand_gamma
                        last_step_size = step_size
                        improved = True
                        break
                    step_size *= 0.5
                if not improved:
                    break
            new_alpha, new_ll, new_gamma = cur_alpha, cur_ll, cur_gamma
            if last_step_size == 0.0:
                # Even the smallest step didn't improve ll; refresh gamma
                # against the new pi so the next iteration still uses a
                # coherent E-step.
                _, new_gamma = _ll(alpha, new_pi)

            delta = new_ll - ll
            alpha, pi, gamma, prev_ll, ll = new_alpha, new_pi, new_gamma, ll, new_ll
            trajectory.append(
                Step(
                    step_idx=2 + em_iter,
                    cost=-ll,
                    delta_cost=-delta,
                    accepted=True,
                    action={
                        "type": "newton_step",
                        "em_iter": int(em_iter),
                        "step_size": float(last_step_size),
                    },
                    state={
                        "log_likelihood": ll,
                        "alpha_norm": float(np.linalg.norm(alpha)),
                    },
                )
            )
            if em_iter > 1 and abs(delta) < self.tol * max(abs(ll), 1.0):
                break
        prev_ll = ll

        labels = gamma.argmax(axis=1).astype(np.int64)
        n_params = k * m2 + (k - 1)
        ll_final = float(prev_ll if prev_ll is not None else float("nan"))
        bic = -2.0 * ll_final + n_params * np.log(max(n, 1))

        return AlgoResult(
            labels=labels,
            extra={
                "log_likelihood": ll_final,
                "bic": float(bic),
                "n_params": int(n_params),
                "n_em_iter": int(em_iter + 1),
                "n_frequencies": int(self.n_frequencies),
                "n_scales": int(self.n_scales),
                "feature_dim": int(m2),
                "alpha_norm": float(np.linalg.norm(alpha)),
            },
            trajectory=trajectory,
        ), float(bic)

    def fit_predict(self, X: np.ndarray, k: Optional[int] = None) -> AlgoResult:
        X = np.asarray(X, dtype=np.float64)
        rng = np.random.default_rng(self.random_state)

        # Sample the basis once so all candidate k values fit in the same
        # Fourier space (BIC comparisons stay apples-to-apples).
        omega, _scales_per_freq = self._sample_omega(X, rng)
        phi = self._features(X, omega)

        if k is not None:
            result, _ = self._fit_one(X, k, phi, omega, rng)
            return result

        if self.k_search is None:
            raise ValueError(
                "k must be provided, or set k_search=(k_lo, k_hi) on the algorithm."
            )

        k_lo, k_hi = self.k_search
        assert k_lo >= 1 and k_hi >= k_lo
        results: list[Tuple[int, AlgoResult, float]] = []
        for kk in range(k_lo, k_hi + 1):
            res, bic = self._fit_one(X, kk, phi, omega, rng)
            results.append((kk, res, bic))
        best_k, best_res, best_bic = min(results, key=lambda t: t[2])
        bic_profile = [{"k": kk, "bic": bic} for kk, _, bic in results]
        # Tack a final summary step onto the chosen model's trajectory.
        best_res.trajectory = (best_res.trajectory or []) + [
            Step(
                step_idx=(best_res.trajectory[-1].step_idx + 1 if best_res.trajectory else 0),
                cost=best_bic,
                accepted=True,
                action={"type": "k_search", "best_k": int(best_k)},
                state={"bic_profile": bic_profile},
            )
        ]
        best_res.extra["k_search_best_k"] = int(best_k)
        best_res.extra["k_search_bic_profile"] = bic_profile
        return best_res
