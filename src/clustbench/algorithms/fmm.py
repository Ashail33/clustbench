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

Heat-kernel per-cluster bandwidth
---------------------------------
Each cluster also carries a learnable scalar ``tau_k >= 0`` that
applies a heat-kernel weighting to its Fourier coefficients,
``W_m(tau_k) = exp(-tau_k * ||omega_m||^2 / 2)``. Small ``tau_k`` lets
the cluster resolve fine structure; large ``tau_k`` smooths it. The
effective amplitudes used to evaluate density are
``alpha_k_eff = alpha_k * W(tau_k)``. ``tau_k`` is updated once per
M-step by a coarse log-scale line search.

BIC and permutation-aware k-search
----------------------------------
Self-normalised log-likelihood is well-defined over the training set,
so the standard BIC

    BIC      = -2 * log_likelihood + p * log(n),
    p        = k * (2M + 1) + (k - 1)          # nominal alpha, tau, pi
    BIC_eff  = -2 * log_likelihood + edf * log(n),
    edf      = Σ_k trace(H_k . (H_k + l2 I)^{-1}) + tau + (k - 1)

are both reported in ``extra``. ``BIC_eff`` uses *effective* degrees
of freedom (Hessian trace), which is the right notion of complexity
for an L2-regularised RFF model — most alpha entries are heavily
shrunken and count as a fraction of a parameter, not a full one.
``k_search`` ranks models by ``BIC_eff``. With ``n_basis_samples > 1``
the BIC is averaged over several independent random Fourier bases, so
the selected ``k`` is robust to the RFF draw.

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
        Base ridge regulariser on ``alpha_k``. Also keeps the Newton
        Hessian positive-definite even when ``Cov_q[phi]`` is
        rank-deficient.
    adaptive_l2 : bool
        Scale the per-cluster ``l2`` by ``max(1, n_features / N_k)``
        each iteration. This protects high-k fits (where each cluster
        has few responsibility-weighted points) from overfitting the
        2M-dim basis without hurting low-k cases.
    damping : float
        Maximum Newton step size in (0, 1]. The actual step is chosen
        by backtracking so the marginal log-likelihood never decreases.
    tol : float
        Stop when the relative change in log-likelihood is below this.
    learn_bandwidth : bool
        Whether to learn a per-cluster heat-kernel bandwidth ``tau_k``.
    tau_init : float
        Initial value of ``tau_k`` (in the diffusion-time units of
        ``exp(-tau * ||omega||^2 / 2)``). ``0.0`` disables initial
        smoothing.
    tau_step : float
        Step size on ``log(tau + 1e-3)`` for the per-cluster line
        search. Larger values explore bandwidth space faster at the
        cost of more line-search candidates.
    k_search : tuple[int, int] | None
        ``(k_lo, k_hi)`` inclusive. When ``k`` is ``None`` at call time,
        fits FMM for every ``k`` in the range and returns the
        minimum-BIC model.
    n_basis_samples : int
        Number of independent random Fourier bases used when
        ``k_search`` is active. BIC is averaged across bases so the
        chosen ``k`` is robust to the RFF draw. For a single ``k`` only
        the first basis is used.
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
        adaptive_l2: bool = True,
        damping: float = 0.2,
        tol: float = 1e-4,
        learn_bandwidth: bool = True,
        tau_init: float = 0.0,
        tau_step: float = 0.5,
        k_search: Optional[Tuple[int, int]] = None,
        n_basis_samples: int = 1,
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
        self.adaptive_l2 = adaptive_l2
        self.damping = damping
        self.tol = tol
        self.learn_bandwidth = learn_bandwidth
        self.tau_init = float(tau_init)
        self.tau_step = float(tau_step)
        self.k_search = k_search
        self.n_basis_samples = max(1, int(n_basis_samples))
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

    @staticmethod
    def _heat_weights(tau: np.ndarray, omega_norm_sq: np.ndarray) -> np.ndarray:
        """Return ``W[k, m] = exp(-0.5 * tau[k] * omega_norm_sq[m])``."""
        return np.exp(-0.5 * tau[:, None] * omega_norm_sq[None, :])

    def _newton_directions(
        self,
        alpha: np.ndarray,         # (k, M2)
        phi: np.ndarray,           # (n, M2)
        data_moments: np.ndarray,  # (k, M2)
        W: np.ndarray,             # (k, M2) heat-kernel weights
        l2_per_cluster: np.ndarray, # (k,)
    ) -> np.ndarray:
        """Per-cluster Newton ascent direction for ``alpha`` at fixed ``tau``.

        All ``k`` per-cluster Hessians/gradients are stacked into a single
        ``(k, M2, M2)`` block and solved with one batched LAPACK call
        (``np.linalg.solve``) — about an order of magnitude faster than
        ``k`` independent ``cho_factor`` invocations.
        """
        n, m2 = phi.shape
        k = alpha.shape[0]
        alpha_eff = alpha * W
        logits = phi @ alpha_eff.T
        log_q = logits - logsumexp(logits, axis=0, keepdims=True)
        q = np.exp(log_q)                                              # (n, k)

        # Per-cluster Cov_q[phi]: vectorise over k using a single batched
        # gemm (faster than k separate matmuls — one BLAS call, no
        # per-cluster Python overhead).
        #   m[j, m]        = sum_i q[i, j] phi[i, m]
        #   cov[j, m, l]   = sum_i q[i, j] phi[i, m] phi[i, l] - m[j, m] m[j, l]
        m_k = q.T @ phi                                                # (k, M2)
        # phi_weighted_T[j, m, i] = q[i, j] * phi[i, m]
        phi_weighted_T = phi.T[None, :, :] * q.T[:, None, :]           # (k, M2, n)
        cov_outer = phi_weighted_T @ phi                               # (k, M2, M2)
        cov = cov_outer - m_k[:, :, None] * m_k[:, None, :]            # (k, M2, M2)

        # Hessian sandwich: H_j = W_j Cov W_j + l2_j I
        W_outer = W[:, :, None] * W[:, None, :]                        # (k, M2, M2)
        H = cov * W_outer
        # Add l2_j * I to each cluster's H.
        H_diag = H.reshape(k, m2 * m2)
        H_diag[:, :: m2 + 1] += l2_per_cluster[:, None]
        H = H_diag.reshape(k, m2, m2)

        g = W * (data_moments - m_k) - l2_per_cluster[:, None] * alpha # (k, M2)

        try:
            # numpy.linalg.solve treats b.ndim == 1 as a single RHS, so
            # we add a trailing axis to get the batched signature
            # (k, M2, M2), (k, M2, 1) -> (k, M2, 1).
            step = np.linalg.solve(H, g[..., None])[..., 0]
        except np.linalg.LinAlgError:
            # Per-cluster lstsq fallback for any singular Hessian.
            step = np.zeros_like(alpha)
            for j in range(k):
                step[j] = np.linalg.lstsq(H[j], g[j], rcond=None)[0]
        return step

    def _basis_action(self) -> dict:
        """Subclasses describe their basis here (appears in the trajectory)."""
        return {
            "type": "fourier_basis",
            "n_frequencies": int(self.n_frequencies),
            "n_scales": int(self.n_scales),
        }

    def _init_alpha(
        self,
        X: np.ndarray,
        k: int,
        phi: np.ndarray,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, dict, dict]:
        """Warm-start alpha from a k-means partition of ``X``.

        Returns ``(alpha, action, state)`` where ``action`` and ``state``
        describe the init for the trajectory.
        """
        eps = 1e-12
        m2 = phi.shape[1]
        km = KMeans(
            n_clusters=k, n_init=3, random_state=self.random_state, max_iter=100
        ).fit(X)
        init_labels = km.labels_
        alpha = np.zeros((k, m2))
        global_mean = phi.mean(axis=0)
        phi_std = phi.std(axis=0) + eps
        init_strength = 1.0 / np.sqrt(m2)
        for j in range(k):
            members = phi[init_labels == j]
            if members.size:
                alpha[j] = init_strength * (members.mean(axis=0) - global_mean) / phi_std
        alpha = alpha + rng.normal(scale=1e-3, size=alpha.shape)
        return (
            alpha,
            {"type": "kmeans_init", "n_clusters": int(k)},
            {"init_inertia": float(km.inertia_), "cost": float(km.inertia_)},
        )

    def _fit_one(
        self,
        X: np.ndarray,
        k: int,
        phi: np.ndarray,
        omega: np.ndarray,
        omega_norm_sq: np.ndarray,
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
                action=self._basis_action(),
                state={"feature_dim": int(m2)},
            )
        )

        # --- Warm-start alpha. The default is a k-means partition of
        # ``X``; subclasses can override ``_init_alpha`` with something
        # cheaper when the basis is informative enough (e.g. LMM uses
        # k-means++ on the eigenvector basis only, skipping the
        # Lloyd loop).
        alpha, init_action, init_state = self._init_alpha(X, k, phi, rng)
        pi = np.ones(k) / k
        tau = np.full(k, self.tau_init if self.learn_bandwidth else 0.0)
        trajectory.append(
            Step(
                step_idx=1,
                cost=float(init_state.get("cost", 0.0)),
                accepted=True,
                action=init_action,
                state=init_state,
            )
        )

        def _ll(alpha_eval: np.ndarray, pi_eval: np.ndarray, tau_eval: np.ndarray) -> Tuple[float, np.ndarray]:
            W = self._heat_weights(tau_eval, omega_norm_sq)
            alpha_eff = alpha_eval * W
            logits = phi @ alpha_eff.T
            log_Zk = logsumexp(logits, axis=0)
            log_pkx = logits - log_Zk[None, :]
            log_post = log_pkx + np.log(pi_eval + eps)[None, :]
            log_marg = logsumexp(log_post, axis=1, keepdims=True)
            log_post = log_post - log_marg
            return float(log_marg.sum()), np.exp(log_post)

        def _search_tau(alpha_in: np.ndarray, tau_in: np.ndarray, pi_in: np.ndarray) -> np.ndarray:
            """Coarse 1D log-search on tau_k per cluster (alpha, pi fixed)."""
            if not self.learn_bandwidth:
                return tau_in
            # Candidate tau values per cluster: multiplicative perturbations
            # plus a "no smoothing" anchor.
            mults = np.array([np.exp(-self.tau_step), 1.0, np.exp(self.tau_step)])
            offsets = np.array([0.0, 0.05])  # additive jitter so tau=0 can leave the floor
            cand_per_cluster: list[np.ndarray] = []
            for j in range(k):
                base_vals = (tau_in[j] + offsets[:, None]) * mults[None, :]
                vals = np.unique(np.concatenate([base_vals.ravel(), [0.0]]))
                cand_per_cluster.append(np.clip(vals, 0.0, 1e3))

            # For each cluster independently, evaluate Q_k(alpha_in, tau_cand)
            # holding alpha, pi at their current values and gamma implied by
            # the current (alpha, pi). We compute Q_k = N_k * L_k where
            # L_k(tau) = (alpha * W(tau)) . d_k - log Z_k(tau).
            # That's just one matmul per candidate per cluster.
            _, gamma_local = _ll(alpha_in, pi_in, tau_in)
            Nk_local = gamma_local.sum(axis=0) + eps
            d_k_local = (gamma_local.T @ phi) / Nk_local[:, None]   # (k, M2)

            new_tau = tau_in.copy()
            for j in range(k):
                best_lk = -np.inf
                best_tau = tau_in[j]
                for tau_cand in cand_per_cluster[j]:
                    Wj = np.exp(-0.5 * tau_cand * omega_norm_sq)
                    alpha_eff_j = alpha_in[j] * Wj
                    logits = phi @ alpha_eff_j
                    log_Z = logsumexp(logits)
                    lk = float(alpha_eff_j @ d_k_local[j] - log_Z)
                    if lk > best_lk:
                        best_lk = lk
                        best_tau = float(tau_cand)
                new_tau[j] = best_tau
            return new_tau

        def _l2_for(Nk_local: np.ndarray) -> np.ndarray:
            """Adaptive ridge per cluster: scale up when N_k < n_features."""
            if not self.adaptive_l2:
                return np.full(Nk_local.shape, self.l2)
            return self.l2 * np.maximum(1.0, m2 / np.maximum(Nk_local, eps))

        # --- EM loop.
        ll, gamma = _ll(alpha, pi, tau)
        em_iter = 0
        for em_iter in range(self.max_iter):
            Nk = gamma.sum(axis=0) + eps
            new_pi = Nk / n
            data_moments = (gamma.T @ phi) / Nk[:, None]
            l2_per_cluster = _l2_for(Nk)

            cur_alpha = alpha
            cur_ll = ll
            cur_gamma = gamma
            last_step_size = 0.0
            W = self._heat_weights(tau, omega_norm_sq)
            for _ in range(self.n_inner_iter):
                direction = self._newton_directions(
                    cur_alpha, phi, data_moments, W, l2_per_cluster
                )
                step_size = self.damping
                improved = False
                for _bt in range(6):
                    cand_alpha = cur_alpha + step_size * direction
                    cand_ll, cand_gamma = _ll(cand_alpha, new_pi, tau)
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
                _, new_gamma = _ll(alpha, new_pi, tau)

            # Update tau via per-cluster line search at the new alpha.
            new_tau = _search_tau(new_alpha, tau, new_pi)
            if not np.allclose(new_tau, tau):
                tau_ll, tau_gamma = _ll(new_alpha, new_pi, new_tau)
                if tau_ll >= new_ll - 1e-9:
                    new_ll = tau_ll
                    new_gamma = tau_gamma
                else:
                    # The grid pick somehow hurt the marginal ll
                    # (per-cluster Q_k can rise while marginal falls if
                    # responsibilities reshuffle). Roll tau back.
                    new_tau = tau

            delta = new_ll - ll
            alpha, pi, gamma, tau, ll = new_alpha, new_pi, new_gamma, new_tau, new_ll
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
                        "tau_mean": float(tau.mean()),
                        "tau_max": float(tau.max()),
                    },
                )
            )
            if em_iter > 1 and abs(delta) < self.tol * max(abs(ll), 1.0):
                break

        labels = gamma.argmax(axis=1).astype(np.int64)

        # Nominal parameter count (every alpha entry counted as 1 d.o.f.).
        tau_params = k if self.learn_bandwidth else 0
        n_params = k * m2 + tau_params + (k - 1)
        bic = -2.0 * ll + n_params * np.log(max(n, 1))

        # Effective d.o.f. accounting for L2 shrinkage:
        #   edf_k = trace(H_k @ (H_k + l2 I)^{-1})
        #         = sum_i lambda_i / (lambda_i + l2)
        # where ``lambda_i`` are eigenvalues of the per-cluster Hessian
        # sandwich ``W_k . Cov_q[phi] . W_k``. With heavy regularisation
        # most alpha components count as a small fraction of a parameter,
        # which keeps the BIC penalty in line with the model's actual
        # complexity.
        W_final = self._heat_weights(tau, omega_norm_sq)
        alpha_eff_final = alpha * W_final
        logits_final = phi @ alpha_eff_final.T
        q_final = np.exp(logits_final - logsumexp(logits_final, axis=0, keepdims=True))
        Nk_final = gamma.sum(axis=0) + eps
        l2_final = _l2_for(Nk_final)
        edf = 0.0
        for j in range(k):
            qj = q_final[:, j]
            phi_w = phi * qj[:, None]
            m_j = phi_w.sum(axis=0)
            cov_j = phi_w.T @ phi - np.outer(m_j, m_j)
            Wj = W_final[j]
            H_j = (Wj[:, None] * cov_j) * Wj[None, :]
            evals = np.linalg.eigvalsh(H_j)
            evals = np.clip(evals, 0.0, None)
            edf += float(np.sum(evals / (evals + l2_final[j])))
        edf += tau_params + (k - 1)
        bic_eff = -2.0 * ll + edf * np.log(max(n, 1))

        return AlgoResult(
            labels=labels,
            extra={
                "log_likelihood": float(ll),
                "bic": float(bic),
                "bic_eff": float(bic_eff),
                "n_params": int(n_params),
                "edf": float(edf),
                "n_em_iter": int(em_iter + 1),
                "n_frequencies": int(self.n_frequencies),
                "n_scales": int(self.n_scales),
                "feature_dim": int(m2),
                "alpha_norm": float(np.linalg.norm(alpha)),
                "tau_mean": float(tau.mean()),
                "tau_min": float(tau.min()),
                "tau_max": float(tau.max()),
                "learn_bandwidth": bool(self.learn_bandwidth),
            },
            trajectory=trajectory,
        ), float(bic_eff)

    def _omega_norm_sq(self, omega: np.ndarray) -> np.ndarray:
        """Squared frequency norms, repeated for the cos/sin pair (length ``2M``)."""
        nq = (omega * omega).sum(axis=1)
        return np.concatenate([nq, nq])

    def _build_basis(
        self, X: np.ndarray, rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Construct the basis features and per-feature heat-kernel arguments.

        Returns ``(omega, phi, omega_norm_sq)`` where ``omega`` is opaque to
        the EM loop (kept for trajectory introspection), ``phi`` has shape
        ``(n, M2)`` and ``omega_norm_sq`` has shape ``(M2,)`` and is used by
        the heat kernel ``exp(-tau * omega_norm_sq / 2)``.

        Subclasses (e.g. ``Lmm``) override this to swap in a different basis.
        """
        omega, _ = self._sample_omega(X, rng)
        phi = self._features(X, omega)
        omega_norm_sq = self._omega_norm_sq(omega)
        return omega, phi, omega_norm_sq

    def fit_predict(self, X: np.ndarray, k: Optional[int] = None) -> AlgoResult:
        X = np.asarray(X, dtype=np.float64)
        rng = np.random.default_rng(self.random_state)

        # For a single ``k`` we only need one basis — sample it and fit.
        if k is not None:
            omega, phi, omega_norm_sq = self._build_basis(X, rng)
            result, _ = self._fit_one(X, k, phi, omega, omega_norm_sq, rng)
            return result

        if self.k_search is None:
            raise ValueError(
                "k must be provided, or set k_search=(k_lo, k_hi) on the algorithm."
            )

        k_lo, k_hi = self.k_search
        assert k_lo >= 1 and k_hi >= k_lo
        ks = list(range(k_lo, k_hi + 1))

        # Multi-basis BIC: fit each k against ``n_basis_samples`` Fourier
        # bases and average BIC across bases. Within one basis, all k
        # share the same Fourier features so their BICs are apples-to-
        # apples; averaging then removes the basis-draw variance.
        per_basis_results: list[dict[int, Tuple[AlgoResult, float]]] = []
        for _basis_idx in range(self.n_basis_samples):
            omega, phi, omega_norm_sq = self._build_basis(X, rng)
            k_map: dict[int, Tuple[AlgoResult, float]] = {}
            for kk in ks:
                res, bic = self._fit_one(X, kk, phi, omega, omega_norm_sq, rng)
                k_map[kk] = (res, bic)
            per_basis_results.append(k_map)

        # Average BIC across bases for each k.
        mean_bic = {kk: float(np.mean([b[kk][1] for b in per_basis_results])) for kk in ks}
        best_k = min(ks, key=lambda kk: mean_bic[kk])
        # Pick the representative model: the basis closest to mean BIC.
        bics_at_best = np.array([b[best_k][1] for b in per_basis_results])
        pick = int(np.argmin(np.abs(bics_at_best - mean_bic[best_k])))
        best_res, best_bic = per_basis_results[pick][best_k]

        bic_profile = [
            {"k": kk, "bic_mean": mean_bic[kk],
             "bic_per_basis": [b[kk][1] for b in per_basis_results]}
            for kk in ks
        ]
        best_res.trajectory = (best_res.trajectory or []) + [
            Step(
                step_idx=(best_res.trajectory[-1].step_idx + 1 if best_res.trajectory else 0),
                cost=best_bic,
                accepted=True,
                action={
                    "type": "k_search",
                    "best_k": int(best_k),
                    "n_basis_samples": int(self.n_basis_samples),
                },
                state={"bic_profile": bic_profile},
            )
        ]
        best_res.extra["k_search_best_k"] = int(best_k)
        best_res.extra["k_search_bic_profile"] = bic_profile
        best_res.extra["k_search_n_basis_samples"] = int(self.n_basis_samples)
        return best_res
