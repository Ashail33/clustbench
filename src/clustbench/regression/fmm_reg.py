"""FmmReg: Mixture-of-Experts on Fourier features (the FMM cousin for regression).

Generative story::

    phi(x) = [cos(omega_1 . x), sin(omega_1 . x), ...]    # M2 features
    pi_k(x) ∝ exp(alpha_k . phi(x))                       # gating, softmax
    y | x, k ~ N(beta_k . phi(x), sigma^2)                # per-expert prediction

    p(y | x) = Σ_k pi_k(x) * N(y; beta_k phi(x), sigma^2)

EM:
  * E-step: responsibilities r_ik = pi_k(x_i) * N(y_i; mu_ik, sigma^2)
    / sum_l (...), where mu_ik = beta_k . phi(x_i).
  * M-step:
      - sigma^2 ← Σ_i,k r_ik (y_i - mu_ik)^2 / N (single shared noise).
      - beta_k ← weighted ridge regression with weights r_:k. Closed
        form: beta_k = (Phi^T diag(r_:k) Phi + lam I)^{-1} Phi^T (r_:k * y).
      - alpha (gating) ← one Newton step on the softmax cross-entropy
        between predicted pi(x_i) and target r_i:. Same machinery as
        the clustering FMM's M-step.

At inference: predict pi_k(x_test) from the gating, beta_k . phi(x_test)
from each expert, blend with pi.
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np
from scipy.special import logsumexp

from ..algorithms.base import Step
from .base import RegResult, Regressor, register_regressor


@register_regressor
class FmmReg(Regressor):
    """Mixture-of-Experts on random Fourier features.

    Parameters
    ----------
    n_components : int
        Number of experts (K).
    n_frequencies : int
        Number of random Fourier frequencies (output dim is 2 *
        ``n_frequencies`` because of the cos/sin pair).
    freq_scale : float or "auto"
        Std of the random Gaussian frequencies. ``"auto"`` uses
        ``1 / median(pairwise distance)`` (RFF heuristic).
    max_iter : int
        Outer EM iterations.
    expert_l2 : float
        Ridge regulariser for per-expert beta solve.
    gating_l2 : float
        Ridge regulariser for gating alpha Newton step.
    gating_damping : float
        Step size for gating Newton update in (0, 1].
    tol : float
        Stop when relative log-likelihood change is below this.
    random_state : int
        RNG seed.
    """

    def __init__(
        self,
        n_components: int = 4,
        n_frequencies: int = 32,
        freq_scale: Any = "auto",
        max_iter: int = 50,
        expert_l2: float = 1e-3,
        gating_l2: float = 1e-2,
        gating_damping: float = 0.5,
        tol: float = 1e-4,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        self.name = "fmm_reg"
        self.n_components = int(n_components)
        self.n_frequencies = int(n_frequencies)
        self.freq_scale = freq_scale
        self.max_iter = int(max_iter)
        self.expert_l2 = float(expert_l2)
        self.gating_l2 = float(gating_l2)
        self.gating_damping = float(gating_damping)
        self.tol = float(tol)
        self.random_state = int(random_state)
        # Learned state — populated by ``fit``.
        self.omega_: np.ndarray | None = None
        self.alpha_: np.ndarray | None = None
        self.beta_: np.ndarray | None = None
        self.sigma2_: float | None = None
        self.trajectory_: list[Step] = []

    @staticmethod
    def _phi(X: np.ndarray, omega: np.ndarray) -> np.ndarray:
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
        return max(float(np.median(d[iu])), 1e-12)

    def _gating_newton_step(
        self,
        alpha: np.ndarray,   # (K, M2)
        phi: np.ndarray,     # (n, M2)
        r: np.ndarray,       # (n, K) soft labels
    ) -> np.ndarray:
        """One damped Newton step on softmax cross-entropy with soft targets.

        The objective is
            L(alpha) = Σ_i Σ_k r_ik (alpha_k . phi_i) - Σ_i logsumexp_k(alpha_k . phi_i)
                      - (l2/2) ||alpha||^2.

        Gradient and Hessian factor cleanly per row of alpha:
            g_k = Σ_i (r_ik - pi_ik) phi_i - l2 alpha_k
            H_k = -Σ_i pi_ik (1 - pi_ik) phi_i phi_i^T - l2 I
        Per-expert solve via Cholesky.
        """
        n, m2 = phi.shape
        k = alpha.shape[0]
        logits = phi @ alpha.T                              # (n, K)
        log_pi = logits - logsumexp(logits, axis=1, keepdims=True)
        pi = np.exp(log_pi)                                 # (n, K)

        eye = np.eye(m2)
        step = np.zeros_like(alpha)
        for j in range(k):
            w = pi[:, j] * (1.0 - pi[:, j]) + 1e-12          # (n,)
            # H_j = -(phi.T diag(w) phi + l2 I)
            phi_w = phi * w[:, None]
            H = phi_w.T @ phi + self.gating_l2 * eye
            g = phi.T @ (r[:, j] - pi[:, j]) - self.gating_l2 * alpha[j]
            try:
                step[j] = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                step[j] = np.linalg.lstsq(H, g, rcond=None)[0]
        return alpha + self.gating_damping * step

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FmmReg":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n, d = X.shape
        K = self.n_components
        rng = np.random.default_rng(self.random_state)
        eps = 1e-12
        self.trajectory_ = []

        # --- 1. Fourier basis.
        if isinstance(self.freq_scale, str) and self.freq_scale == "auto":
            base = 1.0 / self._median_pairwise(X, rng)
        else:
            base = float(self.freq_scale)
        omega = rng.normal(size=(self.n_frequencies, d)) * base
        self.omega_ = omega
        phi = self._phi(X, omega)
        m2 = phi.shape[1]
        self.trajectory_.append(
            Step(step_idx=0, cost=0.0, accepted=True,
                 action={"type": "fourier_basis", "n_frequencies": int(self.n_frequencies)},
                 state={"feature_dim": int(m2), "freq_scale": float(base)})
        )

        # --- 2. Init: cluster the (X, y) joint via k-means in phi, then
        # warm-start beta_k with per-cluster ridge fit on y.
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=K, n_init=3, random_state=self.random_state).fit(phi)
        init_labels = km.labels_

        alpha = np.zeros((K, m2))
        beta = np.zeros((K, m2))
        global_mean_phi = phi.mean(axis=0)
        phi_std = phi.std(axis=0) + eps
        for j in range(K):
            mask = init_labels == j
            if mask.sum() >= 2:
                # alpha_k pushes pi_k high inside cluster j.
                alpha[j] = (1.0 / np.sqrt(m2)) * (phi[mask].mean(axis=0) - global_mean_phi) / phi_std
                # beta_k = ridge fit on points in cluster j.
                Pj = phi[mask]
                yj = y[mask]
                H = Pj.T @ Pj + self.expert_l2 * np.eye(m2)
                beta[j] = np.linalg.solve(H, Pj.T @ yj)
        alpha += rng.normal(scale=1e-3, size=alpha.shape)

        sigma2 = float(np.var(y - (phi @ beta.T)[np.arange(n), init_labels]) + 1e-6)
        self.trajectory_.append(
            Step(step_idx=1, cost=float(km.inertia_), accepted=True,
                 action={"type": "kmeans_warm_start", "n_components": int(K)},
                 state={"sigma2_init": sigma2})
        )

        # --- 3. EM loop.
        prev_ll = -np.inf
        em_iter = 0
        for em_iter in range(self.max_iter):
            # E-step.
            logits = phi @ alpha.T                          # (n, K)
            log_pi = logits - logsumexp(logits, axis=1, keepdims=True)
            preds = phi @ beta.T                            # (n, K) per-expert mean
            residuals = (y[:, None] - preds)
            log_lik = -0.5 * (residuals ** 2) / sigma2 - 0.5 * np.log(2.0 * np.pi * sigma2)
            log_joint = log_pi + log_lik
            log_marg = logsumexp(log_joint, axis=1, keepdims=True)
            log_post = log_joint - log_marg
            r = np.exp(log_post)                            # (n, K), rows sum to 1
            ll = float(log_marg.sum())

            # M-step.
            Nk = r.sum(axis=0) + eps                        # (K,)

            # Update sigma^2 (single shared noise).
            sigma2 = float((r * (residuals ** 2)).sum() / n + 1e-8)

            # Update beta_k via per-expert weighted ridge.
            # (Phi^T diag(r_:k) Phi + lam I) beta_k = Phi^T (r_:k * y)
            for j in range(K):
                wj = r[:, j]
                Pw = phi * wj[:, None]
                Hj = Pw.T @ phi + self.expert_l2 * np.eye(m2)
                bj = Pw.T @ y
                try:
                    beta[j] = np.linalg.solve(Hj, bj)
                except np.linalg.LinAlgError:
                    beta[j] = np.linalg.lstsq(Hj, bj, rcond=None)[0]

            # Update alpha (gating) via one Newton step.
            alpha = self._gating_newton_step(alpha, phi, r)

            delta = ll - prev_ll
            self.trajectory_.append(
                Step(step_idx=2 + em_iter, cost=-ll, delta_cost=-delta, accepted=True,
                     action={"type": "em_step", "em_iter": int(em_iter)},
                     state={"log_likelihood": ll, "sigma2": sigma2,
                            "expert_norm": float(np.linalg.norm(beta)),
                            "gate_norm": float(np.linalg.norm(alpha))})
            )
            if em_iter > 1 and abs(delta) < self.tol * max(abs(ll), 1.0):
                break
            prev_ll = ll

        self.alpha_ = alpha
        self.beta_ = beta
        self.sigma2_ = sigma2
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.omega_ is None or self.alpha_ is None or self.beta_ is None:
            raise RuntimeError("FmmReg.predict called before fit.")
        phi = self._phi(np.asarray(X, dtype=np.float64), self.omega_)
        logits = phi @ self.alpha_.T
        log_pi = logits - logsumexp(logits, axis=1, keepdims=True)
        pi = np.exp(log_pi)                                  # (n, K)
        preds = phi @ self.beta_.T                           # (n, K)
        return (pi * preds).sum(axis=1)

    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> RegResult:
        self.fit(X, y)
        preds = self.predict(X)
        return RegResult(
            predictions=preds,
            extra={
                "n_components": int(self.n_components),
                "n_frequencies": int(self.n_frequencies),
                "feature_dim": int(2 * self.n_frequencies),
                "sigma2": float(self.sigma2_ or 0.0),
            },
            trajectory=self.trajectory_,
        )
