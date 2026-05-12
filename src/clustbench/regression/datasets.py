"""Synthetic regression generators tuned to test mixture-of-experts.

Each generator returns ``(X, y)`` where ``X.shape == (n, d)`` and
``y.shape == (n,)``. The data is designed so a global linear model
(ridge) underperforms while a piecewise-linear model (mixture of
experts) recovers ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RegSpec:
    n_samples: int = 1000
    n_features: int = 5
    n_components: int = 3
    noise: float = 0.1
    seed: int = 1


def gen_piecewise_linear(spec: RegSpec):
    """``K`` linear regimes glued together — each cluster has its own
    linear y = w_k.x + b_k, plus Gaussian noise. Global linear ridge
    fails; per-cluster ridge succeeds.
    """
    rng = np.random.default_rng(spec.seed)
    X = rng.standard_normal((spec.n_samples, spec.n_features))
    # Assign each point to one of K regions via random center proximity.
    centers = rng.standard_normal((spec.n_components, spec.n_features)) * 2.0
    d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
    labels = d2.argmin(axis=1)
    # Per-region coefficients.
    weights = rng.standard_normal((spec.n_components, spec.n_features)) * 1.5
    biases = rng.standard_normal(spec.n_components) * 1.0
    y = np.array([
        X[i] @ weights[labels[i]] + biases[labels[i]] for i in range(spec.n_samples)
    ])
    y = y + rng.normal(scale=spec.noise, size=spec.n_samples)
    return X.astype(np.float32), y.astype(np.float32)


def gen_friedman1(spec: RegSpec):
    """Friedman's well-known nonlinear regression benchmark.

    ``y = 10 sin(pi x_0 x_1) + 20 (x_2 - 0.5)^2 + 10 x_3 + 5 x_4 + noise``
    plus optional irrelevant features that the model must learn to
    ignore (drives toward at least ``5`` features).
    """
    from sklearn.datasets import make_friedman1
    n_features = max(5, spec.n_features)
    X, y = make_friedman1(
        n_samples=spec.n_samples,
        n_features=n_features,
        noise=spec.noise,
        random_state=spec.seed,
    )
    return X.astype(np.float32), y.astype(np.float32)


def gen_sinusoidal(spec: RegSpec):
    """Univariate ``y = sin(pi x) + noise`` with optional irrelevant
    features filling out to ``spec.n_features``."""
    rng = np.random.default_rng(spec.seed)
    x = rng.uniform(-2.0, 2.0, size=spec.n_samples)
    y = np.sin(np.pi * x) + rng.normal(scale=spec.noise, size=spec.n_samples)
    extra = rng.standard_normal((spec.n_samples, max(0, spec.n_features - 1))) * 0.5
    X = np.concatenate([x[:, None], extra], axis=1) if extra.size else x[:, None]
    return X.astype(np.float32), y.astype(np.float32)


def gen_regime_switch_linear(spec: RegSpec):
    """Time-varying linear regression with regime switches.

    The covariates are ``[t, e_1, ..., e_{d-1}]`` where ``t`` is a
    normalised timestamp in ``[0, 1)`` and ``e_*`` are exogenous noise
    features. The dataset is split into ``spec.n_components`` equal
    time segments; each segment has its own linear regression
    coefficients. This is the textbook regime-switching regression
    setup that mixture-of-experts is explicitly designed for.
    """
    rng = np.random.default_rng(spec.seed)
    n = spec.n_samples
    K = spec.n_components
    t = np.arange(n, dtype=np.float64) / n
    exog = rng.standard_normal((n, max(0, spec.n_features - 1)))
    X = np.concatenate([t[:, None], exog], axis=1) if exog.size else t[:, None]
    regime = np.floor(t * K).astype(int).clip(0, K - 1)

    weights = rng.standard_normal((K, X.shape[1])) * 1.5
    biases = rng.standard_normal(K) * 2.0
    y = np.einsum("nd,nd->n", X, weights[regime]) + biases[regime]
    y = y + rng.normal(scale=spec.noise, size=n)
    return X.astype(np.float32), y.astype(np.float32)


def gen_piecewise_polynomial(spec: RegSpec):
    """K segments, each fitting a polynomial of t of degree in {1, 2, 3}.

    ``X = t`` (single feature), ``y`` is piecewise polynomial. Tests
    whether the regressor can pick up local *order* changes (linear in
    one segment, cubic in another) — a thing mixture-of-experts should
    handle by routing each segment to a different expert.
    """
    rng = np.random.default_rng(spec.seed)
    n = spec.n_samples
    K = spec.n_components
    t = np.linspace(-1.0, 1.0, n)
    regime = np.floor((t + 1) / 2 * K).astype(int).clip(0, K - 1)

    orders = rng.integers(1, 4, size=K)
    coefs = [rng.standard_normal(orders[r] + 1) * 2.0 for r in range(K)]
    y = np.zeros(n)
    for r in range(K):
        mask = regime == r
        if not mask.any():
            continue
        tr = t[mask]
        c = coefs[r]
        y[mask] = sum(c[d] * (tr ** d) for d in range(len(c)))
    y = y + rng.normal(scale=spec.noise, size=n)
    return t[:, None].astype(np.float32), y.astype(np.float32)


def gen_sinusoid_drift(spec: RegSpec):
    """Chirp signal: ``y = sin(2π · ω(t) · t)`` with linearly increasing
    frequency ``ω(t) = 1 + (K - 1) · t``.

    Classical non-stationary spectral signal. RFF-based methods with a
    fixed frequency draw will struggle; methods that can adapt the
    feature representation locally (mixture of experts, SIREN-style
    networks) have a fair shot.
    """
    rng = np.random.default_rng(spec.seed)
    n = spec.n_samples
    K = spec.n_components
    t = np.linspace(0.0, 1.0, n)
    omega = 1.0 + (K - 1) * t
    y = np.sin(2.0 * np.pi * omega * t) + rng.normal(scale=spec.noise, size=n)
    return t[:, None].astype(np.float32), y.astype(np.float32)


REG_DATASETS = {
    "piecewise_linear": gen_piecewise_linear,
    "friedman1": gen_friedman1,
    "sinusoidal": gen_sinusoidal,
    "regime_switch_linear": gen_regime_switch_linear,
    "piecewise_polynomial": gen_piecewise_polynomial,
    "sinusoid_drift": gen_sinusoid_drift,
}
