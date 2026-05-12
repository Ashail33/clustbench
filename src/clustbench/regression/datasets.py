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


REG_DATASETS = {
    "piecewise_linear": gen_piecewise_linear,
    "friedman1": gen_friedman1,
    "sinusoidal": gen_sinusoidal,
}
