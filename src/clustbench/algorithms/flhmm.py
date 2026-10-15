"""FLHMM: clustering on Fourier, Laplace and Hilbert transform features.

A tabular data point is not a signal, so each classical transform is
applied to something that *is* a function, and evaluated per point:

* **Fourier block.** The data's characteristic function, sampled through
  random Fourier features ``[cos(w.x), sin(w.x)]`` at multi-scale random
  frequencies (the same basis as :class:`~clustbench.algorithms.fmm.Fmm`).
* **Laplace block.** For each point the Laplace transform of its
  distance distribution to a landmark sample,
  ``L_i(s) = mean_j exp(-s * d_ij / d_med)``, on a log-spaced grid of ``s``.
  Small ``s`` sees the global layout, large ``s`` only the immediate
  neighbourhood, so the curve is a multi-scale local-density signature.
* **Hilbert block.** Along a few projection directions (leading principal
  axes plus random ones) the 1-D density ``f(t)`` of the projected data is
  estimated on a grid and turned into its analytic signal
  ``f + i H[f]``. Each point gets the envelope (local density along that
  direction) and ``cos``/``sin`` of the instantaneous phase at its
  projected coordinate; the phase advances by about one cycle per density
  lobe, so it encodes *which mode* a point sits in.

Each block is centred, standardised and scaled to equal total variance,
then concatenated. ``head="mixture"`` fits the exponential-family EM of
:class:`Fmm` on the combined features (component log-densities linear in
them); ``head="kmeans"`` runs k-means on them instead. ``blocks`` selects a
subset for ablations.
"""
from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import numpy as np

from .base import AlgoResult, register
from .fmm import Fmm

_BLOCKS = ("fourier", "laplace", "hilbert")


def laplace_features(X: np.ndarray, rng: np.random.Generator, n_s: int = 8,
                     n_landmarks: int = 300) -> np.ndarray:
    """``L_i(s) = mean_j exp(-s * d_ij / d_med)`` over landmarks, for log-spaced s."""
    n = X.shape[0]
    idx = rng.choice(n, size=min(n, n_landmarks), replace=False)
    L = X[idx]
    d = np.sqrt(np.maximum(((X[:, None, :] - L[None, :, :]) ** 2).sum(-1), 0.0))  # (n, m)
    med = np.median(d[d > 0]) if np.any(d > 0) else 1.0
    s_grid = np.logspace(-1, 1.5, n_s)  # 0.1 .. ~31.6 in units of 1/d_med
    return np.stack([np.exp(-s * d / med).mean(axis=1) for s in s_grid], axis=1)


def hilbert_features(X: np.ndarray, rng: np.random.Generator, n_dirs: int = 6,
                     n_grid: int = 256, bandwidth: float = 0.05) -> np.ndarray:
    """Envelope and instantaneous phase of the projected density's analytic signal."""
    from scipy.signal import hilbert

    n, d = X.shape
    Xc = X - X.mean(axis=0)
    # Leading principal directions first, then random unit directions.
    try:
        _, _, Vt = np.linalg.svd(Xc[: min(n, 2000)], full_matrices=False)
        dirs = list(Vt[: min(d, max(1, n_dirs // 2))])
    except np.linalg.LinAlgError:
        dirs = []
    while len(dirs) < n_dirs:
        w = rng.standard_normal(d)
        dirs.append(w / (np.linalg.norm(w) + 1e-12))
    feats = []
    for w in dirs:
        t = Xc @ w
        lo, hi = float(t.min()), float(t.max())
        span = max(hi - lo, 1e-12)
        grid = np.linspace(lo - 0.1 * span, hi + 0.1 * span, n_grid)
        h = bandwidth * span
        # Gaussian KDE on the grid via binning + convolution (O(n + grid)).
        counts, edges = np.histogram(t, bins=n_grid, range=(grid[0], grid[-1]))
        step = edges[1] - edges[0]
        kx = np.arange(-4 * h, 4 * h + step, step)
        kern = np.exp(-0.5 * (kx / h) ** 2)
        f = np.convolve(counts.astype(float), kern / kern.sum(), mode="same")
        f = f - f.mean()  # remove DC so the phase tracks the lobes
        analytic = hilbert(f)
        env = np.abs(analytic)
        phase = np.unwrap(np.angle(analytic))
        pos = np.clip(((t - grid[0]) / step).astype(int), 0, n_grid - 1)
        feats.append(np.stack([env[pos], np.cos(phase[pos]), np.sin(phase[pos])], axis=1))
    return np.concatenate(feats, axis=1)


@register
class Flhmm(Fmm):
    """Mixture (or k-means) over Fourier, Laplace and Hilbert transform features."""

    def __init__(
        self,
        blocks: Sequence[str] = _BLOCKS,
        block_weights: Dict[str, float] | None = None,
        head: str = "mixture",
        n_frequencies: int = 32,
        n_scales: int = 3,
        n_laplace: int = 8,
        n_landmarks: int = 300,
        n_dirs: int = 6,
        n_grid: int = 256,
        kde_bandwidth: float = 0.05,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("learn_bandwidth", False)
        kwargs.setdefault("tau_init", 0.0)
        super().__init__(n_frequencies=n_frequencies, n_scales=n_scales, **kwargs)
        self.name = "flhmm"
        bad = set(blocks) - set(_BLOCKS)
        if bad or not blocks:
            raise ValueError(f"blocks must be a non-empty subset of {_BLOCKS}, got {blocks}")
        if head not in ("mixture", "kmeans"):
            raise ValueError("head must be 'mixture' or 'kmeans'")
        self.blocks = tuple(b for b in _BLOCKS if b in blocks)
        self.block_weights = dict(block_weights or {})
        self.head = head
        self.n_laplace = int(n_laplace)
        self.n_landmarks = int(n_landmarks)
        self.n_dirs = int(n_dirs)
        self.n_grid = int(n_grid)
        self.kde_bandwidth = float(kde_bandwidth)
        self._block_dims: Dict[str, int] = {}

    def _basis_action(self) -> dict:
        return {"type": "fourier_laplace_hilbert", "blocks": list(self.blocks),
                "block_dims": dict(self._block_dims), "head": self.head}

    def block_features(self, X: np.ndarray, rng: np.random.Generator | None = None) -> Dict[str, np.ndarray]:
        """Return each transform block, standardised and scaled to unit total variance."""
        X = np.asarray(X, dtype=np.float64)
        rng = rng if rng is not None else np.random.default_rng(self.random_state)
        parts: Dict[str, np.ndarray] = {}
        for name in self.blocks:
            if name == "fourier":
                omega, _ = self._sample_omega(X, rng)
                phi = self._features(X, omega)
            elif name == "laplace":
                phi = laplace_features(X, rng, self.n_laplace, self.n_landmarks)
            else:
                phi = hilbert_features(X, rng, self.n_dirs, self.n_grid, self.kde_bandwidth)
            phi = phi - phi.mean(axis=0, keepdims=True)
            sd = phi.std(axis=0)
            keep = sd > 1e-10
            phi = phi[:, keep] / sd[keep]
            phi = phi / np.sqrt(max(phi.shape[1], 1)) * float(self.block_weights.get(name, 1.0))
            self._block_dims[name] = int(phi.shape[1])
            parts[name] = phi
        return parts

    def transform(self, X: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        """Return the concatenated, equal-variance transform features."""
        return np.concatenate(list(self.block_features(X, rng).values()), axis=1)

    def _build_basis(self, X: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        phi = self.transform(X, rng)
        omega_norm_sq = np.zeros(phi.shape[1])  # heat kernel off (tau = 0)
        return omega_norm_sq.reshape(-1, 1), phi, omega_norm_sq

    def fit_predict(self, X, k=None):
        if self.head == "kmeans":
            from sklearn.cluster import KMeans

            assert k is not None, "k must be given for head='kmeans'"
            Z = self.transform(X)
            km = KMeans(n_clusters=k, n_init=10, random_state=self.random_state).fit(Z)
            return AlgoResult(labels=km.labels_.astype(np.int64),
                              extra={"blocks": list(self.blocks), "block_dims": dict(self._block_dims),
                                     "head": "kmeans", "inertia": float(km.inertia_)})
        result = super().fit_predict(X, k=k)
        if result.extra is not None:
            result.extra.update({"blocks": list(self.blocks), "block_dims": dict(self._block_dims),
                                 "head": "mixture"})
        return result
